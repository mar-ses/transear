"""Contains BLS analysis routines."""

import sys
import time
import socket
import warnings
import numpy as np
import pandas as pd
from astropy.stats import mad_std
from scipy import interpolate
import matplotlib.pyplot as plt
import bls

from . import util_lib
from .util_lib import (mask_flares, mask_floor, mask_transits, fold_on,
                       calc_noise, _fold, MF, LightcurveError)
from .__init__ import HOME_DIR, K2GP_DIR


max_nb = 1900
min_light_curve = 4		# Minimum time period of a lightcurve in days
tce_threshold = 9


# ------------
#
# BLS Analysis
#
# ------------

# BLS result format.
# columns: f, period, power, power_0, power_1, power_2, power_3, ...
# power_i is the power of the ith subtracted BLS run.
#
# BLS peaks format
# columns = ['period', 'duration', 't0', 'depth',
# 			 'power', 'snr', 'valid_flag']
# valid_flag is to be False when depth is zero, or something,
# and can be set false retroactively.

def search_transits(t, f, num_searches=10, R_star=0.1, M_star=0.1,
                    P_min=0.5, P_max=None, nf_tol=3.0, nb_tol=3.0,
                    qms_tol=3.0, snr_threshold=None, ignore_invalid=False,
                    pr_test=True, max_runs=20):
    """Conducts a full search for transit signals.

    A table of periods is extracted, by subsequently masking
    the lightcurve at each transit period. For each peak,
    calculates the BLS power, as well as signal-to-noise ratio.

    TODO BUG: figure out why the BLS peaks are repeated sometimes
              and pevent it happening

    Args:
        t, f (array-like):
        num_searches (int): number of maskings to conduct
        R_star (float)
        M_star (float)
        P_min (float)
        nf_tol (float): frequency resolution
        nb_tol (float): bin resolution for eebls
        qms_tol (float)
        snr_threshold (float): TODO - Crossfield style search
            possibly turn into a specialised function
        ignore_invalid (bool): if True, invalid results don't count
            and are discarded directly, with the BLS not counting
            them as peaks or "searches"
        pr_test (bool): if True, performs a point removal test in
            determining the valid flag

    Returns: (bls_peaks, bls_results)
        bls_peaks (pd.DataFrame): contains the information of each peak.
            columns = ['period', 't0', 'duration', 'depth',
                       'power', 'snr', 'snr_0', 'valid']
        bls_results (pd.DataFrame): power functions of iterations
            columns = ['frequency', 'period', 'power', 'power_0',
                       'power_1', 'power_2', ...]
    """

    # Blocks the run if it would cause a FORTRAN error.
    if len(t) < 10:
        raise InvalidLightcurveError("Lightcurve has less then 10 points.")

    # For the masking component, better to destroy the pandas index.
    if isinstance(t, (pd.DataFrame, pd.Series)):
        t = t.values
    if isinstance(f, (pd.DataFrame, pd.Series)):
        f = f.values

    bls_peaks = pd.DataFrame(index=range(num_searches))
    bls_peaks = bls_peaks.assign(period=np.nan, t0=np.nan, duration=np.nan,
                                 depth=np.nan, power=np.nan, snr=np.nan)
    bls_peaks = bls_peaks.assign(valid_flag=False)

    bls_results = pd.DataFrame(columns=['frequency', 'period',
                                        'power', 'snr'])

    T = max(t) - min(t)
    if P_max is None:
        P_max = 0.1 + T/2.0

    split_info = get_split_info(T=T,
                                P_min=P_min,
                                P_max=P_max,
                                R_star=R_star,
                                M_star=M_star,
                                nf_tol=nf_tol,
                                nb_tol=nb_tol,
                                qms_tol=qms_tol)

    # BUG
    if pd.isnull(R_star) or pd.isnull(M_star):
        print("R_star or M_star are null in bls_tools.search_transits.")
        raise ValueError("R_star or M_star cannot be null in new-style "
                         "bls_tools.search_transits.")

    # i indexes through bls_peaks; starts at 0
    i = 0
    # ti only counts the number of runs, starts at 1
    ti = 1
    while (i < num_searches) and (ti <= max_runs):
        ti += 1

        # Rechecking in case too many points are deleted
        if len(t) < 10:
            # TODO: turn this into an actual and catchable warning
            warnings.warn("Lightcurve was excessively shortened.")
            break
        if (max(t) - min(t)) < min_light_curve:
            # TODO: turn this into an actual and catchable warning
            warnings.warn("Lightcurve was excessively shortened.")
            break
        if split_info is not None \
            and (split_info.fmin < 1 / (max(t) - min(t))).any():
            # This is quite problematic if it happens;
            # must find a way to address this directly, or raise
            # and error
            raise ValueError("Lightcurve got shortened, and now fmin "
                             "has too long a period.")
            break

        (blss, period, power, t0,
        duration, depth, split_info) = calc_smart_bls(t, f,
                                                      R_star=R_star,
                                                      M_star=M_star,
                                                      nf_tol=nf_tol,
                                                      nb_tol=nb_tol,
                                                      qms_tol=qms_tol,
                                                      split_info=split_info,
                                                      verbose=False)

        # Pre-validation
        if depth < 0:
            valid_flag = False
        elif pr_test and not point_removal_test(t, f, period, t0,
                                                duration, depth):
            valid_flag = False
        else:
            valid_flag = True

        # Mask the next array (before skipping)
        tmask = mask_transits(t, t0, period, MF*duration)

        # First and last point must not be removed; instead just subtract
        for idx in [0, -1]:
            if tmask[idx]:
                f[idx] = np.nanmedian(f)
                tmask[idx] = False

        t = t[~tmask]
        f = f[~tmask]

        # Don't note down invalid peaks (but they should be removed)
        if not valid_flag and ignore_invalid:
            continue

        bls_peaks.at[i, 'valid_flag'] = valid_flag

        # The flat power is not useful outside of this loop
        blss = blss.assign(flat_power=flatten_power_spectrum(
                                                        blss,
                                                        N_bins=40,
                                                        bin_by='period',
                                                        flatten_col='power'))
        bls_noise = mad_std(blss.flat_power)
        blss = blss.assign(snr=calc_sn(blss))

        # Note in bls_results
        if i == 0:
            bls_results = blss
        else:
            assert len(bls_results) == len(blss)
            assert (bls_results.frequency == blss.frequency).all()

        bls_results['power_{}'.format(i)] = blss['power']
        bls_results['snr_{}'.format(i)] = blss['snr']

        # Note in bls_peaks
        bls_peaks.at[i, 'period'] = period
        bls_peaks.at[i, 't0'] = t0
        bls_peaks.at[i, 'duration'] = duration
        bls_peaks.at[i, 'depth'] = depth
        bls_peaks.at[i, 'power'] = power
        bls_peaks.at[i, 'noise'] = bls_noise
        bls_peaks.at[i, 'snr'] = blss.loc[blss.period == period, 'snr'].iloc[0]

        # BUG TODO
        if bls_peaks.loc[i, 'noise'] == 0.0:
            print("Zero noise in bls spectrum (causes infinite SNR)!")
            raise ZeroBLSNoiseError("bls_peaks snr is nan. i = {}\n"
                                    "bls_peaks = {}".format(i, bls_peaks),
                                    bls_results=bls_results,
                                    bls_peaks=bls_peaks)
        elif not np.isfinite(bls_peaks.loc[i, 'snr']):
            # With the above, this case is obsolete, keep it for now
            print("Non-finite snr in bls_peaks:", bls_peaks)
            raise ValueError("bls_peaks snr is nan. i = {}\n"
                             "bls_peaks = {}".format(i, bls_peaks))

        assert np.isfinite(bls_peaks.loc[i, 'snr'])

        i += 1

    # Remove NaN peaks (residual if max_runs is reached)
    if ti > max_runs or bls_peaks.period.isnull().any():
        bls_peaks = bls_peaks[~bls_peaks.period.isnull()].copy()
        bls_peaks.index = range(len(bls_peaks))

    return bls_peaks, bls_results


# BLS work-functions
# ------------------

def calc_smart_bls(t, f, split_info=None, verbose=False, **split_kwargs):
    """Main work-function, calculates the bls spectrum of an array.

    Uses EEBLS, programmed by Dan Foreman-Mackey, see:
    https://github.com/dfm/python-bls
    From the algorithm by Kovacs Zucker & Mazeh (2002)

    For a possible improvement, see:
    https://github.com/hpparvi/PyBLS

    NOTE BUG: Current issue is that fractional_duration can be greater
              than the maximum fractional_duration (qma)

    Args:
        t (np.array-like): array of times
        f (np.array-like): array of fluxes (same len as t)
        split_info (pd.DataFrame): the information DataFrame
            controlling the runs. If not given, parameters must be
            input into split_kwargs
        verbose (bool): print information such as run-time, nf
        **split_kwargs: if split_info not given
            P_min, P_max, T, R_star, M_star, P_step,
            nf_tol, nb_tol, qms_tol

    Returns:
        bls_spectrum, best_period, t0, duration, depth, split_info
        bls_spectrum: pd.DataFrame with columns [frequency, period, power]
        split_info: the output values from the bls
                (best_period, best_power, depth,
                fractional_duration, in1, in2)

    Exceptions:
    """

    # This is a temporary fix and shouldn't be relied on
    if abs(np.nanmedian(f)) > 0.1:
        print("WARNING: f is not normalised with median at 0.0.",
              "Subtracting median. Median:", np.nanmedian(f))
        f = f - np.nanmedian(f)
    elif abs(np.nanmedian(f)) > 0.005:
        f = f - np.nanmedian(f)

    assert len(t) == len(f)

    # Prepare parameters
    # ------------------

    T = max(t) - min(t)
    if split_info is not None:
        split_info = split_info.copy()
    elif all(key in split_kwargs for key in ('R_star', 'M_star')):
        if 'P_min' not in split_kwargs:
            split_kwargs['P_min'] = 0.5
        if 'P_max' not in split_kwargs:
            split_kwargs['P_max'] = 0.5 + T/2

        split_info = get_split_info(T=T, **split_kwargs)
    else:
        raise ValueError("split_info not given, and split_kwargs doesn't "
                         "contain enough arguments: {}".format(split_kwargs))

    # Block the run if it would cause a FORTRAN error
    if len(t) < 10:
        raise InvalidLightcurveError("Lightcurve has less than 10 points.")

    if T < min_light_curve:
        raise ShortLightcurveError("Skipping shortened lightcurve "
                                   "(T = {}).".format(T))
    
    if (split_info.fmin < 1/T).any():
        raise InvalidLightcurveError("Max period (1/fmin) greater than "
                                    "lightcurve timescale for one of the "
                                    "splits.")

    if verbose:
        print("Total nf: {}\n".format(split_info.nf.sum()))

    # Perform the distributed BLS fitting
    # -----------------------------------

    # Holds the combined bls information
    bls_spectrum = pd.DataFrame(columns=['frequency', 'period', 'power'])

    for i in split_info.index:
        # The work arrays
        u = np.empty(len(t), dtype=float)
        v = np.empty(len(t), dtype=float)

        # Quick bullshit filter
        if split_info.loc[i, 'nf'] < 1:
            # TODO BUG: what to do here? make args None? Or what?
            # Will have to add a spectrum, if not a fake one
            # Ok, no spectrum addition needed
            print("negative nf")

        t_start = time.time()
        try:
            (power, best_period, best_power, depth,
            fractional_duration, in1, in2) = bls.eebls(
                                                t, f, u, v,
                                                nf=split_info.loc[i, 'nf'],
                                                fmin=split_info.loc[i, 'fmin'],
                                                df=split_info.loc[i, 'df'],
                                                nb=split_info.loc[i, 'nb'],
                                                qmi=split_info.loc[i, 'qmi'],
                                                qma=split_info.loc[i, 'qma'])
        except ValueError as e:
            print("ValueError when running bls.eebls.", split_info.loc[i])
            raise IntentValueError(
                "ValueError when running bls.eebls." + str(e),
                split_info.loc[i])
        t_end = time.time()

        split_info.loc[i, 'run_time'] = t_end - t_start
        split_info.loc[i, 'best_power'] = best_power
        split_info.loc[i, 'best_period'] = best_period
        split_info.loc[i, 'depth'] = depth
        split_info.loc[i, 'fractional_duration'] = fractional_duration
        split_info.loc[i, 'in1'] = in1
        split_info.loc[i, 'in2'] = in2

        # Package results
        frequencies = split_info.loc[i, 'fmin'] \
                + np.arange(split_info.loc[i, 'nf']) * split_info.loc[i, 'df']

        add_spectrum = pd.DataFrame({'frequency':frequencies,
                                     'period':1./frequencies,
                                     'power':power})

        bls_spectrum = bls_spectrum.append(add_spectrum, ignore_index=True)

    if verbose:
        print("Individual run times:", split_info.run_time)
        print("Total run time:", split_info.run_time.sum())

    # Order the total bls spectrum
    bls_spectrum.drop_duplicates(subset='frequency',
                                 keep='first',
                                 inplace=True)
    bls_spectrum.sort_values(by='frequency', inplace=True)
    bls_spectrum.index = range(len(bls_spectrum))

    # Select the strongest peak across the stitched spectrum
    imax = split_info.best_power.idxmax()

    best_period = split_info.loc[imax, 'best_period']
    best_power = split_info.loc[imax, 'best_power']
    depth = split_info.loc[imax, 'depth']
    fractional_duration = split_info.loc[imax, 'fractional_duration']
    in1 = split_info.loc[imax, 'in1']
    in2 = split_info.loc[imax, 'in2']
    nb = split_info.loc[imax, 'nb']

    # in1 is the bin index at the start of the transit
    # in2 is the bin index at the end of transit

    # Convert in1 and in2 to t0 and duration
    duration = fractional_duration * best_period
    t0 = min(t) + best_period * (in1 + in2)/(2*nb)

    # TODO: turn this into an automatic discarding of any 25% duration
    # dips **in split_info**, so that it never removes such a large fraction
    if fractional_duration >= 0.25:
        pd.set_option('display.max_rows', 10)
        pd.set_option('display.max_columns', 20)
        print(split_info, "\nduration = {}".format(fractional_duration),
              "\nsplit_kwargs = {}".format(split_kwargs))

    return (bls_spectrum, best_period, best_power,
            t0, duration, depth, split_info)


def calc_bls(t, f, nf=10000, nb=1500, verbose=False, **bls_kwargs):
    """Main work-function, calculates the bls spectrum of an array.

    Uses EEBLS, programmed by Dan Foreman-Mackey, see:
    https://github.com/dfm/python-bls
    From the algorithm by Kovacs Zucker & Mazeh (2002)

    For a possible improvement, see:
    https://github.com/hpparvi/PyBLS

    Args:
        t (np.array-like): array of times
        f (np.array-like): array of fluxes (same len as t)
        nf (int):
        nb (int):
        verbose (bool): print information such as run-time
        **bls_kwargs: can contain qmi, qma, fmin, fmax etc...

    Returns:
        df_result, best_period, t0, duration, depth, (bls_output**)
            bls_result: pd.DataFrame with columns [f, period, power]
            bls_output**: the output values from the bls
                (power, best_period, best_power, depth,
                fractional_duration, in1, in2)

    Exceptions:
    """

    # This is a temporary fix and shouldn't be relied on
    if np.nanmedian(f) < 0.1:
        f = f - 1.0

    # Blocks the run if it would cause a FORTRAN error.
    if len(t) < 10:
        raise InvalidLightcurveError("Lightcurve has less then 10 points.")

    # Prepare parameters.
    # Observation duration:
    T = max(t) - min(t)
    assert T > 0		# Try to catch this earlier
    # Default frequency limits:
    fmax = 2.			# 12h period is minimum/maximum
    fmin = 2./T			# Require two transits currently
    # TODO: What happens if fmin is too low? Does BLS raise an error?
    # YES.

    # Unpack kwargs and replace the main values
    if 'qmi' in bls_kwargs.keys(): qmi = bls_kwargs['qmi']
    if 'qma' in bls_kwargs.keys(): qma = bls_kwargs['qma']
    if 'fmin' in bls_kwargs.keys(): fmin = bls_kwargs['fmin']
    if 'fmax' in bls_kwargs.keys(): fmax = bls_kwargs['fmax']

    # Frequency step.
    df = (fmax - fmin)/nf
    # Transit ratios/durations:
    qmi = (0.5/24)*fmin 	# assume half hour transit at minimum frequency
    qma = 0.1

    # Run checks on the parameters.
    # Skip all lightcurves that are less than 4 days currently.
    if T < min_light_curve:
        raise ShortLightcurveError("Skipping shortened lightcurve (T = {}).".format(T))
    # fmin must be greater than fmax.
    if fmin > fmax:
        raise ValueError("fmin > fmax - Observation period may be less than a day.")
    assert len(t) == len(f)
    # Period (1/fmin) greater than T
    if fmin < 1/T:
        raise InvalidLightcurveError("Max period (1/fmin) greater than"
                                    "lightcurve timescale.")

    # Run the BLS
    # -----------
    # The work arrays.
    u = np.empty(len(t), dtype=float)
    v = np.empty(len(t), dtype=float)

    start_time = time.time()
    power, best_period, best_power, depth, fractional_duration, in1, in2 = bls.eebls(t, f, u, v, nf, fmin, df, nb, qmi, qma)
    run_time = time.time() - start_time

    # Brief explanation of output:
    # ----------------------------
    # best_period is best-fit period in same units as time
    # best_power is the fit at best_period
    # depth is the depth of transit at best_period
    # in1 is the bin index at the start of the transit
    # in2 is the bin index at the end of transit

    if verbose:
        print("BLS running time: {}".format(run_time))

    # Package results.
    frequencies = fmin + np.arange(nf) * df
    periods = 1./frequencies
    assert len(frequencies) == len(power)
    df_result = pd.DataFrame({'frequency':frequencies, 'period':periods, 'power':power})

    # Convert in1 and in2 to t0 and duration perhaps
    duration = fractional_duration * best_period
    t0 = min(t) + best_period * (in1 + in2)/(2*nb)

    #tt0 = 0

    return df_result, best_period, t0, duration, depth, (power, best_period, best_power, depth, fractional_duration, in1, in2)


# Argument and split utilites
# ---------------------------

# Aim is to be able to detect a transit at impact factor 0.1,
# in terms of period at least. Possibly with a tolerance factor.
# In other words, across a time baseline of N_days, transits at
# the beginning and end must be properly tiled with periods.

# num transits ~ N_days / P
# time difference in centers (at end and beginning) for dP = dP * N_days / P
# Tolerance: time_diff = 0.5 * duration at impact factor b = 0.1

def get_df(P, N_days, R_star, M_star, b=0.0, tol=1.0):
    """Calculates the df required INSIDE the range fmin, fmax.

    Args:
        fmin, fmax (float)
        P (float) [d]
        N_days (float)
        R_star (float) [R_sun]
        M_star (float) [M_sun]
        b (float): default 0.5
        tol (float): divide minimum duration by tol, default 1.0
    """

    dur = util_lib.get_duration(P, R_star, util_lib.get_A(P, M_star), b=b)
    N_transits = N_days / P
    dP_max = (dur / tol) / N_transits

    df = 1/P - 1/(P + dP_max)

    return df

def get_nf(fmin, fmax, *args, **kwargs):
    """Calculates the nf required INSIDE the range fmin, fmax.

    Args:
        fmin, fmax (float)
        P (float) [d]
        N_days (float)
        R_star (float) [R_sun]
        M_star (float) [M_sun]
        b (float): default 0.5
        tol (float): divide minimum duration by tol, default 1.0
    """

    df = get_df(*args, **kwargs)

    nf = ((fmax - fmin) / df)

    if not hasattr(nf, '__len__'):
        return int(nf) + 1
    else:
        return nf.astype(int) + 1

def get_nb(P, R_star, M_star, b=0.0, tol=1.0):
    """Return nb, so that bin width is half a minimum duration.

    NOTE: it doesn't have an upper boundary. Larger than 2000 causes
    error in bls.

    Args:
        P [d], R_star [R_sun], M_star [M_sun], b (0.0 is head on)
        tol (float): divide duration by this number
    """

    dur = util_lib.get_duration(P, R_star, util_lib.get_A(P, M_star), b=b)
    nb = (P / (0.5 * dur / tol))
    if not hasattr(nb, '__len__'):
        return int(nb)
    else:
        return nb.astype(int)

def get_qms(P_1, P_2, R_star, M_star, tol=1.0):
    """Gets the minimum and maximum durations (in qma qmi)

    Uses b=0.5, and b=0.0 as limits too.

    qma and qmi are in duration / period.

    NOTE: very strongly suggested to use tol != 1.0, i.e 3.0+

    Args:
        P_1, P_2 (float): lower and upper periods
        R_star, M_star (float): [R_sun] [M_sun]
        tol (float): divides lower duration, multiplies highest
    """

    # To ensure correct ordering
    if P_2 < P_1:
        temp = P_2
        P_2 = P_1
        P_1 = temp

    higher_duration = util_lib.get_duration(P_2,
                                            R_star=R_star,
                                            A=util_lib.get_A(P_2, M_star),
                                            b=0.0)
    lower_duration = util_lib.get_duration(P_1,
                                            R_star=R_star,
                                            A=util_lib.get_A(P_1, M_star),
                                            b=0.5)

    assert lower_duration < higher_duration

    return (lower_duration/(P_2*tol), tol*higher_duration/(P_1))

def get_split_info(P_min, P_max, T, R_star, M_star,
                   P_step=0.5, nf_tol=3.0, nb_tol=3.0,
                   qms_tol=3.0):

    si = pd.DataFrame()

    for i, p_min in enumerate(np.arange(P_min,
                                        P_max - 0.5,
                                        P_step)):
        si.at[i, 'P_min'] = p_min
        p_max = si.at[i, 'P_max'] = p_min + P_step
        si.at[i, 'fmin'] = max(1.0 / si.loc[i, 'P_max'],
                               1.0 / (T - 0.01))
        si.at[i, 'fmax'] = 1.0 / p_min

        # inverted fmax, fmin error
        if si.at[i, 'fmin'] > si.at[i, 'fmax']:
            # Actually deal with it
            # if this happens, it means T is blocking P_max from going higher
            # This means that fmin can't go so low, and we should stop
            # the loop and leave si as it is
            #
            # BUG
            print(" f_min inversion.\n", '-'*40)
            print(" ".join(["Inversion error,",
                            "P_min = {}, P_max = {},".format(p_min, p_max),
                            "fmin = {},".format(si.at[i, 'fmin']),
                            "fmax = {}".format(si.at[i, 'fmax']),
                            "T = {}".format(T)]))

            si.drop(i, axis='index', inplace=True)
            break

            # BUG: flagging (will never get here)
            # TODO: This part is obsolete.
            # pd.set_option('display.max_columns', 30)
            # pd.set_option('display.max_rows', len(si))
            # err_str = " ".join(["Inversion error,",
            #                     "P_min = {}, P_max = {},".format(p_min, p_max),
            #                     "fmin = {},".format(si.at[i, 'fmin']),
            #                     "fmax = {}".format(si.at[i, 'fmax']),
            #                     "T = {}".format(T)])
            # print(err_str)
            # raise InvertedLimitsError(err_str, si)

        si.at[i, 'nf'] = get_nf(P=p_max,
                                R_star=R_star,
                                M_star=M_star,
                                N_days=T,
                                tol=nf_tol,
                                b=0.0,
                                **si.loc[i, ['fmin', 'fmax']])
        si.at[i, 'df'] = (si.loc[i, 'fmax'] - si.loc[i, 'fmin']) / si.loc[i, 'nf']

        calculated_nb = get_nb(p_max,
                               R_star=R_star,
                               M_star=M_star,
                               b=0.0,
                               tol=nb_tol)
        si.at[i, 'nb'] = min(max_nb, calculated_nb)

        qmi, qma = get_qms(p_min, p_max,
                           R_star=R_star, M_star=M_star,
                           tol=qms_tol)
        si.loc[i, 'qmi'] = qmi
        si.loc[i, 'qma'] = qma

    if not (si['fmax'] > si['fmin']).all() or (si['nf'] < 0.0).any():
        pd.set_option('display.max_columns', 30)
        pd.set_option('display.max_rows', len(si))
        raise InvertedLimitsError(("The fmax/fmin limits have been"
                                   "inverted."), si)

    # Enforce maximum qma
    max_qma = max(si.qma) / qms_tol
    si.loc[si.qma > (max(si.qma) / qms_tol), 'qma'] = max_qma

    # The following must never be true, otherwise a major mistake was made
    assert (qms_tol*si.qmi < si.qma).all()

    return si

# For the above then, as P increases df decreases
# BLS procedure: - divide full period range into sections,
#				 - for each section, take the highest period, calculate df
#				   and determine nf for the period range

# NOTE for the above: most of the 

# Also have an esimator (or something at the beginning) to print out
# the actual total resolution, just in case it gets much larger than 50,000

# Another thing to consider: each BLS section will have its own peak output,
# unless I want to also manual deal with it (no), I need to be able
# to combine the peak outputs, and determine the largest of them.
# Could be as easy as just taking max(power)

# TODO: check again Vanderburg et al and Dressing and Charbonneau if
# they are doing the same


# BLS Validation and peak analysis
# --------------------------------

# TODO: change these to BLS-specific validation

def validate_peaks(bls_peaks, lcf=None, bls_results=None):
    """Combined function that assigns valid_flag to a BLS peak.
    """

    pass

def point_removal_test(t, f, period, t0, duration, depth,
                       num_points=2, change_fraction=0.5):
    """Checks if removing the lowest point(s) changes the depth.

    Args:
        lcf
        period,
        t0,
        depth,
        num_points (int, 2): number of minimum points to remove
        change_fraction (float, 0.5): if depth changes by more
            than this, return false.

    Returns:
        pass_flag: if True, removing the lowest points doesn't
            change the depth too much; if False, the depth
            changed by more than change_fraction*depth.
    """

    if np.nanmedian(f) > 0.1:
        f = f - np.nanmedian(f)
    else:
        f = f.copy()

    f_transit = f[mask_transits(t, t0, period, duration)]

    # Check to make sure f_transit has enough points
    if len(f_transit) <= num_points:
        return False

    depth_0 = - f_transit.mean()
    depth_1 = - f_transit[np.argsort(f_transit)][num_points:].mean()

    if abs((depth_1 - depth_0) / depth_0) > change_fraction:
        return False
    else:
        return True


# Work functions
# --------------

def calc_peak_sed(peak_index, power):
    """Calculates the SED for a single peak.

    Args:
        peak_index (index/array): the spectrum frequency at which
            to evaluate the SED.
        power (np.ndarray-like): the BLS power spectrum.
    """

    pass

def calc_sed(power):
    """Calculate the SED from the power spectrum.

    Expects a **flattened** power spectrum

    SED =

    Args:
        power (np.ndarray-like): the power spectrum (ideally flattened)
    """

    pass

def calc_sn(br, **fps_kwargs):
    """Calculates the S/N as defined in Vanderburg et al (2016).

    S/N = BLS / (MAD(BLS) * 1.4826

    Expects a **flattened** power spectrum.

    Args:
        power (np.ndarray-like): the power spectrum (ideally flattened)

    Returns:
        S/N BLS spectrum
    """

    flat_power = flatten_power_spectrum(br, **fps_kwargs)
    bls_mad = mad_std(flat_power)
    return flat_power / bls_mad

def flatten_power_spectrum(br, N_bins=None, bin_by='period', flatten_col='power'):
    """Flattens the BLS power spectrum by subtraction.

    Intended to bin by period.

    Args:
        br (pd.DataFrame): requires a full standard bls_results.
        N_bins (int): resolution of bins to use for the spline.
        bin_by (str): column in bls_results to use for the uniform
            spacing in bins. Default: 'period'.
        flatten_col (str): column in bls_results to flatten (power).

    Returns:
        flat_power (np.ndarray):
    """

    if N_bins is None:
        N_bins = min(40, int(len(br)//50))
    if N_bins <= 4:
        return br[flatten_col]

    # The bin bounds
    bb = np.linspace(min(br[bin_by]), max(br[bin_by]), N_bins+1)
    bin_medians = np.empty(N_bins, dtype=float)
    loc_medians = np.empty(N_bins, dtype=float)

    for i in range(N_bins):
        bin_medians[i] = np.median(br[(br[bin_by] >= bb[i]) & (br[bin_by] < bb[i+1])][flatten_col])
        loc_medians[i] = np.median(br[(br[bin_by] >= bb[i]) & (br[bin_by] < bb[i+1])][bin_by])

    # Setup the interpolator
    interpolator = interpolate.interp1d(loc_medians, bin_medians, kind='quadratic', fill_value='extrapolate')

    flat_power = br[flatten_col] - interpolator(br[bin_by])

    return flat_power


# Specific visualization tools
# ----------------------------

def plot_bls(bls_result, bls_col='power'):
    """Not empty."""

    plt.plot(bls_result.frequency, bls_result[bls_col], 'k-')
    plt.show()


# Utilities and errors
# --------------------

class ShortLightcurveError(LightcurveError):
    """Lightcurve is less than min_light_curve days long."""
    pass

class InvalidLightcurveError(LightcurveError):
    """Lightcurve has too many points and cannot be searched."""
    pass

class InvertedLimitsError(ValueError):
    """Some type in intent-error in the eebls part."""

    def __init__(self, text, split_info, *args, **kwargs):
        self.split_info = split_info
        super().__init__(text, *args, **kwargs)

class IntentValueError(InvertedLimitsError):
    """Some type in intent-error in the eebls part."""

    pass

class ZeroBLSNoiseError(ValueError):
    """When the noise in the BLS is zero, or the snr is infinity."""

    def __init__(self, text, bls_results, bls_peaks=None, *args, **kwargs):
        self.bls_results = bls_results
        self.bls_peaks = bls_peaks
        super().__init__(text, *args, **kwargs)


# Testing routines
# ----------------

def test_single_bls():
    """Does a BLS on the TRAPPIST lightcurve."""
    from .analysis import highlight_bls_signal

    trappist_loc = "{}/trappist_files/k2gp200164267-c12-detrended-pos.tsv".format(K2GP_DIR)
    lcf = pd.read_csv(trappist_loc, sep='\t')
    lcf = lcf[~mask_flares(lcf.f_detrended)]
    lcf = lcf[~mask_floor(lcf.f_detrended)]

    # Old procedure

    bls_result, best_period, t0, duration, depth, bls_output = calc_bls(t=lcf.t, f=lcf.f_detrended, nf=50000, nb=1500, flims=None, verbose=True)
    highlight_bls_signal(lcf.t, lcf.f_detrended, t0=t0, period=best_period, duration=duration, depth=depth, plot=False)
    mask = mask_transits(lcf.t, t0, best_period, 3*duration)
    lcf = lcf[mask]
    highlight_bls_signal(lcf.t, lcf.f_detrended, t0=t0, period=best_period, duration=duration, depth=depth, plot=False)
    plot_bls(bls_result)

    # New procedure

    bls_result, best_period, best_power, t0, duration, depth, split_info = calc_smart_bls(t=lcf.t, f=lcf.f_detrended, verbose=True)
    highlight_bls_signal(lcf.t, lcf.f_detrended, t0=t0, period=best_period, duration=duration, depth=depth, plot=False)
    mask = mask_transits(lcf.t, t0, best_period, 3*duration)
    lcf = lcf[mask]
    highlight_bls_signal(lcf.t, lcf.f_detrended, t0=t0, period=best_period, duration=duration, depth=depth, plot=False)
    plot_bls(bls_result)

    return bls_result, best_period, t0, duration, bls_output

def test_multi_bls():
    """Does a BLS on the TRAPPIST lightcurve."""
    from .analysis import highlight_bls_signal

    trappist_loc = "{}/trappist_files/k2gp200164267-c12-detrended-pos.tsv".format(K2GP_DIR)
    lcf = pd.read_csv(trappist_loc, sep='\t')
    lcf = lcf[~mask_flares(lcf.f_detrended)]
    lcf = lcf[~mask_floor(lcf.f_detrended)]

    bls_peaks, bls_results = search_transits(lcf.t, lcf.f_detrended, num_searches=5)

    import pdb; pdb.set_trace()

    for i in bls_peaks.index:
        highlight_bls_signal(lcf.t, lcf.f_detrended, bls_peaks.loc[i, 't0'], bls_peaks.loc[i, 'period'], bls_peaks.loc[i, 'duration'], bls_peaks.loc[i, 'depth'])

    return bls_peaks, bls_results

def test_resolution_setting(star_type='mdwarf'):
    if star_type == 'mdwarf':
        Ms, Rs = 0.1, 0.1
    else:
        Ms, Rs = 1.0, 1.0


    P = np.linspace(0.5, 40.0, 200)

    fig, ax = plt.subplots(3)

    ax[0].plot(1/P, get_nf(2.0/80, 2.0, P=P, N_days=80, R_star=Rs, M_star=Ms, tol=1.0), 'k-')
    ax[0].plot(1/P, get_nf(2.0/80, 2.0, P=P, N_days=80, R_star=Rs, M_star=Ms, tol=2.0), 'b-')
    ax[0].plot(1/P, get_nf(2.0/80, 2.0, P=P, N_days=80, R_star=Rs, M_star=Ms, tol=5.0), 'r-')

    ax[1].plot(P, get_nf(2.0/80, 2.0, P=P, N_days=80, R_star=Rs, M_star=Ms, tol=1.0), 'k-')
    ax[1].plot(P, get_nf(2.0/80, 2.0, P=P, N_days=80, R_star=Rs, M_star=Ms, tol=2.0), 'b-')
    ax[1].plot(P, get_nf(2.0/80, 2.0, P=P, N_days=80, R_star=Rs, M_star=Ms, tol=5.0), 'r-')

    ax[2].plot(P, get_nb(P, Rs, Ms, tol=1.0), 'k-')
    ax[2].plot(P, get_nb(P, Rs, Ms, tol=2.0), 'b-')
    ax[2].plot(P, get_nb(P, Rs, Ms, tol=5.0), 'r-')

    plt.show()

def test_qm_setting(star_type='mdwarf'):
    if star_type == 'mdwarf':
        Ms, Rs = 0.1, 0.1
    else:
        Ms, Rs = 1.0, 1.0


    P = np.linspace(0.3, 40.0, 200)

    fig, ax = plt.subplots(2)

    ax[0].plot(P, get_qms(P, 1.0, Rs, Ms, tol=1.0)[1], 'k-')
    ax[1].plot(P, get_qms(P, 1.0, Rs, Ms, tol=1.0)[1]*P*24, 'k-')

    plt.show()






# python-bls usage instructions
# -----------------------------
#
# import bls
# results = bls.eebls(time, flux, u, v, nf, fmin, df, nb, qmi, qma)

# where

#     time is an N-dimensional array of timestamps for the light curve,
#     flux is the N-dimensional light curve array,
#     u and v are N-dimensional empty work arrays,
#     nf is the number of frequency bins to test,
#     fmin is the minimum frequency to test,
#     df is the frequency grid spacing,
#     nb is the number of bins to use in the folded light curve,
#     qmi is the minimum transit duration to test, and
#     qma is the maximum transit duration to test.

# The returned values are

# power, best_period, best_power, depth, q, in1, in2 = results

# where

#     power is the nf-dimensional power spectrum array at frequencies f = fmin + arange(nf) * df,
#     best_period is the best-fit period in the same units as time,
#     best_power is the power at best_period,
#     depth is the depth of the transit at best_period,
#     q is the fractional transit duration,
#     in1 is the bin index at the start of transit, and
#     in2 is the bin index at the end of transit.
