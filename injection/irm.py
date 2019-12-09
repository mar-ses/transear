"""Library for injection modelling of transits.

Contains:
    - injection of transit from observational or
    physical parameters
    - all the stages of recovery

Also uses a Injection object; which can produce its own
batman params easily as a method if required. This is what's
passed around (also perhaps does comparison are reading from
bls output/tf_search output (?).

Also, in all cases the transit fitting results in the form of
p_fit will be passed around and return from the stages.
"""

import sys
import socket
from collections import OrderedDict

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

from .injection_model import InjectionModel, HPNotFoundError
from .signal_validation import (check_parameter_match, validate_signal,
                                find_signal, pre_filter_transit)
from .signal_validation import p_fit_NullError

from ..__init__ import HOME_DIR
from .. import tf_tools, bls_tools, util_lib
from ..bls_tools import IntentValueError, InvertedLimitsError

sys.path.append(HOME_DIR)
from k2gp import k2gp, lcf_tools, gp_tools
from k2gp_dist.__init__ import HP_LOC

# Global variables
tf_snr_cutoff = 5.0

# Keywords for bls and tf routines
bls_keys = ('num_searches', 'P_min', 'P_max', 'nf_tol', 'nb_tol',
            'qms_tol', 'ignore_invalid', 'pr_test', 'max_runs')
tf_keys  = ('bin_type', 'bin_res', 'subtract_results', 'adjust_res',
            'freeze_a', 'overlap_lim')

bls_defaults = {'num_searches':10, 'P_min':0.5, 'P_max':20,
                'ignore_invalid':True, 'pr_test':True,
                'max_runs':20}
tf_defaults = {'bin_type':'regular', 'bin_res':6, 'subtract_result':False,
               'adjust_res':True, 'freeze_a':True, 'overlap_lim':'full'}

# TODO NOTE NOTE NOTE NOTE NOTE NOTE NOTE:
# Super important: A is not a valid parameter to inject transit with;
# rather derive it from the period and M_star (also eccentricity).

# TODO
# Also take care of what happens when t0 is more than 1 period from
# start of lcf when injecting a signal



# Bulk injection/recovery routines
# --------------------------------

def	full_recover(lcf, injection_model, f_col='f_detrended',
                 cascade_failure=True, perform_tf=False,
                 perform_bls=False, perform_dt=True,
                 snr_source='snr_estimate', snr_lim=tf_snr_cutoff,
                 pre_filter=False, **kwargs):
    """Performs all 3 stages of recovery.

    Does the recovery backwards, if a stage is failed,
    subsequent ones are also taken as failed.

    Args:
        lcf (pd.DataFrame)
        injection_model (injection.injection_model.InjectionModel)
        cascade_failure (bool): if True, upon stage failure,
            all subsequent stages are also taken as failed,
            default = True
        **kwargs (dict): for both bls fitting and tf fitting
            dt_kwargs: proc_kw, full_final
            tf_kwargs: num_searches, P_min, P_max, nf_tol, nb_tol,
                qms_tol, ignore_invalid, pr_test, max_runs, f_err
            bls_kwargs: bin_type, bin_res, subtract_results,
                adjust_res, freeze_a, overlap_lim

    Returns:
        tf_flag, bls_flag, dt_flag (booleans), tf_snr:
            For each stage, gives a flag if the signal
            was recovered. Also returns the snr of the
            final recovered signal (if found, else np.nan)
    """

    result = pd.Series({'tf_flag':False, 'bls_flag':False,
                        'dt_flag':False, 'tf_snr':np.nan,
                        'tf_snr_estimate':np.nan,
                        'tf_snr_fit':np.nan})

    if 'f_err' in kwargs.keys():
        f_err = kwargs['f_err'] 
    else:
        f_err = util_lib.calc_noise(lcf[f_col])

    # It's better to write an snr_estimate anyway,
    # so later comparison can be done
    initial_snr = util_lib.estimate_snr(depth=injection_model['depth'],
                                        per=injection_model['per'],
                                        t_base=(max(lcf.t) - min(lcf.t)),
                                        duration=injection_model['duration'],
                                        signoise=f_err,
                                        t_cad=np.nanmedian(np.diff(lcf.t)))

    # TODO: figure this out properly; perhaps I shouldn't do this
    # Basically, is the estimate returned even when the signal
    # has an SNR that's too low? If so, don't put the predicted/initial
    # snr into the estimate
    result['tf_snr_estimate'] = initial_snr
    result['tf_snr_predicted'] = initial_snr

    if pre_filter and not (initial_snr > snr_lim-1.0):
        return result

    # Pre-filters transit and returns False/np.nan if it doesn't pass
    # if pre_filter and not pre_filter_transit(injection_model,
    #                                          lcf.t, lcf[f_col],
    #                                          snr_cutoff=snr_lim - 1.0):
    #     # BUG: testing to see if filter rejects everything
    #     print("Filter failed.")
    #     return result

    if perform_tf:
        result['tf_flag'], p_fit = stage_tf(lcf,
                                            injection_model,
                                            snr_lim=snr_lim,
                                            snr_source=snr_source,
                                            randomise=True,
                                            f_col=f_col,
                                            **(kwargs.copy()))

        if result['tf_flag'] or p_fit is not None:
            # Write the snr even if it's not "found", i.e above the threshold,
            # if the tf routine was actually fitting the correct signal
            result['tf_snr'] = p_fit[snr_source]
            result['tf_snr_estimate'] = p_fit['tf_snr_estimate']
            result['tf_snr_fit'] = p_fit['tf_snr_fit'] \
                                   if 'tf_snr_fit' in p_fit.index \
                                   else np.nan
    else:
        result['tf_flag'] = True

    if perform_bls and (result['tf_flag'] or not cascade_failure):
        result['bls_flag'], _, p_fit = stage_bls(lcf,
                                                injection_model,
                                                snr_lim=snr_lim,
                                                snr_source=snr_source,
                                                f_col=f_col,
                                                **(kwargs.copy()))

        if result['bls_flag'] or p_fit is not None:
            # Write the snr even if it's not "found", i.e above the threshold,
            # if the tf routine was actually fitting the correct signal
            result['tf_snr'] = p_fit[snr_source]
            result['tf_snr_estimate'] = p_fit['snr_estimate']
            result['tf_snr_fit'] = p_fit['snr_fit'] \
                                   if 'snr_fit' in p_fit.index \
                                   else np.nan
    elif not perform_bls:
        result['bls_flag'] = result['tf_flag']

    if perform_dt and (result['bls_flag'] or not cascade_failure):
        result['dt_flag'], _, p_fit, _ = stage_dt(lcf,
                                                  injection_model,
                                                  snr_lim=snr_lim,
                                                  snr_source=snr_source,
                                                  **(kwargs.copy()))

        if result['dt_flag'] or p_fit is not None:
            # Write the snr even if it's not "found", i.e above the threshold,
            # if the tf routine was actually fitting the correct signal
            result['tf_snr'] = p_fit[snr_source]
            result['tf_snr_estimate'] = p_fit['snr_estimate']
            result['tf_snr_fit'] = p_fit['snr_fit'] \
                                   if 'snr_fit' in p_fit.index \
                                   else np.nan
    else:
        result['dt_flag'] = np.nan

    # If the stage is not performed, it's given a fake True value during
    # the recovery to allow the next stage to run. However in the return
    # value; this must be corrected to np.nan in such cases.
    result['tf_flag'] = np.nan if not perform_tf else result['tf_flag']
    result['bls_flag'] = np.nan if not perform_bls else result['bls_flag']

    return result

def	recover_injection(lcf, P, R_p, R_star, M_star, t0=None, inc=None,
                      snr_lim=tf_snr_cutoff, snr_source='snr_estimate',
                      pre_filter=False, **kwargs):
    """Injects a signal and performs all 3 stages of recovery.

    NOTE: should accept the minimal number of parameters.

    Args:
        lcf (pd.DataFrame)
        P, R_p, R_star, M_star (floats): the transit parameters
        t0 (float): if t0 is None, it will be randomly picked
        inc (variable): if a float, will set as the inclination,
            otherwise will randomly draw an inclination < 90.0,
            where it enforces either 'full' overlap, 'half' overlap,
            or 'none' overlap; if inc=None, inc='none' is the default
        **kwargs (dict): passed to full_recover, and not to the
            injection_model.
            To include: cascade_failure, f_err, ... NOT f_col

    Returns:
        tf_flag, bls_flag, dt_flag, tf_snr
    """

    if t0 is None:
        t0 = np.random.rand()*P + min(lcf.t)

    # Substitute in the default if they aren't in already.
    for key in tf_defaults.keys():
        if key not in kwargs:
            kwargs[key] = tf_defaults[key]
    for key in bls_defaults.keys():
        if key not in kwargs:
            kwargs[key] = bls_defaults[key]

    lcf = lcf[['t', 'x', 'y', 'f', 'f_detrended', 'cadence']].copy()

    plot = True if socket.gethostname() == 'telesto' else False

    injection_model = InjectionModel(lcf.t.values, P, t0, R_p,
                                     R_star=R_star, M_star=M_star,
                                     bin_res=20, adjust_res=True)

    # Set the inclination
    if not isinstance(inc, float):
        injection_model.randomise_inclination(code=inc)
    else:
        injection_model['inc'] = inc

    lcf = injection_model.inject_transit(lcf,
                                         f_col=['f', 'f_detrended'],
                                         plot=plot)

    return full_recover(lcf, injection_model,
                        f_col='f_detrended',
                        snr_lim=snr_lim,
                        pre_filter=pre_filter,
                        **kwargs)


# Recovery staging
# ----------------

def stage_tf(lcf, injection_model, f_col='f_detrended', randomise=True,
             snr_source='snr', snr_lim=tf_snr_cutoff, **tf_kwargs):
    """Fits the transit to check if the SNR is high enough.

    TODO: choose bin-type and maybe resolution from the 
    contents of injection_model. IMPORTANT.

    Args:
        lcf (pd.DataFrame)
        injection_model (injection.injection_model.InjectionModel)
        f_col (str): which column in lcf contains the flux and
            the injected transit, default: 'f_detrended'
        randomise (bool): if True, jiggles the parameters in
            injection_model a little bit
        **tf_kwargs (dict): to include: bin_type, bin_res
            bin_type out of: ('none', 'regular', 'nearby', ...)

    Returns:
        validation_flag, p_fit
    """

    lcf = lcf.copy()
    lcf = util_lib.prep_lightcurve(lcf)

    f = lcf[f_col].values
    t = lcf['t'].values

    # Default is no-binning at the moment
    if 'bin_type' not in tf_kwargs:
        tf_kwargs['bin_type'] = 'none'

    tf_keys = tf_kwargs.keys()
    tf_kwargs = {key:tf_kwargs[key] for key in tf_keys if key in tf_kwargs}

    if randomise:
        bls_params = injection_model.jiggle_params(return_in='bls')

    p_fit, _, _, _ = tf_tools.fit_single_transit(
                                        t, f,
                                        R_star=injection_model['R_star'],
                                        M_star=injection_model['M_star'],
                                        **tf_kwargs, **bls_params)

    if snr_source=='snr':
        (_, _, p_fit['snr'],
        p_fit['mcmc_flag']) = tf_tools.sample_transit(
                                        t, f,
                                        R_star=injection_model['R_star'],
                                        M_star=injection_model['M_star'],
                                        **tf_kwargs,
                                        **bls_params)

    validation_flag = validate_signal(p_fit, injection_model,
                                      snr_source=snr_source,
                                      snr_lim=snr_lim)

    return validation_flag, p_fit

def stage_bls(lcf, injection_model, snr_lim=tf_snr_cutoff,
              snr_source='snr', f_col='f_detrended', **kwargs):
    """Fits the transit to check if the SNR is high enough.

    TODO: choose bin-type and maybe resolution from the 
    contents of injection_model. IMPORTANT.

    Args:
        lcf (pd.DataFrame)
        injection_model (injection.injection_model.InjectionModel)
        f_col (str): which column in lcf contains the flux and
            the injected transit, default: 'f_detrended'
        **kwargs (dict): for both bls fitting and tf fitting
            tf_kwargs: bin_type, bin_res, subtract_results
            bls_kwargs: num_searches, nf, nb, qmi, qma, fmin, fmax

    Returns:
        validation_flag, bls_peaks, validated_peak (p_fit-like)
    """

    lcf = lcf.copy()
    lcf = util_lib.prep_lightcurve(lcf)

    f = lcf[f_col].values
    t = lcf['t'].values

    # Extract and separate keyword arguments
    bls_keys = bls_defaults.keys()
    bls_kwargs = {key:kwargs[key] for key in bls_keys if key in kwargs}

    tf_keys = tf_defaults.keys()
    tf_kwargs = {key:kwargs[key] for key in tf_keys if key in kwargs}

    bls_peaks, _ = bls_tools.search_transits(t, f - 1.0, **bls_kwargs)

    bls_peaks = bls_peaks[bls_peaks.valid_flag]

    # BUG TODO
    try:
        bls_peaks = tf_tools.fit_transits(t, f, bls_peaks, calc_snr=False,
                                          R_star=injection_model['R_star'],
                                          M_star=injection_model['M_star'],
                                          **tf_kwargs)
    except TypeError:
        import pdb; pdb.set_trace()

    # Now validate the correct peak in bls_peaks
    dflag, _, idx = find_signal(bls_peaks, injection_model)

    if dflag:
        # Do a proper single-transit fit on the found period
        p_initial = bls_peaks.loc[idx, ['period', 't0',
                                        'depth', 'duration']]
        p_fit, _, = tf_tools.fit_single_transit(
                                        t, f,
                                        R_star=injection_model['R_star'],
                                        M_star=injection_model['M_star'],
                                        **tf_kwargs,
                                        **p_initial)

        if snr_source=='snr':
            (_, _, p_fit['snr'], \
                p_fit['mcmc_flag']) = tf_tools.sample_transit(
                                        t, f,
                                        R_star=injection_model['R_star'],
                                        M_star=injection_model['M_star'],
                                        **tf_kwargs,
                                        **p_initial)       
     
        flag = validate_signal(p_fit, injection_model, snr_lim=snr_lim,
                               snr_source=snr_source)

        return flag, bls_peaks, p_fit
    else:
        return False, bls_peaks, None

def stage_dt(lcf, injection_model, snr_lim=tf_snr_cutoff, snr_source='snr',
             force_classic=False, skip_optimisation=False, **kwargs):
    """Detrend lightcurve and perform transit_search.

    The target_list entry should be emptied into **kwargs,
    containing information such as full_final, proc_kw etc...
    Example:
        stage_dt(lcf, injection_model, **tl.loc[i, ['dt_pv_0', ...]])

    NOTE: lightcurve is expected to be initialised and cleaned properly

    NOTE: performs the detrending on f; producing standard structure
        lcf

    Args:
        lcf (pd.DataFrame)
        injection_model (injection.injection_model.InjectionModel)
        force_classic (bool): if True, prevents quasiperiodict fit
        skip_optimisation (bool): if True, tries to skip the detrending by
            looking up the hyperparameters
        epic (int): MUST be given if skip_optimisation = True
        **kwargs (dict): for both bls fitting and tf fitting
            'epic' (int): must be given if skip_optimisation=True
            dt_kwargs: proc_kw, full_final
            tf_kwargs: bin_type, bin_res, subtract_results
            bls_kwargs: num_searches, nf, nb, qmi, qma, fmin, fmax

    Returns:
        validation_flag, bls_peaks, validated_peak, lcf
    """

    lcf = lcf.copy()

    # Remove dt_ values which are expected from target_list entry
    for key in kwargs:
        if key.startswith('dt_'):
            kwargs[key[3:]] = kwargs.pop(key)

    dt_keys = ('proc_kw', 'full_final', 'ramp')
    dt_kwargs = {key:kwargs[key] for key in dt_keys if key in kwargs}

    # Perform the detrending
    # ----------------------
    skip_optimisation = False if 'epic' not in kwargs else skip_optimisation

    if skip_optimisation:
        try:
            epic = kwargs.pop('epic')
            campaign = kwargs.pop('campaign')
            flux_source = kwargs['flux_source']
            hpdf = pd.read_pickle(HP_LOC)
            hpd = hpdf[(hpdf.epic == int(epic)) \
                     & (hpdf.campaign == int(campaign)) \
                     & (flux_source == hpdf.dt_flux_source)].iloc[0]
        except (FileNotFoundError, IndexError):
            skip_optimisation = False
            if not socket.gethostname() == 'telesto':
                raise HPNotFoundError()

        if not hpd['dt_flag']:
            skip_optimisation = False
        elif force_classic  and hpd['dt_kernel'] == 'qp':
            skip_optimisation = False

    if skip_optimisation:
        lcf = skip_detrend(lcf, hp_data=hpd)
    else:
        lcf = full_detrend(lcf, force_classic, dt_kwargs, **kwargs)

    # Find and validate the transit
    # -----------------------------

    validation_flag, bls_peaks, validated_peak = stage_bls(
        lcf=lcf, injection_model=injection_model, snr_lim=snr_lim,
        snr_source=snr_source, f_col='f_detrended', **kwargs)

    return validation_flag, bls_peaks, validated_peak, lcf


# Detrending work functions
# -------------------------

def skip_detrend(lcf, hp_data):
    """Peform a detrending without optimization.

    Args:
        lcf (pd.DataFrame)
        hp_data (pd.Series): a row of the hp_table
    """

    hp = hp_data.hp
    pv = hp_data.dt_pv
    p_flag = hp_data.dt_kernel in ('qp', 'quasiperiodic')

    if not 'dt_model' in hp_data.index or pd.isnull(hp_data.dt_model):
        model_kw = 'y_offset'
    else:
        model_kw = hp_data.dt_model

    #ramp_flag = hp_data.dt_ramp if 'dt_ramp' in hp_data.index else True

    if p_flag:
        kernel = gp_tools.QuasiPeriodicK2Kernel(pv)
    else:
        kernel = gp_tools.ClassicK2Kernel()

    k2_detrender = gp_tools.K2Detrender(lcf, kernel, model=model_kw)
    # k2_detrender.set_hp(np.array(hp.values()), include_frozen=True)

    # BUG: this should be temporary because not all are in the same format
    # if isinstance(hp, (dict, OrderedDict)):
    #     hpa = np.array([v for v in hp.values()])

    # if len(hpa) == np.sum(k2_detrender.unfrozen_mask):
    #     try:
    #         k2_detrender.set_hp(hpa, include_frozen=False)
    #     except ValueError as e:
    #         raise ValueError(
    #             "Setting hp error.\n"
    #             "hp: {}\n"
    #             "detrender.hp_dict: {}\n"
    #             "detrender.unfrozen_mask: {}\n"
    #             "length: {} vs {}\n"
    #             "".format(hpa, k2_detrender.get_parameter_dict(),
    #                       k2_detrender.unfrozen_mask,
    #                       len(hpa), len(k2_detrender.hp)))
    # else:
    #     try:
    #         k2_detrender.set_hp(hpa, include_frozen=True)
    #     except ValueError as e:
    #         raise ValueError(
    #             "Setting hp error.\n"
    #             "hp: {}\n"
    #             "hp_dict: {}\n"
    #             "detrender.hp_dict: {}\n"
    #             "detrender.unfrozen_mask: {}\n"
    #             "length: {} vs {}\n"
    #             "".format(hpa, hp, k2_detrender.get_parameter_dict(),
    #                       k2_detrender.unfrozen_mask,
    #                       len(hpa), np.sum(k2_detrender.unfrozen_mask)))

    # Checking if any parameters are missing.
    hp_names_given = [key for key in hp.keys()]
    hp_names_k2d = k2_detrender._pname_to_local(k2_detrender.parameter_names)

    if not np.all(np.isin(hp_names_k2d, hp_names_given)):
        missing_names = [n for n in hp_names_k2d if not n in hp_names_given]
        raise ValueError("Not all the detrender parameters were in the "
                         "hp_table.hp dict. Missing: {}\n"
                         "hp_names_k2d: {}\n"
                         "hp_names_given: {}\n"
                         "".format(missing_names,
                                   hp_names_k2d,
                                   hp_names_given))

    # TODO: one issue is still that ramp parameters are given for
    # non-ramp models
    if not np.all(np.isin(hp_names_given, hp_names_k2d)):
        print("skip_detrend: More parameters given than there are in"
              "the model, the likely cause is that the stored",
              "parameters in hp_table are including a ramp model,"
              "isn't properly communicated to the detrender creation",
              "here.\n",
              "\nhp_table.hp dict. Missing:", missing_names,
              "\nhp_names_k2d:", hp_names_k2d,
              "\nhp_names_given:", hp_names_given)

    # BUG: still not working, wtf is hp
    try:
        for hpn, hpv in hp.items():
            k2_detrender.set_parameter(hpn, hpv)
    except ValueError:
        raise ValueError("hp can't be unpacked.\nhp: {}\n"
                         "type(hp): {}".format(hp, type(hp)))

    # Select basis twice to properly identify outliers
    k2_detrender.select_basis()
    k2_detrender.select_basis()
    lcf = k2_detrender.detrend_lightcurve()

    return lcf

def full_detrend(lcf, force_classic, dt_kwargs, **kwargs):
    """Peform the full standardised detrending."""

    # TODO: remove force_classic

    # Periodicity
    if force_classic:
        # Force it to classic and don't log in the sub_list.
        p_flag = False
        pv = np.nan
    elif 'kernel' in kwargs:
        # If values are in the sub_list already.
        p_flag = kwargs.pop('kernel') in ('qp',
                                      'quasiperiodic'
                                      'periodic')
        # If NaN, then does the check automatically
        if 'pv_0' in kwargs.keys() and p_flag:
            pv = kwargs.pop('pv_0')
        elif p_flag:
            p_flag, pv = lcf_tools.detect_lcf_periodicity(lcf)
    else:
        p_flag, pv = lcf_tools.detect_lcf_periodicity(lcf)

    if not p_flag:
        lcf, _, _ = k2gp.detrend_lcf_classic(
                                            lcf,
                                            verbose=False,
                                            plot_all=False,
                                            **dt_kwargs)
    else:
        lcf, _, _ = k2gp.detrend_lcf_quasiperiodic(
                                            lcf,
                                            period=pv,
                                            verbose=False,
                                            plot_all=False,
                                            **dt_kwargs)

    return lcf


# Miscellaneous debugging utils

def print_out(*args):
    print("\n\nParsing bug output.")
    for arg in args:
        print('\n', arg)
    print('\n\n')
    

# Testing
# -------

def test_main():
    # Take some lightcurve kind of at random
    lcf = pd.read_pickle("{}/data/k2/ultracool/diagnostic_targets/"
                      "211892034-C5/k2gp211892034-c05-detrended"
                      "-pos.pickle".format(HOME_DIR))
    lcf['f_detrended'] = lcf['f_detrended'] + 1.0
    lcf['f'] = lcf['f'] + 1.0

    tff, blsf, dtf = list(), list(), list()
    for R_p in (8.0, 2.0, 1.0, 0.5, 0.2, 0.1):
        # tf and bls are not performed by default
        out = recover_injection(lcf,
                                P=20.0,
                                R_p=R_p,
                                R_star=0.117,
                                M_star=0.08,
                                cascade_failure=False,
                                num_searches=3,
                                max_runs=3,
                                pre_filter=True,
                                perform_bls=True,
                                perform_dt=False)
        tff.append(out[0])
        blsf.append(out[1])
        dtf.append(out[2])

    print(tff)
    print(blsf)
    print(dtf)

def test_injection_model():
    from scipy import stats
    from ..tf_tools import TransitFitter

    pd.set_option("display.max_columns", 30)

    # Take some lightcurve kind of at random
    lcf = pd.read_pickle("{}/data/k2/ultracool/diagnostic_targets/"
                      "211892034-C5/k2gp211892034-c05-detrended"
                      "-pos.pickle".format(HOME_DIR))
    lcf['f_detrended'] = lcf['f_detrended'] + 1.0
    lcf['f'] = lcf['f'] + 1.0

    R_p = 1.0
    R_star = 0.117
    M_star = 0.08
    P = 20

    injection_model = InjectionModel(lcf.t.values, P,
                                     t0=(min(lcf.t) + 0.1),
                                     R_p=R_p, M_star=M_star,
                                     R_star=R_star)

    lcf = injection_model.inject_transit(lcf, f_col=['f', 'f_detrended'],
                                         plot=True)
    lcf = util_lib.prep_lightcurve(lcf)

    f = lcf['f_detrended'].values
    t = lcf['t'].values

    bls_peaks, _ = bls_tools.search_transits(t, f - 1.0,
                                             num_searches=3,
                                             max_runs=3,
                                             ignore_invalid=True)

    bls_peaks = tf_tools.fit_transits(t, f, bls_peaks, calc_snr=False,
                                      R_star=injection_model['R_star'],
                                      M_star=injection_model['M_star'],
                                      freeze_a=False)

    print(bls_peaks)

    # Test properties of injection model:
    for p in ('rp', 'R_star', 'M_star', 'a', 'per', 'duration'):
        print("{} = {}".format(p, injection_model[p]))

    # Now validate the correct peak in bls_peaks
    dflag, _, idx = find_signal(bls_peaks, injection_model)

    print("Signal found ({}), index: {}".format(dflag, idx))

    if not dflag:
        print("SIGNAL NOT FOUND.")
        return False
    p_initial = bls_peaks.loc[idx, ['period', 't0',
                                    'depth', 'duration']]

    f_err = stats.sigmaclip(f)[0].std()

    tfitter = TransitFitter.from_bls(t, f, f_err, **p_initial,
                                     bin_res=6, bin_type='regular',
                                     adjust_res=True,
                                     cut_lightcurve=True)

    # The hyperparameters
    print("Hyperparameters\n", '-'*15, '\n',
          tfitter.Prior.get_hpp_dict())

    # Before freezing, print the parameters
    print("Before freezing a on mass, the parameters are:\n",
          tfitter.get_parameter_dict())

    print("Freezing a and estimating based on mass.")

    tfitter.calc_a(R_star=R_star, M_star=M_star, set_value=True)
    tfitter.freeze_parameter('a')

    print("After freezing a on mass, parameters:\n",
          tfitter.get_parameter_dict(True))

    if tfitter.verify_prior(quiet=True):
        print("Prior verification passed.")
    else:
        print("Prior verification failed.")
    