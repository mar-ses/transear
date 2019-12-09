
import time

import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt
from scipy import stats

from .transit_fitter import (TransitFitter, UnorderedLightcurveError,
                             NegativeDepthError)
from .. import util_lib

from ..__init__ import HOME_DIR

# ------------------------
#
# Bulk and usage functions
#
# ------------------------

def fit_transits(t, f, bls_peaks, R_star, M_star, bin_type='regular',
                 bin_res=8, calc_snr=False, subtract_results=False,
                 freeze_a=True, overlap_lim='full', **fit_kwargs):
    """Tries to fit all the peaks in 'bls_peaks' with batman.

    Returns results in an updated bls_peaks. Does NOT clean
    the lightcurve. Remove flares and heavy outliers beforehand.
    **ALSO** expects that lightcurve will have flux normalised
    with a median ~ 1 (not normalised at 0).

    Args:
        t (np.array): time axis
        f (np.array): flux timeseries
        bls_peaks (pd.DataFrame): standard bls_peaks table
        bin_type (str, 'regular'): whether to bit or not ('none')
        bin_res (int): number of binned points to use per
            lightcurve point
        calc_snr (bool): if True, performs an MCMC fit of the transit
            to calculate the signal-to-noise ratio, otherwise the 'snr'
            is calculated from snr_estimate
        subtract_results (bool): if True, subtracts the fitted
            transits before fitting the next
        **fit_kwargs: additional params to `fit_single_transit`,
            e.g. cut_lightcurve, iterations, burn,
                 nwalkers, adjust_res, f_err

    Returns:
        bls_peaks but updated with the fitted parameters, under
            columns such as tf_period etc...
    """

    # Temporary re-normalization for old-style lightcurves
    if np.median(f) < 0.1:
        f = f + 1.0

    if isinstance(bls_peaks, pd.Series):
        bls_peaks = pd.DataFrame(bls_peaks).T

    params = ('per', 't0', 'rp', 'a', 'depth', 'duration', 'w', 'u1', 'u2',
              'ecc', 'inc', 'log_llr', 'snr_estimate', 'snr')
    if calc_snr:
        params = params + ('snr_fit', 'mcmc_flag')

    # Columns must be added in any case
    for col in ('period',) + params:
        bls_peaks['tf_{}'.format(col)] = np.nan

    for col in ('tf_mcmc_flag',):
        if not calc_snr:
            continue

        bls_peaks[col] = bls_peaks[col].astype(bool)
        bls_peaks[col] = False

    # Only do valid peaks
    if 'valid_flag' in bls_peaks.columns:
        valid_mask = bls_peaks.valid_flag
    else:
        valid_mask = bls_peaks.depth > 0

    for i, ix in enumerate(bls_peaks[valid_mask].index):
        p_initial = bls_peaks.loc[ix, ['period', 't0',
                                       'depth', 'duration']]

        if p_initial.isnull().any():
            raise ValueError('p_initial contained NaN.')

        p_fit, _ = fit_single_transit(t, f, bin_type=bin_type,
                                      bin_res=bin_res,
                                      freeze_a=freeze_a,
                                      overlap_lim=overlap_lim,
                                      R_star=R_star,
                                      M_star=M_star,
                                      **fit_kwargs,
                                      **p_initial)

        if calc_snr:
            _, _, p_fit['snr_fit'], p_fit['mcmc_flag'] = sample_transit(
                                      t, f, bin_type=bin_type,
                                      bin_res=bin_res,
                                      freeze_a=freeze_a,
                                      overlap_lim=overlap_lim,
                                      R_star=R_star,
                                      M_star=M_star,
                                      **fit_kwargs,
                                      **p_initial)
            p_fit['snr'] = p_fit['snr_fit']
        else:
            p_fit['snr'] = p_fit['snr_estimate']

        # Write the results
        bls_peaks.loc[ix, 'tf_period'] = p_fit['per']
        for p in params:
            bls_peaks.at[ix, 'tf_{}'.format(p)] = p_fit[p]

        # Subtract the fitted transit IF it is significant (snr > 1)
        # and if the parameters are physically sensible (harder).
        if subtract_results:
            raise NotImplementedError("Can't subtract transits yet.")

        # Subtract the fitted transit IF it is significant (snr > 1)
        # and if the parameters are physically sensible (harder).

    return bls_peaks

def fit_single_transit(t, f, bin_type='regular', bin_res=6,
                       cut_lightcurve=True, f_err=None,
                       adjust_res=True, freeze_a=True,
                       overlap_lim='full', **fit_params):
    """Fits a single transit from bls parameters.

    Returns the results as the p_fit DataFrame.

    Does NOT clean the lightcurve. Remove flares and heavy outliers
    beforehand. Expects that lightcurve will have flux normalised
    with a median ~ 1 (not normalised at 0).

    Args:
        t (np.array): time axis
        f (np.array): flux timeseries
        bin_type (str, 'regular'): whether to bit or not ('none')
        bin_res (int): number of binned points to use per
            lightcurve point
        cut_lightcurve (bool): cuts non-required parts of lightcurve
        adjust_res (bool): re-adjusts bin_res to make it have a
            a minimum feasible number of points per transit
        freeze_a (bool): sets a at the beginning and freezes it;
            it True, R_star and M_star **must** be given
        overlap_lim (str or int): sets the minimum overlap required
            by the prior, as a prior on inclination
        **fit_params (dict): REQUIRES all the keyword arguments
            such as:
            M_star, R_star, BLS-PARAMS

    Returns:
        p_fit, batman.params
    """

    mcmc_keys = ('iterations', 'burn', 'nwalkers', 'plot_posterior')
    mcmc_params = dict(p_0=None)
    for key in mcmc_keys:
        if key in fit_params:
            mcmc_params[key] = fit_params.pop(key)

    # Temporary re-normalization for old-style lightcurves
    if abs(np.nanmedian(f)) > 0.1:
        f = f + 1.0 - np.nanmedian(f)

    if f_err is None:
        f_err = stats.sigmaclip(f)[0].std()

    # BUG
    try:
        tfitter = TransitFitter.from_bls(t, f, f_err, **fit_params,
                                     bin_res=bin_res, bin_type=bin_type,
                                     adjust_res=adjust_res,
                                     cut_lightcurve=cut_lightcurve,
                                     overlap_lim=overlap_lim)
    except ValueError as e:
        print("\n Prior verification failure detected\n", "-"*50)
        print("TransitFitter first creation.")
        print("**fit_params\n", fit_params)
        if hasattr(e, 'parameter_dict'):
            print("Parameter dict\n", e.parameter_dict)
        raise e from None


    if freeze_a:
        tfitter.enforce_M_star(R_star=fit_params['R_star'],
                               M_star=fit_params['M_star'])
        tfitter['R_star'] = fit_params['R_star']

    # tfitter = TransitFitter(t, f, f_err, **fit_params, bin_res=bin_res,
    # 						bin_type=bin_type, adjust_res=adjust_res,
    # 						cut_lightcurve=cut_lightcurve)

    # BUG: still causes an error occasionally
    p_fit, params, _ = tfitter.optimize_parameters(show_fit=False)

    return p_fit, params

def sample_transit(t, f, bin_type='regular', bin_res=6,
                   cut_lightcurve=True, adjust_res=True,
                   freeze_a=True, overlap_lim='full',
                   **fit_params):
    """Samples physical parameters for a BLS transit signal.

    Does NOT clean the lightcurve. Remove flares and heavy outliers
    beforehand. Expects that lightcurve will have flux normalised
    with a median ~ 1 (not normalised at 0).

    Args:
        t (np.array): time axis
        f (np.array): flux timeseries
        bin_type (str, 'regular'): whether to bit or not ('none')
        bin_res (int): number of binned points to use per
            lightcurve point
        cut_lightcurve (bool): cuts non-required parts of lightcurve
        adjust_res (bool): re-adjusts bin_res to make it have a
            a minimum feasible number of points per transit
        freeze_a (bool): sets a at the beginning and freezes it;
            it True, R_star and M_star **must** be given
        overlap_lim (str or int): sets the minimum overlap required
            by the prior, as a prior on inclination
        **fit_params (dict): requires all the keyword arguments
            such as:
            M_star, R_star, BLS-PARAMS
            Also contains the mcmc parameters:
            iterations, burn, nwalkers, plot_posterior
            and finall f_err

    Returns:
        result_df, chain_df, snr, mcmc_flag
    """

    mcmc_keys = ('iterations', 'burn', 'nwalkers', 'plot_posterior')
    mcmc_params = dict(p_0=None)
    for key in mcmc_keys:
        if key in fit_params:
            mcmc_params[key] = fit_params.pop(key)

    # Temporary re-normalization for old-style lightcurves
    if np.nanmedian(f) < 0.5:
        f = f + 1.0 - np.nanmedian(f)

    if f_err is None:
        f_err = stats.sigmaclip(f)[0].std()

    tfitter = TransitFitter.from_bls(t, f, f_err, **fit_params,
                                     bin_res=bin_res, bin_type=bin_type,
                                     adjust_res=adjust_res,
                                     cut_lightcurve=cut_lightcurve,
                                     overlap_lim=overlap_lim)

    if freeze_a:
        tfitter.calc_a(R_star=fit_params['R_star'],
                       M_star=fit_params['M_star'],
                       set_value=True)
        tfitter.freeze_parameter('a')
        tfitter['R_star'] = fit_params['R_star']

    chain_df, result_df, snr, mcmc_flag = tfitter.sample_posteriors(**mcmc_params)

    return result_df, chain_df, snr, mcmc_flag


# Testing
# -------

def test_basic(cut_lc=False):
    """Performed on telesto/home - on TRAPPIST"""

    from .. import bls_tools, util_lib

    lcf = get_test_lcf()

    # Peforms BLS first
    bls_peaks, _ = bls_tools.search_transits(lcf.t, lcf.f_detrended,
                                             num_searches=5,
                                             ignore_invalid=True,
                                             max_runs=5)

    # Test automated transit search
    tstart = time.time()
    bls_peaks = fit_transits(lcf.t, lcf.f_detrended, bls_peaks,
                             calc_snr=False, cut_lightcurve=cut_lc)
    tend = time.time()
    print("Run-time optimization:", tend-tstart)

    # Test a single fit by parts; with 'from_stellar_params'
    # Set up the object
    bls_params = bls_peaks.iloc[0][['period', 't0', 'depth', 'duration']]
    print("Fitting:")
    print(bls_params)

    f_err = util_lib.calc_noise(lcf.f_detrended)
    tfitter = TransitFitter.from_bls(lcf.t, lcf.f_detrended, f_err,
                                     **bls_params, bin_res=8,
                                     bin_type='regular')

    if cut_lc:
        tfitter.cut_lightcurve()

    p_fit, _, _ = tfitter.optimize_parameters(show_fit=False)

    print("Active parameters:", tfitter.parameter_names)
    print("Starting:")
    print(p_fit)

    tstart = time.time()
    chain, _, snr, _ = tfitter.sample_posteriors(iterations=1000, burn=1000)

    tend = time.time()
    print("Run-time MCMC:", tend-tstart)
    print("rp_snr:", snr)

    fig = corner.corner(chain)
    fig.show()

    bls_peaks = bls_peaks[bls_peaks.valid_flag]

    tstart = time.time()
    bls_peaks = fit_transits(lcf.t, lcf.f_detrended, bls_peaks,
                             calc_snr=True, cut_lightcurve=cut_lc,
                             plot_posterior=True)
    tend = time.time()

    for i in range(len(bls_peaks)):
        print(bls_peaks.loc[i])
    print("Run-time full search:", tend-tstart)

    import pdb; pdb.set_trace()

    print("Ending test.")

def test_snr_fit(cut_lc=False, deep=False):
    """Test the new 'fixed a' transit fitting."""

    from .. import bls_tools, util_lib

    lcf = get_test_lcf()

    if deep:
        num_searches = 10
        max_runs = 25
    else:
        num_searches = 5
        max_runs = 5

    # Peforms BLS first
    bls_peaks, _ = bls_tools.search_transits(lcf.t, lcf.f_detrended,
                                             num_searches=num_searches,
                                             ignore_invalid=True,
                                             max_runs=max_runs)

    R_star, M_star = 0.117, 0.08
    
    # Test a single fit by parts; with 'from_stellar_params'
    # Set up the object
    bls_params = bls_peaks.iloc[0][['period', 't0', 'depth', 'duration']]
    print("Fitting:")
    print(bls_params)

    f_err = util_lib.calc_noise(lcf.f_detrended)
    tfitter = TransitFitter.from_stellar_params(lcf.t, lcf.f_detrended,
                                                f_err,
                                                per=bls_params['period'],
                                                t0=bls_params['t0'],
                                                rp=bls_params['depth']**0.5,
                                                R_star=R_star,
                                                M_star=M_star,
                                                bin_res=8,
                                                bin_type='regular',
                                                cut_lightcurve=cut_lc,
                                                overlap_lim='full')

    tstart = time.time()
    chain, _, snr, _ = tfitter.sample_posteriors(iterations=1000, burn=1000)
    tend = time.time()
    print("Run-time MCMC:", tend-tstart)
    print("rp_snr:", snr)

    fig = corner.corner(chain)
    fig.suptitle('Full overlap, frozen a')
    fig.show()

    # Show the distributions
    sample_transit(lcf.t, lcf.f_detrended, R_star=R_star, M_star=M_star,
                   **bls_params, cut_lightcurve=cut_lc, bin_res=8,
                   adjust_res=6, freeze_a=True, overlap_lim='partial',
                   plot_posterior=True)
    sample_transit(lcf.t, lcf.f_detrended, R_star=R_star, M_star=M_star,
                   **bls_params, cut_lightcurve=cut_lc, bin_res=8,
                   adjust_res=6, freeze_a=False, overlap_lim='partial',
                   plot_posterior=True)

    plt.show()

    # Test automated transit search
    tstart = time.time()
    bls_peaks = fit_transits(lcf.t, lcf.f_detrended, bls_peaks,
                             R_star=R_star, M_star=M_star,
                             calc_snr=True, cut_lightcurve=cut_lc,
                             plot_posteriors=True)
    tend = time.time()
    print("Run-time optimization:", tend-tstart)


def test_times(cut_lc=False):
    """Performed on telesto/home - on TRAPPIST"""

    from .. import bls_tools, util_lib

    lcf = pd.read_pickle("{}/data/trappist/k2gp246199087-c12-detrended-tpf.pickle".format(HOME_DIR))

    if np.nanmedian(lcf.f_detrended) < 0.8:
        lcf['f_detrended'] = lcf.f_detrended + 1.0

    lcf = util_lib.prep_lightcurve(lcf, 4, 6)

    # Peforms BLS first
    bls_peaks, _ = bls_tools.search_transits(lcf.t, lcf.f_detrended,
                                             num_searches=5,
                                             ignore_invalid=True,
                                             max_runs=5)

    # Set up the object
    bls_params = bls_peaks.iloc[0][['period', 't0', 'depth', 'duration']]
    print("Fitting:")
    print(bls_params)

    f_err = util_lib.calc_noise(lcf.f)
    tfitter = TransitFitter(lcf.t, lcf.f, f_err, **bls_params, bin_res=8,
                            bin_type='regular')

    if cut_lc:
        tfitter.cut_lightcurve()

    print("evaluate_model runtime:", get_runtime(tfitter.evaluate_model,
                                                 dict(), repeat_arg=True))

    print("lnposterior runtime:", get_runtime(tfitter.lnposterior, dict(),
                                              repeat_arg=True))

    print("lnprior runtime:", get_runtime(tfitter.lnprior, dict(),
                                          repeat_arg=True))

    print("get_parameter_vector runtime:",
          get_runtime(tfitter.get_parameter_vector,
                        dict(), repeat_arg=True))

    print("set_parameter_vector runtime:",
          get_runtime(tfitter.set_parameter_vector,
                        dict(vector=tfitter.get_parameter_vector()),
                      repeat_arg=True))

def get_runtime(func, arg_array, N_times=100, get_output=False, repeat_arg=False):
    if repeat_arg:
        arg_array = np.array([arg_array]*N_times)

    t_start_0 = time.time()
    output = func(**arg_array[0])
    t_end_0 = time.time()

    t_start = time.time()
    for i in range(1, N_times):
        output = func(**arg_array[i])
    t_end = time.time()

    # Check if caching occurred:
    if (t_end_0 - t_start_0) > 2*(t_end - t_start)/(N_times - 1):
        print("Caching likely to have occurred.\nFirst run:",
              t_end_0 - t_start_0, "\naverage of other runs:",
              (t_end - t_start)/(N_times - 1))

    if not get_output:
        return (t_end - t_start + t_end_0 - t_start_0)/N_times
    else:
        return (t_end - t_start + t_end_0 - t_start_0)/N_times, output

def get_test_lcf():
    """Loads the TRAPPIST-1 C12 lightcurve file."""

    lcf = pd.read_pickle("{}/data/trappist/k2gp246199087-c12-detrended-tpf.pickle".format(HOME_DIR))

    if np.nanmedian(lcf.f_detrended) < 0.8:
        lcf['f_detrended'] = lcf.f_detrended + 1.0

    lcf = util_lib.prep_lightcurve(lcf, 4, 6)

    return lcf


