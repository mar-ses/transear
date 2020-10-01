"""Functions for checking recovered signals and validating detections."""

import numpy as np
from .. import util_lib


# Signal validation
# -----------------

def check_parameter_match(p_fit, injection_model, aliases=True):
    """Check p_fit corresponds to same parameters as injection_model.

    NOTE: use this before calculating the SNR, to check if this is
    the injected signal, and not waste SNR calculations on signals
    that weren't injected.

    NOTE: This is also known as transit validation.

    Args:
        p_fit (pd.Series or dict): the p_fit object returns by tf_tools;
        injection_model (InjectionModel): the injection_model to compare to

    Returns:
        bool: True if parameters are validated (not necessarily detected)
    """

    if 'period' in p_fit.index and 'per' not in p_fit.index:
        p_fit['per'] = p_fit['period']

    # Validation tolerances (relative):
    P_tol = 0.01			# 1% tolerance
    rp_tol = 1.00			# 100% tolerance	(is this even necessary?)
    t0_tol = 0.50			# in fractions of a **duration**
    duration_tol = 0.50		# 50% tolerance

    # TODO: change rp_tol check to ratio instead of difference
    P = injection_model['per']
    rp = injection_model['rp']
    t0 = injection_model['t0']
    duration = injection_model['duration']
    # t0 and period must accomodate aliases if required
    t0_fit = p_fit['t0']
    per_fit = p_fit['per']

    # Check for nan values
    if p_fit[['t0', 'per', 'duration', 'rp']].isnull().any():
        raise p_fit_NullError(
            'p_fit contains null values:' + str(p_fit[p_fit.isnull()].index))
        print('p_fit contains null values in',
              p_fit[p_fit.isnull()].index)
        return False

    # Boring parameters
    if abs(rp - p_fit['rp']) > rp*rp_tol:
        return False
    if abs(duration - p_fit['duration']) > duration*duration_tol:
        return False
    
    # Set up aliases
    # --------------
    # Current aliases are 1/2, 2 and 3; check this.
    P_alias = P * np.array([1/3, 1/2, 1, 2, 3])
    t0_alias = np.empty_like(P_alias)
    alias_match = False		# keeps track of if an alias is found

    # For each alias, calculate the t0_alias that would
    # line up with t0_fit
    for i, pa in enumerate(P_alias):
        # Each alias would assume to have t0 at the t0 (fit), except
        # they might then be staggered or out-of-phase.
        # Thus, for each alias, translate t0 (alias) to as close
        # as possible to the t0 (fit)
        t0_alias[i] = t0 + pa*int((t0 - t0_fit)/pa)

    # Period and t0
    # P_mask is True for P_aliases within tolerance of the period fit
    P_mask = np.abs(per_fit - P_alias) < P_alias*P_tol
    t0_mask = np.abs(t0_fit - t0_alias) < duration*t0_tol
    if P_mask[P_alias == P] and t0_mask[P_alias == P]:
        # i.e fit matches the *actual* P and t0
        alias_match = False
    elif aliases and (P_mask & t0_mask).any():
        # i.e fit matches an aliased P and t0 (both at the same time)
        alias_match = True
    else:
        return False

    # Currently just print a message/warning
    if alias_match:
        print("ALIAS MATCH DETECTED.")

    # If it gets to this point, the parameters match
    return True

def validate_signal(p_fit, injection_model, snr_lim, snr_source='snr'):
    """Tests detection of the injection_model in the transit_fit.

    Args:
        p_fit (pd.Series): the p_fit object returned by tf_tools
        injection_model (InjectionModel): the injection_model to
            compare to
        snr_lim (float): how many sigmas to consider a "detection"
        snr_source (str): key in p_fit to use as the snr

    Return:
        found, matched (bools)
        found: True if signal is properly validated; i.e matching
            parameters and SNR over threshold
        matched: True if parameters are matched, but if SNR is not
            over the threshold
    """

    # All parameter matching is contained in check_parameter_match
    match = check_parameter_match(p_fit, injection_model)
    if not match:
        return False, match
    elif p_fit[snr_source] > snr_lim:
        # Signal is matched and has SNR over the limit
        # From here, can check other attributes (snr, ch_flag, etc...)
        return True, match
    else:
        # Signal matches the transit but the SNR is below the limit
        return False, match

def find_signal(bls_peaks, injection_model):
    """Finds the injected signal in the bls_peaks.

    NOTE: new update; no longer performs validation.

    Args:
        bls_peaks (pd.DataFrame): the bls_peaks object with
            columns added by tf_tools.fit_transits
        injection_model (InjectionModel): the injection_model to
            compare to

    Return:
        found_flag, p_fit-like, idx: returns the peaks in
            bls_peaks, and its index
    """

    bls_peaks = bls_peaks.copy()

    p_fit_keys = ['t0', 'rp', 'a', 'depth', 'duration', 'w', 'u1', 'u2',
                  'ecc', 'inc', 'period']
    rename_dict = {'tf_{}'.format(key):key for key in p_fit_keys}

    try:
        bls_peaks = bls_peaks[list(rename_dict.keys())]
    except KeyError:
        raise KeyError("tf_ keys not found in bls_peaks. " \
                       "May have forgotten to run a transit fit.")

    bls_peaks = bls_peaks.rename(columns=rename_dict)
    bls_peaks['per'] = bls_peaks['period']

    for idx in bls_peaks.index:
        pflike = bls_peaks.loc[idx]

        if pflike[p_fit_keys].isnull().any():
            continue

        found_flag = check_parameter_match(pflike, injection_model)

        if found_flag:
            return True, pflike, idx
        else:
            continue

    return False, None, None

def pre_filter_transit(injection_model, t, f, snr_cutoff=1.0):
    """Checks if the expected snr is greater than a cutoff.

    NOTE: currently not used; I do this manually in irm.

    Args:
        injection_model (transear.injection.InjectonModel)
        t, f (np.array)
        snr_cutoff (float=1.0)

    Returns:
        True if snr_estimate > snr_cutoff
        False otherwise
    """

    # TODO: decide if I need f or if I can just pull it out of the injection
    # model.
    snr_estimate = util_lib.estimate_snr(depth=injection_model['depth'],
                                         per=injection_model['per'],
                                         t_base=(max(t) - min(t)),
                                         duration=injection_model['duration'],
                                         signoise=util_lib.calc_noise(f),
                                         t_cad=np.nanmedian(np.diff(t)))

    return snr_estimate > snr_cutoff


class p_fit_NullError(ValueError):
    def __init__(self, error_str, *args, bls_peaks=None):
        self.error_str = error_str
        self.bls_peaks = bls_peaks
        super().__init__(error_str, *args)





# def find_signal(bls_peaks, injection_model, snr_lim=tf_snr_cutoff,
# 				**validate_kwargs):
# 	"""Validates a transit_fit based on the injection_model.

# 	Args:
# 		bls_peaks (pd.DataFrame): the bls_peaks object with
# 			columns added by tf_tools.fit_transits
# 		injection_model (InjectionModel): the injection_model to
# 			compare to
# 		**validate_kwargs

# 	Return:
# 		bool (+ p_fit-like): True if signal is found and
# 			properly validated, along with p_fit; None
# 			otherwise
# 	"""

# 	p_fit_keys = ('t0', 'rp', 'a', 'depth', 'duration', 'w', 'u1', 'u2',
# 				  'ecc', 'inc', 'snr', 'period')
# 	rename_dict = {'tf_{}'.format(key):key for key in p_fit_keys}

# 	# BUG
# 	print("Entering debugging.")
# 	print("rename_dict:", rename_dict)
# 	print("1 bls_peaks.columns:", bls_peaks.columns)
# 	bls_peaks = bls_peaks[list(rename_dict.keys())]
# 	print("2 bls_peaks.columns:", bls_peaks.columns)
# 	bls_peaks = bls_peaks.rename(columns=rename_dict)
# 	print("3 bls_peaks.columns:", bls_peaks.columns)
# 	bls_peaks = bls_peaks.rename(columns={'period':'per'})
# 	print("4 bls_peaks.columns:", bls_peaks.columns)

# 	for i in bls_peaks.index:
# 		pflike = bls_peaks.loc[i]
# 		validation_flag = validate_signal(pflike,
# 										  injection_model,
# 										  snr_lim=snr_lim,
# 										  **validate_kwargs)
# 		if validation_flag:
# 			return True, pflike
# 		else:
# 			continue

# 	return False, None
