"""Contains BLS analysis routines.
"""

import sys
import time
import numpy as np
import pandas as pd
from astropy.stats import mad_std
from scipy import interpolate
import matplotlib.pyplot as plt
import bls

from .util_lib import (mask_flares, mask_floor, mask_transits, fold_on,
					   calc_noise, _fold, MF)
from .__init__ import HOME_DIR, K2GP_DIR

# Idea: have one basic pythonic wrapper work-function around BLS.
# Stuff comes around it, but this thing needs to be utterly general
# and loopable.

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
# columns = ['period', 'duration', 't0', 'depth', 'power', 'snr', 'valid_flag']
# valid_flag is to be False when depth is zero, or something, and can be set false
# retroactively.

# Minimum time period of a lightcurve
min_light_curve = 4				# in days
tce_threshold = 9

def search_transits(t, f, num_searches=5, nf=50000, nb=1800, **bls_kwargs):
	"""Conducts a full search for transit signals.

	A table of periods is extracted, by subsequently masking
	the lightcurve at each transit period. For each peak,
	calculates the BLS power, as well as signal-to-noise ratio.

	Args:
		t, f (array-like):
		num_searches (int): number of maskings to conduct.
		nf (int): frequency resolution
		nb (int): bin resolution for eebls
		**bls_kwargs: pass qmi, qma, fmin, fmax

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

	# fmin and fmax need to be hard set here to avoid problems in bls_results
	T = max(t) - min(t)
	if 'fmin' not in bls_kwargs.keys():
		bls_kwargs['fmin'] = 2./T	# Require two transits currently
	if 'fmax' not in bls_kwargs.keys():
		bls_kwargs['fmax'] = 2.		# 12h period is minimum/maximum

	bls_peaks = pd.DataFrame(index=range(num_searches))
	# To ensure the correct dtypes
	bls_peaks = bls_peaks.assign(period=np.nan, t0=np.nan, duration=np.nan, depth=np.nan, power=np.nan, snr=np.nan)
	bls_peaks = bls_peaks.assign(valid_flag=True)

	# TODO: Change to a while loop.
	for i in range(num_searches):
		# Rechecking in case too many points are deleted
		if len(t) < 10:
			break
		if (max(t) - min(t)) < min_light_curve:
			# Catch it early
			break
		if bls_kwargs['fmin'] < 1/(max(t) - min(t)):
			# Would break BLS too
			# Other option is to recalculate fmin, but unlikely to be useful
			break

		bls_r, period, t0, duration, depth, bls_o = calc_bls(t=t, f=f, nf=nf, nb=nb, **bls_kwargs)

		# The flat power is not useful outside of this loop
		bls_r = bls_r.assign(flat_power=flatten_power_spectrum(bls_r, N_bins=40, bin_by='period', flatten_col='power'))
		bls_noise = mad_std(bls_r.flat_power)
		bls_r = bls_r.assign(snr=calc_sn(bls_r))

		# Note in bls_results
		if i == 0:
			bls_results = bls_r
		else:
			assert len(bls_results) == len(bls_r)
			assert (bls_results.frequency == bls_r.frequency).all()
		bls_results['power_{}'.format(i)] = bls_r['power']
		bls_results['snr_{}'.format(i)] = bls_r['snr']

		# Note in bls_peaks
		bls_peaks.at[i, 'period'] = period
		bls_peaks.at[i, 't0'] = t0
		bls_peaks.at[i, 'duration'] = duration
		bls_peaks.at[i, 'depth'] = depth
		bls_peaks.at[i, 'power'] = bls_o[2]
		bls_peaks.at[i, 'noise'] = bls_noise
		# Get the nearest highest SNR
		df = (max(bls_results.frequency) - min(bls_results.frequency))/nf
		bls_peaks.at[i, 'snr'] = max(bls_results.loc[(bls_results.frequency.between(1/period - 8*df, 1/period + 8*df)), 'snr'])

		# Pre-validation
		# Convert this into deletion and automatic filtration.
		if depth < 0:
			bls_peaks.loc[i, 'valid_flag'] = False
		else:
			bls_peaks.loc[i, 'valid_flag'] = True
			# TCE checking
			if bls_peaks.loc[i, 'snr'] > 0:
				# TODO:
				pass


		# Mask the next array
		tmask = mask_transits(t, t0, period, MF*duration)
		t = t[~tmask]
		f = f[~tmask]

	return bls_peaks, bls_results


# BLS work-function
# -----------------

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

# Further work-functions
# ----------------------

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

def flatten_power_spectrum(br, N_bins=40, bin_by='period', flatten_col='power'):
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

# --------------------------------
#
# BLS Validation and peak analysis
#
# --------------------------------

def validate_peaks(bls_peaks, lcf=None, bls_results=None):
	"""Combined function that assigns valid_flag to a BLS peak.
	"""

	pass

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


# ----------------
#
# Testing routines
#
# ----------------

def test_single_bls():
	"""Does a BLS on the TRAPPIST lightcurve."""
	from .analysis import highlight_bls_signal

	trappist_loc = "{}/trappist_files/k2gp200164267-c12-detrended-pos.tsv".format(K2GP_DIR)
	lcf = pd.read_csv(trappist_loc, sep='\t')
	lcf = lcf[~mask_flares(lcf.f_detrended)]
	lcf = lcf[~mask_floor(lcf.f_detrended)]

	bls_result, best_period, t0, duration, depth, bls_output = calc_bls(t=lcf.t, f=lcf.f_detrended, nf=50000, nb=1500, flims=None, verbose=True)

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

	bls_peaks, bls_results = search_transits(lcf.t, lcf.f_detrended, num_searches=4, nf=10000)

	for i in bls_peaks.index:
		highlight_bls_signal(lcf.t, lcf.f_detrended, bls_peaks.loc[i, 't0'], bls_peaks.loc[i, 'period'], bls_peaks.loc[i, 'duration'], bls_peaks.loc[i, 'depth'])

	return bls_peaks, bls_results

# ----------------------------
#
# Specific visualization tools
#
# ----------------------------

# TODO: Plot the BLS spectrum (maybe not even needed), more importantly,
# plot the folded lightcurve on some BLS parameters from the table.
# with highlighted transit, including predicted depth.

def plot_bls(bls_result, bls_col='power'):
	"""Not empty."""

	plt.plot(bls_result.frequency, bls_result[bls_col], 'k-')
	plt.show()



# --------------------
#
# Utilities and errors
#
# --------------------

class ShortLightcurveError(Exception):
	"""Lightcurve is less than min_light_curve days long."""
	pass

class InvalidLightcurveError(Exception):
	"""Lightcurve has too many points and cannot be searched."""
	pass






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


# OLD AND ARCHIVE

def fold_on_old(t, period, t0=None):
	"""Work function for folding the time axis, centered on t0.

	Args:
		t (np.array-like): array of time values (doesn't need to be sorted).
		period (float): period to fold on.
		t0 (float): time to center the fold on."""

	if t0 is None:
		t0 = min(t)

	# Make sure t0 is the earliest possible INSIDE the lightcurve.
	if min(t) < t0 - period:
		t0 = t0 - (period * (t0 - min(t))//period)

	tc = t - t0
	t_length = max(t) - min(t)
	t_offset = max(0, min(t) - t0)
	num_folds = int((t_length + t_offset) // period) + 1

	# For holding the folded times.
	tf = np.empty(len(t), dtype=float)
	# To make it possible to check if all have been assigned to.
	tf[:] = np.nan
	for i in range(num_folds + 1):
		mask = (tc < (i + 0.5)*period) & (tc >= (i - 0.5)*period)
		tf[mask.values] = tc[mask.values] - i*period

	return tf

















