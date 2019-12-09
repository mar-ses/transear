"""Library for utility functions used by various processes in transear.

Folding functions, masking functions, etc...
"""

import numpy as np
import pandas as pd
from scipy import stats

from astropy import units, constants as const

#from .__init__ import HOME_DIR, K2GP_DIR

# Transit masking cushion
MF = 1.4


# Standardised lightcurve preparation
# -----------------------------------

def prep_lightcurve(lcf, flare_sig=None, floor_sig=None, base_val=0.3, quiet=False):
	"""Standardised preparation routine for transit search.

	To be used in transit searching, and injection/recovery modelling.

	1. Check length
	2. Remove floor and flares
	3. Check ordering and fix if required

	Args:
		lcf (pd.DataFrame):
		floor_sig (float): number of sigmas away from median
			to cut outliers off (negative outliers)
		flare_sig (float): number of sigmas above the median
			to use as a cutoff for flares
		base_val (float): absolute value below the median
			within which not to cut out outliers. In other
			words, the largest transit that wouldn't be cut out

	Returns:
		lcf (pd.DataFrame): copy
	"""

	# Check length
	if len(lcf) <= 50:
		raise LessThan50PointsError("Lightcurve has less than 50 " \
									"points, which means we can't " \
									"calculate the noise.")

	# Check ordering and duplication
	if not (np.diff(lcf.t.values) > 0.0).all():
		#num_diffs = np.sum(np.diff(lcf.t.values) <= 0.0)
		#print("Lightcurve {} was unordered with {} "\
		#	  "negative diffs.".format(sl.loc[i, 'epic'],
		#			  				   num_diffs))
		if quiet:
			lcf = lcf.sort_values(by='t')
		else:
			raise LightcurveError("Lightcurve was unordered.")

	if np.sum(lcf.duplicated(subset='cadence') \
			& lcf.duplicated(subset='t')):
		if quiet:
			lcf = lcf.drop_duplicates(['t', 'cadence'], keep='first')
		else:
			raise DuplicatedCadencesError	

	lcf = lcf.copy()

	if flare_sig is not None:
		lcf = lcf[~mask_flares(lcf.f_detrended, flare_sig)]
	if floor_sig is not None:
		lcf = lcf[~mask_floor(lcf.f_detrended, floor_sig, base_val)]

	lcf = lcf[lcf.f_detrended > 0.0]	# line in the sand

	lcf.index = range(len(lcf))

	# Only checks that not all points were removed by floor and flare.
	if len(lcf) == 0 or np.isnan(lcf.t).all():
		raise ZeroLengthError("Lightcurve preparation seems to have " \
							  "removed all lightcurve points.")

	return lcf




# Lightcurve utilities
# --------------------------

def bin_regular(t, f, npb):
	"""Bins the flux with npb points per bin.

	NOTE: arrays MUST be sorted in time, and must be numpy

	Args:
		t, f, npb

	Returns:
		t_binned, f_binned
	"""

	if isinstance(t, pd.Series):
		t = t.values
	if isinstance(f, pd.Series):
		f = f.values

	# Deal with nan values
	nanmask = np.isnan(f)
	f = f[~nanmask]
	t = t[~nanmask]

	N = len(t)
	N_pad = N%npb				# the padded remainder
	N_unpad = int(N//npb)		# how many unpadded binned points

	t_binned = np.empty(N_unpad + bool(N_pad), dtype=float)
	f_binned = np.empty(N_unpad + bool(N_pad), dtype=float)

	# unpad the array
	if N_pad:
		t_binned[-1] = np.nanmean(t[-N_pad:])
		f_binned[-1] = np.nanmean(f[-N_pad:])

	t_binned[:N_unpad] = np.average(np.reshape(t[:N_unpad*npb], (-1, npb)),
									axis=1)
	f_binned[:N_unpad] = np.average(np.reshape(f[:N_unpad*npb], (-1, npb)),
									axis=1)

	return t_binned, f_binned

def fold_on(t, period, t0=None, symmetrize=True):
	"""Work function for folding the time axis, translated to 0.

	i.e t0 gets translated to 0 in the output.

	Args:
		t (np.array-like): array of time values (doesn't need to be sorted).
		period (float): period to fold on.
		t0 (float): time to center the fold on."""

	if t0 is None:
		t0 = min(t)

	# Make sure t0 is the earliest possible INSIDE the lightcurve.
	if t0 > (min(t) + period):
		t0 = t0 - (period * (t0 - min(t))//period)

	t_folded = _fold(t, period) - t0

	if symmetrize:
		# Symmetrize the fold.
		t_folded[t_folded >= period/2] -= period
		t_folded[t_folded < -period/2] += period

	return t_folded

def mask_transits(t, t0, period, duration):
	"""Returns a mask ndarray which flags all the in-transit points in t.

	True means in-transit. t0 doesn't need to be within or before the times in t.

	Args:
		t (array-like): array of times to search and flag in.
		t0 (float): reference transit
		period (float):
		duration (float): mutliply this by a factor if we want cushioning.
			This is a duration, not a half-duration.

	Returns:
		mask (np.ndarray): mask of IN-TRANSIT points.
	"""

	# Rough stuff, useless but oh well.
	if isinstance(t, pd.DataFrame):
		t = t.t

	# i.e if t0 is not the earliest.
	if (min(t) + period) < t0:
		t0 = t0 - (period * (t0 - min(t))//period)

	t_length = max(t) - min(t)
	t_offset = max(0, min(t) - t0)
	num_transits = int((t_length + t_offset) // period) + 1

	mask = np.zeros_like(t, dtype=bool)

	# Cushions it a bit on both sides of the lightcurve just in case.
	for i in range(-1, num_transits + 1):
		mask = mask | ((t < (t0 + i*period + duration/2))
					   & (t > (t0 + i*period - duration/2)))

	return mask

def mask_flares(f, sig_factor=4):
	"""Removes all the points more than n sigmas above the median.

	Args:
		f (np.ndarray-like): the "flux" column to mask
		sig_factor (float): number of sigmas to allow above the zero-point

	Returns:
		mask (pd.Series): True for flares
	"""

	noise_level = calc_noise(ts=f, chunk_size=20)
	mask = f > (np.nanmedian(f) + sig_factor*noise_level)
	return mask

def mask_floor(f, sig_factor=6, base_val=0.05):
	"""Removes all the points more than n sigmas above the median.

	Args:
		f (np.ndarray-like): the "flux" column to mask
		sig_factor (float): number of sigmas to allow above the zero-point
		base_val (float): if sig_factor*sigma would be less than this, then
			this is used. intended to prevent the deletion of 2% transits,
			which are feasible in our survey.

	Returns:
		mask (pd.Series): True for outliers
	"""

	noise_level = calc_noise(ts=f, chunk_size=20)
	sig_mask = f < (np.nanmedian(f) - sig_factor*noise_level)
	floor_mask = f < (np.nanmedian(f) - base_val)
	return sig_mask & floor_mask

def calc_noise(ts, n_chunks=1000, chunk_size=50):
	"""Performs local-average calculation of the noise of a curve.

	In this case, intended to be used for the BLS spectrum noise.

	Args:
		ts (np.ndarray-like): the series to get the noise from.
		n_chunks (int): number of chunks to calculate (accuracy).
		chunk_size (int): size of each chunk, should be less
			than the size of the expected red-noise and real
			variation.

	Returns:
		sigma (float): the noise level.
	"""

	# This should carry a warning, possibly cause an exception.
	if chunk_size > len(ts):
		return stats.sigmaclip(ts)[0].std()

	# Split the data into chunks.
	#import pdb; pdb.set_trace()
	start_nums = ts[:-chunk_size].sample(n_chunks, replace=True).index
	sigma_list = []
	for index in start_nums:
		chunk = ts.iloc[index:index+chunk_size]
		sigma_list.append(stats.sigmaclip(chunk)[0].std())
	#sigma = np.nanpercentile(sigma_list, 30)
	sigma = np.median(sigma_list)

	return sigma


# Transit utilities
# -----------------

def estimate_snr(depth, per, t_base, duration, signoise, t_cad):
	return depth * np.sqrt(t_base * duration / (per * t_cad)) / signoise


# Standard orbital calculations
# -----------------------------

# TODO: consider if it's better to keep the unit objects throughout

def f_p(P, nf, rf=2):
	"""Calculates the dP at a period.

	Args:
		P
		nf: number of frequency points
		rf: range of frequency points (f_max - f_mins)
	"""

	f = 1/P
	df = rf / nf
	f_next = f + df
	return P - 1/f_next

def get_duration(P, R_star, A, b=0.0):
	"""Units: P [d], R_star [R_sun], A [AU], duration [d]

	"""

	if isinstance(P, (pd.Series, pd.DataFrame)):
		P = P.values
	if isinstance(R_star, (pd.Series, pd.DataFrame)):
		R_star = R_star.values
	if isinstance(A, (pd.Series, pd.DataFrame)):
		A = A.values
	if isinstance(b, (pd.Series, pd.DataFrame)):
		b = b.values

	duration = np.sqrt(1 - b**2) * R_star*const.R_sun * P*units.day / (np.pi*A*units.au)

	return duration.to('d').value

def get_A(P, M_star):
	"""Units: P in days, M_star in M_suns, A in AU

	NOTE: this is not in the same units as Kreigberg's, i.e
	not divided by the stellar radius

	Calculation:
	------------
	v**2 / A = GM_star / A**2
	v = 2piA / P	-> A = P * v / 2pi
						 = (P/2pi) * sqrt(GM_star / A)
	A = (P/2pi)**2/3 * (GM_star)**1/3

	TODO: figure out why I previously had more parameters
	than degrees of freedom
	"""

	if isinstance(P, (pd.Series, pd.DataFrame)):
		P = P.values
	if isinstance(M_star, (pd.Series, pd.DataFrame)):
		M_star = M_star.values

	A = (0.5 * P * units.day / np.pi)**(2/3)\
		* (const.G * M_star * const.M_sun)**(1/3)

	return A.to('AU').value

def calc_a(P, M_star, R_star):
	"""Calculates the semi-major axis in a circular orbit.

	NOTE: in units of stellar radius

	Args:
		P (float): days
		M_star (float): in solar masses
		R_star (float): in solar radii

	Returns:
		a = A / R_star
	"""

	if isinstance(P, (pd.Series, pd.DataFrame)):
		P = P.values
	if isinstance(R_star, (pd.Series, pd.DataFrame)):
		R_star = R_star.values
	if isinstance(M_star, (pd.Series, pd.DataFrame)):
		M_star = M_star.values

	T_term = (0.5*P*units.day/np.pi)**(2.0/3.0)
	M_term = (const.G*M_star*const.M_sun)**(1.0/3.0)

	a = (T_term * M_term / (R_star*const.R_sun))
	return a.to('').value

	#return (T_term * M_term / (R_star*const.R_sun)).to('')


# Hidden Utilities
# ----------------

def _fold(t, period):
	"""Folds on the first value.
	"""

	tf = np.empty(len(t), dtype=float)
	tf[:] = np.nan

	t0 = min(t)
	t_length = max(t) - min(t)
	num_folds = int(t_length // period) + 1

	for i in range(num_folds + 1):
		mask = ((t - t0) < (i+1)*period) & ((t - t0) >= i*period)
		if isinstance(mask, pd.Series):
			mask = mask.values
		tf[mask] = t[mask] - i*period

	return tf


# Exception definitions
# ---------------------

class LightcurveError(ValueError):
	"""Holds all specific types of lightcurve error."""
	# Define init, to store the lcf if possible.
	pass

# When prepping lightcurve

class ZeroLengthError(LightcurveError):
	"""When a lightcurve has zero valid points."""
	pass

class LessThan50PointsError(LightcurveError):
	"""When a lightcurve has less than 50 points."""
	pass

class DuplicatedCadencesError(LightcurveError):
	"""Lightcurve points are duplicated.
	Will cause failure of gp or transit search.
	"""
	pass


