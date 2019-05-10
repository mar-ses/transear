"""Library for injection modelling of transits.

Contains:
	- injection of transit from observational or
	physical parameters
	- all the stages of recovery

Also uses a transit_model object; which can produce its own
batman params easily as a method if required. This is what's
passed around (also perhaps does comparison are reading from
bls output/tf_search output (?).

Also, in all cases the transit fitting results in the form of
p_fit will be passed around and return from the stages.
"""

import os
import sys
import copy
import time
import socket
from collections import OrderedDict

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import norm

from astropy import units, constants as const
import batman

from .__init__ import HOME_DIR
from . import tf_tools, bls_tools, util_lib
sys.path.append(HOME_DIR)
from k2gp import k2gp, lcf_tools, gp_tools
from k2gp_dist.__init__ import HP_LOC


# Global variables
tf_snr_cutoff = 5.0

# TODO NOTE NOTE NOTE NOTE NOTE NOTE NOTE:
# Super important: A is not a valid parameter to inject transit with;
# rather derive it from the period and M_star (also eccentricity).

# TODO
# Also take care of what happens when t0 is more than 1 period from
# start of lcf when injecting a signal


# Recovery stages
# ---------------

def stage_tf(lcf, transit_model, f_col='f_detrended', randomise=True,
			 snr_lim=tf_snr_cutoff, **tf_kwargs):
	"""Fits the transit to check if the SNR is high enough.

	TODO: choose bin-type and maybe resolution from the 
	contents of transit_model. IMPORTANT.

	Args:
		lcf (pd.DataFrame)
		transit_model (Transit_Model)
		f_col (str): which column in lcf contains the flux and
			the injected transit, default: 'f_detrended'
		randomise (bool): if True, jiggles the parameters in
			transit_model a little bit
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
	tf_keys = ('bin_type', 'bin_res', 'subtract_results')
	kwk = tf_kwargs.copy().keys()
	tf_kwargs = {key:tf_kwargs.pop(key) for key in kwk if key in tf_keys}

	if randomise:
		bls_params = transit_model.jiggle_params(return_in='bls')

	# p_fit = tf_tools.fit_single_transit(t, f, **tf_kwargs, **bls_params)
	# BUG:
	p_fit, params = tf_tools.fit_single_transit(t, f, return_params=True,
												calc_snr=True,
												cut_lightcurve=True,
												adjust_res=True,
												**tf_kwargs, **bls_params)

	validation_flag = validate_signal(p_fit, transit_model,
									  snr_lim=snr_lim)

	return validation_flag, p_fit

def stage_bls(lcf, transit_model, snr_lim=tf_snr_cutoff,
			  f_col='f_detrended', **kwargs):
	"""Fits the transit to check if the SNR is high enough.

	TODO: choose bin-type and maybe resolution from the 
	contents of transit_model. IMPORTANT.

	Args:
		lcf (pd.DataFrame)
		transit_model (Transit_Model)
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

	kwk = kwargs.copy().keys()

	bls_keys = ('num_searches', 'nf', 'nb', 'qmi', 'qma', 'fmin', 'fmax')
	bls_kwargs = {key:kwargs.pop(key) for key in kwk if key in bls_keys}

	tf_keys = ('bin_type', 'bin_res', 'subtract_results')
	tf_kwargs = {key:kwargs.pop(key) for key in kwk if key in tf_keys}

	bls_peaks, _ = bls_tools.search_transits(t, f, **bls_kwargs)
	bls_peaks = bls_peaks[bls_peaks.depth > 0.0]
	bls_peaks = tf_tools.fit_transits(t, f, bls_peaks, calc_snr=True,
									  cut_lightcurve=True, adjust_res=True,
									  **tf_kwargs)

	# Now validate the correct peak in bls_peaks
	validation_flag, validated_peak = find_signal(bls_peaks, transit_model,
												  snr_lim=snr_lim)

	return validation_flag, bls_peaks, validated_peak

def stage_dt(lcf, transit_model, snr_lim=tf_snr_cutoff,
			 force_classic=False, skip_optimisation=False, **kwargs):
	"""Detrend lightcurve and perform transit_search.s

	TODO: Takes the raw lightcurve and performs the full detrending.
	Performs stage_bls on it (pass along the real values). Validation
	done by stage_bls.

	The target_list entry should be emptied into **kwargs,
	containing information such as full_final, proc_kw etc...
	Example:
		stage_dt(lcf, transit_model, **tl.loc[i, ['dt_pv_0', ...]])

	NOTE: lightcurve is expected to be initialised and cleaned properly

	NOTE: performs the detrending on f; producing standard structure
		lcf

	Args:
		lcf (pd.DataFrame)
		transit_model (TransitModel)
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

	# Remove all the dt_ values which are expected when emptying out
	# the target_list entry
	for key in kwargs:
		if key.startswith('dt_'):
			kwargs[key[3:]] = kwargs.pop(key)

	kwk = kwargs.copy().keys()
	dt_keys = ('proc_kw', 'full_final', 'ramp')
	dt_kwargs = {key:kwargs.pop(key) for key in kwk if key in dt_keys}

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
		# Otherwise do it the good old way
		lcf = full_detrend(lcf, force_classic, dt_kwargs, **kwargs)

	# Find and validate the transit
	# -----------------------------

	validation_flag, bls_peaks, validated_peak = stage_bls(
														lcf,
														transit_model,
														snr_lim=snr_lim,
														f_col='f_detrended',
														**kwargs)

	return validation_flag, bls_peaks, validated_peak, lcf

# def fast_detrend(lcf, force_classic, dt_kwargs, **kwargs):
# 	"""Peform the full standardised detrending."""

# 	# Periodicity
# 	if force_classic:
# 		# Force it to classic and don't log in the sub_list.
# 		p_flag = False
# 		pv = np.nan
# 	elif 'kernel' in kwargs:
# 		# If values are in the sub_list already.
# 		p_flag = kwargs.pop('kernel') in ('qp',
# 									  'quasiperiodic'
# 									  'periodic')
# 		# If NaN, then does the check automatically
# 		if not 'pv_0' in kwargs.keys():
# 			p_flag, pv = lcf_tools.detect_lcf_periodicity(lcf)
# 		else:
# 			pv = kwargs.pop('pv_0')
# 	else:
# 		p_flag, pv = lcf_tools.detect_lcf_periodicity(lcf)

# 	if not p_flag:
# 		lcf, _, _ = k2gp.detrend_lcf_classic(
# 											lcf,
# 											verbose=False,
# 											plot_all=False,
# 											**dt_kwargs)
# 	else:
# 		lcf, _, _ = k2gp.detrend_lcf_quasiperiodic(
# 											lcf,
# 											period=pv,
# 											verbose=False,
# 											plot_all=False,
# 											**dt_kwargs)

# 	return lcf

def skip_detrend(lcf, hp_data):
	"""Peform a detrending without optimization.

	Args:
		lcf (pd.DataFrame)
		hp_data (pd.Series): a row of the hp_table
	"""

	hp = hp_data.hp
	pv = hp_data.dt_pv
	p_flag = hp_data.dt_kernel in ('qp', 'quasiperiodic')
	ramp_flag = hp_data.dt_ramp if 'dt_ramp' in hp_data.index else True

	if isinstance(hp, (dict, OrderedDict)):
		hp = np.array([v for v in hp.values()])

	if p_flag:
		kernel = gp_tools.QuasiPeriodicK2Kernel(pv)
	else:
		kernel = gp_tools.ClassicK2Kernel()

	k2_detrender = gp_tools.K2Detrender(lcf, kernel, ramp=ramp_flag)
	# k2_detrender.set_hp(np.array(hp.values()), include_frozen=True)

	# BUG: this should be temporary because not all are in the same format
	if len(hp) == np.sum(k2_detrender.unfrozen_mask):
		k2_detrender.set_hp(hp, include_frozen=False)
	else:
		k2_detrender.set_hp(hp, include_frozen=True)

	lcfb = k2_detrender.select_basis()
	lcfb = k2_detrender.select_basis()
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


# Bulk injection/recovery routines
# --------------------------------

def	full_recover(lcf, transit_model, f_col='f_detrended',
				 cascade_failure=True, perform_tf=False,
				 perform_bls=False, perform_dt=True,
				 snr_lim=tf_snr_cutoff,
				 **kwargs):
	"""Performs all 3 stages of recovery.

	Does the recovery backwards, if a stage is failed,
	subsequent ones are also taken as failed.

	Args:
		lcf (pd.DataFrame)
		transit_model (TransitModel)
		cascade_failure (bool): if True, upon stage failure,
			all subsequent stages are also taken as failed,
			default = True
		**kwargs (dict): for both bls fitting and tf fitting
			dt_kwargs: proc_kw, full_final
			tf_kwargs: bin_type, bin_res, subtract_results
			bls_kwargs: num_searches, nf, nb, qmi, qma, fmin, fmax

	Returns:
		tf_flag, bls_flag, dt_flag (booleans), tf_snr:
			For each stage, gives a flag if the signal
			was recovered. Also returns the snr of the
			final recovered signal (if found, else np.nan)
	"""

	tf_flag = False
	bls_flag = False
	dt_flag = False

	if perform_tf:
		tf_flag, p_fit = stage_tf(lcf,
								  transit_model,
								  snr_lim=snr_lim,
								  randomise=True,
								  f_col=f_col,
								  **(kwargs.copy()))
		tf_snr = p_fit['snr']
	else:
		tf_flag = True
		tf_snr = np.nan

	if perform_bls and (tf_flag or not cascade_failure):
		bls_flag, _, p_fit = stage_bls(lcf,
									   transit_model,
									   snr_lim=snr_lim,
									   f_col=f_col,
									   **(kwargs.copy()))
		tf_snr = p_fit['snr']
	elif not perform_bls:
		bls_flag = tf_flag
		tf_snr = tf_snr if not np.isnan(tf_snr) else np.nan

	if perform_dt and (bls_flag or not cascade_failure):
		dt_flag, _, p_fit, _ = stage_dt(lcf,
										transit_model,
										snr_lim=snr_lim,
										**(kwargs.copy()))
		tf_snr = p_fit['snr']
	else:
		tf_snr = tf_snr if not np.isnan(tf_snr) else np.nan

	return tf_flag, bls_flag, dt_flag, tf_snr

def	recover_injection(lcf, P, R_p, R_star, M_star, t0=None,
					  snr_lim=tf_snr_cutoff, **kwargs):
	"""Injects a signal and performs all 3 stages of recovery.

	NOTE: should accept the minimal number of parameters.

	Args:
		lcf (pd.DataFrame)
		P, t0, R_p (floats): the transit parameters
			If t0 is None, it will be randomly picked
		**kwargs (dict): passed to full_recover, to include
			cascade_failure, ...
			NOT f_col

	Returns:
		tf_flag, bls_flag, dt_flag
	"""

	# TODO: THIS IS WRONG; replace the entry of A with Mstar and ecc
	# Merely a temporary fix for tests
	if t0 is None:
		t0 = np.random.rand()*P + min(lcf.t)
	A = util_lib.calc_a(P, M_star, R_star)

	if 'bin_type' not in kwargs:
		kwargs['bin_type'] = 'regular'

	lcf = lcf[['t', 'x', 'y', 'f', 'f_detrended']]

	plot = True if socket.gethostname() == 'telesto' else False
	# transit_model = TransitModel(P, t0, rr, A)
	transit_model = TransitModel.from_injection(P, t0, R_p, A,
												R_star=R_star,
												M_star=M_star)
	lcf = transit_model.inject_transit(lcf, f_col=['f', 'f_detrended'], plot=plot)

	tf_flag, bls_flag, dt_flag, tf_snr = full_recover(lcf, transit_model,
													  f_col='f_detrended',
													  snr_lim=snr_lim,
													  **kwargs)

	return tf_flag, bls_flag, dt_flag, tf_snr


# Signal validation
# -----------------

def validate_signal(p_fit, transit_model, snr_lim=tf_snr_cutoff):
	"""Validates a transit_fit based on the transit_model.

	Args:
		p_fit (pd.Series): the p_fit object return by tf_tools;
			for bls_peaks; use the find_signal wrapper
		transit_model (TransitModel): the transit_model to
			compare to

	Return:
		bool: True if signal is properly validated (found)
	"""

	# Validation tolerances (relative):
	P_tol = 0.01			# 1% tolerance
	rr_tol = 1.00			# 100% tolerance	(is this even necessary?)
	t0_tol = 0.01			# absolute tolerance of 100% of duration
							# NOTE: this must be checked for all period
							#		intervals
	duration_tol = 0.50		# 50% tolerance

	# TODO: change rr_tol check to ratio instead of difference

	P = transit_model['P']
	rr = transit_model['rr']
	t0 = transit_model['t0']
	duration = transit_model['duration']

	# Check the peak is actually found
	if abs(P - p_fit['per']) > P*P_tol:
		return False
	if abs(rr - p_fit['rp']) > rr*rr_tol:
		return False
	if abs(duration - p_fit['duration']) > duration*duration_tol:
		return False

	# Check t0 more in-depth
	# First: move t0 of p_fit in the closest period interval to P
	t0_fit = p_fit['t0']
	per_fit = p_fit['per']
	if abs(t0 - t0_fit) > per_fit:
		t0_fit = t0_fit + per_fit*((t0 - t0_fit)//per_fit)
	assert abs(t0 - t0_fit) < per_fit, "The above operation failed."
	if abs(t0 - t0_fit) > t0*t0_tol:
		return False

	# Fit is on the correct one; check snr
	if p_fit['snr'] > snr_lim:
		return True
	else:
		return False

def find_signal(bls_peaks, transit_model, snr_lim=tf_snr_cutoff,
				**validate_kwargs):
	"""Validates a transit_fit based on the transit_model.

	Args:
		bls_peaks (pd.DataFrame): the bls_peaks object with
			columns added by tf_tools.fit_transits
		transit_model (TransitModel): the transit_model to
			compare to
		**validate_kwargs

	Return:
		bool (+ p_fit-like): True if signal is found and
			properly validated, along with p_fit; None
			otherwise
	"""

	p_fit_keys = ('t0', 'rp', 'a', 'depth', 'duration', 'w', 'u1', 'u2',
				  'ecc', 'inc', 'snr', 'period')
	rename_dict = {'tf_{}'.format(key):key for key in p_fit_keys}

	# BUG
	print("Entering debugging.")
	print("rename_dict:", rename_dict)
	print("1 bls_peaks.columns:", bls_peaks.columns)
	bls_peaks = bls_peaks[list(rename_dict.keys())]
	print("2 bls_peaks.columns:", bls_peaks.columns)
	bls_peaks = bls_peaks.rename(columns=rename_dict)
	print("3 bls_peaks.columns:", bls_peaks.columns)
	bls_peaks = bls_peaks.rename(columns={'period':'per'})
	print("4 bls_peaks.columns:", bls_peaks.columns)

	for i in bls_peaks.index:
		pflike = bls_peaks.loc[i]
		validation_flag = validate_signal(pflike,
										  transit_model,
										  snr_lim=snr_lim,
										  **validate_kwargs)
		if validation_flag:
			return True, pflike
		else:
			continue

	return False, None


# Transit model object
# --------------------

class TransitModel(object):
	"""Contains the parameters of the transit model.

	Internally, it's a wrapper around a pd.Series.

	Must decide the parametrisation it will contain (the
	other values are then subject to conversion.

	Currently the basic internals are in observational
	parameters; the physical parameters are extracted
	by computing them from the data.

	TODO: binning as well (internal?)

	"""

	# TODO: needs to be figured out 
	# NOTE: P, t0, rr are compulsory
	# NOTE: Rstar, Mstar, e, u are not
	# 		compulsory and will be set default values.
	# NOTE: currently, I believe that actually Rstar is compulsory,
	#		and so is Mstar. This will be maintained until further
	#		notice.
	# NOTE: w and e will be assumed 0.0 for the entire beginning
	#		of the study.
	# NOTE: careful around units; A is currently in same units
	#		as Rstar (but not in units OF Rstar like batman)
	_stored_parameter_names = ('P', 't0', 'rr', 'A', 'ecc', 'inc',
							   'w', 'u', 'Rstar', 'Mstar')
	_derived_parameter_names = ('depth', 'Rp', 'duration', 'b')

	def __init__(self, P, t0, rr, A, ecc=0.0, inc=90.0, w=0.0,
				 u=None, Rstar=None, Mstar=None):
		"""Main creation routine from stored parameters.

		Args:
			P
			t0
			rr
			A
			e
			w
			u
			Rstar
			Mstar
		"""

		if u is None:
			u = [0.1, 0.3]
		if Rstar is None:
			Rstar = 0.1			# The trappist radius (in solar radii)
		if Mstar is None:
			Mstar = 0.081		# The trappist mass (in solar masses)

		self['P'] = P
		self['t0'] = t0
		self['rr'] = rr
		self['A'] = A
		self['ecc'] = ecc
		self['inc'] = inc
		self['w'] = w
		self['u'] = u
		self['Rstar'] = Rstar
		self['Mstar'] = Mstar

		# Set the conversion dict (_to_derived, _to_stored)
		# Alternatively; have a _convert_param method

	@classmethod
	def from_injection(cls, P, t0, R_p, A, R_star=0.1, M_star=None):
		"""Initiates a TransitModel object from derived parameters.

		TODO: change this to inject solely from R_star and M_star,
			  without requiring (A); a very simple proposition.

		Args:
			P (float): days
			t0 (float): days
			R_p (float): R_earth
			A (float): in units of Rstar (not R_sun)
				TODO: improve this
			R_star (float): in R_sun, default is 0.1
			M_star (float): in M_sun, TODO
		"""

		Rstar = 0.1		# in R_sun
		rr = (R_p * const.R_earth / (R_star*const.R_sun)).to('')

		# Convert to observational parameters
		# ...

		return cls(P=P, t0=t0, rr=rr, A=A, Rstar=Rstar)

	@classmethod
	def from_batman(cls, batman_params):
		"""Initiates a TransitModel object from batman object."""

		# Convert to observational parameters
		# ...

		if batman_params.limb_dark != 'quadratic':
			raise ValueError("Only working with quadratic limb-darkening.")

		t0 = batman_params.t0
		P = batman_params.per
		rr = batman_params.rp
		A = batman_params.a
		inc = batman_params.inc
		ecc = batman_params.ecc
		w = batman_params.w
		u = batman_params.u

		return cls(t0=t0, P=P, rr=rr, A=A, inc=inc, ecc=ecc, w=w, u=u)

	@classmethod
	def from_bls(cls, per, t0, depth, duration):
		"""Models the transit from bls parametrisation.

		Assumes default Rstar of 0.1R_*.
		"""

		Rstar = 0.1
		rr = np.sqrt(depth)
		A = per / (np.pi * duration)

		return cls(P=per, t0=t0, rr=rr, A=A,
						Rstar=Rstar)

	def __copy__(self):
		a = type(self)(1, 1, 1, 1)		# foundation
		for p in self._stored_parameter_names:
			a[p] = self[p]
		return a

	# Internals and properties
	# ------------------------

	def __getitem__(self, name):
		return self.get_parameter(name)

	def __setitem__(self, name, value):
		self.set_parameter(name, value)

	def get_parameter(self, name):
		if name in self._derived_parameter_names:
			return self._getter_dict[name](self)
		elif name == 'u':
			# Don't want external modification of the list
			return list(getattr(self, 'u')).copy()
		elif name in self._stored_parameter_names:
			return getattr(self, name)

	def set_parameter(self, name, value):
		if name in self._derived_parameter_names:
			raise NotImplementedError
		elif name in 'ecc':
			# Physical bounds
			if (value > 1) or (value < 0):
				raise ValueError("e must be in the interval [0, 1].")
			setattr(self, 'ecc', float(value))
		elif name in 'u':
			setattr(self, 'u', list(value))
		elif name in self._stored_parameter_names:
			setattr(self, name, float(value))

	# Conversion methods
	# ------------------

	def get_depth(self):
		return self['rr']**2

	def get_b(self):
		return self['A'] * np.cos(2*np.pi*self['inc']/360.0)

	def get_Rp(self):
		return self['rr'] * self['Rstar']

	def get_duration(self):
		return self['P'] / (np.pi*self['A'])

	_getter_dict = {'b':get_b,
					'Rp':get_Rp,
					'depth':get_depth,
					'duration':get_duration}

	# Production methods
	# ------------------

	def get_bls_params(self):
		"""Produces a dictionary with the bls parameters."""

		return {'period':self['P'], 't0':self['t0'], 'depth':self['depth'],
				'duration':self['duration']}

	def get_batman_params(self):
		"""Produces batman.params from the internal state."""

		params = batman.TransitParams()

		params.t0 = self['t0']
		params.per = self['P']
		params.rp = self['rr']
		params.a = self['A']
		params.inc = self['inc']
		params.ecc = self['ecc']
		params.w = self['w']
		params.limb_dark = 'quadratic'
		params.u = copy.copy(self['u'])

		return params

	def generate_model(self, t, binning=None):
		"""Generates a transit model at the times in t.

		NOTE: acts as a wrapper around... actually...

		Args:
			t (array): times are which to calculate the model
			binning (bool): whether to use binning or not
				TODO: clarify this.

		Returns:
			f (array): the fluxes at times t, normalised at 0.0
		"""

		if isinstance(t, pd.Series):
			t_model = t.copy().values
		else:
			t_model = np.copy(t)

		params = self.get_batman_params()
		m = batman.TransitModel(params, t_model)

		f_model = m.light_curve(params) - 1.0

		return f_model

	def inject_transit(self, lcf, f_col=None, plot=False):
		"""Injects the transit signal into a copy of the lcf."""

		lcf = lcf.copy()
		if f_col is None:
			f_col = ['f', 'f_detrended']
		elif not hasattr(f_col, '__len__'):
			f_col = [f_col]
		f_model = self.generate_model(lcf.t)

		if plot:
			plt.plot(lcf.t, lcf.f + f_model, '.', c='0.5', alpha=0.5)
			plt.plot(lcf.t, lcf.f_detrended + f_model, 'k.', alpha=0.8)
			plt.plot(lcf.t, f_model + np.nanmedian(lcf.f_detrended),
					 'r-', alpha=0.8)
			plt.show()

		for col in f_col:
			lcf[col] = lcf[col] + f_model
		return lcf

	def jiggle_params(self, return_in='bls', save=False):
		"""Randomises the parameters with very small changes.

		To jiggle:
			P, t0, rr, A, e

		Args:
			save (bool): if True, save the parameter values
			return_in (str): which format to return it in
				options: 'bls' (default), 'batman', 'model'
		"""

		if not save:
			param_memory = [self[p] for p in self._stored_parameter_names]

		self['P'] = self['P']*(1 + 0.001*norm.rvs())
		self['t0'] = self['t0'] + self['duration']*0.1*norm.rvs()
		self['rr'] = self['rr']*(1 + 0.05*norm.rvs())
		self['A'] = self['A']*(1 + 0.05*norm.rvs())
		self['ecc'] = abs(self['ecc'] + 0.05*norm.rvs())

		if return_in == 'bls':
			out = self.get_bls_params()
		elif return_in == 'model':
			out = copy.copy(self)
		elif return_in == 'batman':
			out = self.get_batman_params()
		else:
			raise NotImplementedError

		if not save:
			for i, p in enumerate(self._stored_parameter_names):
				self[p] = param_memory[i]

		return out


# Errors
# ------

class HPNotFoundError(Exception):
	pass


# Miscellaneous debugging utils

def print_out(*args):
	print("\n\nParsing bug output.")
	for arg in args:
		print('\n', arg)
	print('\n\n')


# Testing
# -------

def main():
	# Take some lightcurve kind of at random
	lcf = pd.read_pickle("{}/data/k2/ultracool/diagnostic_targets/"
					  "211892034-C5/k2gp211892034-c05-detrended"
					  "-pos.pickle".format(HOME_DIR))

	tff, blsf, dtf = list(), list(), list()
	for R_p in (2.0, 1.0, 0.5, 0.2, 0.1):
		out = recover_injection(lcf,
								P=4.0,
								R_p=R_p,
								R_star=0.14,
								M_star=0.08,
								cascade_failure=False)
		tff.append(out[0])
		blsf.append(out[1])
		dtf.append(out[2])

	print(tff)
	print(blsf)
	print(dtf)
	