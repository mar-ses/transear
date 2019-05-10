"""Functions and procedures for fitting transit models to lightcurves."""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_powell
from scipy.stats import norm, truncnorm
from astropy import units, constants as const
import batman
import emcee
import corner

from . import util_lib
from .__init__ import HOME_DIR, K2GP_DIR

# TODO: figure out if there is too much degeneracy between:
# eccentricities and w; as well as a, inclination and limb-darkening
# CURRENTLY: to assume zero eccentricity always I guess.
# Although eccentricity should be pinned down since a is known-ish from
# duration... This is altogether terrible.

# ------------------------
#
# Bulk and usage functions
#
# ------------------------

def fit_transits(t, f, bls_peaks, bin_type='regular', bin_res=8, subtract_results=False):
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
			lightcurve point.
		subtract_results (bool): if True, subtracts the fitted
			transits before fitting the next.


	Returns:
		bls_peaks but updated with the fitted parameters, under
			columns such as tf_period etc...
	"""

	# Temporary re-normalization for old-style lightcurves
	if np.nanmedian(f) < 0.1:
		f = f + 1.0

	params = ('t0', 'rp', 'a', 'depth', 'duration',
			  'w', 'u1', 'u2', 'ecc', 'inc', 'snr')
	for p in params:
		bls_peaks['tf_{}'.format(p)] = np.nan
	bls_peaks['tf_period'] = np.nan

	for i, ix in enumerate(bls_peaks.index):
		p_initial = bls_peaks.loc[ix, ['period', 't0',
									   'depth', 'duration']]

		# Skip negative depth signals
		if p_initial['depth'] < 0:
			continue

		p_fit = fit_single_transit(t, f, bin_type, bin_res, **p_initial)

		# Write the results
		bls_peaks.loc[ix, 'tf_period'] = p_fit['per']
		for p in params:
			bls_peaks.loc[ix, 'tf_{}'.format(p)] = p_fit[p]
			
		if subtract_results:
			raise NotImplementedError("Can't subtract transits yet.")

		# tfitter = TransitFitter(t, f, f_err, **p_initial, bin_res=bin_res,
		# 						bin_type=bin_type)
		# p_fit, _, _ = tfitter.optimize_parameters(p_initial, show_fit=False)
		
		# bls_peaks.loc[ix, 'tf_t0'] = p_fit['t0']
		# bls_peaks.loc[ix, 'tf_rp'] = p_fit['rp']
		# bls_peaks.loc[ix, 'tf_a'] = p_fit['a']
		# bls_peaks.loc[ix, 'tf_depth'] = p_fit['depth']
		# bls_peaks.loc[ix, 'tf_duration'] = p_fit['duration']
		# bls_peaks.loc[ix, 'tf_w'] = p_fit['w']
		# bls_peaks.loc[ix, 'tf_u1'] = p_fit['u1']
		# bls_peaks.loc[ix, 'tf_u2'] = p_fit['u2']
		# bls_peaks.loc[ix, 'tf_ecc'] = p_fit['ecc']
		# bls_peaks.loc[ix, 'tf_inc'] = p_fit['inc']
		# bls_peaks.loc[ix, 'tf_snr'] = p_fit['snr']

		# Subtract the fitted transit IF it is significant (snr > 1)
		# and if the parameters are physically sensible (harder).

	return bls_peaks

# TODO: a function that fits a single transit from BLS parameters.
def fit_single_transit(t, f, bin_type='regular', bin_res=8, return_params=False, **bls_params):
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
		**bls_kwargs (dict): requires all the keyword arguments
			such as

	Returns:
		p_fit (pd.DataFrame): contains all the parameters of
			the fit.
	"""

	# Temporary re-normalization for old-style lightcurves
	if np.median(f) < 0.1:
		f = f + 1.0

	f_err = util_lib.calc_noise(f)

	tfitter = TransitFitter(t, f, f_err, **bls_params, bin_res=bin_res,
							bin_type=bin_type)
	p_fit, params, _ = tfitter.optimize_parameters(pd.Series(bls_params),
											  show_fit=False)

	if not return_params:
		return p_fit
	else:
		return p_fit, params


# --------------------
#
# Transit prior object
#
# --------------------

class TransitPrior(object):
	"""The prior object, can be called as a function.

	Priors are (currently) normal functions on each parameter,
	except the limb-darkening parameters, which is uniform and
	forbidden from summing to greater than one.

	Attributes:
		num_params (int): number of parameters for the prior.
		param_names (list of str): names of the prior arguments.

	Methods:
		calc_lnprior (function): takes the parameter vector as
			argument, returns the ln of the prior probability.
			Also defined as the function __call__ operator, i.e ().
	"""

	param_names = ['per', 't0', 'rp', 'a', 'inc', 'ecc', 'w', 'u']
	num_params = len(param_names)

	def __init__(self, ):
		"""
		"""
		raise NotImplementedError

	def __call__(self, p):
		"""
		"""
		raise NotImplementedError

		if self.num_params != len(p):
			raise ValueError("Vector was of the wrong dimension (expected: {}, received: {}".format(self.num_params, len(p)))

		# The limb-darkening
		if p[-1][0] + p[-1][1] > 1:
			return -np.inf



	def calc_lnprior(self, p):
		"""
		"""
		return self(p)


# ---------------------
#
# Transit fitter object
#
# ---------------------

class TransitFitter(object):
	"""Contains an instance of transit-fit, and performs the fits.

	TODO: remove the private attributes from this doc, move them to
	a comment section perhaps.
	TODO: change all 'period' to 'per' in internal context

	Example usage:
		tfitter(lcf.t, lcf.f_detrended, f_err,
				**bls_peaks.loc[i, ['period',
									't0', 'depth',
									'duration']])
		fit_out, _, _ = tfitter.optimize_parameters()

	Fitting parameters, i.e the "vector":
		period, t0, rp, a, inc, ecc, w, u1, u2 (quadratic limb-darkening)
		TODO: Update the fitting parameters, especially u must be removed.

	Attributes:
		m (batman.TransitModel): the transit model
		params (batman.TransitParams): the transit parameters
		bss (float): the step size used in batman

		_initial_parameters (BLS format): the initial values
		_param_names (tuple): full list of possible parameters
		_vector_active (tuple): ordered parameters in the pvector
		_vector_dim (int): dimension of the pvector

		f_data (np.array): the flux data
		t_data (np.array): the times at the flux data
		f_err (float or np.array): the flux uncertainties
	Binning attributes:
		_t_model (np.array): optional, if using binning the
			grid of times at which the model is calculated,
			to simulate binning. Only points near the
			transit will be padded.
		_t_bins (np.array): the bin boundary points to
			produce a f_data-like grid from _t_model.
			len(_t_bins) = len(t_data) + 1
		_bin_indices (np.array): same shape as f_model (not f_data),
			is the list of indices in f_data, to which each point in
			f_model must be summed to.
		_bin_freq (np.array or int): number of padded points per bin.
		_bin_type (int): 0 for none, 1 for regular, 2 for irregular
			1 assumes that each bin contains an equal number of adjacent
			points in f_model; while 2 bins according to _t_bins and
			_bin_indices.
	Prior attributes:
		_lower_a_lim, _upper_a_lim (float): limits for the
			uniform prior on the semi-major axis.

	Methods:
		TODO
		_bin_model ()
	Internal state methods:
		set_bin_mode ():
			sets the internal binning from model to data.
			Both at the object attribute level, and by selecting which
			method is used as _bin_model
		reset_step_size
	Utility methods:
		_to_params
		_to_vector
	"""

	_param_names = ('per', 't0', 'rp', 'a', 'inc', 'ecc', 'w', 'u1', 'u2')

	def __init__(self, t, f, f_err, period, t0, depth, duration, prior=None, bin_type='regular', bin_res=8):
		"""Sets the initial value and possibly the prior if not given.

		The lightcurve must be sorted in ascending order by time.

		Args:
			t (np.array): time points in the lightcurve
			f (np.array): flux values in the lightcurve
			f_err (float or np.array): flux uncertainty in lightcurve
			period, t0, depth, duration (floats): the values to
				initialise the object with. Suggested usage is to
				unpack a dictionary: **bls_peaks.loc[i, ['period',...]]
			prior (TransitPrior): the object/function that calculates
				the prior at each point (currently not implemented).
			bin_type (str): out of ['none', 'regular', 'nearby']
				Values other than the above assume 'none'.
			bin_res (int): if binning is not 'none', then this is the
				number of model points used to average in each bin.

		Returns:
			...

		Raises:
			UnorderedLightcurveError: the time points in the lightcurve
				are not in ascending order.

		NOTE: Current expectation is that we initialise with BLS
		parameters (period, duration, t0, depth) and produce
		batman parameters (a, rp, etc...). Conversion must be done
		outside of this scope.

		TODO: Currently this sets the prior manually, without the
		TransitPrior object, though this should be changed.

		TODO: consider whether an instance array of bin boundaries
		needs to be made, for more efficient binning.
		"""

		# Checks on input
		if not np.all(np.diff(t) >= 0):
			raise UnorderedLightcurveError("Lightcurve timeseries is not ordered by time.")

		# Save the data
		self.f_data = np.array(f)
		self.t_data = np.array(t)
		self.f_err = f_err

		# Convert the necessary values
		if depth < 0:
			raise NegativeDepthError
		rp0 = np.sqrt(depth)
		# v = 2 pi a / period = 2 R_star / duration
		a0 = period / (np.pi*duration)

		# Save the values
		self._initial_parameters = pd.Series({'period':period,
											  't0':t0,
											  'depth':depth,
											  'duration':duration})
		self._initial_parameters['rp'] = rp0
		self._initial_parameters['a'] = a0

		self.params = batman.TransitParams()

		# Set the initial-value hyperparameters
		params = self.params
		params.t0 = t0
		params.per = period
		params.rp = rp0							# in units of stellar radius
		params.a = a0							# in units of stellar radius
		params.inc = 90.						# initial value
		params.ecc = 0.							# initial value
		params.w = 90.							# initial value
		params.limb_dark = "quadratic"
		params.u = [0.1, 0.3]					# should I fit this too?
												# NOTE: in fit split as u1, u2

		# Initialise the model
		self.set_bin_mode(bin_type, bin_res)
		self.m = batman.TransitModel(params, self._t_model)
		self.bss = self.m.fac
		self.m = batman.TransitModel(params, self._t_model, fac=self.bss)

		# Physical arguments for priors and initialization
		# For a 0.5 M_S mass star, a in terms of period is:
		# T = 2*pi*sqrt(a**3 / GM_*)
		# a = (T/2pi)**(2/3) * (0.5GM_S)**(1/3)
		# Upper limit, divide by Jupiter radius, 0.5 M_sun
		# Lower limit, take TRAPPIST-1 mass (0.08M_*), and like 0.2 Solar radii
		M_fac = (0.5*period*units.day/np.pi)**(2/3) * (const.G)**(1/3)
		self._lower_a_lim = (M_fac*(0.05*const.M_sun)**(1/3) / (0.2*const.R_sun)).to('')
		self._upper_a_lim = (M_fac*(0.5*const.M_sun)**(1/3) / const.R_jup).to('')

		# Set the fitting parameters
		self.set_active_vector(('per', 't0', 'rp', 'a', 'inc'))


	# Probability and internal methods
	# --------------------------------

	def lnprior(self, pvector):
		"""ln-prior is based on Gaussians at the initial values.

		Args:
			params (batman.TransitParams): optional, uses object
				values by default.

		Returns:
			log_e of the model likelihood at pvector
		"""
		# a
		if (pvector[3] < self._lower_a_lim) or (pvector[3] > self._upper_a_lim):
			return - np.inf

		# period
		lnpdf_per = norm.logpdf(pvector[0],
						self._initial_parameters.period,
						self._initial_parameters.duration)
		# t0
		lnpdf_t0 = norm.logpdf(pvector[1],
						self._initial_parameters.t0,
						self._initial_parameters.duration)
		# rp
		lnpdf_rp = norm.logpdf(pvector[2],
						self._initial_parameters.rp,
						self._initial_parameters.rp)
		return lnpdf_per + lnpdf_t0 + lnpdf_rp

	def evaluate_model(self, pvector, bin_to_data=True):
		"""Calculates a transit model at _t_model."""

		if isinstance(pvector, batman.TransitParams):
			f_model = self.m.light_curve(pvector)
		else:
			for i, p in enumerate(self._vector_active):
				if p in ('u1', 'u2'):
					self.params.u[int(p[-1])-1] = pvector[i]
				else:
					setattr(self.params, p, pvector[i])
			f_model = self.m.light_curve(self.params)

		if bin_to_data:
			return self.bin_model(f_model)
		else:
			return f_model

	def lnposterior(self, pvector):
		"""Calculates a model, returns the lnpdf fit to the data."""

		f_out = self.evaluate_model(pvector, bin_to_data=True)
		lnpdf = (self.lnprior(pvector) + lnlikelihood_gauss(f_out, self.f_data, self.f_err))

		return lnpdf

	def calc_snr(self, params=None):
		"""Calculate the SNR of the transit model vs. flat line.

		Defined as likelihood(model|params) / likelihood(flat-line).
		"""

		if params is None:
			params = self.params
		elif isinstance(params, batman.TransitParams):
			pass
		else:
			params = self._to_params(params)

		model_line = self.m.light_curve(params)
		data_line = self.bin_model(model_line)
		model_lnlikelihood = lnlikelihood_gauss(data_line,
												self.f_data,
												self.f_err)

		null_line = np.median(self.f_data)
		null_lnlikelihood = lnlikelihood_gauss(null_line,
											   self.f_data,
											   self.f_err)

		return model_lnlikelihood - null_lnlikelihood


	# Optimization and process methods
	# --------------------------------

	def optimize_parameters(self, initial_pvector=None, show_fit=False, save_result=True):
		"""Finds the MAP transit parameters to fit our data.

		Args:
			initial_pvector (np.array): if None, starts from the
				initial values given to __init__. Otherwise, if
				it's a Series, it assumes the initial values are
				under ['period', 't0', 'rp']
				TODO: this needs to be improved
			show_fit (bool): if True, prints data and plots fit
			save_result (bool): if True, saves fit to self.params

		Returns:
			(wres, result_params, pvector)
			wres (pd.Series): contains all the params values under
				keyword indexes, plus derived values (duration)
			result_params (batman.TransitParams): best-fit params
			pvector (np.array): array of the best-fit active vector
		"""

		if initial_pvector is None:
			initial_pvector = self._to_vector(**self._initial_parameters[['period', 't0', 'rp']])
		elif isinstance(initial_pvector, (pd.Series)):
			if 'rp' not in initial_pvector:
				initial_pvector = initial_pvector.copy()
				initial_pvector['rp'] = np.sqrt(initial_pvector['depth'])
			initial_pvector = self._to_vector(**initial_pvector[['period', 't0', 'rp']])

		# We need the negative of the lnposterior
		# pvector is in context of the TransitFitter object
		# ovector is in context of the optimization (with 'frozen' values)
		# Currently frozen: ecc = 0, u = [0.1, 0.3]
		self.params.ecc = 0.0
		self.params.u = [0.1, 0.3]
		def ofunc(ovector):
			"""The negative of the lnposterior."""
			return -self.lnposterior(ovector)

		# Peform the fit
		result = fmin_powell(ofunc, initial_pvector, disp=False)

		# Extract and wrap the results
		result_params = self._to_params(result)
		wres = pd.Series(result, index=self._vector_active)

		if show_fit:
			print(wres)
			snr = self.calc_snr(result)
			print("SNR of model is: {}".format(snr))
			f_plot = self.bin_model(self.m.light_curve(result_params))
			fig, ax = plt.subplots()
			ax.plot(self.t_data, self.f_data, 'k.')
			ax.plot(self.t_data, f_plot, 'r-')
			fig.show()

		# Add the derived and frozen parameters
		wres['duration'] = wres['per'] / (np.pi*wres['a'])
		wres['depth'] = wres['rp']**2
		wres['snr'] = self.calc_snr(result)
		wres['u'] = self.params.u
		for p in self._param_names[:-2]:
			if p not in self._vector_active:
				wres[p] = getattr(self.params, p)
		for i, p in enumerate(['u1', 'u2']):
			if p not in self._vector_active:
				wres[p] = getattr(self.params, 'u')[i]

		if save_result:
			self.params = result_params

		return wres, result_params, result

	def get_posteriors(self, show_fit=False, save_result=True):
		"""

		NOTE: starts based on the stored parameters, with jitter.
		"""
		
		# Extract initial values
		initial_pvector = self.get_
		

		# Peform the fit
		result = fmin_powell(ofunc, initial_pvector[:5], disp=False)

		# Extract and wrap the results
		result_params = self._to_params(result)
		wres = pd.Series(result, index=self._vector_names[:len(result)])

		if show_fit:
			print(wres)
			fig, ax = plt.subplots()
			ax.plot(self.t_data, self.f_data, 'k.')
			ax.plot(self.t_data, self.m.light_curve(result_params), 'r-')
			fig.show()

		if save_result:
			self.params = result_params

		return wres, result_params, pvector


	# Internal utility methods
	# ------------------------

	def _to_params(self, pvector, params=None):
		"""Helper function, updates params from values of pvector.

		Doesn't save to self unless params=None.

		Args:
			pvector (np.array):
			params (batman.TransitParams): if None, uses self.params

		Returns:
			params (batman.TransitParams): the updated params
		"""

		if params is None:
			params = self.params

		for i, p in enumerate(self._vector_active):
			if p in ('u1', 'u2'):
				params.u[int(p[-1])-1] = pvector[i]
			else:
				setattr(params, p, pvector[i])

		return params

	def _to_vector(self, cut_to_active=True, **param_kwargs):
		"""Converts the argument into vector for internal processing.

		NOTE: u must be passed as a list u, not as u1 and u2."""

		if 'depth' in param_kwargs:
			param_kwargs['rp'] = np.sqrt(param_kwargs['depth'])

		pv = np.empty(len(self._param_names), dtype=float)
		for i, p in enumerate(self._param_names[:-2]):
			if p in param_kwargs:
				pv[i] = param_kwargs[p]
			elif p not in ('u1', 'u2'):
				pv[i] = getattr(self.params, p)
		# Alternative for period
		if 'per' not in param_kwargs and 'period' in param_kwargs:
			pv[self._param_names.index('per')] = param_kwargs['period']
		# Deal with u directly
		if 'u' in param_kwargs:
			pv[self._param_names.index('u1')] = param_kwargs['u'][0]
			pv[self._param_names.index('u2')] = param_kwargs['u'][1]
		else:
			pv[self._param_names.index('u1')] = self.params.u[0]
			pv[self._param_names.index('u2')] = self.params.u[1]

		if cut_to_active:
			pvout = np.empty(self._vector_dim, dtype=float)
			for i, p in enumerate(self._param_names):
				if p in self._vector_active:
					pvout[i] = pv[i]
		else:
			pvout = pv

		return pvout

	def bin_model(self, f_model):
		"""Bins the model output into the same dimension as the data.

		Uses self._bin_type to check the type of binning to use.

		Args:
			f_model (np.array)
		Returns:
			f_binned (np.array)
		"""

		if self._bin_type == 'none':
			return f_model
		if self._bin_type == 'regular':
			return bin_model_regular(f_model, num_per_bin=self._bin_freq)

	def set_bin_mode(self, bin_type, bin_res=4):
		"""Initialises the bin mode for model-to-data conversion.

		Args:
			bin_type (str): 'regular', 'nearby', 'none' or other
			bin_res (int): if 'regular', then this is the number of
				flux points binned into each data bin (the resolution).

		Returns:
			None: only changes object internal state.

		TODO: 'nearby' bin-mode requires ._bin_freq to be already set.
		Nevertheless, may still need to be moved to its own place.
		"""

		# Binning procedure
		# -----------------
		# Bin boundaries (assumes equally spaced, minus some gaps)
		ts = np.median(self.t_data[1:] - self.t_data[:-1])	# median time-step
		t_bins = np.empty(len(self.t_data)+1, dtype=float)
		t_bins[0] = 1.5*self.t_data[0] - 0.5*self.t_data[1]
		t_bins[-1] = 1.5*self.t_data[-1] - 0.5*self.t_data[-2]
		t_bins[1:-1] = self.t_data[:-1] \
				+ 0.5*(self.t_data[1:] - self.t_data[:-1])
		self._t_bins = t_bins

		if bin_type == 'nearby':
			raise NotImplementedError
		elif bin_type == 'regular':
			self._bin_type = 'regular'
			self._bin_indices = np.sort(list(range(len(self.t_data)))*bin_res)
			self._bin_freq = bin_res
			# Can't be done w.r.t t_bins, as it is irregular around gaps
			t_model = np.empty(len(self.t_data)*bin_res, dtype=float)
			for i in range(len(self.t_data)):
				t_model[i*bin_res:(i+1)*bin_res] = np.linspace(
														self.t_data[i]-ts/2,
														self.t_data[i]+ts/2,
														bin_res+1,
														endpoint=False)[1:]
			self._t_model = t_model
		else:
			# No binning case is default
			self._bin_type = 'none'
			self._t_model = self.t_data
			self._bin_indices = np.array(range(len(self.t_data)))
			self._bin_freq = 1

		if hasattr(self, 'params'):
			self.m = batman.TransitModel(self.params, self._t_model)
			self.bss = self.m.fac
			self.m = batman.TransitModel(self.params, self._t_model, fac=self.bss)

	def set_active_vector(self, active_list=('per', 't0', 'rp', 'a', 'inc')):
		"""Defines the parameters which will be fitted.

		Args:
			active_list (tuple, ordered): possible values are
				['per', 't0', 'rp', 'a', 'inc', 'ecc', 'w', 'u']

		Returns:
			None

		Raises:
			ValueError: if active_list contains invalid inputs
		"""

		for p in active_list:
			if p not in self._param_names:
				raise ValueError("active_list contains invalid vector names.")

		self._vector_active = active_list
		self._vector_dim = len(active_list)

	def reset_step_size(self):
		"""Resets the step size (done for speed) to new value.
		"""
		raise NotImplementedError


# -----------------
#
# Utility functions
#
# -----------------

def lnlikelihood_gauss(f_model, f_data, f_err):
	"""ln-likelihood of fluxes in f_model, vs f_data.
	This is optional and subject to selection.
	"""

	return - 0.5 * np.sum((f_model - f_data)**2 / f_err**2)

def calc_a(P, M_star, R_star):
	"""Calculates the semi-major axis in a circular orbit.

	Args:
		P (float): days
		M_star (float): in solar masses
		R_star (float): in solar radii

	Returns:
		a = A / R_star
	"""

	# Convert from series if required
	if isinstance(P, (pd.Series, pd.DataFrame)):
		P = P.values
	if isinstance(M_star, (pd.Series, pd.DataFrame)):
		M_star = M_star.values
	if isinstance(R_star, (pd.Series, pd.DataFrame)):
		R_star = R_star.values

	T_term = (0.5*P*units.day/np.pi)**(2.0/3.0)
	M_term = (const.G*M_star*const.M_sun)**(1.0/3.0)

	return (T_term * M_term / (R_star*const.R_sun)).to('')


# Various model-binning functions
# -------------------------------

def bin_model_regular(f_model, num_per_bin):
	"""Bins from model output to data, assuming regular binning.

	i.e: bin i in the output is the average of points
	f_model[i*num_per_bin:(i+1)*num_per_bin]

	Args:
		f_model (np.array): must have dimension num_per_bin*len(data)
		num_per_bin (int): number of points in each bin. If it's not
			a factor of len(f_model), causes an error.

	Returns:
		f_binned (np.array): the binned data

	Raises:
		AssertionError: when len(f_model) is not divisible by num_per_bin
	"""

	assert len(f_model) % num_per_bin == 0

	f_binned = np.zeros(int(len(f_model)//num_per_bin), dtype=float)
	index_base = np.array(range(len(f_binned))) * num_per_bin

	for i in range(num_per_bin):
		f_binned += f_model[index_base + i]

	return f_binned / num_per_bin


# ----------
#
# Exceptions
#
# ----------

class UnorderedLightcurveError(Exception):
	pass

class NegativeDepthError(ValueError):
	pass


# ------------------------
#
# Running tests
#
# ------------------------

TEST_NOISE = 0.0018

def test_basic():
	"""Performed on telesto/home - TRAPPIST"""

	from . import bls_tools, util_lib

	lcf = pd.read_pickle("{}/data/trappist/k2gp246199087-c12-detrended-tpf.pickle".format(HOME_DIR))

	if np.nanmedian(lcf.f_detrended) < 0.8:
		lcf['f_detrended'] = lcf.f_detrended + 1.0

	lcf = util_lib.prep_lightcurve(lcf, 4, 6)

	# Peforms BLS first
	bls_peaks, _ = bls_tools.search_transits(lcf.t, lcf.f_detrended,
											 num_searches=5,
											 ignore_invalid=True)

	tstart = time.time()
	bls_peaks = fit_transits(lcf.t, lcf.f_detrended, bls_peaks)

	tend = time.time()
	print("Run-time optimization:", tend-tstart)

	print("Ending test.")

def test_times():
	"""Performed on telesto/home - on TRAPPIST"""

	from . import bls_tools, util_lib

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





def test_trappist(sc=False, reg=True, show_fit=False):
	"""BLS followed by transit search on TRAPPIST."""

	from .util_lib import mask_flares, mask_floor
	from .bls_tools import search_transits
	from .analysis import highlight_bls_peaks

	if reg is True:
		bin_type = 'regular'
	else:
		bin_type = 'none'

	if sc:
		trappist_loc = "{}/trappist_files/sc_detrend/trappist_21_o5.tsv".format(K2GP_DIR)
	else:
		trappist_loc = "{}/trappist_files/k2gp200164267-c12-detrended-pos.tsv".format(K2GP_DIR)

	lcf = pd.read_csv(trappist_loc, sep='\t')
	lcf = lcf[~mask_flares(lcf.f_detrended)]
	lcf = lcf[~mask_floor(lcf.f_detrended)]

	lcf_noise = np.std(lcf.f_detrended)

	bls_peaks, bls_results = search_transits(lcf.t, lcf.f_detrended, num_searches=6, nf=20000)

	for i in range(len(bls_peaks)):
		p_test = bls_peaks.loc[i, ['period', 't0', 'depth', 'duration']]

		tstart = time.time()
		tfitter = TransitFitter(lcf.t.values, lcf.f_detrended.values + 1, f_err=lcf_noise, **p_test, bin_type=bin_type)
		print("Object creation time: {}".format(time.time() - tstart))

		tstart = time.time()
		fit_params, _, _ = tfitter.optimize_parameters(p_test, show_fit=show_fit)
		print("Optimization time: {}".format(time.time() - tstart))

		# Result reinsertion

		snr = tfitter.calc_snr(fit_params.values)
		bls_peaks.loc[i, 'tf_snr'] = snr
		bls_peaks.loc[i, 'tf_per'] = fit_params['per']
		bls_peaks.loc[i, 'tf_t0'] = fit_params['t0']
		bls_peaks.loc[i, 'tf_rp'] = fit_params['rp']
		bls_peaks.loc[i, 'tf_a'] = fit_params['a']
		bls_peaks.loc[i, 'tf_inc'] = fit_params['inc']
		bls_peaks.loc[i, 'tf_duration'] = fit_params['duration']

		#print("SNR of model is: {}".format(snr))
	highlight_bls_peaks(lcf, bls_results, bls_peaks[bls_peaks.tf_snr > 9], title=None, plot=False)

	print(bls_peaks)

def test_ubelix_run():
	from . import bls_tools, analysis

	lcf = pd.read_pickle('k2_data/trappist/k2gp246199087-c12-detrended-tpf.pickle')
	bls_loc = 'k2_data/trappist/k2gp246199087-c12-bls-peaks.csv'
	blsr_loc = 'k2_data/trappist/k2gp246199087-c12-bls-results.csv'

	lcf = lcf[~bls_tools.mask_flares(lcf.f_detrended, 4)]
	lcf = lcf[~bls_tools.mask_floor(lcf.f_detrended, 6, base_val=0.3)]

	f = lcf['f_detrended'].values
	t = lcf['t'].values

	if os.path.exists(bls_loc) and os.path.exists(blsr_loc):
		bls_peaks = pd.read_csv(bls_loc)
		bls_results = pd.read_csv(blsr_loc)
	else:
		bls_peaks, bls_results = bls_tools.search_transits(t, f, num_searches=5, nf=50000, nb=1800)
		bls_peaks.to_csv(bls_loc, index=False)
		bls_results.to_csv(blsr_loc, index=False)

	bls_peaks = bls_peaks[bls_peaks.depth > 0]
	bls_peaks = fit_transits(t, f, bls_peaks.iloc[:5])
	bls_peaks = bls_peaks[bls_peaks.tf_snr > 0]

	analysis.highlight_bls_peaks(lcf, bls_results, bls_peaks, source='tf')

	print(bls_peaks)

# Lightcurve production
# ---------------------

def get_test_lightcurve(noise=0.001, n_points=2000):
	"""Generates a simple noisy test lightcurve with 2 transits."""

	t = np.linspace(0, 75, n_points)

	per = 7.0

	M_fac = (0.5*per*units.day/np.pi)**(2/3) * (0.5*const.G)**(1/3)
	a = (M_fac*(0.08*const.M_sun)**(1/3) / (1.3*const.R_jup)).to('')
	print("Real value of a: {}".format(a))

	transit_params = pd.Series({'period':per, 't0':3., 'rp':0.08, 'a':a, 'inc':89.2, 'ecc':0.0, 'u':[0.1, 0.3]})
	first_guess = pd.Series({'period':per+0.01, 't0':3.03, 'rp':0.06, 'a':a, 'inc':90, 'ecc':0, 'u':[0.13, 0.24]})

	# Set the initial-value hyperparameters
	# -------------------------------------
	params = batman.TransitParams()
	params.t0 = transit_params['t0']
	params.per = transit_params['period']
	params.rp = transit_params['rp']
	params.a = transit_params['a']
	params.inc = transit_params['inc']
	params.ecc = transit_params['ecc']
	params.w = 90.							# initial value
	params.limb_dark = "quadratic"
	params.u = transit_params['u']

	# Initialise the model
	m = batman.TransitModel(params, t)
	f_model = m.light_curve(params)
	f_data = f_model + noise*np.random.randn(len(t))

	lcf = pd.DataFrame({'t':t, 'f_model':f_model, 'f':f_data})

	return lcf, transit_params, first_guess

