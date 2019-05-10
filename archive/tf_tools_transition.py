"""Functions and procedures for fitting transit models to lightcurves."""

import os
import time
import warnings
from collections import OrderedDict

import batman
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units, constants as const
from scipy.optimize import fmin_powell
from scipy.stats import norm, truncnorm

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

def fit_transits(t, f, bls_peaks, bin_type='regular', bin_res=8,
				 calc_snr=True, subtract_results=False, **fit_params):
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
			to calculate the signal-to-noise ratio
		subtract_results (bool): if True, subtracts the fitted
			transits before fitting the next
		**fit_params: additional params to `fit_single_transit`,
			e.g. cut_lightcurve, iterations, burn, nwalkers

	Returns:
		bls_peaks but updated with the fitted parameters, under
			columns such as tf_period etc...
	"""

	# Temporary re-normalization for old-style lightcurves
	if np.median(f) < 0.1:
		f = f + 1.0

	params = ('t0', 'rp', 'a', 'depth', 'duration', 'w',
			  'u1', 'u2', 'ecc', 'inc', 'log_llr')
	if calc_snr:
		params = params + ('snr',)

	# f_err = util_lib.calc_noise(f)

	for i, ix in enumerate(bls_peaks.index):
		p_initial = bls_peaks.loc[ix, ['period', 't0',
									   'depth', 'duration']]

		p_fit = fit_single_transit(t, f, bin_type, bin_res,
								   calc_snr=calc_snr,
								   **fit_params, **p_initial)

		# Write the results
		bls_peaks.loc[ix, 'tf_period'] = p_fit['per']
		for p in params:
			bls_peaks.loc[ix, 'tf_{}'.format(p)] = p_fit[p]

		# Subtract the fitted transit IF it is significant (snr > 1)
		# and if the parameters are physically sensible (harder).
		if subtract_results:
			raise NotImplementedError("Can't subtract transits yet.")

		# # Write the results
		# bls_peaks.loc[ix, 'tf_period'] = p_fit['per']
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
def fit_single_transit(t, f, bin_type='regular', bin_res=6,
					   calc_snr=True, cut_lightcurve=False,
					   return_chain=False, return_params=False,
					   **fit_params):
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
		**fit_params (dict): requires all the keyword arguments
			such as:
			M_star, R_star (possible), ....

	Returns:
		p_fit (pd.DataFrame): contains all the parameters of
			the fit.
	"""

	mcmc_keys = ('iterations', 'burn', 'nwalkers')
	mcmc_params = dict(initial_pvector=None)
	for key in mcmc_keys:
		if key in fit_params:
			mcmc_params[key] = fit_params.pop(key)

	# Temporary re-normalization for old-style lightcurves
	if np.median(f) < 0.1:
		f = f + 1.0

	f_err = util_lib.calc_noise(f)

	tfitter = TransitFitter(t, f, f_err, **fit_params, bin_res=bin_res,
							bin_type=bin_type)

	if cut_lightcurve:
		tfitter.cut_lightcurve()

	p_fit, params, _ = tfitter.optimize_parameters(show_fit=False)

	if calc_snr:
		mcmc_chain, p_fit['snr'] = tfitter.sample_posteriors(**mcmc_params)

	if not return_params and not return_chain:
		return p_fit
	elif not return_params:
		return p_fit, mcmc_chain
	elif not return_chain:
		return p_fit, params
	else:
		return p_fit, mcmc_chain, params


# --------------------
#
# Transit prior object
#
# --------------------

class BasicTransitPrior(object):
	"""The prior object, can be called as a function.

	Priors are (currently) normal functions on each parameter,
	except the limb-darkening parameters, which is uniform and
	forbidden from summing to greater than one.

	Doesn't "know" about frozen parameters; it works on the full
	parameter space, and expects perfectly aligned input.

	Attributes:
		num_params (int): number of parameters for the prior.
		param_names (list of str): names of the prior arguments.

	Methods:
		calc_lnprior (function): takes the parameter vector as
			argument, returns the ln of the prior probability.
			Also defined as the function __call__ operator, i.e ().
	"""

	_param_names = ('per', 't0', 'rp', 'a', 'inc', 'ecc', 'w', 'u1', 'u2')

	def __init__(self, period, t0, rp, duration,
				 time_baseline=40, verbose=False):
		"""TODO
		"""

		self.set_hyperparameters(period, t0, rp, duration,
								 time_baseline=time_baseline,
								 verbose=verbose)

	def __call__(self, p):
		"""Must work on all the parameters, active or not."""

		return self.calc_lnprior(p)

	def __len__(self):
		return len(self._param_names)

	@property
	def num_params(self):
		return len(self)

	def calc_lnprior(self, p):
		"""
		"""

		if self.num_params != len(p):
			raise ValueError("Vector was of the wrong dimension, expected:",
							 self.num_params, "received:", len(p))

		# The limb-darkening
		if p[-2] + p[-1] > 1:
			return -np.inf

		# Eccentricity; basic
		if not 0.0 <= p[self._param_names.index('ecc')] < 1.0:
			return -np.inf

		# a
		if (p[3] < self._lower_a_lim) or (p[3] > self._upper_a_lim):
			return - np.inf

		# inc
		if np.sin(np.deg2rad(abs(p[4] - 90))) > (1/p[3]):
			# i.e not overlapping with stellar disk
			return - np.inf

		# Hard limits on the period
		if abs(p[0] - self._period) > (3*self._duration/self._num_transits):
			return - np.inf

		# period
		lnpdf_per = norm.logpdf(p[0], self._period,
								self._duration / self._num_transits)
		# t0
		lnpdf_t0 = norm.logpdf(p[1], self._t0, self._duration)
		# rp
		lnpdf_rp = norm.logpdf(p[2], self._rp, 2*self._rp)

		return lnpdf_per + lnpdf_t0 + lnpdf_rp

	def set_hyperparameters(self, period, t0, rp, duration, time_baseline=40, verbose=False):
		"""Sets the hyperparameters of the prior.

		Argument behind limits of a:
		- upper limit: Jupiter radius start of 0.5 M_sun
		- lower limit: 0.2 R_sun star of TRAPPIST-1 mass (0.08M_sun)

		NOTE: latter should be lowered for the mass actually.

		Args:
			period (float, days): used to determine the limits on a
			t0
			rp
			duration
			time_baseline (float, 40): observational time baseline,
				i.e how many transit have been observed fully, to set
				limits on the prior uncertainty on period
		"""

		self._duration = duration
		self._period = period
		self._t0 = t0
		self._rp = rp
		self._num_transits = time_baseline / period
		
		# Physical arguments for priors and initialization
		# For a 0.5 M_S mass star, a in terms of period is:
		# T = 2*pi*sqrt(a**3 / GM_*)
		# a = (T/2pi)**(2/3) * (0.5GM_S)**(1/3)
		# Upper limit, divide by Jupiter radius, 0.5 M_sun
		# Lower limit, take TRAPPIST-1 mass (0.08M_*), and like 0.2 Solar radii

		M_fac = (0.5*period*units.day/np.pi)**(2/3) * (0.5*const.G)**(1/3)
		self._lower_a_lim = (M_fac*(0.05*const.M_sun)**(1/3) / (0.4*const.R_sun)).to('')
		self._upper_a_lim = (M_fac*(0.5*const.M_sun)**(1/3) / const.R_jup).to('')

		if verbose:
			print("A limits set at: [{}, {}]".format(self._lower_a_lim,
												 self._upper_a_lim))

class PhysicalTransitPrior(object):
	"""The prior object, can be called as a function.

	Priors are (currently) normal functions on each parameter,
	except the limb-darkening parameters, which is uniform and
	forbidden from summing to greater than one.

	Doesn't "know" about frozen parameters; it works on the full
	parameter space, and expects perfectly aligned input.

	Attributes:
		num_params (int): number of parameters for the prior.
		param_names (list of str): names of the prior arguments.

	Methods:
		calc_lnprior (function): takes the parameter vector as
			argument, returns the ln of the prior probability.
			Also defined as the function __call__ operator, i.e ().
	"""

	_param_names = ('per', 't0', 'rp', 'a', 'inc', 'ecc', 'w', 'u1', 'u2')

	def __init__(self, period, t0, rp, duration, verbose=False):
		"""TODO
		"""
		
		self.set_hyperparameters(period, t0, rp, duration, verbose=verbose)

	def __call__(self, p):
		"""Must work on all the parameters, active or not."""

		return self.calc_lnprior(p)

	def __len__(self):
		return len(self._param_names)

	@property
	def num_params(self):
		return len(self)

	def calc_lnprior(self, p):
		"""
		"""

		if self.num_params != len(p):
			raise ValueError("Vector was of the wrong dimension, expected:",
							 self.num_params, "received:", len(p))

		# The limb-darkening
		if p[-2] + p[-1] > 1:
			return -np.inf

		# Eccentricity; basic
		if not 0.0 <= p[self._param_names.index('ecc')] < 1.0:
			return -np.inf

		# a
		if (p[3] < self._lower_a_lim) or (p[3] > self._upper_a_lim):
			return - np.inf

		# period
		lnpdf_per = norm.logpdf(p[0], self._period, self._duration)
		# t0
		lnpdf_t0 = norm.logpdf(p[1], self._t0, self._duration)
		# rp
		lnpdf_rp = norm.logpdf(p[2], self._rp, self._rp)

		return lnpdf_per + lnpdf_t0 + lnpdf_rp

	def set_hyperparameters(self, period, t0, rp, duration, verbose=False):
		"""Sets the hyperparameters of the prior.

		Argument behind limits of a:
		- upper limit: Jupiter radius start of 0.5 M_sun
		- lower limit: 0.2 R_sun star of TRAPPIST-1 mass (0.08M_sun)

		NOTE: latter should be lowered for the mass actually.

		Args:
			period (float, days): used to determine the limits
				on a.
		"""

		self._duration = duration
		self._period = period
		self._t0 = t0
		self._rp = rp		
		
		# Physical arguments for priors and initialization
		# For a 0.5 M_S mass star, a in terms of period is:
		# T = 2*pi*sqrt(a**3 / GM_*)
		# a = (T/2pi)**(2/3) * (0.5GM_S)**(1/3)
		# Upper limit, divide by Jupiter radius, 0.5 M_sun
		# Lower limit, take TRAPPIST-1 mass (0.08M_*), and like 0.2 Solar radii

		M_fac = (0.5*period*units.day/np.pi)**(2/3) * (0.5*const.G)**(1/3)
		self._lower_a_lim = (M_fac*(0.05*const.M_sun)**(1/3) / (0.4*const.R_sun)).to('')
		self._upper_a_lim = (M_fac*(0.5*const.M_sun)**(1/3) / const.R_jup).to('')

		if verbose:
			print("A limits set at: [{}, {}]".format(self._lower_a_lim,
												 self._upper_a_lim))


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

	TODO - NOTE: in order to introduce sampling of stellar radius
				 and mass, change the storage of mass and radius
				 into params; should be the easiest way and is
				 not expected to have negative side-effects. Otherwise
				 expand the parameter_names; frozen_mask must include
				 these parameters etc... It's a bit more difficult.

	NOTE: the key for working is that frozen/unfrozen, get_parameters
		  and so on only include _param_names. The derived parameters
		  are separate.

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

		f_data (np.array): the flux data
		t_data (np.array): the times at the flux data
		f_err (float or np.array): the flux uncertainties

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
	_derived_parameter_names = ('depth', 'Rp', 'duration', 'b')
	_additional_stored_names = ('R_star', 'M_star')

	def __init__(self, t, f, f_err, period, t0, depth, duration,
				 R_star=1.4, M_star=0.1, prior=None,
				 bin_type='regular', bin_res=6):
		"""Sets the initial value and possibly the prior if not given.

		The lightcurve must be sorted in ascending order by time.

		NOTE: Current expectation is that we initialise with BLS
		parameters (period, duration, t0, depth) and produce
		batman parameters (a, rp, etc...). Conversion must be done
		outside of this scope.

		Args:
			t (np.array): time points in the lightcurve
			f (np.array): flux values in the lightcurve
			f_err (float or np.array): flux uncertainty in lightcurve
			period, t0, depth, duration (floats): the values to
				initialise the object with. Suggested usage is to
				unpack a dictionary: **bls_peaks.loc[i, ['period',...]]
			prior (TransitPrior): the object/function that calculates
				the prior at each point; must be callable.
			bin_type (str): out of ['none', 'regular', 'nearby']
				Values other than the above assume 'none'.
			bin_res (int): if binning is not 'none', then this is the
				number of model points used to average in each bin.

		Returns:
			...

		Raises:
			UnorderedLightcurveError: the time points in the lightcurve
				are not in ascending order.

		TODO: consider whether an instance array of bin boundaries
		needs to be made, for more efficient binning.
		"""

		# Checks on input
		if not np.all(np.diff(t) >= 0):
			raise UnorderedLightcurveError("Lightcurve timeseries is not ordered by time.")
		if depth < 0:
			raise NegativeDepthError
		if np.nanmedian(f) < 0.8:
			warnings.warn(("Lightcurve median is below 0.8. Must" +\
						   "be normalised at 1.0."),
						   Warning)
		if np.any(np.isnan([depth, duration, period,t0, M_star, R_star])):
			raise ValueError("Some fit parameters were NaN.")

		# Save the data
		self.f_data = np.array(f)
		self.t_data = np.array(t)
		self.f_err = f_err

		# Set star parameters
		self['M_star'] = M_star
		self['R_star'] = R_star

		# Set bin mode - TODO: check if this is the right place
		if bin_type == 'regular':
			self.set_regular_bin_mode(bin_res)
		else:
			self.set_no_bin_mode()

		# Convert the necessary values
		rp0 = np.sqrt(depth)
		a0 = period / (np.pi*duration)

		# Set prior
		if prior is None:
			self.Prior = BasicTransitPrior(period, t0, rp0, duration,
										   time_baseline=(max(t) - min(t)))
		else:
			self.Prior = prior

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
		self.m = batman.TransitModel(params, self._t_model)
		self.bss = self.m.fac
		self.m = batman.TransitModel(params, self._t_model, fac=self.bss)

		# Initialise internals
		self._unfrozen_mask = np.ones_like(self._param_names, dtype=bool)

		# Set the fitting parameters
		self.set_active_vector(('per', 't0', 'rp', 'a', 'inc'))

		# Last line check (temporary) to make sure that the internal
		# parameter names match the batman parametrisation
		assert np.all([hasattr(params, n) for n in self._param_names[:-2]])

	def __len__(self):
		return self.unfrozen_mask.sum()

	def __getitem__(self, name):
		if name in ('u1', 'u2'):
			u = getattr(self.params, 'u')
			return u[('u1', 'u2').index(name)]
		elif name in self._param_names:
			return getattr(self.params, name)
		elif name == 'u':
			return getattr(self.params, 'u')
		elif name in self._derived_parameter_names:
			return self._getter_dict[name](self)
		elif name in self._additional_stored_names:
			return getattr(self, name)
		else:
			raise ValueError("{} not found in batman.params.".format(name))

	def __setitem__(self, name, value):
		if name in ('u1', 'u2'):
			self.params.u[('u1', 'u2').index(name)] = value
		elif name in self._param_names:
			setattr(self.params, name, value)
		elif name == 'u':
			setattr(self.params, 'u', value)
		elif name in self._derived_parameter_names:
			raise NotImplementedError('Cannot set by derived params.')
		elif name in self._additional_stored_names:
			setattr(self, name, float(value))
		else:
			raise ValueError("{} not found in batman.params.".format(name))

	def full_size(self):
		return len(self._param_names)

	# Conversion methods

	def get_depth(self):
		return self['rp']**2

	def get_b(self):
		return self['a'] * np.cos(2*np.pi*self['inc']/360.0)

	def get_Rp(self):
		return self['rp'] * self['Rstar']

	def get_duration(self):
		return self['per'] / (np.pi*self['a'])

	_getter_dict = {'b':get_b,
					'Rp':get_Rp,
					'depth':get_depth,
					'duration':get_duration}

	# Properties and internals
	# ------------------------

	@property
	def unfrozen_mask(self):
		return self._unfrozen_mask

	def get_parameter_names(self, include_frozen=False):
		if not include_frozen:
			return tuple(name for i, name in enumerate(self._param_names)
						 if self.unfrozen_mask[i])
		else:
			return tuple(self._param_names)

	# By default, doesn't include frozen parameters
	parameter_names = property(get_parameter_names)

	def get_parameter_vector(self, include_frozen=False):
		return np.array(
					[self[n]
					for n in self.get_parameter_names(include_frozen)])

	def set_parameter_vector(self, vector, include_frozen=False):
		for i, n in enumerate(self.get_parameter_names(include_frozen)):
			self[n] = vector[i]

	# By default, doesn't include frozen parameters
	parameter_vector = property(get_parameter_vector, set_parameter_vector)

	def get_parameter_dict(self, include_frozen=False):
		return OrderedDict(zip(
			self.get_parameter_names(include_frozen=include_frozen),
			self.get_parameter_vector(include_frozen=include_frozen)
		))

	def freeze_parameter(self, name):
		"""Freeze a single parameter."""

		self._unfrozen_mask[self._param_names.index(name)] = False

	def thaw_parameter(self, name):
		"""Freeze a single parameter."""

		self._unfrozen_mask[self._param_names.index(name)] = True

	def set_active_vector(self, names):
		for name in self._param_names:
			if name in names:
				self.thaw_parameter(name)
			else:
				self.freeze_parameter(name)

	# Probability and model evaluation
	# --------------------------------

	def lnprior(self, pvector=None):
		"""TODO: are we using pvectors or not?"""

		if pvector is not None:
			self.set_parameter_vector(pvector)

		p = self.get_parameter_vector(include_frozen=True)

		return self.Prior(p)

	def lnlikelihood(self, pvector=None):

		if pvector is not None:
			self.set_parameter_vector(pvector)

		f_model = self.evaluate_model(bin_to_data=True)
		return - 0.5 * np.sum((f_model - self.f_data)**2 / self.f_err**2)

	def lnposterior(self, pvector=None):

		if pvector is not None:
			self.set_parameter_vector(pvector)
			
		return self.lnprior() + self.lnlikelihood()

	def neg_lnposterior(self, pvector=None):

		if pvector is not None:
			self.set_parameter_vector(pvector)
			
		return -(self.lnprior() + self.lnlikelihood())

	def evaluate_model(self, pvector=None, bin_to_data=True):

		if pvector is not None:
			self.set_parameter_vector(pvector)
			f_model = self.m.light_curve(self.params)
		elif isinstance(pvector, batman.TransitParams):
			f_model = self.m.light_curve(pvector)
		else:
			f_model = self.m.light_curve(self.params)

		if bin_to_data:
			return self.bin_model(f_model)
		else:
			return f_model

	def calc_likelihood_ratio(self, params=None):
		"""Calculate the likelihood ratio of the model vs. flat line.

		NOTE: somewhat obsolete, used to be the "SNR"

		NOTE: also deprecated though still functions

		Defined as likelihood(model|params) / likelihood(flat).
		"""

		if params is None:
			params = self.params
		elif isinstance(params, batman.TransitParams):
			pass
		else:
			self.set_parameter_vector(params)
			params = self.params

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

	def optimize_parameters(self, initial_pvector=None, show_fit=False):
		"""Finds the MAP transit parameters to fit our data.

		Args:
			initial_pvector (np.array): if None, starts from the
				initial values given to __init__. Otherwise, if
				it's a Series, it assumes the initial values are
				under ['period', 't0', 'rp']
				TODO: this needs to be improved
			show_fit (bool): if True, prints data and plots fit

		Returns:
			(wres, result_params, pvector)
			wres (pd.Series): contains all the params values under
				keyword indexes, plus derived values (duration)
			result_params (batman.TransitParams): best-fit params
			pvector (np.array): array of the best-fit active vector
		"""

		# TODO: bring up to speed

		if initial_pvector is None:
			initial_pvector = self.get_parameter_vector()
		elif isinstance(initial_pvector, (pd.Series, dict)):
			initial_pvector = pd.Series(initial_pvector)
			self.set_active_vector(initial_pvector.index)
		else:
			self.set_parameter_vector(initial_pvector)

		result = fmin_powell(self.neg_lnposterior,
							 initial_pvector,
							 disp=False)

		# Extract and wrap the results
		self.set_parameter_vector(result)
		result_params = self.params
		wres = pd.Series(self.get_parameter_dict(include_frozen=True))
		wres['u'] = self['u']
		wres['duration'] = wres['per'] / (np.pi*wres['a'])
		wres['depth'] = wres['rp']**2
		wres['log_llr'] = self.calc_likelihood_ratio(result)

		# Optional visualization (for testing)
		if show_fit:
			print(wres)
			llh_ratio = self.calc_likelihood_ratio()
			print("Likelihood ratio of model is:", llh_ratio)
			f_plot = self.bin_model(self.m.light_curve(result_params))
			fig, ax = plt.subplots()
			ax.plot(self.t_data, self.f_data, 'k.')
			ax.plot(self.t_data, f_plot, 'r-')
			fig.show()

		return wres, result_params, result

	def sample_posteriors(self, initial_pvector=None, iterations=2000,
						  burn=2000, nwalkers=None):
		"""

		NOTE: starts based on the stored parameters, with jitter.

		TODO: internal convergence test perhaps if time allows long
			  enough chains.

		Args:
			iterations (int): number of iterations (after burn) to
				use to populate the posterior (total samples will be
				equal to nwalkers*iterations)
			burn (int): number of iterations to burn through before
				the main sampling
			initial_pvector (array): initial value around which to
				jitter the walkers
			nwalkers (int): number of affine-invariant samplers to use,
				if None will default to 4*ndim
		"""

		# Extract initial values
		if initial_pvector is None:
			initial_pvector = self.get_parameter_vector()

		ndim = len(initial_pvector)
		nwalkers = 4*ndim if nwalkers is None else nwalkers

		# Initial walker position array
		p0 = initial_pvector + 1e-6*np.random.randn(nwalkers, ndim)

		# Sampling procedure
		sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnposterior)
		sampler.run_mcmc(p0, iterations + burn)
		chain = sampler.flatchain[burn:]

		# NOTE: SNR currently doesn't tell us if the radius converged
		# to the right value or at all (it could converge to 0 and
		# have a completely messed up phase)

		chain_df = pd.DataFrame(chain, columns=self.parameter_names)

		# Calculate SNR of transit depth
		if 'rp' in self.parameter_names:
			rp_idx = self.parameter_names.index('rp')
			rp_chain = chain[:, rp_idx]
			rp_med = np.median(rp_chain)
			rp_sig = np.std(rp_chain)
			rp_snr = rp_med / rp_sig
		else:
			rp_snr = None


		return chain_df, rp_snr

	# Internal utility methods
	# ------------------------

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
			return bin_model_regular(f_model, num_per_bin=self._bin_res)

	def set_no_bin_mode(self):
		self._bin_type = 'none'
		self._t_model = self.t_data
		self._bin_indices = np.array(range(len(self.t_data)))
		self._bin_res = 1

		# Reset the transit model and resolution
		if hasattr(self, 'params'):
			self.m = batman.TransitModel(self.params, self._t_model)
			self.bss = self.m.fac
			self.m = batman.TransitModel(self.params, self._t_model, fac=self.bss)

	def set_regular_bin_mode(self, bin_res=4):
		"""Initialises the bin mode for model-to-data conversion.

		Args:
			bin_res (int): the number of flux points binned into
				each data bin (the resolution).

		Returns:
			None
		"""

		# Binning procedure
		# -----------------
		# Bin boundaries (assumes equally spaced, minus some gaps)
		ts = np.median(self.t_data[1:] - self.t_data[:-1])	# time-step
		t_bins = np.empty(len(self.t_data)+1, dtype=float)
		t_bins[0] = 1.5*self.t_data[0] - 0.5*self.t_data[1]
		t_bins[-1] = 1.5*self.t_data[-1] - 0.5*self.t_data[-2]
		t_bins[1:-1] = self.t_data[:-1] \
				+ 0.5*(self.t_data[1:] - self.t_data[:-1])
		self._t_bins = t_bins

		self._bin_type = 'regular'
		self._bin_indices = np.sort(list(range(len(self.t_data)))*bin_res)
		self._bin_res = bin_res
		# Can't be done w.r.t t_bins, as it is irregular around gaps
		t_model = np.empty(len(self.t_data)*bin_res, dtype=float)
		for i in range(len(self.t_data)):
			t_model[i*bin_res:(i+1)*bin_res] = np.linspace(
													self.t_data[i]-ts/2,
													self.t_data[i]+ts/2,
													bin_res+1,
													endpoint=False)[1:]
		self._t_model = t_model

		# Reset the transit model and resolution
		if hasattr(self, 'params'):
			self.m = batman.TransitModel(self.params, self._t_model)
			self.bss = self.m.fac
			self.m = batman.TransitModel(self.params, self._t_model, fac=self.bss)

	def cut_lightcurve(self, num_durations=3):
		"""Cuts the lightcurves points that aren't near transits.

		NOTE: cuts based on the current stored parameters.

		Maintains the previous bin mode (though it recaculates it)

		Args:
			num_durations: number of durations to either side of
				transit to cut (i.e total length is 2*num_durations)
		"""

		t = self.t_data
		t0 = self['t0']
		dur = self['duration']
		per = self['per']

		if t0 > (min(t) + per):
			t0 = t0 - per*((t0 - np.nanmin(t))//per)

		mask = np.zeros_like(t, dtype=bool)

		for i in range(int((np.nanmax(t) - np.nanmin(t))/per) + 1):
			tt = t0 + i*per
			mask[(t >= (tt - dur*num_durations))\
				& (t < (tt + dur*num_durations))] = True

		# Set up and store the cut
		self.t_data = self.t_data[mask]
		self.f_data = self.f_data[mask]

		# Reset the bin model
		if self._bin_type == 'regular':
			self.set_regular_bin_mode(self._bin_res)
		elif self._bin_type == 'none':
			self.set_no_bin_mode()
		else:
			raise NotImplementedError("Bin model still hasn't been",
									  "added here.")


	def reset_step_size(self):
		"""Resets the step size (done for speed) to new value.
		"""
		raise NotImplementedError

	# Other
	# -----

	def plot_model(self, show=True):
		fig, ax = plt.subplots()

		ax.plot(self.t_data, self.f_data, 'k.')
		ax.plot(self._t_model, self.evaluate_model(bin_to_data=False), 'r-')

		if show:
			plt.show()
		else:
			fig.show()






# Utility functions
# -----------------

def lnlikelihood_gauss(f_model, f_data, f_err):
	"""ln-likelihood of fluxes in f_model, vs f_data.
	This is optional and subject to selection."""

	return - 0.5 * np.sum((f_model - f_data)**2 / f_err**2)

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

def calc_a(P, M_star, R_star):
	"""Calculates the semi-major axis in a circular orbit.

	Args:
		P (float): days
		M_star (float): in solar masses
		R_star (float): in solar radii

	Returns:
		a = A / R_star
	"""

	T_term = (0.5*P*units.day/np.pi)**(2.0/3.0)
	M_term = (const.G*M_star*const.M_sun)**(1.0/3.0)

	return (T_term * M_term / (R_star*const.R_sun)).to('')


# Exceptions
# ----------

class UnorderedLightcurveError(Exception):
	pass

class NegativeDepthError(ValueError):
	pass

# Testing
# -------

def test_basic(cut_lc=False):
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

	tstart = time.time()
	bls_peaks = fit_transits(lcf.t, lcf.f_detrended, bls_peaks,
							 calc_snr=False, cut_lightcurve=cut_lc)

	tend = time.time()
	print("Run-time optimization:", tend-tstart)

	# Set up the object
	bls_params = bls_peaks.iloc[0][['period', 't0', 'depth', 'duration']]
	print("Fitting:")
	print(bls_params)

	f_err = util_lib.calc_noise(lcf.f_detrended)
	tfitter = TransitFitter(lcf.t, lcf.f_detrended, f_err, **bls_params,
							bin_res=8, bin_type='regular')

	if cut_lc:
		tfitter.cut_lightcurve()

	p_fit, _, _ = tfitter.optimize_parameters(show_fit=False)

	print("Active parameters:", tfitter.parameter_names)
	print("Starting:")
	print(p_fit)

	tstart = time.time()
	chain, snr = tfitter.sample_posteriors(iterations=1000, burn=1000)

	tend = time.time()
	print("Run-time MCMC:", tend-tstart)
	print("rp_snr:", snr)

	fig = corner.corner(chain)
	fig.show()

	bls_peaks = bls_peaks[bls_peaks.valid_flag]

	tstart = time.time()
	bls_peaks = fit_transits(lcf.t, lcf.f_detrended, bls_peaks,
							 calc_snr=True, cut_lightcurve=cut_lc)
	tend = time.time()

	for i in range(len(bls_peaks)):
		print(bls_peaks.loc[i])
	print("Run-time full search:", tend-tstart)

	import pdb; pdb.set_trace()

	print("Ending test.")

def test_times(cut_lc=False):
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



