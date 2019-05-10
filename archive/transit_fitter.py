
import warnings
import copy
from collections import OrderedDict

import batman
import corner
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units, constants as const
from scipy.optimize import fmin_powell
from scipy.stats import norm, truncnorm, percentileofscore

from .priors import BasicTransitPrior, PhysicalTransitPrior
from .. import util_lib
from ..__init__ import HOME_DIR



# Transit fitter object
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
				 R_star=0.14, M_star=0.1, prior=None,
				 bin_type='regular', bin_res=6, adjust_res=False,
				 cut_lightcurve=False):
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
			warnings.warn(("Lightcurve median is below 0.8. Must " \
						   + "be normalised at 1.0."),
						   Warning)
		if np.any(np.isnan([depth, duration, period, t0, M_star, R_star])):
			raise ValueError(("Some fit parameters were NaN. "
							+ "Values: {}".format([depth, duration,
												   period, t0, M_star,
												   R_star])))

		# Save the data
		self.f_data = np.array(f)
		self.t_data = np.array(t)
		self.f_err = f_err

		# Set star parameters (NOTE this seems irrelevant)
		self['M_star'] = M_star
		self['R_star'] = R_star

		# Convert the necessary values
		rp0 = np.sqrt(depth)
		a0 = period / (np.pi*duration)	# likely to be overestimate

		# Set prior
		if prior is None:
			self.Prior = BasicTransitPrior(period, t0, rp0, duration,
										   time_baseline=(max(t) - min(t)))
		else:
			self.Prior = prior

		if a0 > self.Prior._upper_a_lim:
			# implies high impact factor (b) in reality
			# (but not considered in bls duration)
			a0 = self.Prior._upper_a_lim - 0.0001
		elif a0 < self.Prior._lower_a_lim:
			# implies a super-heavy and compact object (unlikely)
			a0 = self.Prior._lower_a_lim + 0.0001

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

		if cut_lightcurve:
			self.cut_lightcurve()

		# Set bin mode - needs to be here to adjust resolution
		if bin_type == 'regular':
			self.set_regular_bin_mode(bin_res, adjust_res=adjust_res)
		else:
			self.set_no_bin_mode()

		# Initialise the model
		self.m = batman.TransitModel(params, self._t_model)
		self.bss = self.m.fac
		self.m = batman.TransitModel(params, self._t_model, fac=self.bss)

		# Initialise internals
		self._unfrozen_mask = np.ones_like(self._param_names, dtype=bool)

		# Set the fitting parameters
		self.set_active_vector(('per', 't0', 'rp', 'a', 'inc'))

		# Make sure the initial values and priors are valid
		self.verify_prior(info_str='inside __init__.')

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

		self.verify_prior(info_str='inside optimize_parameters.')

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

	def sample_posteriors(self, p_0=None, iterations=2000,
						  burn=2000, nwalkers=None, plot_posterior=False):
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
			p_0 (array): initial value around which to
				jitter the walkers
			nwalkers (int): number of affine-invariant samplers to use,
				if None will default to 2*ndim

		Returns:
			chain_df, results_df, rp_snr, ch_flag
		"""

		# Extract initial values
		if p_0 is None:
			p_0 = self.get_parameter_vector()

		self.set_parameter_vector(p_0)
		self.verify_prior(info_str='inside sample_posteriors')

		ndim = len(p_0)
		nwalkers = 2*ndim if nwalkers is None else nwalkers

		# Initial walker position array
		p0 = p_0 + 1e-6*np.random.randn(nwalkers, ndim)

		# Sampling procedure
		sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnposterior)
		sampler.run_mcmc(p0, iterations + burn)
		chain = sampler.flatchain[burn:]

		# NOTE: SNR currently doesn't tell us if the radius converged
		# to the right value or at all (it could converge to 0 and
		# have a completely messed up phase)

		chain_df = pd.DataFrame(chain, columns=self.parameter_names)

		# Calculate the result dataframe
		result_df = pd.DataFrame(columns=self.parameter_names,
								 index=['median', 'lower', 'upper'])

		for i, p in enumerate(self.parameter_names):
			med = np.median(chain[:, i])
			low = med - np.percentile(chain[:, i], 16)
			high = np.percentile(chain[:, i], 84) - med

			result_df.loc['median', p] = med
			result_df.loc['lower', p] = low
			result_df.loc['upper', p] = high

		# Calculate SNR of transit depth
		if 'rp' in self.parameter_names:
			rp_idx = self.parameter_names.index('rp')
			rp_chain = chain[:, rp_idx]
			rp_med = min(np.median(rp_chain), 1.0)
			rp_sig = np.std(rp_chain)
			rp_snr = rp_med / rp_sig
		else:
			rp_snr = None

		# If parameter initial values are extreme in the posterior
		ch_flag = False
		for p in ('rp', 'per'):
			if p in self.parameter_names:
				p_idx = self.parameter_names.index(p)
				p_percentile = percentileofscore(chain[:, i], p_0[p_idx])

				if abs(p_percentile - 50.0) > 47.0:
					ch_flag = True

		if plot_posterior:
			fig = corner.corner(chain_df)
			fig.show()

		return chain_df, result_df, rp_snr, ch_flag

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

	def set_regular_bin_mode(self, bin_res=4, adjust_res=False):
		"""Initialises the bin mode for model-to-data conversion.

		Args:
			bin_res (int): the number of flux points binned into
				each data bin (the resolution).
			adjust_res (bool): If True, will adjust the bin_resolution
				based on the duration; to make sure at least 3 points
				are within a duration.

		Returns:
			None
		"""

		# Adjust resolution if below the minimum
		if adjust_res:
			bin_res = self.adjust_res(bin_res, max_points=50000,
									  points_per_dur=3)

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

	def adjust_res(self, bin_res=None, max_points=50000, points_per_dur=3):
		"""Adjusts bin_res for minimum number of points per transit.

		Args:
			bin_res (float): proposed bin_res; i.e minimum bin_res to
				use. If None, the return bin_res will be either the
				minimum resolution, or max_res.
			max_points (int): maximum number of points to have in a
				a lightcurve, for computational efficiency. If lightcurve
				is to be cut; perform the cut first. Default: 50,000
			points_per_dur (int): minimum number of points per duration
				to use; i.e to provide semi-accurate modelling.
				Default: 3 (low-ball)

		Returns:
			bin_res (int): adjusted bin_res
		"""

		# If done without 'duration'
		if not hasattr(self, 'params'):
			if not hasattr(self.params, 'duration'):
				warnings.warn(("'duration' not entered yet, cannot " \
							   + "adjust the bin_res."))
				return bin_res

		# Adjust resolution if below the minimum
		max_res = int(max_points / len(self.t_data))  # max is 50000 points

		if bin_res is None:
			bin_res = max_res

		# Argument:  ts / bin_res = duration / points_per_dur
		ts = np.median(self.t_data[1:] - self.t_data[:-1])
		min_bin_res = min(max_res, ts*points_per_dur/self['duration'])
		min_bin_res = int(np.ceil(min_bin_res))

		# Inadequate sampling warning
		if min_bin_res == max_res:
			warnings.warn(("Maximum resolution reached on bin_res; " \
						   + "sampling may be inadequate."))

		return max(bin_res, min_bin_res)

	def cut_lightcurve(self, num_durations=3):
		"""Cuts the lightcurves points that aren't near transits.

		NOTE: cuts based on the current stored parameters.

		TODO: make the cut-non-permanent

		Maintains the previous bin mode (though it recaculates it)

		Args:
			num_durations: number of durations to either side of
				transit to cut (i.e total length is 2*num_durations)
		"""

		t = self.t_data
		t0 = self['t0']
		dur = max(self['duration'], 2*30.0/24.0)
		per = self['per']

		if t0 > (min(t) + per):
			t0 = t0 - per*((t0 - np.nanmin(t))//per)

		mask = np.zeros_like(t, dtype=bool)

		for i in range(int((np.nanmax(t) - np.nanmin(t))/per) + 1):
			tt = t0 + i*per
			mask[(t >= (tt - dur*num_durations))\
				& (t < (tt + dur*num_durations))] = True

		# Save the uncut lightcurve
		if not hasattr(self, 't_saved'):
			self.t_saved = self.t_data.copy()
			self.f_saved = self.f_data.copy()

		# Set up and store the cut
		self.t_data = self.t_data[mask]
		self.f_data = self.f_data[mask]

		# Reset the bin model
		if not hasattr(self, 'bin_type'):
			pass
		elif self._bin_type == 'regular':
			self.set_regular_bin_mode(self._bin_res)
		elif self._bin_type == 'none':
			self.set_no_bin_mode()
		else:
			raise NotImplementedError("Bin model still hasn't been",
									  "added here.")

	def undo_cut(self):
		"""Restores the original lightcurve from memory."""

		if hasattr(self, 't_saved'):
			self.t_data = self.t_saved.copy()
			self.f_data = self.f_saved.copy()
		else:
			raise ValueError("Lightcurve hasn't been cut yet.",
							 "Or at least, t_saved doesn't exist.")

	def reset_step_size(self):
		"""Resets the step size (done for speed) to new value.
		"""
		raise NotImplementedError

	# Other
	# -----

	def verify_prior(self, vector=None, quiet=False, include_frozen=True,
					 info_str=None):
		"""Verifies the current vector with the prior."""

		initial_vector = np.copy(self.get_parameter_vector(include_frozen=True))

		if vector is None:
			vector = np.copy(np.copy(self.get_parameter_vector(include_frozen=include_frozen)))

		self.set_parameter_vector(vector, include_frozen=include_frozen)

		if np.isfinite(self.lnprior()):
			self.set_parameter_vector(initial_vector, include_frozen=True)
			return True
		elif quiet:
			self.set_parameter_vector(initial_vector, include_frozen=True)
			return False
		else:
			parameter_dict = self.get_parameter_dict(include_frozen=True)

			names = ('period', 'duration', 't0', 'rp', 'lower_a_lim',
					 'upper_a_lim')
			hpp = {name:getattr(self.Prior, '_' + name) for name in names}

			raise InvalidInitializationError(
				" Hyperparameters are out of bounds. Info str: {}".format(
											info_str),
				vector=vector, hpp=hpp,
				parameter_dict=parameter_dict
			)

	def plot_model(self, show=True):
		fig, ax = plt.subplots()

		ax.plot(self.t_data, self.f_data, 'k.')
		ax.plot(self._t_model, self.evaluate_model(bin_to_data=False), 'r-')

		if show:
			plt.show()
		else:
			fig.show()

	def plot_posterior(self, show=True):
		raise NotImplementedError

	def evaluate_model_at(self, t, pvector=None, p_fit=None, params=None):
		"""Evaluates transit model at specific set of times."""

		# Whether to reduce f to a len(t) array or keep as Nxlen(t)
		reduce_dim = False

		# Set up the params objects
		if pvector is not None:
			old_pvector = self.get_parameter_vector()
			if not np.ndim(pvector) == 2:
				pvector = np.expand_dims(pvector, axis=0)
				reduce_dim = True

			params = np.empty(len(pvector), dtype=object)
			for i in range(len(pvector)):
				self.set_parameter_vector(pvector[i])
				params[i] = copy.deepcopy(self.params)

			self.set_parameter_vector(old_pvector)

		elif p_fit is not None:
			old_pvector = self.get_parameter_vector()
			if not np.ndim(p_fit) == 2:
				if not isinstance(p_fit, pd.Series):
					p_fit = pd.Series(p_fit)
				p_fit = p_fit.to_frame().T
				reduce_dim = True
			elif not isinstance(p_fit, pd.DataFrame):
				p_fit = pd.DataFrame(p_fit)

			params = np.empty(len(p_fit), dtype=object)
			for i, idx in enumerate(p_fit.index):
				for name in self.get_parameter_names(include_frozen=True):
					if name in p_fit:
						self[name] = p_fit.loc[idx, name]
				params[i] = copy.deepcopy(self.params)

			self.set_parameter_vector(old_pvector)

		elif params is not None:
			if not np.ndim(params) == 2:
				params = np.expand_dims(params, axis=0)
				reduce_dim = True
			for i, param in enumerate(params):
				params[i] = copy.deepcopy(param)
		else:
			params = np.array([copy.deepcopy(self.params)])

		m = batman.TransitModel(params[0], t)
		bss = m.fac
		m = batman.TransitModel(params[0], t, fac=bss)

		f = np.empty([len(params), len(t)], dtype=object)
		for i in range(len(params)):
			f[i] = m.light_curve(params[i])

		if reduce_dim:
			f = f[0]

		return f


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


# Exceptions
# ----------

class UnorderedLightcurveError(Exception):
	pass

class NegativeDepthError(ValueError):
	pass

class InvalidInitializationError(ValueError):
	"""Thrown when the prior verification fail."""

	def __init__(self, message=None, vector=None, hpp=None,
				 parameter_dict=None):

		self.parameter_vector = vector
		self.hpp = hpp
		self.parameter_dict = parameter_dict

		if message is None:
			message = ''
		message += "\nParsing:\n" +\
				   "parameter\t: {}\n\n".format(parameter_dict) +\
				   "hp\t\t: {}\n\n".format(vector) +\
				   "hpp\t\t: {}\n".format(hpp)

		super().__init__(message)
