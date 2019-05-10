"""Library for modelling transits for injection

Contains:
	- TransitModel object; with utility function definitions
"""

import os
import sys
import copy
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from astropy import units, constants as const
import batman

from ..__init__ import HOME_DIR
from .. import util_lib
from ..tf_tools import transit_model
sys.path.append(HOME_DIR)

# Transit model object
# --------------------

class InjectionModel(transit_model.TransitModel):
	"""Contains the parameters of the transit model."""

	def __init__(self, t, P, t0, R_p, M_star, R_star, ecc=0.0,
				 bin_type='regular', bin_res=20, adjust_res=False,
				 **kwargs):
		"""Initiates a TransitModel object for injection.
		
		A is derived from R_star, M_star and P; ecc can be added later.
		Only difference here is that R_p is given in *earth units*.

		Args:
			P (float): days
			t0 (float): days
			R_p (float): R_earth
			R_star (float): in R_sun
			M_star (float): in M_sun
			ecc (float): eccentry, default is 0.0
			**kwargs
		"""

		# Calculate first value of a
		rp = (R_p * const.R_earth / (R_star*const.R_sun)).to('')

		if ecc != 0.0:
			raise NotImplementedError(("Non-zero eccentricity still not "
									   "possible."))

		# Calculate first value of a
		M_fac = (0.5*P*units.day/np.pi)**(2/3) * (0.5*const.G)**(1/3)
		a0 = (M_fac*(M_star*const.M_sun)**(1/3) / (R_star*const.R_sun)).to('').value

		super().__init__(t=t, per=P, t0=t0, rp=rp, a=a0, R_star=R_star,
						 bin_type=bin_type, bin_res=bin_res,
						 adjust_res=adjust_res, **kwargs)
		self.M_star_0 = M_star

		self.set_active_vector([])

	# Production methods
	# ------------------

	def get_bls_params(self):
		"""Produces a dictionary with the bls parameters."""

		return {'period':self['P'], 't0':self['t0'], 'depth':self['depth'],
				'duration':self['duration']}

	def get_batman_params(self):
		"""Produces batman.params from the internal state."""

		return self.params

	def inject_transit(self, lcf, f_col=None, plot=False, adjust_res=True):
		"""Injects the transit signal into a copy of the lcf.
		
		NOTE: injects into both f and f_detrended; but NOT f_temporal.
		
		Args:
			lcf (pd.DataFrame)
			f_col (list): list of columns where to add the transit
			plot (bool): plot model and lightcurve
			adjust_res (bool): if entering a new light,
				to adjust bins_res
		"""

		lcf = lcf.copy()
		if f_col is None:
			f_col = ['f', 'f_detrended', 'f_temporal']
		elif not hasattr(f_col, '__len__'):
			f_col = [f_col]

		if not (lcf.t == self.t_data).all():
			self.set_timeseries(lcf.t.values, self._bin_res, self._bin_type,
								adjust_res=adjust_res)

		f_model = self.evaluate_model() - 1.0

		if plot:
			plt.plot(lcf.t, lcf.f + f_model, '.', c='0.5', alpha=0.5)
			plt.plot(lcf.t, lcf.f_detrended + f_model, 'k.', alpha=0.8)
			plt.plot(lcf.t, f_model + np.nanmedian(lcf.f_detrended),
					 'r-', alpha=0.8)
			plt.show()

		for col in f_col:
			lcf[col] = lcf[col] + f_model
		return lcf

	def jiggle_params(self, return_in='bls', jiggle_ecc=False, save=False):
		"""Randomises the parameters with very small changes.

		To jiggle:
			P, t0, rp, A, e

		Args:
			save (bool): if True, save the parameter values
			return_in (str): which format to return it in
				options: 'bls' (default), 'batman', 'model'
		"""

		if not save:
			saved_params = copy.deepcopy(self.params)

		self['P'] = self['P']*(1 + 0.001*norm.rvs())
		self['t0'] = self['t0'] + self['duration']*0.1*norm.rvs()
		self['rp'] = self['rp']*(1 + 0.05*norm.rvs())
		self['a'] = self['a']*(1 + 0.05*norm.rvs())
		if jiggle_ecc:
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
			self.params = saved_params

		return out

	def randomise_inclination(self, code='none', set_value=True):
		"""Picks a random inclination in degrees, below 90.0.
		
		Args:
			code: 'none' allows any degree of overlap (b < 1 + rp)
				'half' requires half-overlap (b < 1)
				'full' requires full-overlap (b < 1 - rp)
				default: 'none'
		"""

		max_inc = 90.0

		if inc is None or inc in ('none', 'free'):
			# Default is automatically minimal ('none') overlap enforcement,
			# i.e b < 1 + rp
			min_inc = np.rad2deg(np.arccos((1 + self['rp'])/self['a']))
		elif inc in ('half',):
			# Default is automatically minimal ('none') overlap enforcement,
			# i.e b < 1 + rp
			min_inc = np.rad2deg(np.arccos(1/self['a']))
		elif inc in ('full',):
			# Default is automatically minimal ('none') overlap enforcement,
			# i.e b < 1 + rp
			min_inc = np.rad2deg(np.arccos((1 - self['rp'])/self['a']))
		elif not isinstance(inc, float):
			raise ValueError("inc argument not recognised:", inc)

		inc = (max_inc-min_inc)*np.random.random() + min_inc

		if set_value:
			self['inc'] = inc

		return inc









# Transit model object (OBSOLETE /////)
# --------------------

class TransitModel(object):
	"""Contains the parameters of the transit model."""

	def __init__(self, P, t0, rr, A, ecc=0.0, inc=90.0, w=0.0,
				 u=None, R_star=None, M_star=None):
		"""Main creation routine from stored parameters.

		Args:
			P
			t0
			rr
			A
			e
			w
			u
			R_star
			M_star
		"""

		if u is None:
			u = [0.1, 0.3]
		if R_star is None:
			R_star = 0.1			# The trappist radius (in solar radii)
		if M_star is None:
			M_star = 0.081		# The trappist mass (in solar masses)

		self['P'] = P
		self['t0'] = t0
		self['rr'] = rr
		self['A'] = A
		self['ecc'] = ecc
		self['inc'] = inc
		self['w'] = w
		self['u'] = u
		self['R_star'] = R_star
		self['M_star'] = M_star

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
			A (float): in units of R_star (not R_sun)
				TODO: improve this
			R_star (float): in R_sun, default is 0.1
			M_star (float): in M_sun, TODO
		"""

		R_star = 0.1		# in R_sun
		rr = (R_p * const.R_earth / (R_star*const.R_sun)).to('')

		# Convert to observational parameters
		# ...

		return cls(P=P, t0=t0, rr=rr, A=A, R_star=R_star)

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

		Assumes default R_star of 0.1R_*.
		"""

		R_star = 0.1
		rr = np.sqrt(depth)
		A = per / (np.pi * duration)

		return cls(P=per, t0=t0, rr=rr, A=A,
						R_star=R_star)

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
		return self['rr'] * self['R_star']

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

	def randomise_inclination(self, code='none', set_value=True):
		"""Picks a random inclination in degrees, below 90.0.
		
		Args:
			code: 'none' allows any degree of overlap (b < 1 + rp)
				'half' requires half-overlap (b < 1)
				'full' requires full-overlap (b < 1 - rp)
				default: 'none'
		"""

		max_inc = 90.0

		if inc is None or inc in ('none', 'free'):
			# Default is automatically minimal ('none') overlap enforcement,
			# i.e b < 1 + rp
			min_inc = np.rad2deg(np.arccos((1 + self['rp'])/self['a']))
		elif inc in ('half',):
			# Default is automatically minimal ('none') overlap enforcement,
			# i.e b < 1 + rp
			min_inc = np.rad2deg(np.arccos(1/self['a']))
		elif inc in ('full',):
			# Default is automatically minimal ('none') overlap enforcement,
			# i.e b < 1 + rp
			min_inc = np.rad2deg(np.arccos((1 - self['rp'])/self['a']))
		elif not isinstance(inc, float):
			raise ValueError("inc argument not recognised:", inc)

		inc = (max_inc-min_inc)*np.random.random() + min_inc

		if set_value:
			self['inc'] = inc

		return inc


# Errors
# ------

class HPNotFoundError(Exception):
	pass

