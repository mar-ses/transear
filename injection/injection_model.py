"""Library for modelling transits for injection

Contains:
    - TransitModel object; with utility function definitions
"""

import sys
import copy

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from astropy import units, constants as const
#import batman

from ..__init__ import HOME_DIR
#from .. import util_lib
from ..tf_tools import transit_model
sys.path.append(HOME_DIR)

# Transit model object
# --------------------

class InjectionModel(transit_model.TransitModel):
    """Contains the parameters of the transit model."""

    def __init__(self, t, P, t0, R_p, M_star, R_star, ecc=0.0,
                 bin_type='regular', bin_res=20, adjust_res=False,
                 multiply_in=True, **kwargs):
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
        rp = (R_p * const.R_earth / (R_star*const.R_sun)).to('').value

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
        self._multin = multiply_in
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
            self.set_timeseries(lcf.t.values,
                                bin_res=self._bin_res,
                                bin_type=self._bin_type,
                                adjust_res=adjust_res)

        f_model = self.evaluate_model() - 1.0

        if plot:
            plt.plot(lcf.t, lcf.f + f_model, '.', c='0.5', alpha=0.5)
            plt.plot(lcf.t, lcf.f_detrended + f_model, 'k.', alpha=0.8)
            plt.plot(lcf.t, f_model + np.nanmedian(lcf.f_detrended),
                     'r-', alpha=0.8)
            plt.show()

        for col in f_col:
            if not self._multin:
                lcf[col] = lcf[col] + f_model
            else:
                # multiply in; need to make the baseline be 1.0
                lcf[col] = lcf[col] * (f_model + 1.0)
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

        if code is None or code in ('none', 'free'):
            # Default is automatically minimal ('none') overlap enforcement,
            # i.e b < 1 + rp
            min_inc = np.rad2deg(np.arccos((1 + self['rp'])/self['a']))
        elif code in ('half',):
            # Default is automatically minimal ('none') overlap enforcement,
            # i.e b < 1 + rp
            min_inc = np.rad2deg(np.arccos(1/self['a']))
        elif code in ('full',):
            # Default is automatically minimal ('none') overlap enforcement,
            # i.e b < 1 + rp
            min_inc = np.rad2deg(np.arccos((1 - self['rp'])/self['a']))
        else:
            raise ValueError("code argument not recognised:", code)
        # elif not isinstance(code, float):
        # 	raise ValueError("inc argument not recognised:", inc)

        inc = (max_inc-min_inc)*np.random.random() + min_inc

        if set_value:
            self['inc'] = inc

        return inc


# Errors
# ------

class HPNotFoundError(Exception):
    pass

