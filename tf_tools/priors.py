

import numpy as np
from astropy import units, constants as const
from scipy.stats import norm, truncnorm


from ..__init__ import HOME_DIR, K2GP_DIR



# Transit prior object
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

    _param_names = ('per', 't0', 'rp', 'a', 'inc', 'ecc', 'w', 'u1', 'u2',
                    'R_star')
    _hpp_names = ('_duration', '_period', '_t0', '_rp', '_num_transits',
                  '_lower_a_lim', '_upper_a_lim', '_overlap_lim')

    def __init__(self, period, t0, rp, duration,
                 **hpp_kwargs):
        """TODO

        Args:
            period, t0, rp, duration: required values
            **hpp_kwargs:
                time_baseline=40,
                overlap_lim='partial',
                stellar_class='ultracool',
                radius_constraint='stellar',
                verbose=False
        """

        self.set_hyperparameters(period, t0, rp, duration,
                                 **hpp_kwargs)

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
        if p[-3] + p[-2] > 1:
            return -np.inf

        # Eccentricity; basic
        if not 0.0 <= p[self._param_names.index('ecc')] < 1.0:
            return -np.inf

        # rp basic
        if not 0.0 < p[2] < self._r_max:
            return -np.inf

        # a
        if (p[3] < self._lower_a_lim) or (p[3] > self._upper_a_lim):
            return - np.inf

        # inc
        if p[4] > 90.0 or p[4] < 0.0:
            # i.e keep it on one side for easier mcmc working
            return -np.inf
        if p[3]*np.cos(np.deg2rad(p[4])) > 1 + p[2]*self._overlap_lim:
            # NOTE: this entirely relies on inc < 90
            # p[3]*np.cos(np.deg2rad(p[4])) == b
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

    def verify_lnprior(self, p):
        """Returns which parameter is out of bounds."""

        if self.num_params != len(p):
            raise ValueError("Vector was of the wrong dimension, expected:",
                             self.num_params, "received:", len(p))

        # The limb-darkening
        if p[-3] + p[-2] > 1:
            return 'u_basic'

        # Eccentricity; basic
        if not 0.0 <= p[self._param_names.index('ecc')] < 1.0:
            return 'ecc'

        # rp basic
        if not 0.0 < p[2] < self._r_max:
            return 'rp_basic'

        # a
        if (p[3] < self._lower_a_lim) or (p[3] > self._upper_a_lim):
            return 'a_out_of_bounds'

        # inc
        if p[4] > 90.0 or p[4] < 0.0:
            # i.e keep it on one side for easier mcmc working
            return 'inc_basic_bounds'
        if p[3]*np.cos(np.deg2rad(p[4])) > 1 + p[2]*self._overlap_lim:
            # NOTE: this entirely relies on inc < 90
            # p[3]*np.cos(np.deg2rad(p[4])) == b
            return 'inc_overlap'

        # Hard limits on the period
        if abs(p[0] - self._period) > (3*self._duration/self._num_transits):
            return 'period'

        # period
        lnpdf_per = norm.logpdf(p[0], self._period,
                                self._duration / self._num_transits)
        # t0
        lnpdf_t0 = norm.logpdf(p[1], self._t0, self._duration)
        # rp
        lnpdf_rp = norm.logpdf(p[2], self._rp, 2*self._rp)

        return lnpdf_per + lnpdf_t0 + lnpdf_rp

    def set_hyperparameters(self, period, t0, rp, duration,
                            time_baseline=40, verbose=False,
                            overlap_lim='partial', stellar_class='ultracool',
                            radius_constraint='stellar'):
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
            overlap_lim (str): the minimum amount of overlap required;
                if 'full': planet must fully occult (olim = -1.0)
                if 'partial': planet must half-occult (olim = 0.0)
                if 'none': planet can graze at its minimum (olim = 1.0)
                Alternatively, give a float corresponding to fraction
                of planetarty radius that must overlap
            stellar_class (str): 'M', 'ultracool'
                for setting a-limits
        """

        # Other limits
        # ------------
        self._duration = duration
        self._period = period
        self._t0 = t0
        self._rp = rp
        self._num_transits = time_baseline / period

        # Planet radius constraints
        # -------------------------

        if radius_constraint == 'large':
            self._r_max = 2.0       # Planet can be twice the stellar radius
        elif radius_constraint in ('star', 'stellar'):
            self._r_max = 1.0       # Up to stellar radius
        elif radius_constraint == 'small':
            self._r_max = 0.5
        else:
            raise ValueError(("Unrecognised radius constraint: "
                              "{}".format(radius_constraint)))

        # Stellar constraints go into limits on the semi-major axis
        # ---------------------------------------------------------

        # Physical arguments for priors and initialization
        # For a 0.5 M_S mass star, a in terms of period is:
        # T = 2*pi*sqrt(a**3 / GM_*)
        # a = (T/2pi)**(2/3) * (0.5GM_S)**(1/3)
        # Upper limit, divide by Jupiter radius, 0.5 M_sun
        # Lower limit, take TRAPPIST-1 mass (0.08M_*), and like 0.2 Solar radii

        if stellar_class == 'M':
            max_mass = 0.5
            min_mass = 0.050        # 50 Jupiter masses
            max_rad = 0.4
            min_rad = 0.1
        elif stellar_class =='ultracool':
            max_mass = 0.5
            min_mass = 0.025        # 25 Jupiter masses
            max_rad = 0.4
            min_rad = 0.04          # 0.4 Jupiter radii
        else:
            raise NotImplementedError("Other stellar classes not" \
                                      "available yet for 'a' prior.")

        M_fac = (0.5*period*units.day/np.pi)**(2/3) * (0.5*const.G)**(1/3)
        self._lower_a_lim = (M_fac*(min_mass*const.M_sun)**(1/3) / (max_rad*const.R_sun)).to('')
        self._upper_a_lim = (M_fac*(max_mass*const.M_sun)**(1/3) / (min_rad*const.R_sun)).to('')

        if isinstance(overlap_lim, str):
            if 'full': self._overlap_lim = -1.0
            elif 'partial': self._overlap_lim = 0.0
            elif 'none': self._overlap_lim = 1.0
            else:
                raise ValueError('overlap_lim not recognised:', overlap_lim)
        elif isinstance(overlap_lim, (int, float)):
            self._overlap_lim = overlap_lim
        else:
            raise ValueError('overlap_lim not recognised:', overlap_lim)

        if verbose:
            print("A limits set at: [{}, {}]".format(self._lower_a_lim,
                                                     self._upper_a_lim))

    def get_hpp_dict(self):
        hpp = dict()

        for hpp_name in self._hpp_names:
            hpp[hpp_name] = getattr(self, hpp_name)

        return hpp

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



# Base object (TODO: interface object)
# ------------------------------------

# TODO: this is actually a completely general prior model object;
# can be shared with the gp package.

class TransitPrior(object):
    """Interface for the prior object, can be called as a function.

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

    _param_names = (None,)
    _hpp_names = (None,)

    def __init__(self, *hpp_args, **hpp_kwargs):
        """Sets the hyperparameter values."""
        self.set_hyperparameters(*hpp_args, **hpp_kwargs)

    def __call__(self, p):
        """Must work on all the parameters, active or not."""
        return self.calc_lnprior(p)

    def __len__(self):
        return len(self._param_names)

    @property
    def num_params(self):
        return len(self)

    def calc_lnprior(self, p):
        """Calculate the log-prior for the **full** parameters p."""
        # TODO: must be made more general (somehow)
        raise NotImplementedError("Placeholder method for interface.")

    def set_hyperparameters(self, *hpp_args, **hpp_kwargs):
        """Sets the hyperparameters of the prior."""
        raise NotImplementedError

    def get_hpp_dict(self):
        hpp = dict()

        for hpp_name in self._hpp_names:
            dict[hpp_name] = getattr(self, hpp_name)

        return hpp


