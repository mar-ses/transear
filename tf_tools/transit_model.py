"""A transit model wrapper around Batman.

Expected to be inherited by the TransitFitter object, which
extends the object by introducing probabilities, fitting,
priors and so on...

This object must contain all the property methods which
involve the batman params; as well as model evaluation at
certain sets of points, and also including model binning.

Additional expectation: replacement of the model in
injection. (TODO)

TODO: StellarTransitModel - child object which replaces the
    specification of 'a', and possibly other parameters, with
    M_star and R_star. To be figure out. Possibly also replace
    rp with R_p, the actual planet radius.

    NOTE: will involve re-casting the derived vs. stored vs.
    base parameters; redoing the init and classmethod, and
    some of the internals. The idea is that R_star, R_p and
    M_star are going to be base parameters, and a, rp and so
    on are the derived parameters outside; but the internals
    must automatically and perpetually maintain the correction
    from R_star, R_p, and M_star into the internal batman.params
"""

import copy
import warnings
from collections import OrderedDict

import batman
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units, constants as const

from .. import util_lib
from ..__init__ import HOME_DIR

# Transit modelling object
# ------------------------

class TransitModel(object):
    """Contains batman.TransitParams and batman.TransitModel.

    Expected to be inherited by the TransitFitter object, which
    extends the object by introducing probabilities, fitting,
    priors and so on...

    This object must contain all the property methods which
    involve the batman params; as well as model evaluation at
    certain sets of points, and also including model binning.
    Also contains the concept of "frozen" parameters.

    Still requires the knowledge of a series of "times" in which
    to product the model; though flux isn't required as this
    doesn't care about comparisons.

    NOTE: regarding parameter names, period is 'per', and 'u'
    is split into u1 and u2.

    Additional initialisation class methods:
        .from_batman(t, per, t0, rp, a, ..., **kwargs): creates
            the object from the raw batman params; i.e using 'a'
        .from_bls(t, per, t0, depth, duration, R_star, M_star, ...):
        .from_physical(t, per, t0, rp, R_star, M_star, ...):
    """

    _param_names = ('per', 't0', 'rp', 'a', 'inc', 'ecc', 'w', 'u1', 'u2')
    _additional_param_names = ('R_star',)
    _pvector_names = _param_names + _additional_param_names
    _derived_parameter_names = ('depth', 'R_p', 'duration', 'b', 'M_star')

    def __init__(self, t, per, t0, rp, a, R_star=None,
                 bin_type='regular', bin_res=6, adjust_res=False,
                 **kwargs):
        """Initalise an instance of the model on a set of times.

        The parameters must be given in terms of "batman" params,
        not BLS (implement class methods for that). Except for 'a',
        which is given in terms of R_star and M_star.

        The times **MUST** be sorted in ascending order by time.

        Args:
            t (np.array): time points in the lightcurve
            per, t0, rp, a (floats): the values to
                initialise the object with. Suggested usage is to
                unpack a dictionary: **bls_peaks.loc[i, ['period',...]]
            bin_type (str): out of ['none', 'regular', 'nearby']
                Values other than the above assume 'none'
            bin_res (int): if binning is not 'none', then this is the
                number of model points used to average in each bin
            adjust_res (bool): if True, will readjust the bin resolution
                to make sure a minimum number of points are included
                per bin; based on the estimate duration
            **kwargs: additional batman params to set,
                include: inc, ecc, w, **u**

        Raise:
            UnorderLightCurveError: the time points in the lightcurve
                are not in ascending order
        """

        # Checks on input
        if rp < 0:
            raise NegativeDepthError

        inputs = [per, t0, rp, a] + [p for k,p in kwargs.items()
                                       if k not in ('u', 'limb_dark')]
        if np.any(np.isnan(inputs)):
            raise ValueError(
                ("Some fit parameters were NaN. "
               + "Values: {}".format([per, t0, rp, a] \
                                   + [p for p in kwargs.values()])))

        self.params = batman.TransitParams()

        # Set the initial-value hyperparameters
        self.params.t0 = t0
        self.params.per = per
        self.params.rp = rp					# in units of stellar radius
        self.params.a = a					# in units of stellar radius
        self.params.inc = 90. if 'inc' not in kwargs else kwargs['inc']
        self.params.ecc = 0. if 'ecc' not in kwargs else kwargs['ecc']
        self.params.w = 90. if 'w' not in kwargs else kwargs['w']
        self.params.limb_dark = "quadratic"
        self.params.u = [0.1, 0.3]  if 'u' not in kwargs\
                                    else copy.copy(kwargs['u'])

        # From now; R_star MUST exist; though it may be np.nan
        if R_star is None:
            # R_star is nan should be OK; unless the prior
            # includes R_star
            self['R_star'] = np.nan
        else:
            self['R_star'] = R_star

        # Set the model timeseries
        self.set_timeseries(t, bin_res=bin_res, adjust_res=adjust_res,
                            bin_type=bin_type)

        # Initialise internals
        self._unfrozen_mask = np.ones_like(self._pvector_names, dtype=bool)
        self.set_active_vector(('per', 't0', 'rp', 'a', 'inc'))

        # Last line check (temporary) to make sure that the internal
        # parameter names match the batman parametrisation
        assert np.all([hasattr(self.params, n) for n in self._param_names[:-2]])


    # Class methods for initialisation
    # --------------------------------

    @classmethod
    def from_batman(cls, t, batman_params, **kwargs):
        """Initiates a TransitModel object from batman object."""

        # Convert to observational parameters
        # ...

        if batman_params.limb_dark != 'quadratic':
            raise ValueError("Only working with quadratic limb-darkening.")

        t0 = batman_params.t0
        per = batman_params.per
        rp = batman_params.rp
        a = batman_params.a
        inc = batman_params.inc
        ecc = batman_params.ecc
        w = batman_params.w
        u = batman_params.u

        return cls(t=t, t0=t0, per=per, rp=rp, a=a, inc=inc, ecc=ecc,
                   w=w, u=u, **kwargs)

    @classmethod
    def from_bls(cls, t, per, t0, depth, duration, **kwargs):
        """Models the transit from bls parametrisation.

        Assumes default R_star of 0.1R_*.
        """

        rp = np.sqrt(depth)
        a = per / (np.pi * duration)	# Likely to be overestimate

        return cls(t=t, per=per, t0=t0, rp=rp, a=a, **kwargs)

    @classmethod
    def from_stellar_params(cls, t, per, t0, rp, R_star,
                            M_star, freeze_a=True, **kwargs):
        """With a fixed `a` based on stellar mass.

        """

        # Calculate first value of a
        M_fac = (0.5*per*units.day/np.pi)**(2/3) * (0.5*const.G)**(1/3)
        a0 = (M_fac*(M_star*const.M_sun)**(1/3) / (R_star*const.R_sun)).to('').value

        obj = cls(t=t, per=per, t0=t0, rp=rp, a=a0,
                  R_star=R_star, **kwargs)
        
        if freeze_a:
            obj.freeze_parameter('a')

        return obj


    # Internal operator work-methods
    # ------------------------------

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
        elif name in self._additional_param_names:
            return getattr(self, name)
        elif name in self._derived_parameter_names:
            return self._getter_dict[name](self)
        # elif name in self._additional_stored_names:
        # 	return self._additional_getter_dict[name]()
        else:
            raise ValueError("{} not found in batman.params.".format(name))

    def __setitem__(self, name, value):
        if name in ('u1', 'u2'):
            self.params.u[('u1', 'u2').index(name)] = value
        elif name in self._param_names:
            setattr(self.params, name, value)
        elif name == 'u':
            setattr(self.params, 'u', value)
        elif name in self._additional_param_names:
            setattr(self, name, value)
        elif name in self._derived_parameter_names:
            raise NotImplementedError('Cannot set by derived params.')
        # elif name in self._additional_stored_names:
        # 	self._additional_setter_dict[name](value)
        else:
            raise ValueError("{} not found in batman.params.".format(name))

    def __copy__(self):
        arg_dict = {key:self[key] for key in self._pvector_names[:-2]}
        arg_dict['u'] = self['u']

        a = type(self)(t=copy.copy(self.t_data),
                       bin_type=self._bin_type,
                       bin_res=self._bin_res,
                       adjust_res=False,
                       **arg_dict)

        return a

    def full_size(self):
        return len(self._pvector_names)


    # Properties and parameters
    # -------------------------

    @property
    def unfrozen_mask(self):
        return self._unfrozen_mask

    def get_parameter_names(self, include_frozen=False):
        if not include_frozen:
            return list(name for i, name in enumerate(self._pvector_names)
                         if self.unfrozen_mask[i])
        else:
            return list(self._pvector_names)

    # By default, doesn't include frozen parameters
    parameter_names = property(get_parameter_names)

    def get_parameter_vector(self, include_frozen=False):
        # TODO: speed up by removing looped function calls;
        # instead, rewrite the stuff right in here.
        # Allow the dict-access to take list parameters
        return np.array(
                    [self[n]
                    for n in self.get_parameter_names(include_frozen)])

    def set_parameter_vector(self, vector, include_frozen=False):
        # TODO: speed up by removing looped function calls;
        # instead, rewrite the stuff right in here.
        for i, n in enumerate(self.get_parameter_names(include_frozen)):
            # Allow the dict-access to take list parameters
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

        self._unfrozen_mask[self._pvector_names.index(name)] = False

    def thaw_parameter(self, name):
        """Freeze a single parameter."""

        self._unfrozen_mask[self._pvector_names.index(name)] = True

    def set_active_vector(self, names):
        for name in self._pvector_names:
            if name in names:
                self.thaw_parameter(name)
            else:
                self.freeze_parameter(name)

    # Conversion methods

    def get_depth(self):
        return self['rp']**2

    def get_b(self):
        return self['a'] * np.cos(2*np.pi*self['inc']/360.0)

    def get_R_p(self, R_earth=True):
        """Returns planet radius in earth units by default.
        
        Args:
            R_earth (bool): if True, in earth units, otherwise in R_sun
        """
        
        # In earth units
        if R_earth:
            R_star = self['R_star'] * (const.R_sun / const.R_earth).to('')
        else:
            R_star = self['R_star']
        return self['rp'] * R_star

    def get_duration(self):
        b_factor = np.sqrt(1 - np.cos(self['inc'])**2 / self['a']**2)
        return b_factor * self['per'] / (np.pi*self['a'])

    def get_M_star(self):
        M_fac = (0.5 * self['per'] * units.day / np.pi)**(2/3) \
              * (0.5*const.G)**(1/3)
        M_star = (((self['a'] * self['R_star'] * const.R_sun) \
                 / (M_fac * const.M_sun**(1/3)))**3).to('').value
        return M_star

    _getter_dict = {'b':get_b,
                    'R_p':get_R_p,
                    'depth':get_depth,
                    'duration':get_duration,
                    'M_star':get_M_star}


    # Model evaluation
    # ----------------

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

    def evaluate_model_at(self, t, pvector=None, p_fit=None, params=None):
        """Evaluates transit model at specific set of times.
)
        TODO: currently terrible and over-engineered; change
        this to just plot at specific times, and then instead
        do the looping over things and putting in parameters
        separate (?), or not perhaps.

        Args:
            t (np.array): times at which to calculate model
            pvector (np.array): to enter if you have an array
                that functions like the parameter_vector in this
                object.
            p_fit (pandas Object): named columns or values, or
                dict; enter here if inputting the chain output
                from sample_posteriors.
            params (batman.TransitParams): enter the direct
                params objects.
        """

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
        elif self._bin_type == 'regular':
            assert len(f_model) % self._bin_res == 0

            f_binned = np.zeros(int(len(f_model)//self._bin_res),
                                dtype=float)
            index_base = np.array(range(len(f_binned))) * self._bin_res

            for i in range(self._bin_res):
                f_binned += f_model[index_base + i]

            return f_binned / self._bin_res
            #return bin_model_regular(f_model, num_per_bin=self._bin_res)
        else:
            raise ValueError("self._bin_type not recognised.")

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
                                      points_per_dur=5)

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
        ts = np.nanmedian(self.t_data[1:] - self.t_data[:-1])
        min_bin_res = min(max_res, ts*points_per_dur/self['duration'])
        min_bin_res = int(np.ceil(min_bin_res))

        # Inadequate sampling warning
        if min_bin_res == max_res:
            warnings.warn(("Maximum resolution reached on bin_res; " \
                           + "sampling may be inadequate."))

        return max(bin_res, min_bin_res)

    def reset_step_size(self):
        """Resets the step size (done for speed) to new value.
        """
        raise NotImplementedError


    # Other
    # -----

    def set_timeseries(self, t, bin_res, adjust_res, bin_type='regular'):
        """Sets the timeseries on which to calculate the model."""

        # Checks on input
        if not np.all(np.diff(t) >= 0):
            raise UnorderedLightcurveError("Lightcurve timeseries is not ordered by time.")

        # Save the data
        self.t_data = np.array(t)
    
        # Set bin mode - needs to be here to adjust resolution
        if bin_type == 'regular':
            self.set_regular_bin_mode(bin_res, adjust_res=adjust_res)
        else:
            self.set_no_bin_mode()

        # Initialise the model
        self.m = batman.TransitModel(self.params, self._t_model)
        self.bss = self.m.fac
        self.m = batman.TransitModel(self.params, self._t_model,
                                     fac=self.bss)

    def calc_a(self, R_star, M_star, per=None, set_value=False):
        per = self['per'] if per is None else per
    
        M_fac = (0.5*per*units.day/np.pi)**(2/3) * const.G**(1/3)
        a = (M_fac*(M_star*const.M_sun)**(1/3) / (R_star*const.R_sun)).to('').value

        if set_value:
            self['a'] = a

        return a

    def enforce_M_star(self, R_star, M_star, per=None):
        """Basic enforcement of a stellar mass, by freezing a.

        Args:
            R_star, M_star, per
        """

        a = self.calc_a(R_star, M_star, per)
        
        self['a'] = a
        self.freeze_parameter('a')

    def plot_model(self, show=True):
        fig, ax = plt.subplots()

        ax.plot(self.t_data, self.evaluate_model(bin_to_data=True), 'k.')
        ax.plot(self._t_model, self.evaluate_model(bin_to_data=False),
                'r-', alpha=0.7, zorder=-1)

        if show:
            plt.show()
        else:
            fig.show()


class StellarTransitModel(TransitModel):
    """Overloaded to use R_star, R_p and M_star as fittings parameters.

    TODO: have an internal stored R_p_unit, R_star_unit, M_star_unit.

    TODO: current issue - Can't really set this thing as a vector;
          I need to figure out what affects what and in what order
          basically. Like if I set_vector for period and M_star,

          fgd
    """

    _param_names = ('per', 't0', 'rp', 'a', 'inc', 'ecc', 'w', 'u1', 'u2')
    _pvector_names = ('per', 't0', 'R_p', 'R_star', 'M_star',
                      'inc', 'ecc', 'w', 'u1', 'u2')
    _derived_parameter_names = ('depth', 'duration', 'b', 'A')
    _additional_stored_names = ('R_p', 'R_star', 'M_star')

    def __init__(self, t, per, t0, R_p, M_star, R_star,
                 bin_type='regular', bin_res=6,
                 adjust_res=False, **kwargs):
        """

        NOTE: R_p is in earth radii. R_star and M_star in solar units.
        """

        rp0 = (R_p*const.R_earth / (R_star*const.R_sun)).to('').value

        # Calculate first value of a
        M_fac = (0.5*per*units.day/np.pi)**(2/3) * (0.5*const.G)**(1/3)
        a0 = (M_fac*(M_star*const.M_sun)**(1/3) / (R_star*const.R_sun)).to('').value

        super().__init__(t=t, per=per, t0=t0, rp=rp0, a=a0,
                         bin_type=bin_type, bin_res=bin_res,
                         adjust_res=adjust_res, **kwargs)

        self['R_star'] = R_star


    # Class methods for initialisation
    # --------------------------------

    @classmethod
    def from_batman(cls, t, batman_params, R_star, **kwargs):
        """Initiates a TransitModel object from batman object."""

        # Convert to observational parameters
        # ...

        if batman_params.limb_dark != 'quadratic':
            raise ValueError("Only working with quadratic limb-darkening.")

        t0 = batman_params.t0
        per = batman_params.per
        rp = batman_params.rp
        a = batman_params.a
        inc = batman_params.inc
        ecc = batman_params.ecc
        w = batman_params.w
        u = batman_params.u

        R_p = (rp * R_star * const.R_sun / const.R_earth).to('').value
        M_fac = (0.5*per*units.day/np.pi)**(2/3) * (0.5*const.G)**(1/3)
        M_star = (((a*R_star*const.R_sun) / (M_fac*(const.M_sun)**(1/3)))**3).to('').value

        return cls(t=t, t0=t0, per=per, R_p=R_p, R_star=R_star,
                   M_star=M_star, inc=inc, ecc=ecc, w=w, u=u, **kwargs)

    @classmethod
    def from_bls(cls, t, per, t0, depth, duration, **kwargs):
        """Models the transit from bls parametrisation.

        Assumes default R_star of 0.1R_*.
        """

        rp = (np.sqrt(depth) * R_star * const.R_sun / const.R_earth).to('').value

        a = per / (np.pi * duration)
        M_fac = (0.5*per*units.day/np.pi)**(2/3) * (0.5*const.G)**(1/3)
        M_star = (((a*R_star*const.R_sun) / (M_fac*(const.M_sun)**(1/3)))**3).to('').value

        return cls(t=t, per=per, t0=t0, R_p=R_p, R_star=R_star,
                   M_star=M_star, **kwargs)


    # Internal operator work-methods
    # ------------------------------

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
        elif name == 'R_star':
            return getattr(self, 'R_star')
        elif name in self._additional_stored_names:
            return self._additional_getter_dict[name]()
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
        elif name == 'R_star':
            setattr(self, 'R_star', value)
        elif name in self._additional_stored_names:
            self._additional_setter_dict[name](value)
        else:
            raise ValueError("{} not found in batman.params.".format(name))

    # Conversion methods

    # def get_depth(self):
    #     return self['rp']**2

    # def get_b(self):
    #     return self['a'] * np.cos(2*np.pi*self['inc']/360.0)

    # def get_duration(self):
    #     return self['per'] / (np.pi*self['a'])

    # _getter_dict = {'b':get_b,
    #                 'R_p':get_R_p,
    #                 'depth':get_depth,
    #                 'duration':get_duration}


# Utility functions
# -----------------

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
