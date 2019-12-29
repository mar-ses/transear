"""Object for fitting transit models to lightcurve data.

Straps on probabilistic methods, as well as optimisation to
the standard tf_tools.TransitModel object.
"""

import copy
import warnings

import corner
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_powell
from scipy.stats import percentileofscore
from astropy import units, constants as const

import batman

from .transit_model import TransitModel
from .priors import BasicTransitPrior, PhysicalTransitPrior
from .. import util_lib
from ..__init__ import HOME_DIR

# Transit fitter object
# ---------------------

class TransitFitter(TransitModel):
    """Contains an instance of transit-fit, and performs the fits.

    TODO - NOTE: in order to introduce sampling of stellar radius
                 and mass, change the storage of mass and radius
                 into params; should be the easiest way and is
                 not expected to have negative side-effects. Otherwise
                 expand the parameter_names; frozen_mask must include
                 these parameters etc... It's a bit more difficult.

    NOTE: the key for working is that frozen/unfrozen, get_parameters
          and so on only include _param_names. The derived parameters
          are separate.

    NOTE: current aim, shouldn't care about inner parameters;
    that stuff is contained in the BaseClass (Parent); i.e
    TransitModel, or alternatively another parent class transit
    model.


    Attributes:
        m (batman.TransitModel): the transit model
        params (batman.TransitParams): the transit parameters
        bss (float): the step size used in batman

        f_data (np.array): the flux data
        t_data (np.array): the times at the flux data
        f_err (float or np.array): the flux uncertainties
    """

    def __init__(self, t, f, f_err, per, t0, rp, a, R_star=0.117,
                 prior=None, bin_type='regular', bin_res=6,
                 adjust_res=False, cut_lightcurve=False,
                 overlap_lim='partial', stellar_class='ultracool',
                 radius_constraint='stellar',
                 **kwargs):
        """

        Args:

            **kwargs: go solely to transit_model,
                
        """

        # prior kwargs
        if 'overlap_lim' in kwargs:
            overlap_lim = kwargs.pop('overlap_lim')
        else:
            overlap_lim = 'partial'

        # Initialise the model
        super().__init__(t=t, per=per, t0=t0, rp=rp, a=a,
                         bin_type=bin_type, bin_res=bin_res,
                         adjust_res=adjust_res, **kwargs)

        # Checks on input
        if np.nanmedian(f) < 0.8:
            warnings.warn(("Lightcurve median is below 0.8. Must " \
                           + "be normalised at 1.0."),
                           Warning)

        # Save the data
        self.f_data = np.array(f)
        self.f_err = f_err

        # Set prior
        if prior is None:
            self.Prior = BasicTransitPrior(per, t0, rp, self['duration'],
                                           time_baseline=(max(t) - min(t)),
                                           overlap_lim=overlap_lim,
                                           stellar_class=stellar_class,
                                           radius_constraint=radius_constraint)
        else:
            self.Prior = prior

        # Save the values
        self._initial_parameters = pd.Series(self.get_parameter_dict(include_frozen=True))
        self._initial_parameters = self._initial_parameters.append(
                    pd.Series({'R_star':R_star}))

        if cut_lightcurve:
            self.cut_lightcurve()

        # Set bin mode - needs to be here to adjust resolution
        # 				 after cutting the lightcurve
        if bin_type == 'regular':
            self.set_regular_bin_mode(bin_res, adjust_res=adjust_res)
        else:
            self.set_no_bin_mode()

        # Make sure the initial values and priors are valid
        self.verify_prior(info_str='inside __init__.',
                          quiet=(prior is not None))

    # Class methods - must override and extend the base-class methods
    # -------------

    @classmethod
    def from_batman(cls):
        raise NotImplementedError

    @classmethod
    def from_transit_model(cls, f, f_err, transit_model, **kwargs):
        """

        NOTE: only works for the standard TransitModel object.

        Args:
            **kwargs: prior, cut_lightcurve
        """

        # To check if the semi-major axis was adjusted
        a0 = transit_model['a']

        obj = cls(transit_model.t_data, f, f_err,
                  transit_model['per'], transit_model['t0'],
                  transit_model['rp'], transit_model['a'],
                  transit_model['R_star'],
                  bin_type=transit_model._bin_type,
                  bin_res=transit_model._bin_res,
                  **kwargs)

        # Other things to copy
        obj.unfrozen_mask = copy.copy(transit_model.unfrozen_mask)

        if a0 != obj['a']:
            warnings.warn('`from_transit_model`:',
                          'The semi-major axis was modified.')

        return obj

    @classmethod
    def from_bls(cls, t, f, f_err, period, t0, depth, duration,
                 R_star=0.117, prior=None, bin_type='regular', bin_res=6,
                 adjust_res=False, cut_lightcurve=False,
                 overlap_lim='partial', stellar_class='ultracool',
                 radius_constraint='stellar', **kwargs):
        """Can directly take the bls_peaks.loc[...] results.

        i.e .from_bls(t, f, f_err, bls_peaks.loc[i, [...]])

        Args:
            t, f, f_err, period, t0, depth, duration
            **kwargs: get passed to self.__init__()
                R_star, prior, bin_type, bin_res, adjust_res,
                cut_lightcurve, **transit_model_kwargs
        """

        # Convert the necessary values
        rp0 = np.sqrt(depth)
        a0 = period / (np.pi*duration)	# likely to be overestimate	
        
        # Create the prior directly to avoid issues with a
        if prior is None:
            prior = BasicTransitPrior(period=period, t0=t0, rp=rp0,
                                      duration=duration,
                                      time_baseline=(max(t) - min(t)),
                                      overlap_lim=overlap_lim,stellar_class='ultracool',
                                      radius_constraint='stellar',)

        # Adjust a to the prior
        if a0 > prior._upper_a_lim:
            # caused high impact factor (b) in reality
            a0 = prior._upper_a_lim - 0.0001
        elif a0 < prior._lower_a_lim:
            # implies a super-heavy and compact star (unlikely)
            a0 = prior._lower_a_lim + 0.0001

        # Adjust r to the prior
        if rp0 > prior._r_max:
            rp0 = prior._r_max - 1e-6

        # Ignore the a warning
        obj = cls(t, f, f_err, period, t0, rp0, a0, R_star=R_star,
                  prior=prior, bin_type=bin_type, bin_res=bin_res,
                  adjust_res=adjust_res, cut_lightcurve=cut_lightcurve,
                  overlap_lim=overlap_lim, **kwargs)

        # Add depth and duration
        obj._initial_parameters = obj._initial_parameters.append(
                    pd.Series({'depth':depth, 'duration':duration}))

        return obj

    @classmethod
    def from_stellar_params(cls, t, f, f_err, per, t0, rp, R_star,
                            M_star, freeze_a=True, **kwargs):
        """Initiates based on stellar mass and radius instead of a.

        By default, freezes a.
        """

        # Calculate first value of a
        M_fac = (0.5*per*units.day/np.pi)**(2/3) * (0.5*const.G)**(1/3)
        a0 = (M_fac*(M_star*const.M_sun)**(1/3) / (R_star*const.R_sun)).to('').value

        obj = cls(t, f, f_err, per=per, t0=t0, rp=rp, a=a0,
                  R_star=R_star, **kwargs)

        if freeze_a:
            obj.freeze_parameter('a')

        obj._initial_parameters['M_star'] = M_star
        obj._initial_parameters['R_star'] = R_star

        return obj

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

    def calc_ll_ratio(self, params=None):
        """Estimate the SNR for basic statistical principles.
        NOTE: assumes a minor ingress/egress.

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

    def estimate_snr(self, count_points=True,  subtract_transit=True,
                     nearby_baseline=True):
        """Estimate the SNR for a basic gaussian assumption.

        Args:
            count_points (bool): if True, counts the point in-transit
                otherwise, estimates points in-transit (more stable)
            subtract_transit (bool=True): currently unused (transit
                is subtracted anyway for the depth)
            nearby_baseline (bool=True): if False, for the depth
                references uses the lightcurve median, otherwise, uses
                the median from the nearby points out-of-transit.
                Attempts to select the nearest 50 points.
            params (batman.TransitParams)
        """

        # Needed for a few things, the cadence time and total time
        t_cad = np.nanmedian(np.diff(self.t_data))
        t_base = np.nanmax(self.t_data) - np.nanmin(self.t_data)

        # adjust for ingress/egress
        adj_dur = self['duration'] * (1.0 - self['rp'])
        t_mask = mask_transits(self.t_data, self['t0'], self['per'],
                               duration=adj_dur)

        if nearby_baseline:
            # Number of points to aim for in the baseline measurement
            num_points_baseline = 45
            min_points_baseline = 20

            # The time needed to achieve the above points
            baseline_window = t_cad*num_points_baseline*self['per']/t_base

            baseline_mask = mask_transits(
                self.t_data, self['t0'], self['per'],
                duration=self['duration'] + baseline_window + t_cad)
            transit_mask = mask_transits(self.t_data, self['t0'], self['per'],
                                         duration=self['duration'] + t_cad)
            # Remove the actual transit
            baseline_mask = baseline_mask & ~transit_mask

            if baseline_mask.sum() > min_points_baseline:
                # TODO:
                print("Number of baseline points:", baseline_mask.sum())
                f0 = np.nanmedian(self.f_data[baseline_mask])
            else:
                # TODO:
                print("Not enough baseline points:", baseline_mask.sum())
                f0 = np.nanmedian(self.f_data)
        else:
            # Otherwise, or if there weren't enough points to set the
            # baseline
            f0 = np.nanmedian(self.f_data)
        
        depth = f0 - np.nanmean(self.f_data[t_mask])

        # If there is no points in-transit, the snr doesn't exist
        if np.sum(t_mask) == 0:
            return 0.0

        if subtract_transit:
            f_lcf = self.f_data - self.evaluate_model() + f0
        else:
            f_lcf = self.f_data.copy()

        # NOTE: cannot calculate noise directly; megaflares, such as
        # those in trappist-1, completely mess this up. Instead, need
        # to use the GP noise, or sigma-clip the noise out.
        #signoise = np.nanstd(f_lcf)

        if count_points:
            npoints = np.sum(t_mask)
        else:
            npoints = t_base * adj_dur / (self['per'] * t_cad)

        return depth * np.sqrt(npoints) / self.f_err

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
        wres['snr_estimate'] = self.estimate_snr()

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
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                        dim=ndim,
                                        lnpostfn=self.lnposterior)

        pos, _, _ = sampler.run_mcmc(p0, burn)
        sampler.reset()
        sampler.run_mcmc(pos, iterations)

        chain = sampler.flatchain

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
            depth_med = min(np.median(rp_chain**2), 1.0)
            depth_sig = np.std(rp_chain**2)
            depth_snr = depth_med / depth_sig
        else:
            depth_snr = None

        # If parameter initial values are extreme in the posterior
        ch_flag = False
        for p in ('rp', 'per'):
            if p in self.parameter_names:
                p_idx = self.parameter_names.index(p)
                p_percentile = percentileofscore(chain[:, i], p_0[p_idx])

                if abs(p_percentile - 50.0) > 47.0:
                    ch_flag = True

        if plot_posterior:
            # BUG
            try:
                truths = self._initial_parameters[self.parameter_names]
            except KeyError:
                import pdb; pdb.set_trace()
            fig = corner.corner(chain_df, truths=truths)
            fig.show()

        return chain_df, result_df, depth_snr, ch_flag

    # Internal utility methods
    # ------------------------

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
        elif isinstance(self.Prior, BasicTransitPrior):
            parameters = self.get_parameter_vector(include_frozen=True)
            prior_code = self.Prior.verify_lnprior(parameters)
            parameter_dict = self.get_parameter_dict(include_frozen=True)

            names = ('period', 'duration', 't0', 'rp', 'lower_a_lim',
                     'upper_a_lim')
            hpp = {name:getattr(self.Prior, '_' + name) for name in names}

            raise InvalidInitializationError(
                "Hyperparameters are out of bounds. "\
                "Prior code string: {}\n"
                "Info str: {}".format(prior_code, info_str),
                vector=vector, hpp=hpp,
                parameter_dict=parameter_dict,
                M_star=self['M_star'],
                R_star=self['R_star']
            )
        else:
            print('Prior verification failed for new prior.')
            import pdb; pdb.set_trace()

    def plot_model(self, show=True):
        """Plot the current model; along with the data.

        NOTE: overrides TransitModel's method, adding the f_data.
        """
        
        fig, ax = plt.subplots()

        ax.plot(self.t_data, self.f_data, 'k.')
        ax.plot(self._t_model, self.evaluate_model(bin_to_data=False), 'r-')

        if show:
            plt.show()
        else:
            fig.show()

    def plot_posterior(self, chain, num_samples=None, 
                       N_durations=None, show=True):
        """Plots models of a chain, folded on transit.

        NOTE: plots a maximum of 100 samples.

        Args:
            chain (pd.DataFrame): the direct output of
                tfitter.sample_posterior()
            num_samples (int): (max) number of subsamples to plot
            N_durations (float): number of durations either side of
                folded center to plot (window half-width)
            show (bool): show plot or not
        """

        if num_samples is not None and num_samples < len(chain):
            chain = chain.sample(num_samples, replace=False)
        elif len(chain) > 100:
            chain = chain.sample(100, replace=False)

        # Fold time and cut the window
        window = self['per']/2 if N_durations is None \
                               else N_durations*self['duration']

        if window < 0:
            print("Invalid window, period:", self['per'],
                  "duration:", self['duration'],
                  "semi-major axis:", self['a'])
            window = self['per'] / 2.0
            import pdb; pdb.set_trace()

        t_folded = util_lib.fold_on(self.t_data, period=self['per'],
                                    t0=self['t0'], symmetrize=True)
        window_mask = (t_folded > -window) & (t_folded < window)
        t_folded = t_folded[window_mask]
        f_folded = self.f_data[window_mask]
        
        # Calculate the models; t must first be centered on the
        # actual t0, it can't start off folded.
        t = np.linspace(self['t0'] - window,
                        self['t0'] + window,
                        10 * 24 * 2 * window)	# 10 points per hour
        f = self.evaluate_model_at(t, p_fit=chain)

        fig, ax = plt.subplots()

        ax.plot(t_folded, f_folded, '.',
                c='0.7', alpha=0.8, zorder=-1)

        for i in range(len(chain)):
            # Plot, but recenter it on 0.0 by subtracting t0
            ax.plot(t - self['t0'], f[i], 'k-', alpha=0.7)

        ax.set_xlabel('Time, days')
        ax.set_ylabel('Normalise flux')
        ax.set_xlim([min(t_folded), max(t_folded)])

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


# Exceptions
# ----------

class UnorderedLightcurveError(Exception):
    pass

class NegativeDepthError(ValueError):
    pass

class InvalidInitializationError(ValueError):
    """Thrown when the prior verification fail."""

    def __init__(self, message=None, vector=None, hpp=None,
                 parameter_dict=None, M_star=None, R_star=None):

        self.parameter_vector = vector
        self.hpp = hpp
        self.parameter_dict = parameter_dict
        self.parameter_dict['M_star'] = M_star
        self.parameter_dict['R_star'] = R_star

        if message is None:
            message = ''
        message += "\nParsing:\n" +\
                   "parameter\t: {}\n\n".format(parameter_dict) +\
                   "hp\t\t: {}\n\n".format(vector) +\
                   "hpp\t\t: {}\n".format(hpp)

        super().__init__(message)



