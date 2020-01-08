"""Data analysis for single lightcurves, focusing transit search."""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
# BUG NOTE: temporary fix
try:
    import batman
except ModuleNotFoundError:
    pass

from .util_lib import (mask_flares, mask_floor, mask_transits, fold_on,
                       calc_noise, bin_regular, _fold, MF)
from .__init__ import K2GP_DIR


# ---------------------------------
#
# General
#
# ---------------------------------

# ---------------------------------
#
# Transit fitting focused
#
# ---------------------------------

# ---------------------------------
#
# BLS focused
#
# ---------------------------------

def highlight_bls_peaks(lcf, bls_results, bls_peaks, title=None, show=True,
                        source='bls', clean_lcf=True, normalise_phase=True,
                        plot_dx=True, bls_col='snr', bls_axis='period',
                        min_P_top=1.0, maximise=False, bin_folds=True):
    """Plots N folds, BLS spectrums, and the lightcurve on one large plot.

    N is determined by the dimension of bls_peaks.
    All selection (i.e valid_flag) must be done prior to this, including
    lcf selection (i.e o_flag, g_flag etc...).

    Maximum number of folds/peaks on the plot: 5 (currently)

    Can also plot the TransitFitter results instead

    Arguments:
        lcf (pd.DataFrame): assumes we are taking 'f_detrended'
        bls_results (pd.DataFrame)
        bls_peaks (pd.DataFrame)
        title (str): title to enter as Figure title
        plot (bool): whether to plt.show afterwards
        source (str): 'bls' or 'tf', to plot values from transit
            fitting, or from the bls. Default is 'bls'.

    Returns:
        None
    """

    if source == 'bls':
        per_array = bls_peaks['period']
        t0_array = bls_peaks['t0']
        depth_array = bls_peaks['depth']
        duration_array = bls_peaks['duration']
    elif source == 'tf':
        # The rest of the parameters (for transit modelling)
        # are entered directly from bls_peaks if source=='tf'
        per_array = bls_peaks['tf_period']
        t0_array = bls_peaks['tf_t0']
        depth_array = bls_peaks['tf_depth']
        duration_array = bls_peaks['tf_duration']
    else:
        raise ValueError("Invalid source entered.")

    npeaks = min(8, len(bls_peaks))
    if len(bls_peaks) > npeaks:
        bls_peaks = bls_peaks.iloc[:npeaks]

    # Data production
    if (np.diff(lcf.t) < 0).all():
        lcf = lcf.sort_values(by='t')

    if clean_lcf:
        lcf_fold = lcf[~mask_flares(lcf.f_detrended, sig_factor=4)]
        lcf_fold = lcf_fold[~mask_floor(lcf_fold.f_detrended,
                                        sig_factor=6,
                                        base_val=0.3)]

    # Params for later
    rms = calc_noise(lcf.f_detrended)			# Check usage
    f0 = np.nanmedian(lcf.f_detrended)
    if abs(f0) < 0.01:
        f0 = 0.0

    # Figure setup
    # ------------
    fig = plt.figure()

    # Outer figure gridspec
    gs = gridspec.GridSpec(2, 2, wspace=0.0, hspace=0.0,
                           height_ratios=[1, npeaks])

    ax_lcf = fig.add_subplot(gs[0, :])
    # ax_fold setup
    gs_fold = gridspec.GridSpecFromSubplotSpec(npeaks, 1, subplot_spec=gs[1, 0], wspace=0.0, hspace=0.0)
    ax_fold = np.empty(npeaks, dtype=object)
    for i in range(npeaks):
        ax_fold[i] = fig.add_subplot(gs_fold[i])
    # ax_fold setup
    gs_bls = gridspec.GridSpecFromSubplotSpec(npeaks, 1, subplot_spec=gs[1, 1], wspace=0.0, hspace=0.0)
    ax_bls = np.empty(npeaks, dtype=object)
    for i in range(npeaks):
        ax_bls[i] = fig.add_subplot(gs_bls[i])

    # Setup the colourmap
    cmap = cm.get_cmap('inferno')
    cl = cmap(np.linspace(0.4, 0.9, npeaks))

    # Plot lcf
    # --------

    # Raw
    if 'f_raw' in lcf:
        f_raw = lcf.f_raw
    else:
        f_raw = lcf.f
    ax_lcf.plot(lcf.t, f_raw, color='0.7', alpha=0.4, zorder=-100, linestyle='none', marker='.')
    # Temporal
    ax_lcf.plot(lcf.t, lcf.f_temporal + lcf.f_detrended \
                - np.nanmedian(lcf.f_detrended), color='0.3',
                alpha=0.7, zorder=-1, linestyle='none', marker='.')
    # Detrended
    ax_lcf.plot(lcf.t, lcf.f_detrended, color='b', alpha=0.7, zorder=1, linestyle='none', marker='.')

    # Set here to prevent the axis limits resetting later
    ax_lcf.set_xlim(min(lcf.t), max(lcf.t))
    ax_lcf.set_ylim(ax_lcf.get_ylim()[0], max(lcf_fold.f))

    # Highlight transit events
    for i, idx in enumerate(bls_peaks.index):
        t0 = t0_array[idx]
        P = per_array[idx]

        # If the period is too low, don't plot the points.
        if P < min_P_top:
            continue

        if (min(lcf.t) + P) < t0:
            t0 = t0 - (P * (t0 - min(lcf.t))//P)
        t_times = [t0]
        while t_times[-1] < max(lcf.t):
            t_times.append(t_times[-1] + P)

        ax_lcf.vlines(t_times, *ax_lcf.get_ylim(), colors=cl[i], linestyles='dashed')

    if plot_dx:
        dx = np.sqrt(np.diff(lcf.x)**2 + np.diff(lcf.y)**2)
        dx_t = (lcf.t.iloc[:-1].values + lcf.t.iloc[1:].values) / 2.0

        norm_factor = 0.2 * (ax_lcf.get_ylim()[1] - ax_lcf.get_ylim()[0])
        norm_factor = norm_factor / max(dx)
        dx_norm = norm_factor * dx + ax_lcf.get_ylim()[0]

        ax_lcf.plot(dx_t, dx_norm, 'r-')

    # Add text
    # ax_lcf.text(
    # 	x=np.percentile(lcf.t, 3),
    # 	y=min(lcf.f_detrended),
    # 	s=("rp = {}\nsnr = {}".format(bls_peaks.loc[idx, 'tf_rp'],
    # 									bls_peaks.loc[idx, 'tf_snr'])),
    # 	bbox=dict(facecolor='white', alpha=0.6),
    # 	color='red',
    # 	alpha=0.9)

    # Plot folds
    # ----------

    fold_colours = ('black', 'black', 'black', 'black', 'black')
    for i, idx in enumerate(bls_peaks.index):
        if source == 'bls':
            highlight_bls_signal(
                lcf_fold.t, lcf_fold.f_detrended,
                normalise_phase=normalise_phase,
                t0=t0_array[idx],
                period=per_array[idx],
                duration=duration_array[idx],
                depth=depth_array[idx],
                show=False,
                ax=ax_fold[i],
                color=fold_colours[i],
                alpha=0.7,
                mlc=cl[i],
                bin_fold=bin_folds)
        elif source == 'tf':
            highlight_tf_signal(
                lcf_fold.t, lcf_fold.f_detrended,
                normalise_phase=normalise_phase,
                show=False,
                ax=ax_fold[i],
                color=fold_colours[i],
                alpha=0.7,
                mlc=cl[i],
                bin_fold=bin_folds,
                **bls_peaks.loc[idx, ['tf_period',
                                      'tf_t0',
                                      'tf_rp',
                                      'tf_a',
                                      'tf_ecc',
                                      'tf_inc',
                                      'tf_w',
                                      'tf_u1',
                                      'tf_u2']])

        # BUG
        try:
            ax_fold[i].set_ylim(max(0, min(lcf_fold.f) - 0.1*rms),
                                max(lcf_fold.f) + 0.1*rms)
        except ValueError:
            import pdb; pdb.set_trace()
        #ax_fold[i].set_yticks([f0, f0 - depth_array[idx]])

        if normalise_phase:
            ax_fold[i].set_xlim(-0.25, 0.25)

        if source == 'tf':
            # Add text
            xlim = ax_fold[i].get_xlim()
            ylim = ax_fold[i].get_ylim()
            ax_fold[i].text(
                x=xlim[0],
                y=ylim[0] + 0.05*(ylim[1] - ylim[0]),
                s=("rp = {:.02g}\n"
                   "snr = {:.02g}".format(bls_peaks.loc[idx, 'tf_rp'],
                                          bls_peaks.loc[idx, 'tf_snr'])),
                bbox=dict(facecolor='white', alpha=0.6),
                color='red',
                alpha=0.9)

        # Remove previous transit
        tmask = mask_transits(
                    lcf_fold.t,
                    t0=t0_array[idx],
                    period=per_array[idx],
                    duration=MF*duration_array[idx])
        lcf_fold = lcf_fold[~tmask]

    # Plot bls
    # --------

    # Note, this has a lot of points, could do with binning or sampling.
    for i, idx in enumerate(bls_peaks.index):
        y_bls = bls_results["{}_{}".format(bls_col, idx)]
        x_bls = bls_results[bls_axis]

        ax_bls[i].plot(x_bls, y_bls, 'k-')
        ax_bls[i].set_xlim(min(x_bls),
                           max(x_bls))

        # Add text
        h_dur = duration_array[idx]
        h_durh = h_dur * 24
        d_per = per_array[idx]

        ax_bls[i].text(
            x=np.percentile(x_bls, 3),
            y=np.percentile(y_bls, 5),
            s=("n.{}\n"
               "P = {:.03g}d, bls_snr = {:.02g}\n"
               "dur = {:.02g}d = {:.02g}h".format(idx, d_per,
                                                    bls_peaks.loc[idx, 'snr'],
                                                    h_dur, h_durh)),
            bbox=dict(facecolor='white', alpha=0.8),
            color='red'
        )

    # Standard structural aesthetics
    # ------------------------------

    # show only the outside spines
    all_axes = fig.get_axes()
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)

    # ax_lcf
    ax_lcf.xaxis.set_ticks_position('top')
    ax_lcf.xaxis.set_label_position('top')
    #ax_lcf.yaxis.set_ticks_position('right')	# Maybe to be deleted
    #ax_lcf.yaxis.set_label_position('right')
    # choose the y-axis ticks
    #ax_lcf.set_yticks([(f0 + i*rms) for i in range(-2,3)])
    # TODO: y axis ticks at nearest 0.01, or 0.001 i.e order of magnitude
    # based on the depth and noise perhaps (i.e minimum 2 big ticks,
    # with small /10)

    # ax_fold
    for i, idx in enumerate(bls_peaks.index):
        if not (i + 1) == len(bls_peaks):
            #ax_fold[i].xaxis.set_visible(False)
            ax_fold[i].spines['bottom'].set_visible(True)
            ax_fold[i].spines['bottom'].set_color('black')
        # elif normalise_phase:
        # 	ax_fold[i].set_xticks([-0.5, -0.25, 0.0, 0.25])

    # ax_bls
    for i, idx in enumerate(bls_peaks.index):
        #ax.set_yticks([f0, f0 - bls_peaks.iloc[i].depth])
        ax_bls[i].set_yticks([np.nanmedian(bls_results["{}_{}".format(bls_col, idx)]), 0.8*max(bls_results["{}_{}".format(bls_col, idx)])])
        if not (i + 1) == len(ax_bls):
            ax_bls[i].xaxis.set_visible(False)
            ax_bls[i].spines['bottom'].set_visible(True)
            ax_bls[i].spines['bottom'].set_color('black')
        else:
            #ax_bls[i].set_xticks([0.0, 0.25, 0.50, 0.75])
            pass
        ax_bls[i].yaxis.set_ticks_position('right')
        ax_bls[i].yaxis.set_label_position('right')

    # Other aesthetics
    if title is not None:
        fig.suptitle(title)

    # Maximise the figure to the screen
    if maximise:
        mng = fig.canvas.manager
        if plt.get_backend() == 'QT4Agg':
            mng.window.showMaximized()
        elif plt.get_backend() == 'wxAgg':
            mng.window.Maximize(True)
        elif plt.get_backend() == 'TkAgg':
            mng.window.state('zoomed')
        else:
            print("unknown backend, couldn't maximise: {}".format(mng))

    if show:
        plt.show(block=True)
    else:
        fig.show()


# Work functions
# --------------

def highlight_bls_signal(t, f, t0, period, duration, depth=None, show=True,
                         ax=None, normalise_phase=True, tf_offset=None,
                         title=None, mlc='red', bin_fold=True, **fold_kwargs):
    """Folds the lightcurve on the BLS signal and highlights the *transit.

    Intended as a work function or utility when ax is passed.

    Args:
        t (np.array-like): time axis
        f (np.array-like): the flux values at t
        t0
        period
        duration
        depth
        show (bool): whether to plt.show() the plot or not
        ax (matplotlib.Axes): axis on which to plot
        normalise_phase (bool): to normalise phase to -0.5, 0.5
        tf_offset (array-like len: 2): scalar [t, f] translation vector,
            to move the entire fold somewhere (i.e for plotting multiple)
            folds on the same axes
        title (str): the title to put on the plot.
            If 'auto', lists the parameter values.
        mlc (str/mpl color): color to use for the model line
        bin_fold (bool): if True, attempts to bin the folds
        **fold_kwargs: arguments to pass to plt.plot for the lcf points
            expect: alpha, zorder, color, marker, etc...

    Returns:
        fig, ax
    """

    #t_folded = _fold(t, period)
    t_folded = fold_on(t, period=period, t0=t0, symmetrize=True)
    assert not np.any(np.isnan(t_folded))

    if normalise_phase:
        t_folded = t_folded / period
        duration = duration / period

    if tf_offset is not None:
        t_folded = t_folded + tf_offset[0]
        f = f + tf_offset[1]

    f0 = np.nanmedian(f[~mask_transits(t, t0, period, duration*1.5)])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Unpack the kwargs to the plot of the flux points
    if 'color' not in fold_kwargs: fold_kwargs['color'] = 'black'
    if 'marker' not in fold_kwargs: fold_kwargs['marker'] = '.'
    if 'linestyle' not in fold_kwargs: fold_kwargs['linestyle'] = 'none'
    if 'alpha' not in fold_kwargs: fold_kwargs['alpha'] = 0.7

    ax.plot(t_folded, f, **fold_kwargs)
    # The zero-line
    ax.plot([min(t_folded), -duration/2], [f0, f0], '--', c=mlc)
    ax.plot([duration/2, max(t_folded)], [f0, f0], '--', c=mlc)
    if depth is not None:
        ax.plot([-duration/2, -duration/2], [f0, f0 - depth], '-', c=mlc)
        ax.plot([duration/2, duration/2], [f0, f0 - depth], '-', c=mlc)
        ax.plot([-duration/2, duration/2], [f0 - depth, f0 - depth], '-', c=mlc, zorder=10)
    else:
        ax.axvline(-duration/2, c=mlc)
        ax.axvline(duration/2, c=mlc)

    # Fold if required
    if bin_fold and np.sum(np.abs(t_folded) < duration/2) > 12:
        # Only if folded contains more than 12 points
        # Up to 8 bins in duration (nb; npb is number of points per bin)
        num_points = np.sum(np.abs(t_folded) < duration/2)

        sort_args = np.argsort(t_folded)
        t_folded = t_folded[sort_args]
        f_folded = f[sort_args]

        if num_points <= 18:
            npb = 3
        else:
            npb = max(4, int(num_points // 8))

        t_bin, f_bin = bin_regular(t_folded, f_folded, npb)

        # TODO: ISSUE IS THAT SOME F_BINS ARE NAN
        #import pdb; pdb.set_trace()

        ax.plot(t_bin, f_bin, c='b', alpha=0.8, zorder=5)


    if title == 'auto':
        hduration = duration * 24
        fig.suptitle("P: {:.02f} - $t_0$: {:.04f}\n".format(period, t0) + \
                    "depth: {:.02g} - duration (hr): {:.02g}".format(depth, hduration))
    elif isinstance(title, str):
        fig.suptitle(title)

    if show:
        plt.show()

    return fig, ax

def highlight_tf_signal(t, f, show=True, ax=None, normalise_phase=True,
                        tf_offset=None, title=None, mlc='red',
                        return_points=False, bin_fold=True, **batman_kwargs):
    """Folds the lightcurve on the BLS signal and highlights the *transit.

    Intended as a work function or utility when ax is passed.

    Args:
        t (np.array-like): time axis
        f (np.array-like): the flux values at t
        t0
        period
        duration
        depth
        show (bool): whether to plt.show() the plot or not
        ax (matplotlib.Axes): axis on which to plot
        normalise_phase (bool): to normalise phase to -0.5, 0.5
        tf_offset (array-like len: 2): scalar [t, f] translation vector,
            to move the entire fold somewhere (i.e for plotting multiple)
            folds on the same axes
        title (str): the title to put on the plot.
            If 'auto', lists the parameter values.
        mlc (str/mpl color): color to use for the model line
        return_points (bool): if True, return the points
            that were plotted
        **batman_kwargs: arguments to batman for plotting
            t0, per, t0, a, rp, inc, ecc=0.0, w=90.0, u1=0.1, u2=0.3
            Can also take them all with a 'bls_' affix
        **fold_kwargs (implicit): arguments in batman_kwargs that
            aren't recognized are treated as fold_kwargs for the
            plotting colours and markers. Expected:
            alpha, zorder, color, marker, ...


    Returns:
        fig, ax + (points_df if return_points)
    """

    # Convert the batman_kwargs
    bkeys = ('per', 't0', 'a', 'rp', 'inc', 'ecc', 'w', 'u1', 'u2')
    bdefaults = {'ecc':0.0, 'w':90.0, 'u1':0.1, 'u2':0.3}

    # Remove all 'bls_' affixes
    for key in list(batman_kwargs.keys()):
        if key.startswith('tf_'):
            batman_kwargs[key[3:]] = batman_kwargs.pop(key)

    # Rename period
    if 'period' in batman_kwargs:
        batman_kwargs['per'] = batman_kwargs.pop('period')

    # Move extra arguments to fold_kwargs
    fold_kwargs = dict()
    for key in list(batman_kwargs.keys()):
        if key not in bkeys:
            fold_kwargs[key] = batman_kwargs[key]
            del batman_kwargs[key]

    # Insert defaults
    for key in bkeys:
        if (key not in batman_kwargs or np.isnan(batman_kwargs[key])) and key in bdefaults:
            batman_kwargs[key] = bdefaults[key]
        elif key not in batman_kwargs:
            raise ValueError("{} wasn't entered.".format(key))
    batman_kwargs['u'] = [batman_kwargs.pop('u1'), batman_kwargs.pop('u2')]

    period = batman_kwargs['per']
    t0 = batman_kwargs['t0']
    depth = batman_kwargs['rp']**2
    duration = batman_kwargs['per']/(np.pi*batman_kwargs['a'])

    # t_folded is symmetric, and centered on the transit at x=0
    t_folded = fold_on(t, period=period, t0=t0, symmetrize=True)
    assert not np.any(np.isnan(t_folded))

    # Set model time
    t_base = np.linspace(min(t_folded), max(t_folded), 1000)
    t_transit = np.linspace(-duration, +duration, 1000)
    t_model = np.sort(np.concatenate([t_base, t_transit]))

    # Model the transit
    params = batman.TransitParams()
    for key in batman_kwargs:
        setattr(params, key, batman_kwargs[key])
    # Since it's centered:
    params.t0 = 0.0
    params.limb_dark = 'quadratic'
    m = batman.TransitModel(params, t_model)
    f_model = m.light_curve(params)

    # Expects it to be normalised on 1, not zero
    f0 = np.nanmedian(f[~mask_transits(t, t0, period, duration*1.5)])
    if f0 < 0.1:
        f_model = f_model - 1.0

    # Transformations
    if normalise_phase:
        t_folded = t_folded / period
        t_model = t_model / period
        duration = duration / period
    if tf_offset is not None:
        t_folded = t_folded + tf_offset[0]
        t_model = t_model + tf_offset[0]
        f = f + tf_offset[1]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Unpack the kwargs to the plot of the flux points
    if 'color' not in fold_kwargs: fold_kwargs['color'] = '0.5'
    if 'marker' not in fold_kwargs: fold_kwargs['marker'] = '.'
    if 'linestyle' not in fold_kwargs: fold_kwargs['linestyle'] = 'none'
    if 'alpha' not in fold_kwargs: fold_kwargs['alpha'] = 0.6

    ax.plot(t_folded, f, **fold_kwargs)
    ax.plot(t_model, f_model, '-', c=mlc, zorder=10)

    # Fold if required
    if bin_fold and np.sum(np.abs(t_folded) < duration/2) > 8:
        # Only if folded contains more than 12 points
        # Up to 8 bins in duration (nb; npb is number of points per bin)
        num_points = np.sum(np.abs(t_folded) < duration/2)

        # tb and fb folded should have the same plot as t_fold vs f;
        # however, they are sorted in tb_folded
        sort_args = np.argsort(t_folded)
        # For this to work however, both need to be numpy arrays
        if isinstance(t_folded, np.ndarray):
            tb_folded = t_folded[sort_args]
        else:
            tb_folded = t_folded.values[sort_args]
        if isinstance(f, np.ndarray):
            fb_folded = f[sort_args]
        else:
            fb_folded = f.values[sort_args]

        if num_points <= 12:
            npb = 4
        else:
            npb = max(5, int(num_points // 4))

        t_bin, f_bin = bin_regular(tb_folded, fb_folded, npb)

        ax.plot(t_bin, f_bin, marker='.', linestyle='-', c='b',
                alpha=0.8, zorder=5)

    if title == 'auto':
        hduration = duration * 24
        fig.suptitle("P: {:.02f} - $t_0$: {:.04f}\n".format(period, t0) + \
                    "depth: {:.02g} - duration (hr): {:.02g}".format(depth, hduration))
    elif isinstance(title, str):
        fig.suptitle(title)

    if show:
        plt.show()

    if not return_points:
        return fig, ax
    else:
        pdf = pd.DataFrame({'t':t_folded, 'f':f})
        return fig, ax, pdf


# -----------------------
#
# Testing
#
# -----------------------

def test_highlight():
    """Tests the main peak highlight function trappist 1."""

    bls_peaks = pd.read_csv("{}/trappist_files/bls_tests/bls_peaks.csv".format(K2GP_DIR))
    bls_results = pd.read_pickle("{}/trappist_files/bls_tests/bls_results.pickle".format(K2GP_DIR))
    lcf = pd.read_csv("{}/trappist_files/k2gp200164267-c12-detrended-pos.tsv".format(K2GP_DIR), sep='\t')

    highlight_bls_peaks(lcf, bls_results, bls_peaks)

