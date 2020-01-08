"""Tabulates the candidates, and checks for new candidates."""

import numpy as np
import pandas as pd



def get_signal_df():
    """Combines the dicts into a df with epic, flag, comment."""

    df = pd.DataFrame(columns=['epic', 'flag', 'comment'])

    for epic, comment in candidate_dict.items():
        idx = len(df)
        df.loc[idx, 'epic'] = epic
        df.loc[idx, 'flag'] = 'candidate'
        df.loc[idx, 'comment'] = comment

    for epic, comment in not_sure_dict.items():
        idx = len(df)
        df.loc[idx, 'epic'] = epic
        df.loc[idx, 'flag'] = 'not_sure'
        df.loc[idx, 'comment'] = comment

    for epic, comment in discarded_signal_dict.items():
        idx = len(df)
        df.loc[idx, 'epic'] = epic
        df.loc[idx, 'flag'] = 'discarded_signal'
        df.loc[idx, 'comment'] = comment

    for epic, comment in known_planet_dict.items():
        idx = len(df)
        df.loc[idx, 'epic'] = epic
        df.loc[idx, 'flag'] = 'known_planet'
        df.loc[idx, 'comment'] = comment

    for epic, comment in false_positive_dict.items():
        idx = len(df)
        df.loc[idx, 'epic'] = epic
        df.loc[idx, 'flag'] = 'false_positive'
        df.loc[idx, 'comment'] = comment

    for epic, comment in variability_dict.items():
        idx = len(df)
        df.loc[idx, 'epic'] = epic
        df.loc[idx, 'flag'] = 'variability'
        df.loc[idx, 'comment'] = comment

    for epic, comment in interesting_noise_dict.items():
        idx = len(df)
        df.loc[idx, 'epic'] = epic
        df.loc[idx, 'flag'] = 'interesting_noise'
        df.loc[idx, 'comment'] = comment

    for epic, comment in flagged_as_noise_dict.items():
        idx = len(df)
        df.loc[idx, 'epic'] = epic
        df.loc[idx, 'flag'] = 'flagged_as_noise'
        df.loc[idx, 'comment'] = comment

    # Now I must combine the rows which have the same epic repeated.
    duplicates = df[df.epic.duplicated(keep=False)]
    df = df[~df.epic.duplicated()]

    # for _, (epic, flag, comment) in duplicates.iterrows():
    for epic in duplicates.epic.unique():
        idx = df[df.epic == epic].index[0]

        # import pdb; pdb.set_trace()

        df.loc[idx, 'flag'] = ",".join(
            duplicates.loc[duplicates.epic == epic, 'flag'])

        df.loc[idx, 'comment'] = '\n'.join(
            duplicates.loc[duplicates.epic == epic, 'comment'])

    assert not df.epic.duplicated().any()

    return df


# Candidate dicts
# ---------------

# Re-detrend everything with power-based BLS search

candidate_dict = {
    # All these C5 targets are from the Praeseps cluster
    211926133: ("C5: 2/3 signals, 7/9 for 11.9:15, 4.95:7 for 8.41:11.9 "
                "11.9d, snr 6.7, rp 0.12; 1.99 Re"
                "15.3d, snr 7.5, rp 0.12; 1.89 Re "
                "3rd signal: "
                "8.41d, snr 8.7, rp 0.095; ~1.58 Re, might be 8.5 (5:7) "
                "If there is a target, it's this. "
                "However, alias checks are required; the lower "
                "BLS peaks seem to actually have bigger power. Praesepe. "
                "Issue is that the high-snr peak relies on the existence "
                "of the other peaks for validation. So hard to put it in. "
                "Gmag: 18.2, 2MASS Jmag 14.96, M5.5V, Praesepe"),
    246036782: ("C12: 2 targets, 13.4d and 16.2d (lost, snr<7), 5:6.04 "
                "13.4d, snr 7.2, rp 0.12, 1.48 Re "
                "Gmag: 17.5, 2MASS Jmag 13, M8V (2500K)"),
}

not_sure_dict = {
    212028729: ("C5: 3.7d, R=0.11, 1.2% serious target. Count as possible "
                "detection. SEND TO BRICE. However, since then it changed "
                "drastically, the two periods are now different. "
                "Second try: 9.47d, snr 7.5, rp 0.18; 1.2d, snr 7.2, rp 0.12. "
                "Something is going on here; possibly aliasing. "
                "Look into it further."),
    246794779: ("C13: 5.74d, snr 8.4, rp 0.093. Hard to say, 2 points "
                "dragging it quite down. SNR still too high - check the SNR "
                "directly on this one. DISCARD."),
    211997576: ("C5: 3.23d, snr 7.7, rp 0.12, looks decent actually "
                "(FURTHER LOOK). However previously had 8.86d, very "
                "questionable, on the limit (7.1). Likely noise to be "
                "honest. Nah  I don't believe it actually. DISCARD for now. "
                "Third try: 10.2d, DISCARD. These aren't even aliases."),
    212007145: ("C5: BLS is dense, but 7.67d, snr 7.8, rp 0.11 actually looks "
                "alright, check for aliases."),
    211994941: ("C5: flagged it as variability, but if that's removed, "
                "check it again for this signal."),
    211990019: ("C5: 1.11d and 4.98d, might aliases, planets too small "
                "though."),
    211845069: ("C5: not the main one, but check for aliases."),
    211328277: ("C5: 11.6d, snr 11, rp 0.27; but not transit dip. "
                "Might be alias though."),
    211075914: ("C4: Single strong transit. Question is, what is it? "
                "Could be binary. It's like 15%."),
    211010818: ("C4: 1.22d, snr 9.4, rp 0.091. Might have potential. "
                "Pleiades. Check the aliases. Got removed with new SNR."),
    211088144: ("C4: 11.5d, snr 7.2, rp 0.11. Might be a target. Pleiades. "
                "Potential second target. Looks like alias is more likely. "
                "Check TPF. If it's good, could be an unresolved binary. "
                "Lost to alias though: 9.73dm snr 6.7."),
    210981512: ("C4: 2.67d, snr 8.2, 0.098 rp, very strong, looks real. "
                "STUDY FURTHER.  Second signal is 18.8d, snr 7.1, rp 0.11; "
                "weak though. TPF and star need a deeper look for a second "
                "target. POSSIBLE REMOVAL. Potential SECOND star based on "
                "dt_rms though. Period changed completely though. "
                "First try: 0.737d; that's not even an alias is it? OCCR."),
    211897893: ("C5: 6.64d, snr 7.6, rp 0.079; looks alright actually. Small planet though."),
    212068782: ("C5: 8.98d, snr 8.3, rp 0.08; could be, but planet is too small for cutoff."),
    211984495: ("C5: 4.73d, snr 7, rp 0.13. Looks alright. FURTHER LOOK. "
                "TARGET FOR OCCR. No too much red noise; but keep as "
                "possible not sure target for SAINTEX"),
    220220365: ("C8: 3.92d, snr 7.2, rp 0.18, check the alias at 3.8 or "
                "whatever. Alias went away, main dip ramps into it too much "
                "DISCARD."),
    211088075: ("C4: no reason to ignore, 19.8d. "
                "Second search: 9.9d, snr 7.5, rp 0.14. Actually not bad. "
                "Plus another target at 1.32d, snr 7.2, rp 0.096. "
                "Figure out the aliasing, but this is not a target: snr fell. "
                "Check at lower periods and power-based search."),
    211993420: ("C5: 19.4d, snr 7.2, rp 0.092. Dip noise is much lower though, "
                "seems more like random alignment "
                "of below-average points. Need to perhaps figure out how "
                "to remove these too frankly. Too common. Disappeared "
                "in new runs. DISCARD."),
    212086091: ("C5: 6.01d, snr 7.8, rp 0.084; points nearby as just as "
                "low as transit. Check for aliases, otherwise DISCARD. "
                "As of the latest, the main peak is below the SNR limit."),
}

discarded_signal_dict = {
    211093202: ("C4: 2 with snr below limit (10.3 and 15.3, looks like "
                "there is possible aliasing here. DISCARDING FOR NOW but "
                "FURTHER LOOK. Previously: "
                "Serious candidate, snr 7.6, rp=0.087"),
    212089563: ("C5: 8.33d, snr 7.8, rp 0.071. Looks like the other signals. "
                "Also, might be second TPF target. DISCARD."),
    248651399: ("C14: 8.11d, ignorable, very skewed downward. "
                "New SNR method removes it (DISCARD)."),
    211915709: ("C5: got removed, not a signal: DISCARD."),
    211896805: ("C5: no reason to ignore, 16.5d. Signal lost. DISCARD."),
    211025779: ("C4: 12.7d, on the limit, don't know how the dnr is so high, "
                "but not a great signal. New SNR method loses signal."),
    211936457: ("C5: COMPLETELY GONE. DISCARDED. Previously: two targets on "
                "9.64 orbit. Unless one is an alias, this is impossible. "
                "However CHECK IT FURTHER, shapes are quite nice. "
                "Check if some sort of binary."),
    211938530: ("C5: Completely gone and dusted; DISCARD. HOWEVER, "
                "what is the weird Gaussian spike in the stellar brightness "
                "at 7155d? Check what happened in the TPF; because otherwise "
                "it might be gravitational lensing or something exotic. "
                "Multiple events in fact. Previously: no reason to ignore"),
    211913858: ("C4: Removed from tl; previously: No reason to ignore."),
    211078274: ("C4: 16.6d, snr 9.2, rp 0.097; aliased from earlier. "
                "Second target: 17.4d, 7.4 snr, 0.085 rp, looks almost better. "
                "However, think this is noise. First one definitely."
                "another target on the tpf. POSSIBLE REMOVAL."),
    251357067: ("C16: 7.88d, snr Lots of red noise similar to the dip in "
                "the fold. Second signal is trash. "
                "Check for aliases, otherwise to be DISCARDED."),
    211029507: ("C4: To many similarly low points in the fold. DISCARD."),
    210423437: ("C4: Several: 12.3d, 12d, 2.7d. Red noise in lightcurve. "
                "DISCARD. Looks like unusual stellar variability that "
                "the gp didn't lock on to."),
    211129571: ("C4: 10.6d, snr 7.6, rp 0.18; red noise is left over. "
                "DISCARD."),
    211086170: ("C4: 10.3d, snr 7.2, rp 0.13; possible as usual, "
                "but same old C4. Pleiades. Eclipsing binary candidate on: "
                "https://talk.planethunters.org/#/boards/BPH0000007/discussions/DPH0000qr5"),
    211893704: ("C5: 9.47d, snr 10, rp 0.17; high red noise, DISCARD. "
                "Left-over stellar variability, perhaps of a different "
                "period."),
    210822528: ("C4: 15.9d, snr 7.1, rp 0.086. Likely trash. "
                "This star needs to calm itself down. "
                "Guess where this star is? DISCARD."),
    211029757: ("C4: 13.5d, snr 8.1, rp 0.17. Looks unlikely, a bit weird. "
                "DISCARD. Pleiades."),
    211907293: ("C5: 7.36d, snr 8.5, rp 0.16. There is a spike at some "
                "point which causes an outlier. DISCARD."),
    210457230: ("C4: 9.76d, snr 7.4, rp 0.15. So many similar outliers "
                "though. Probably DISCARD. Hyades."),
    211135624: ("C4: 14.1d, snr 7.2, rp 0.13; however there is stellar "
                "variability; very slight though. Find why it was not "
                "detected. Pleiades."),
    211940389: ("C5: 8.64d, snr 7.1, rp 0.098, not worth it, looks like "
                "just an accumulation of low points."),
    211138722: ("C4: 9.19d and 1,59d. Not sure how  this got past the "
                "point removal, second signal is flat."),
    211082185: ("C4: 13.9d, snr 7.4, rp 0.17, lots of similar such signals "
                "in the fold. DISCARD."),
    211970944: ("C5: 3.05d, snr 7.1, could be something, though there are "
                "lots of these low outliers nearby. Check aliases, "
                "but probably discard, snr on the limit anyway."),
    212031039: ("C5: 12.6d, snr 7, rp 0.071; terrible BLS spectrum, "
                "peaks aren't aliases. Lots of dips. DISCARD."),
    246395463: ("C12: 9.02, too much red noise, BLS is all over the place. "
                "DISCARD."),
    211088777: ("C4: 1.68d, snr 7.9, rp 0.084, DISCARD."),
    210989751: ("C4: 11.7d, aligned with the cadence rate and only 6 "
                "transits. DISCARD."),
    211725787: ("C5: 1.77, doesn't look like anything but check the "
                "other BLS peak. DISCARD."),
    212110421: ("C5: too skewed downward. DISCARD. The BLS spectrum nearly "
                "makes one's eyes water."),
    211080412: ("C4: 1.88 and 12.9, but noise changes size halfway through, "
                "and is underestimated. DISCARD."),
    211938817: ("C5: a couple of very low outliers on one transit, "
                "nothing on the others. Why not rejected by the "
                "point-rejection filter? DISCARD."),
    211953966: ("C5: 6.06; one point is like a 100 sigma outlier. "
                "Otherwise flat. DISCARD."),
    212106472: ("C5: 6.79d: too much on the BLS, not even the highest peaks, "
                "not aliasing. DISCARD."),
    211887567: ("C5: 12.5d: BLS spectrum looks more like white noise "
                "than the actual detrended lightcurve. DISCARD."),
    211887287: ("C5: 10.2d: too many points at the same level as the transit "
                "around the tranit. DISCARD."),
    248622974: ("C14: 2.2d: BLS is also bad, check for aliases. In fact how "
                "is that even the highest peak."),
    211089146: ("C4: 3 signals but lightcurve is messed up and noisy. "
                "DISCARD."),
    211955381: ("C5: 10.3d, snr 9.3, rp 0.13; this is almost certainly "
                "aliased, and relies on only 3 points below the line "
                "on one transit; the rest are more normal. Could be a single "
                "transit, but for the occurrence rate, "
                "this is to be DISCARDED."),
    211993420: ("C5: 19.4d is too skewed, low snr anyway, and BLS is dense."),
    212820594: ("C6: 8.32d, outliers lead into it"),
    212090825: ("C18: just a spike"),
    212411722: ("C6: very skewed, check the SNR manually."),
    212085656: ("C5: BLS is a forest, can't even see the peak"),
    246008196: ("C12: low points all around transit"),
    212178513: ("C18: outlier spike"),
    212012745: ("C5: check again"),
    251407310: ("C16: noise in the beginning as usualy for C16"),
    212068777: ("C5: similar dips all around"),
    201744267: ("C14: unfinished detrending"),
    201426001: ("C10: on the edge"),
    201181297: ("C10: red noise"),
    212090825: ("C5: spike is due to 1 point"),
    211997576: ("C5: nearby outlier points are similar."),
    211003659: ("C4: 5.38d, snr 7.8, rp 0.12; DISCARD. "),
    210933157: (""),
    212028729: ("C5: was not_sure, but points on the first are not "
                "the only dip"),
    212069325: ("C5"),
    212022062: ("C5"),
    211013627: ("C4"),
    245976296: ("C12: noise at start triggers the dip but it's the only "
                "feature"),
    212162176: ("C16: noise at start"),
    211071351: ("C4"),
    211091536: ("C4"),
    210774807: ("C4: stellar period"),
    211072835: ("C4: noise"),
    211145665: ("C4: stellar period is still in"),
    211075945: ("C4"),
    211048321: ("C4"),
    246403896: ("C12"),
    210371851: ("C4: only one significant dip"),
    211803674: ("C5: interesting variability however"),
    211303324: ("C5: previous CANDIDATE, but DISCARD. No common signal in "
                "campaigns 18 and 5. On the edge anyway (snr on 18 got "
                "wiped out)"),
    212094548: ("C5: BLS"),
    211060762: ("C4: noise at start"),
    211076441: ("C4: only one significant dip"),
    212078515: ("C5: red noise"),
    211891774: ("C5: red noise"),
    211111611: ("C4: red noise"),
    201482905: ("C10"),
    210468157: ("C4: red noise"),
    211937565: ("C5: BLS"),
    211052796: ("C4: BLS has nothing"),
    212069380: ("C5"),
    212016898: ("C5: could even be a small planet, but below our cutoff, "
                "also noisy. 2.63d."),
    211973504: ("C5"),
    246016139: ("C12: red noise"),
    211925319: ("C5"),
    211010149: ("C4"),
    210990028: ("C4"),
    211977997: ("C5, not even teh first BLS signal, and all poor dips"),
    211037417: ("C4"),
    251397724: ("C16"),
    211075914: ("C4: Single strong transit. Question is, what is it? Could "
                "be binary. It's like 15%. "
                "FURTHER LOOK, this is is something in any case. "
                "There is also a LONG PERIOD SIGNAL (43d)."),
    211063723: ("C4: only 1 real dip but check aliases"),
    211997857: ("C5: 15.5d, snr 8.3, rp 0.13 and 1.98d, snr 8.9, rp 0.08. "
                "Second looks like it can't possibly have such high SNR. "
                "Flat outside of the beginning. DISCARD."),
    228803953: ("C10: 3.94d, snr 7.9, rp 0.15, very few points in-transit "
                "though, 0.63h (so inclined). Check the transit parameters. "
                "Allowing low inclination might be the problem. "
                "OCCR TARGET but see about the inclination/b parameter. "
                "Lots of similar shapes nearby though. DISCARD."),
    211897893: ("C5: 6.64d, snr 7.6, rp 0.079; looks alright actually. "
                "Small planet though."),
    212068782: ("C5: 8.98d, snr 8.3, rp 0.08; could be, but planet is too "
                "small for cutoff."),
}

known_planet_dict = {
    210490365: ("C4: Is this a binary? 0.11rp. 3.48d. Known planet, this is "
                "a sub-Neptune, but star is M4.5 (M=0.29). Issue however: "
                "1% transit; by our reckoning will be half the size due to "
                "star's youth; see: "
                "https://authors.library.caltech.edu/65299/1/Mann_2016p46.pdf"),

}


false_positive_dict = {
    211079188: ("C4: 2.64d, snr 5.8 (underestimate), 0.085 rp. "
                "POTENTIAL BINARY ACCODING TO KRUSE/LUGER. "
                "Duration is much longer than maximum expected though; "
                "star must be much bigger (check if it's a younger star). "
                "Also, could be binary. This seemed to alias with something "
                "else, not sure how that works. Star has 0.175d variability "
                "period. Fold on the variability period. Might be an "
                "additional period in there. False positive according to: "
                "https://arxiv.org/pdf/1907.10806.pdf"),
    212002525: ("C5: DEFINITE TRANSIT, snr like 200, 5.81d. "
                "This is an eclipsing binary: "
                "https://arxiv.org/pdf/1706.03084.pdf"),
}

variability_dict = {
    212016014: ("C5: 0.845d, looks like stellar variability (slow trend). "
                "Check if the new SNR method works on this."),
    251357067: ("C16: 0.876d, why didn't this get flagged?"),
    211103969: ("C4: variability still in it."),
    212070886: ("C5: variability at 0.579d, why not flagged or detrended?"),
    246005947: ("C12: very strong variability, why not flagged as qp, check "
                "if new SNR works."),
    210957162: ("C4: very strong variability, why not flagged as qp, check "
                "if new SNR works."),
    211998837: ("C5: clear variability, what is the LS signal strength?"),
    211938530: ("C5: What is the weird Gaussian spike in the stellar "
                "brightness at 7155d? Check the TPF; because otherwise "
                "it might be gravitational lensing or something exotic. "
                "Multiple events in fact. Previously was a candidate."),
    211994941: ("C4: Very low level of variability visible on fold, but "
                "too low for LS apparently. Something close to 0.2d."),
    210926194: ("C4: what in the world. This might be some insane stellar "
                "variability. Look into further."),
    211096320: ("C4: looks like 2 overlapping variability patterns, "
                "only very short, 1 of a couple of days and potentially "
                "even one slightly longer."),
    249639465: ("C15: Think this is some sort of relativistic beaming or "
                "whatever it was; variability period is below 0.1d. "
                "I think 1.04/13 or 1.04/15 or something. "
                "TARGET OF INTEREST. Find it 6in publications."),
    211927174: ("C5: some VERY large changes in variability paradigm"),    
    211915085: ("C5: some kind of phenomenon, but I saw this in another "
                "lightcurve, check what the relation is. 2 HUGE dips, "
                r"one very long. 50\% brightness."),
    211928574: ("C5"),
}

interesting_noise_dict = {
    211984495: ("I don't see how this is SNR 7. Plot the actual fitted "
                "transits and check the noise."),
    204407831: ("C15, strong signal, but I think it's stellar. "
                "Check Lomb-Scargle. Also check new SNR "
                "- old snr is 7.4, rp=0.095"),
    211078274: ("C4: 11.8d, look at SNR, white noise, etc directly. "
                "Check that one point on the 3rd transit, why isn't it there "
                "in the fold?"),
    212016014: ("C5: 0.845d, looks like stellar variability (slow trend). "
                "Check if the new SNR method works on this."),
    211045147: ("C4: 19.2d, slow trend into a dip, "
                "check if new SNR still flags this."),
    212070886: ("C5: variability at 0.579d, why not flagged or detrended, "
                "check new SNR method."),
    246005947: ("C12: very strong variability, why not flagged as qp, "
                "check if new SNR works."),
    210957162: ("C4: very strong variability, why not flagged as qp, "
                "check if new SNR works."),
    211984100: ("C5: how is this signal snr 7.5?"),
    212001425: ("C5: this is not a signal, why is its SNR 7.7 and "
                "depth so high?"),
    211064435: ("C4: Why doesn't the noise selector remove this? "
                "Do it manually, see what's going on."),
    211051975: ("C4: Should absolutely be removed by the noise classifier."),
    246794779: ("C13: 5.74d, snr 8.4, rp 0.093. Hard to say, "
                "2 points dragging it quite down. SNR still too high - "
                "check the SNR directly on this one."),
    211007016: ("C4: dip far below 0. I think this must be a background "
                "mis-calculation or something, also it's only a single dip."),
}

# Signals permanently flagged as noise (when it's abundantly clear)
flagged_as_noise_dict = {
    210871940: ("Lots of red noise."),
    211007016: ("C4: messy lightcurve. Pleiades."),
    210917578: ("C4: Noise at start triggers all the signals. Pleiades"),
    212217562: ("C16: Noise bit at the start."),
    212216119: ("C16: Noisy bit at the start. Perhaps try to remove these if we find them?"),
    212044515: ("C16: Something very strange at the start; a dip into negative values (not transit-like)."),
    211041750: ("C4: Too much point drift noise for the GP apparently. Need to implemented the 'block' "
                "centered kernels."),
    211078745: ("C4: Some weird kind of flappy noise."),
    212027121: ("C16: Again something at the start; seems to be a C16 thing."),
    212183961: ("C16: Think star was barely in the TPF."),
    246711015: ("C13: Check the TPF."),
    211928367: ("C5: Multiple types of overlapping noise. What is red-giant stochastic variability supposed "
                "to look like?"),
    210926194: ("C4: what in the world. This might be some insane stellar variability. Look into further."),
    212156288: ("C16: another classic. Need to remove these, they might be messing with the irm."),
    212022056: ("C16"),
    210674207: ("C4: Stellar variability was hard to remove, signals is weak anyway."),
    211077607: ("C4: Second half has some kind of drops."),
    211096320: ("C4: stellar variability"),
    210915466: ("C4: stellar variability is actually shorter."),
    211077682: ("C4: Systematic noise is too high. Check if increasing hyperparameter bounds in GP does "
                "the trick."),
    245978296: ("C12: Some interesting variability in the lightcurve. Check it again."),
    211107496: ("C4: Section of lightcurve is compromised or something."),
    211161968:  "",
    212217563: ("C16"),
    212216110: ("C16"),
    212021699: ("C16"),
    211029076: ("C4"),
    210978953: ("C4"),
    211080604: ("C4"),
    211927174: ("C5"),
    245973927: ("C12"),
    210405752: ("C4"),
    212143886: ("C5"),
    211915085: ("C5"),
    211131711: ("C4"),
    211073598: ("C4"),
    210742017: ("C4"),
    211727975: ("C5"),
    246018096: ("C6"),
    211928574: ("C5"),
    211154974: ("C4"),
    220189125: ("C8"),
    211013191: ("C4"),
    220197688: ("C8"),
    211075914: ("C4: Single strong transit. Question is, what is it? Could be binary. It's like 15%. "
                "FURTHER LOOK, this is is something in any case. There is also a LONG PERIOD SIGNAL (43d)."),
}

