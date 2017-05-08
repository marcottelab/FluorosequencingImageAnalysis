#!/home/boulgakov/anaconda3/bin/python


"""
Library for for fitting step functions to peptide photometries.

Characterizing the photometry of labeled peptides is one of the pillars of
fluorosequencing. Discrete steps in fluorescence are a frequently-encountered
fluor behavior. They may, for example, occur during photobleaching or may be
used as a fluor-counting strategy. This behavior is an idealization of
fluorophore physics: we are hoping that fluors are either on or off, with their
emission being a constant, with some noise around this mean. With this in mind,
we try to fit "step functions" to these steps to infer the status of labeled
peptides' fluors.

The name "step functions" is due to their resemblence to steps when plotted in
Cartesian coordinates. In Cartesian nomenclature, frames of a field of view
(representing time) comprise the horizontal axis, and Spot luminosity across
the frames the vertical axis. Hence, the steps are a series of horizontal lines
interpolating the regions of luminosity that are constant (within noise),
connected by vertical steps. Such a series of steps is also sometimes called a
"step train".

For convenience, instead of discussing fitting steps to the data, we will
discuss the equivalent task of fitting a consecutive series of horizontal
"plateaus" -- i.e. the horizontal lines -- across frames. Fitting a plateau
across a series of frames means that these frames' deviations in luminosity
from the fitted plateau are minimized, and that we are thus inferring that
during these frames the fluorescence of the Spot is constant within noise.

These plateaus can be fully characterized by three numbers: (starting frame,
ending frame, height). "height" is luminosity. Note that the plateau defined by
this tuple covers both the starting and ending frame, inclusively (i.e. it does
not follow the Python list syntax of the last number in foo[a:b] being member #
b - 1). A step fit is a sequence of such plateaus as represented by a list:
[(plateau_0_start, plateau_0_stop, plateau_0_height),
 (plateau_1_start, plateau_1_stop, plateau_1_height),
 (plateau_2_start, plateau_2_stop, plateau_2_height),

 ...

 (plateau_n_start, plateau_n_stop, plateau_n_height)]

This library provides algorithms to fit such plateaus to a sequence of
numbers -- ostensibly representing a peptide's luminosity recorded through
time.
"""


import logging
import datetime
import time
import numpy as np
import itertools
from operator import itemgetter
from scipy.signal import medfilt
from scipy.stats import ttest_ind, linregress
import math


#if the calling application does not have logging setup, stepfitting_library
#will log to NullHandler by default
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _plateau_squared_residuals(luminosities, plateau):
    """
    Calculate squared residuals for a sequence of points deviating from a
    horizontal line fit.

    Arguments:
        luminosities: List of luminosities of a Spot through frames.
        plateau: Plateau to compare residuals against.

    Returns:
        Sum of squares of distances of points and the plateau.
    """
    start, stop, height = plateau
    return sum([(lum - height)**2 for lum in luminosities[start:stop + 1]])


def _plateaus_squared_residuals(luminosities, plateaus):
    """
    Sums _plateau_squared_residuals for multiple plateaus.
    """
    return sum(_plateau_squared_residuals(luminosities, plateau)
               for plateau in plateaus)


def _fit_plateau(luminosities, starting_frame, stopping_frame):
    """
    Get the best-fit plateau to a sequence of luminosities that minimizes
    the sum of their squared residuals.

    Arguments:
        luminosities: List of luminosities of a Spot through frames.
        starting_frame and stopping_frame: The line should fit all points
            starting with starting_frame and ending with stopping_frame.

    Returns:
        Plateau at the height of the best-fitting horizontal line.
    """
    if not 0 <= starting_frame <= stopping_frame < len(luminosities):
        raise ValueError("Invalid (starting_frame, stopping_frame): " +
                         str((starting_frame, stopping_frame)) +
                         " with len(luminosities) = " +
                         str(len(luminosities)))
    return (starting_frame, stopping_frame,
            np.mean(luminosities[starting_frame: stopping_frame + 1]))


def _split_plateau(luminosities, plateau, forbidden_splits=None,
                   min_step_magnitude=5000):
    """
    Split a plateau into two plateaus, such that the split yields the smallest
    possible sum of squares of residues.

    Arguments:
        luminosities: List of luminosities of the Spot through frames.
        plateau: Plateau to split.
        forbidden_splits: If not None, this is a list of splits that are not
            allowed. Each split is represented by a tuple (stop, start), where
            stop and start are the consecutive frames between which a split is
            not allowed.
        min_step_magnitude: Minimum size of allowed fitted step, i.e. the
            difference in height of the two resulting plateaus.

    Returns:
        (left_plateau, left_plateau_residuals,
         right_plateau, right_plateau_residuals,
         total_residuals)

        left_plateau and right_plateau are returned as None if a possible split
        is not found within the constraints given.
    """
    logger = logging.getLogger()
    logging.debug("stepfitting_library._split_plateau locals() = " +
                  str(locals()))
    start, stop, height = plateau
    if not 0 <= start <= stop < len(luminosities):
        raise ValueError("plateau start and stop does not fit within " +
                         "luminosities; locals() = " + str(locals()))
    #initialize forbidden_splits as a set
    if forbidden_splits is None:
        forbidden_splits = set()
    else:
        forbidden_splits = set(forbidden_splits)
    #stores best split found so far
    #(left_plateau, left_residuals,
    # right_plateau, right_residuals,
    # total_residuals)
    best_split = (None,
                  len(luminosities) *
                      (np.amax(luminosities) - np.amin(luminosities))**2,
                  None,
                  len(luminosities) *
                      (np.amax(luminosities) - np.amin(luminosities))**2,
                  2 * len(luminosities) *
                      (np.amax(luminosities) - np.amin(luminosities))**2)
    for s in range(start, stop):
        if (s, s + 1) in forbidden_splits:
            continue
        left_plateau = _fit_plateau(luminosities, start, s)
        right_plateau = _fit_plateau(luminosities, s + 1, stop)
        if abs(left_plateau[2] - right_plateau[2]) < min_step_magnitude:
            continue
        left_residuals = _plateau_squared_residuals(luminosities,
                                                    left_plateau)
        right_residuals = _plateau_squared_residuals(luminosities,
                                                     right_plateau)
        total_residuals = left_residuals + right_residuals
        if total_residuals <= best_split[4]: #<= instead of < for flat case
            best_split = (left_plateau, left_residuals,
                          right_plateau, right_residuals,
                          total_residuals)
    logger.debug("stepfitting_library._split_plateau: returning best_split " +
                 "= " + str(best_split))
    return best_split


def _best_split(luminosities, plateaus, bestfit_plateaus=None,
                min_step_length=2, min_step_magnitude=5000):
    """
    Find the plateau split amongst plateaus that minimizes the total sum of
    squared residuals across all plateaus.

    Arguments:
        luminosities: List of luminosities of the Spot through frames.
        plateaus: Existing sequence of plateaus to try splitting.
        bestfit_plateaus: If not None, indicates that we're performing a
            counter-fit and thus must constrain any new split not to align with
            any boundary of any plateau in bestfit_plateaus. Furthermore, this
            constrains any split so that there cannot be more than one counter-
            fit plateau split per bestfit plateau. This option is designed
            specifically for chi_squared_step_fitter.
        min_step_length: Minimum length of fitted steps.
        min_step_magnitude: Minimum size of allowed fitted step.

    Returns:
        The sequence of plataus given, with one of them split into two such
        that this split has the lowest sum of squared residues possible.
        Returns None if plateaus cannot be split (due to e.g. bestfit_plateaus
        blocking all possible splits, or all plateaus already being
        min_step_length, etc.).
    """
    logger = logging.getLogger()
    logger.debug("stepfitting_library._best_split locals() = " + str(locals()))
    forbidden_splits = []
    #append to forbidden_splits the bestfit steps: counterfits cannot share
    #these splits
    if bestfit_plateaus is not None:
        for p, (start, stop, height) in enumerate(bestfit_plateaus[:-1]):
            next_start, next_stop, next_height = bestfit_plateaus[p + 1]
            forbidden_splits.append((stop, next_start))
    #append to forbidden_splits counterfit plateaus' invalid splits that
    #would result in more than one counterfit step per bestfit step
    if bestfit_plateaus is not None:
        all_counterfit_starts = [start
                                 for (start, stop, height) in plateaus]
        all_counterfit_stops = [stop
                                for (start, stop, height) in plateaus]
        for p, (start, stop, height) in enumerate(bestfit_plateaus):
            for f in range(start, stop + 1):
                if f in all_counterfit_starts:
                    assert f == 0 or f - 1 in all_counterfit_stops
                    forbidden_splits += [(u, u + 1)
                                         for u in range(start, stop)]
    #append to forbidden splits plateaus that are already shorter than
    #min_step_length
    for p, (start, stop, height) in enumerate(plateaus):
        if stop - start < min_step_length:
            forbidden_splits += [(u, u + 1) for u in range(start, stop)]
    #append to forbidden splits those cases that would result in a plateau
    #shorter than min_step_length
    for p, (start, stop, height) in enumerate(plateaus):
        for u in range(start, stop):
            if u - start < min_step_length or stop - u < min_step_length:
                forbidden_splits.append((u, u + 1))
    best_split_index = None
    best_split_residuals = len(luminosities) * (np.amax(luminosities) -
                                                np.amin(luminosities))**2
    best_split_results = None
    for p, plateau in enumerate(plateaus):
        start, stop, height = plateau
        (left_plateau, left_residuals,
         right_plateau, right_residuals,
         total_residuals) = \
                          _split_plateau(luminosities=luminosities,
                                         plateau=plateau,
                                         forbidden_splits=forbidden_splits,
                                         min_step_magnitude=min_step_magnitude)
        if (left_plateau is not None and
            right_plateau is not None and
            total_residuals < best_split_residuals):
            best_split_index, best_split_residuals = p, total_residuals
            best_split_results = (left_plateau, left_residuals,
                                  right_plateau, right_residuals,
                                  total_residuals)
    if best_split_index is not None:
        assert best_split_results is not None
        (best_left_plateau, best_left_residuals, best_right_plateau,
         best_right_residuals, best_total_residuals) = best_split_results
        return_result = (plateaus[:best_split_index] +
                         [best_left_plateau, best_right_plateau] +
                         plateaus[best_split_index + 1:])
    else:
        return_result = None
    logger.debug("stepfitting_plateau._best_split: returning result = " +
                 str(return_result))
    return return_result


def _fit_steps(luminosities, num_plateaus, bestfit_plateaus=None,
               existing_fit=None, min_step_length=2,
               min_step_magnitude=5000):
    """
    Utility function for chi_squared_step_fitter containing the fitting iteration loop.

    Arguments:
        luminosities: List of luminosities of the Spot through frames.
        num_plateaus: Number of plateaus to fit to the luminosities. Cannot be
            greater than the number of luminosities.
        bestfit_plateaus: If not None, this list of plateaus is taken as the
            set of best-fitting plateaus against which this function is to make
            a counter-fit. (See chi_squared_step_fitter for details about counter-fits.) If
            bestfit_plateaus is not None, then num_plateaus must be
            len(bestfit_plateaus) + 1.
        existing_fit: If not None, initializes algorithm with these existing
            plateaus. Thus, _fit_steps tries to find the best fit that is some
            combination of plateau splits performed on those in existing_fit.
            If this is given, num_plateaus must be less than or equal to
            len(existing_fit). This function does not check whether
            bestfit_plateaus and existing_fit contradict each other if both are
            given.
        min_step_length: Minimum length of fitted steps.
        min_step_magnitude: Minimum size of allowed fitted steps.

    Returns:
        If bestfit_plateaus is None, _fit_steps returns a best-fit on
        luminosities as a list of num_plateaus number of plateaus. If
        bestfit_plateaus is not None, then _fit_steps returns the counter-fit
        on luminosities, given bestfit_plateaus.
    """
    logger = logging.getLogger()
    if len(luminosities) < num_plateaus:
        raise ValueError("num_plateaus = " + str(num_plateaus) +
                         " is greater than len(luminosities) = " +
                         str(len(luminosities)))
    if (bestfit_plateaus is not None and
        len(bestfit_plateaus) + 1 != num_plateaus):
        raise ValueError("len(bestfit_plateaus) + 1 = " +
                         str(len(bestfit_plateaus) + 1) +
                         " != num_plateaus = " + str(num_plateaus))
    if existing_fit is not None and num_plateaus < len(existing_fit):
        raise ValueError("num_plateaus = " + str(num_plateaus) + " but " +
                         "len(existing_fit) = " + str(len(existing_fit)))
    #Initialize the algorithm
    if existing_fit is None:
        initial_plateau = _fit_plateau(luminosities, 0, len(luminosities) - 1)
        plateaus = [initial_plateau]
    else:
        plateaus = existing_fit
    while len(plateaus) < num_plateaus:
        new_plateaus = _best_split(luminosities=luminosities,
                                   plateaus=plateaus,
                                   bestfit_plateaus=bestfit_plateaus,
                                   min_step_length=min_step_length,
                                   min_step_magnitude=min_step_magnitude)
        if new_plateaus is None:
            logger.debug("stepfitting_library._fit_steps: _best_split could " +
                         "not split plateaus further; locals() = " +
                         str(locals()))
            break    #no further splits could be done by _best_split
        else:
            plateaus = new_plateaus
    logger.debug("stepfitting_library._fit_steps: returning plateaus = " +
                 str(plateaus))
    return plateaus


def chi_squared_step_fitter(luminosity_sequence, num_steps_multiplier=1,
                            num_steps=None, min_step_length=2,
                            min_step_magnitude=0.0, ignore_counterfits=False):
    """
    A step-fitting algorithm for Spot luminosity.

    Taken from "Assembly dynamics of microtubules at molecular resolution,"
    by Kerssemakers et al.    doi:10.1038/nature04928

    We strongly recommend reading through Supplementary Methods 3 of the paper
    for a clear graphical illustration of the algorithm.

    Input to the algorithm is a sequence of a Spot's luminosity as it varies
    with time across frames. The algorithm tries to find step-wise increases or
    decreases in the Spot's luminosity. The decreases, for example, may be due
    to fluor photobleaching, fluor removal via Edman, etc.

    We hope that the behavior of the Spots due to fluors is (as in the paper)
    "a step train with with steps of varying size and duration, hidden in
    Gaussian noise with RMS amplitude sigma."

    Note that the plateaus cannot have horizontal gaps between them: all frames
    must be covered.

    The quality of a series of plateau fits across a sequence of frames is
    evaluated by the total sum of squares of all luminosity deviations for all
    frames from their respective fitted plateau. We are searching for the
    sequence of best-fitting plateaus, with the lowest sum of squared
    deviations, but at the same time not over-fit.

    In summary, the plateau-fitting algorithm works as follows:
        1. Initializes  by fitting one plateau across all frames.
        2. For subsequent iterations, the algorithm repeatedly splits one
            fitted plateau into two segments, re-fitting each of the segments
            to their respective frames. This is equivalent to adding a step. At
            each iteration, the plateau that the algorithm chooses to split is
            the one whose split results in the largest vertical distance
            between the newly-created segments.
        3. Iterative splitting continues until either there are no more
            plateaus left to split (i.e. there is one plateau per frame) or
            some specified number of plateaus is reached.
        4. At some point, with enough plateaus fitted, it is probable that the
            algorithm starts to overfit the data. Following Kerssemakers et
            al., each the number of plateaus increases, we perform what they
            call a "counter-fit" ("counterfit"?), and compare the quality of
            the counter-fit with the best fit. A counter-fit is composed of
            plateaus whose connecting steps occur in the middle of the best-
            fitted plateaus, with one step per best-fitting plateau. The
            counter-fitted plateaus are also best-fitted to their respective
            intervals, however these intervals are constrained as just
            described. Just as for the best-fitting plateau sequence, we
            compute a total sum of squared deviations for the counter-fit
            plateaus. The ratio of this sum from the counter-fit, divided by
            the same sum from the best-fit is defined as the "step-indicator"
            S. Kerssemakers et al.: "When the number of steps in the best fit
            is very close to the real number of steps in the data, the value of
            S will be large... If however the data are severley under- or
            overfitted, or when the data consist of gradual non-stepped growth,
            the value for S will be close to 1." Thus, we perform counterfits
            for the increasingly refined series of best-fits, and compute their
            S's. The number of best-fit steps with the highest S is chosen as
            the correct number of plateaus/steps.

    Arguments:
        luminosity_sequence: Sequence of a Spot's photometries through the
            frames as a list.
        num_steps_multiplier: Keep adding steps to the fit until their number
            equals or is greater than num_steps_multiplier *
            len(luminosity_sequence). Must be in the open interval (0, 1). The
            algorithm will attempt to fit at most len(luminosity_sequence) - 2
            number of steps. (The -2 is to allow for a counter-fit.)
        num_steps: If not None, overrides num_steps_multiplier. Must be between
            1 and len(luminosity_sequence) - 1, inclusive.
        min_step_length: Minimum length of fitted steps.
        min_step_magnitude: Minimum size of allowed fitted steps.
        ignore_counterfit: If True, return the largest number of plateaus
            fitted, regardless of counterfits.

    Returns:
        List of horizontal plateaus fit to the luminosity sequence. Each
        plateau is specified by (start_frame, stop_frame, height).

        [( start_frame_1, stop_frame_1, height_1),
         (  stop_frame_1, stop_frame_2, height_2),

         ...

         (stop_frame_n-1, stop_frame_n, height_n)
        ]
    """
    logger = logging.getLogger()
    if not 0 < num_steps_multiplier <= 1:
        raise ValueError("num_steps_multiplier has an invalid value of " +
                         str(num_steps_multiplier))
    if (num_steps is not None and
        not 0 < num_steps < len(luminosity_sequence)):
        raise ValueError("num_steps has an invalid value of " +
                         str(num_steps) +
                         " vs len(luminosity_sequence) = " +
                         str(len(luminosity_sequence)))
    #if not explicitly given, compute the number of steps to (over)fit to
    if num_steps is None:
        num_steps = min(int(np.ceil(num_steps_multiplier *
                            len(luminosity_sequence))),
                        len(luminosity_sequence) - 2)
    #num_steps implies this num_plateaus to fit
    num_plateaus = num_steps + 1
    logger.debug("stepfitting_library.chi_squared_step_fitter locals() = " +
                 str(locals()))
    #plateau_fits stores the best-fit, the counter-fit, and S for each of
    #the increasing number of plateaus.
    #
    #Each member is (best_fit, counter_fit, S).
    #
    #Each of the best_fit's or counter_fit's is a list of plateaus in the
    #same list format as in this function's docstring's "Returns" section.
    plateau_fits = []
    for p in range(1, num_plateaus + 1):
        logger.debug("stepfitting_library.chi_squared_step_fitter: at " +
                     "plateau fit p = " + str(p) + " at time " +
                     str(datetime.datetime.fromtimestamp(time.time())))
        if len(plateau_fits) > 0:
            existing_fit = plateau_fits[-1][0]
        else:
            existing_fit = None
        best_fit = _fit_steps(luminosities=luminosity_sequence,
                              num_plateaus=p,
                              bestfit_plateaus=None,
                              existing_fit=existing_fit,
                              min_step_length=min_step_length,
                              min_step_magnitude=min_step_magnitude)
        #check to see if we can't increase number of best-fits due to some
        #constraint
        if (len(plateau_fits) > 0 and
            len(best_fit) == len(plateau_fits[-1][0])):
            break
        bestfit_residuals = \
                  _plateaus_squared_residuals(luminosities=luminosity_sequence,
                                              plateaus=best_fit)
        counter_fit = _fit_steps(luminosities=luminosity_sequence,
                                 num_plateaus=p + 1,
                                 bestfit_plateaus=best_fit,
                                 existing_fit=None,
                                 min_step_length=0,#counterfits must allow
                                                   #length 0
                                 min_step_magnitude=min_step_magnitude)
        counterfit_residuals = \
                  _plateaus_squared_residuals(luminosities=luminosity_sequence,
                                              plateaus=counter_fit)
        if float(bestfit_residuals) != 0:
            S = float(counterfit_residuals) / float(bestfit_residuals)
        else:
            logger.debug("stepfitting_library.chi_squared_step_fitter: " +
                         "bestfit_residuals = 0, locals() = " + str(locals()))
            S = 10**10
        plateau_fits.append((best_fit, counter_fit, S))
    if ignore_counterfits:
        return_fit = sorted(plateau_fits, key=lambda x:len(x[0]),
                            reverse=True)[0][0]
    else:
        return_fit = sorted(plateau_fits, key=lambda x:x[2], reverse=True)[0][0]
    logger.debug("stepfitting_library.chi_squared_step_fitter: returning " +
                 "fit = " + str(return_fit))
    return return_fit


def plateau_value(plateaus, frame):
    """
    Get value of fitted plateaus at a frame.

    Arguments:
        plateaus: List of fitted plateaus.
        frame: Frame number to get value for.

    Returns:
        Height of the plateau at frame.
    """
    logger = logging.getLogger()
    logger.debug("stepfitting_library.plateau_value locals() = " +
                 str(locals()))
    for (start, stop, height) in plateaus:
        if start <= frame <= stop:
            value = height
            break
    else:
        raise ValueError("frame " + str(frame) + " is outside of " +
                         "plateaus " + str(plateaus))
    return value


def mean_filter(luminosities, rank):
    raise DeprecationWarning("This function was made, but not used. I'm not "
                             "sure it handles edges the way I want it to "
                             "right now.")
    if rank < 1:
        raise ValueError("Rank cannot be less than one.")
    padded_luminosities = ([luminosities[0] for r in range(rank)] +
                            luminosities +
                           [luminosities[-1] for r in range(rank)])
    return [np.mean(padded_luminosities[L:L + 2 * rank + 1])
            for L, lum in enumerate(luminosities)]


def _pairwise(iterable):
    """
    Produces an iterable that yields "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    From Python itertools recipies.

    e.g.

    a = _pairwise([5, 7, 11, 4, 5])
    for v, w in a:
        print [v, w]

    will produce

    [5, 7]
    [7, 11]
    [11, 4]
    [4, 5]

    The name of this function reminds me of Pennywise.
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def _triplewise(iterable):
    """
    Produces an iterable that yields "s -> (s0, s1, s2), (s1, s2, s3),
                                           (s2, s3, s4), ..."

    Variant of _pairwise.

    a = _triple([5, 7, 11, 4, 5])
    for u, v, w in a:
        print [u, v, w]

    will produce

    [5, 7, 11]
    [7, 11, 4]
    [11, 4, 5]
    """
    a, b, c = itertools.tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return itertools.izip(a, b, c)


def plateaus_to_steps(plateaus):
    """
    Converts step-fitted plateaus as to a list of steps. Steps only happen
    between frames.

    Arguments:
        plateaus: List of fitted plateaus.

    Returns:
        List of steps in the form

        [(pre_step1_frame_#, post_step1_frame_#, step1_magnitude),
         (pre_step2_frame_#, post_step2_frame_#, step2_magnitude),
                ...
         (pre_stepN_frame_#, post_stepN_frame_#, stepN_magnitude)]

        Each step's tuple contains three values. pre_stepI_frame_# is the frame
        index of the frame directly preceeding the I'th step,
        post_stepI_frame_# is the frame index of the frame directly following
        the I'th step. stepI_magnitude is the size of the step, with steps
        increasing in luminosity being positive.
    """
    steps = []
    #uses _pairwise to generate pairs of plateaus
    for (plateau_A, plateau_B) in _pairwise(plateaus):
        start_A, stop_A, height_A = plateau_A
        start_B, stop_B, height_B = plateau_B
        steps.append((stop_A, start_B, height_B - height_A))
    return steps


def last_step_info(steps, frame):
    """
    Get information about the last step that took place before the frame.

    Arguments:
        steps: Steps representing the output of plateaus_to_steps.
        frame: Frame # for which to get the last preceeding step's information.

    Returns:
        (last_step_num, last_step_position, last_step_magnitude)

        where last_step_num is the (0-indexed) number of the last step,
        last_step_position is the (0-indexed) frame # after which the step took
        place (i.e. the step is between this frame # and # + 1), and
        last_step_magnitude is the magnitude of the step (increase in
        luminosity is positive).

        Returns (None, None, None) if there was not step before frame.
    """
    if frame < 0:
        raise ValueError("frame must be a positive integer.")
    return_values = None, None, None
    for s, (step_A, step_B) in enumerate(_pairwise(steps)):
        pre_frame_A, post_frame_A, magnitude_A = step_A
        pre_frame_B, post_frame_B, magnitude_B = step_B
        if post_frame_A <= frame <= pre_frame_B:
            return_values = (s, pre_frame_A, magnitude_A)
            break
    else:
        if len(steps) == 0:
            return_values = None, None, None
        else:
            last_pre_frame, last_post_frame, last_magnitude = steps[-1]
            if frame >= last_pre_frame:
                return_values = (len(steps) - 1, last_pre_frame,
                                 last_magnitude)
    return return_values


def frame_plateau(plateaus, frame):
    """
    Get the plateau containing frame, and the plateau's index in plateaus.
    """
    return_plateau = None, None, None
    return_index = None
    for p, plateau in enumerate(plateaus):
        start, stop, height = plateau
        if start <= frame <= stop:
            return_plateau = start, stop, height
            return_index = p
            break
    return return_plateau, return_index


def _consecutive_integers(integers):
    """
    Given a list of integers, returns sub-lists of consecutive integers.

    From http://stackoverflow.com/questions/2361945/detecting-consecutive\
    -integers-in-a-list

    e.g. given

    integers = [ 1, 4,5,6, 10, 15,16,17,18, 22, 25,26,27,28]

        this function will return

    [[1], [4, 5, 6], [10], [15, 16, 17, 18], [22], [25, 26, 27, 28]]
    """
    consecutive_integers = []
    for k, g in itertools.groupby(enumerate(integers),
                                  lambda (i, x):i - x):
        consecutive_integers.append(map(itemgetter(1), g))
    return consecutive_integers


def _merge_plateaus(luminosities, plateau_a, plateau_b):
    """
    Given two consecutive plateaus, return a single plateau fitted to
    luminosities spanned by both.
    """
    start_a, stop_a, height_a = plateau_a
    start_b, stop_b, height_b = plateau_b
    if stop_a + 1 != start_b:
        raise ValueError("Merged plateaus must be consecutive. locals() = " +
                         str(locals()))
    return _fit_plateau(luminosities, start_a, stop_b)


#Not sure how to do this yet...
#def _repeat_filter(target_filter):
#    """
#    To be used as a decorator for applying filters to a plateaus repeatedly.
#    Specifically, applies target_filter (len(plateaus) - 1) number of times.
#
#    plateaus must be the second argument passed to target_filter.
#    """
#    def target_filter_wrap(*args, **kwargs):
#        plateaus = kwargs['plateaus']
#        filtered_plateaus = target_filter(*args, **kwargs)
#        for n in range(len(plateaus) - 1):
#            filtered_plateaus = target_filter(plateaus=filtered_plateaus,
#                                              *args, **kwargs)
#    return target_filter_wrap



def _filter_upsteps_singlepass(luminosities, plateaus):
    """
    Utility function for filter_upsteps. Will merge at least one existing
    upstep every time it is called. To guarantee all upsteps are eliminated,
    this function must be applied at least as many times as there are plateaus,
    minus one.

    Arguments:
        luminosities: List of luminosities of a Spot through frames.
        plateaus: Plateaus to filter.

    Returns:
        List of plateaus with at least one upstep removed via plateau merging.
    """
    if len(plateaus) < 2:
        downstep_filtered = plateaus
    else:
        downstep_filtered = []
        for a, b in _pairwise(plateaus):
            start_a, stop_a, height_a = a
            start_b, stop_b, height_b = b
            #Check to see if a was merged in the prior iteration.
            if len(downstep_filtered) > 0:
                last_start, last_stop, last_height = downstep_filtered[-1]
                if stop_a == last_stop:
                    continue
                else:
                    assert last_stop + 1 == start_a
            if height_b > height_a:
                merged = _merge_plateaus(luminosities, a, b)
                downstep_filtered.append(merged)
            else:
                downstep_filtered.append(a)
        #If the last two plateaus were not merged, append the last b.
        last_start, last_stop, last_height = plateaus[-1]
        last_flt_start, last_flt_stop, last_flt_height = downstep_filtered[-1]
        if last_stop != last_flt_stop:
            downstep_filtered.append(plateaus[-1])
    return downstep_filtered


def filter_upsteps(luminosities, plateaus):
    """
    Merges plateaus until no upsteps remain.

    Upsteps are defined as a step from a plateau with a lower height to a
    plateau with a higher height.

    Note that this function does not do any kind of global fit optimization. It
    merely iteratively merges plateaus until no upsteps are left. The algorithm
    immediately merges an upstep as soon as it finds one. It does not check
    whether this new merge produces a new upstep or not. Every successive merge
    to remove an upstep may generate a new one. It is hence quite possible to
    feed in a sequence of plateaus that will be merged until there is only one
    left.

    Arguments:
        luminosities: List of luminosities of a Spot through frames.
        plateaus: Plateaus to filter.

    Returns:
        List of plateaus with upsteps removed via plateau merging.
    """
    filtered_plateaus = plateaus
    for n in range(len(plateaus) - 1):
        filtered_plateaus = _filter_upsteps_singlepass(luminosities,
                                                       filtered_plateaus)
    return filtered_plateaus


def _filter_small_steps_singlepass(luminosities, plateaus, min_magnitude=None,
                                   min_noise_ratio=None):
    """
    Utility function for filter_small_steps. Will merge at least one existing
    upstep every time it is called. To guarantee all small steps are
    eliminated, this function must be applied at least as many times as there
    are plateaus, minus one.

    Arguments:
        luminosities: List of luminosities of a Spot through frames.
        plateaus: Plateaus to filter.
        min_magnitude: Remove any steps smaller than this size.
        min_noise_ratio: Remove any steps that are smaller than

            min_noise_ratio * max(sqrt(plateau A squared residuals),
                                  sqrt(plateau B squared residuals))

            where plateau A and plateau B are the two plateaus around the step.

        Note that min_magnitude and min_noise criteria are applied in an OR
        fashion: i.e. both can be provided, and a step is removed if either is
        satisfied. If the parameter is None, it is not applied.
        Consequentially, if both criteria are None, no steps are filtered.

    Returns:
        List of plateaus with at least one small step removed via plateau
        merging.
    """
    if min_magnitude is not None and min_magnitude < 0:
        raise ValueError("min_magnitude < 0 makes no sense. locals() = " +
                         str(locals()))
    if min_noise_ratio is not None and min_noise_ratio < 0:
        raise ValueError("min_noise_ratio < 0 makes no sense. locals() = " +
                         str(locals()))
    if len(plateaus) < 2:
        magnitude_filtered = plateaus
    else:
        magnitude_filtered = []
        for a, b in _pairwise(plateaus):
            start_a, stop_a, height_a = a
            start_b, stop_b, height_b = b
            #Check to see if a was merged in the prior iteration.
            if len(magnitude_filtered) > 0:
                last_start, last_stop, last_height = magnitude_filtered[-1]
                if stop_a == last_stop:
                    continue
                else:
                    assert last_stop + 1 == start_a
            #step size to be tested against the criteria
            step_size = abs(height_a - height_b)
            #booleans to track whether merging is necessary
            noise_ratio_merge, magnitude_merge = False, False
            #if min_noise_ratio was given, compute max noise for plateaus and
            #test
            if min_noise_ratio is not None:
                max_noise = \
                    max(math.sqrt(_plateau_squared_residuals(luminosities, a)),
                        math.sqrt(_plateau_squared_residuals(luminosities, b)))
                if step_size < max_noise * min_noise_ratio:
                    noise_ratio_merge = True
            #if min_magnitude given, test it
            if (min_magnitude is not None and
                step_size < min_magnitude):
                magnitude_merge = True
            #if either of the criteria indicates a merge needs to take place,
            #do it
            if noise_ratio_merge or magnitude_merge:
                merged = _merge_plateaus(luminosities, a, b)
                magnitude_filtered.append(merged)
            else:
                magnitude_filtered.append(a)
        #If the last two plateaus were not merged, append the last b.
        last_start, last_stop, last_height = plateaus[-1]
        last_flt_start, last_flt_stop, last_flt_height = magnitude_filtered[-1]
        if last_stop != last_flt_stop:
            magnitude_filtered.append(plateaus[-1])
    return magnitude_filtered


def filter_small_steps(luminosities, plateaus, min_magnitude=None,
                       min_noise_ratio=None):
    """
    Merges plateaus until no steps smaller than a specified size remain.

    Note that this function does not do any kind of global fit optimization. It
    merely iteratively merges plateaus until no small steps are left. The
    algorithm immediately merges a small step as soon as it finds one. It does
    not check whether this new merge produces a new small step or not. Every
    successive merge to remove a small step may generate a new one. It is hence
    quite possible to feed in a sequence of plateaus that will be merged until
    there is only one left.

    Arguments:
        luminosities: List of luminosities of a Spot through frames.
        plateaus: Plateaus to filter.
        min_magnitude: Remove any steps smaller than this size.
        min_noise_ratio: Remove any steps that are smaller than

            min_noise_ratio * max(sqrt(plateau A squared residuals),
                                  sqrt(plateau B squared residuals))

            where plateau A and plateau B are the two plateaus around the step.

        Note that min_magnitude and min_noise criteria are applied in an OR
        fashion: i.e. both can be provided, and a step is removed if either is
        satisfied. If the parameter is None, it is not applied.
        Consequentially, if both criteria are None, no steps are filtered.

    Returns:
        List of plateaus with small steps removed via plateau merging.
    """
    if min_magnitude is not None and min_magnitude < 0:
        raise ValueError("min_step_magnitude < 0 makes no sense. " +
                         "locals() = " + str(locals()))
    if min_noise_ratio is not None and min_noise_ratio < 0:
        raise ValueError("min_step_noise_ratio < 0 makes no sense. " +
                         "locals() = " + str(locals()))
    filtered_plateaus = plateaus
    for n in range(len(plateaus) - 1):
        filtered_plateaus = \
                _filter_small_steps_singlepass(luminosities,
                                               filtered_plateaus,
                                               min_magnitude=min_magnitude,
                                               min_noise_ratio=min_noise_ratio)
    return filtered_plateaus


def sliding_t_fitter(luminosity_sequence, window_radius=20, p_threshold=0.001,
                     median_filter_size=None, downsteps_only=False,
                     min_step_magnitude=None):
    """
    Fits steps to a luminosity sequence using a sliding-window two-tailed
    Welch's t-test.

    Between every frame, this algorithm computes a two-tailed Welch's t-test,
    with frame luminosities to its left being compared to frame luminosities on
    its right. If the two-tailed p-value is below a specified threshold, we
    reject the null hypothesis of equal average luminosities for frames to the
    left and right, and claim that a step occurs.

    Currently, this uses the default scipy.stats.ttest_ind function to compute
    t and p values. For those inter-frame positions whose p values are above a
    threshold, their t values are compared to adjacent frames and those frames
    with a local t maxima are chosen as steps.

    Once the locations of steps is found, each plateau between each two steps
    is fitted as the mean of the luminosities in that interval.

    An alternative t-test variant that is currently not implemented but may be
    be useful later is described below:

    ===BELOW NOT (YET?) IMPLEMENTED===

    The method below is taken from "A Comparison of Step-Detection Methods: How
    Well Can You Do?" by Carter et al. doi: 10.1529/biophysj.107.110601 and
    "Mechanics of the kinesin step" by Carter & Cross doi: 10.1038/nature03528.
    This method is not the one currently implemented.

    Between every frame, this algorithm computes a two-tailed Welch's t-test,
    with frame luminosities to its left being compared to frame luminosities on
    its right:

        (mean(left_frames) - mean(right_frames))
    t = ----------------------------------------
        sqrt(var(left_frames)/num_left_frames +
             var(right_frames)/num_right_frames)

    with degrees of freedom determined by

             (var(left_frames) + var(right_frames))
             --------------------------------------
             max(num_left_frames, num_right_frames)
    df = ---------------------------------------------------------
             (var(left_frames)**2 + var(right_frames)**2)
             -------------------------------------------------
             (max(num_left_frames, num_right_frames) *
              (max(num_left_frames, num_right_frames) - 1))

    ===ABOVE NOT (YET?) IMPLEMENTED===

    Arguments:
        luminosity_sequence: Sequence of a Spot's photometries through the
            frames as a list.
        window_radius: Radius to use for t-test window, i.e. a window_radius
            number of frames on each side will be included in the test.
        p_threshold: Threshold of p-value to use as a cutoff for the t-test.
        median_filter_size: If not None, first apply median filter of this
            size.
        downsteps_only: If True, will only fit downward steps. Any upward
            trends are ignored.
        min_step_magnitude: If not None, will remove steps that are smaller in
            magnitude than this number.

    Returns:
        List of horizontal plateaus fit to the luminosity sequence. Each
        plateau is specified by (start_frame, stop_frame, height).

        [( start_frame_1, stop_frame_1, height_1),
         (  stop_frame_1, stop_frame_2, height_2),

         ...

         (stop_frame_n-1, stop_frame_n, height_n)
        ]
    """
    logger = logging.getLogger()
    if median_filter_size is not None:
        luminosity_sequence = medfilt(luminosity_sequence,
                                      kernel_size=median_filter_size)
    step_positions = []
    step_positions_by_radius = []
    ftp_by_radius = []
    for radius in range(5, window_radius):
        step_positions_by_radius.append([])
        ftp_by_radius.append({})
        for f, frame in enumerate(luminosity_sequence):
            left_frames = luminosity_sequence[f - radius:f]
            right_frames = luminosity_sequence[f:f + radius]
            t, p = ttest_ind(left_frames, right_frames, equal_var=False)
            if p < p_threshold:
                step_positions_by_radius[-1].append(f)
                ftp_by_radius[-1].setdefault(f, (t, p))
    if len(step_positions_by_radius) > 0:
        step_intersection = set(step_positions_by_radius[0])
    else:
        step_intersection = set()
    for steps in step_positions_by_radius:
        step_intersection &= set(steps)
    step_positions = sorted(list(step_intersection))
    consecutive_step_positions = _consecutive_integers(step_positions)
    filtered_step_positions = []
    for grouping in consecutive_step_positions:
        max_t_frame, max_t = \
                         sorted([(frame, ftp_by_radius[-1][frame][0])
                                 for frame in grouping], key=lambda x:x[0])[-1]
        filtered_step_positions.append(max_t_frame)
    step_positions = filtered_step_positions
    logger.debug("stepfitting_library.sliding_t_fitter: " +
                 "step_positions_by_radius = " +
                 str(step_positions_by_radius) + "; step_intersection = " +
                 str(step_intersection))
    if len(step_positions) > 0:
        start, stop = 0, step_positions[0] - 1
        plateau = _fit_plateau(luminosity_sequence, start, stop)
        plateaus = [plateau]
    else:
        start, stop = 0, len(luminosity_sequence) - 1
        plateau = _fit_plateau(luminosity_sequence, start, stop)
        plateaus = [plateau]
    for f1, f2 in _pairwise(step_positions):
        start, stop = f1, f2 - 1
        plateau = _fit_plateau(luminosity_sequence, start, stop)
        plateaus.append(plateau)
    if len(step_positions) > 0:
        start, stop = step_positions[-1], len(luminosity_sequence) - 1
        plateau = _fit_plateau(luminosity_sequence, start, stop)
        plateaus.append(plateau)
    #If requested, remove any upward steps by merging plateaus.
    if downsteps_only:
        plateaus = filter_upsteps(luminosity_sequence, plateaus)
    #If requested, remove steps below min_step_magnitude.
    if min_step_magnitude is not None:
        plateaus = filter_small_steps(luminosity_sequence, plateaus,
                                      min_magnitude=min_step_magnitude)
    return plateaus
    #This block is for the Carter et al. method.
    #for f, frame in enumerate(luminosity_sequence):
    #    left_frames = luminosity_sequence[f - window_radius:f]
    #    right_frames = luminosity_sequence[f:f + window_radius]
    #    mL, mR = float(np.mean(left_frames)), float(np.mean(right_frames))
    #    vL, vR = float(np.var(left_frames)), float(np.var(right_frames))
    #    NL, NR = float(len(left_frames)), float(len(right_frames))
    #    N = float(max(NL, NR))
    #    t = (mL - mR) / math.sqrt(vL / NL + vR / NR)
    #    df = ((vL + vR) / N) / ((vL**2 + vR**2) / (N * (N - 1)))
    #    t_tests.append(t)
    #    degrees_freedom.append(df)


def chung_kennedy_filter(luminosities,
                         #window_lengths=(2, 3, 4, 5, 6, 7, 8, 16),
                         window_lengths=range(2, 17),
                         #window_lengths=(2, 4, 8, 16),
                         M=10, p=2):
    """
    An implementation of "Forward-backward non-linear filtering technique for
    extracting small biological signals from noise," by Chung & Kennedy.
    doi:10.1016/0165-0270(91)90118-J

    See the paper for explanations of the algorithm and parameters in detail.

    Arguments:
        luminosities: Sequence of luminosities -- as a list -- to fit. Must be
            longer than length 2.
        window_lengths: List or tuple of window lengths to use for the backward
            and forward predictors.
        M: Size of window over which predictors are to be compared.
        p: Exponent in predictors' weighting factors.

    Returns:
        Sequence of filtered luminosities as a list.
    """
    logger = logging.getLogger()
    if not len(luminosities) > 2:
        raise ValueError("luminosities must have len(luminosities) > 2; " +
                         "currently len(luminosities) = " +
                         str(len(luminosities)))
    filtered_luminosities = []
    #Prepare front and back predictors for each frame, for each window_length.
    #
    #front_predictors is a dictionary containing the front predictors for all
    #frames, for each window length. front_predictors[w][k] will give the front
    #predictor for window length w, at frame k.
    #
    #back_predictors is the corresponding dictionary for back predictors.
    front_predictors = {w: [None for L, luminosity in enumerate(luminosities)]
                        for w in window_lengths}
    back_predictors = {w: [None for L, luminosity in enumerate(luminosities)]
                       for w in window_lengths}
    #Loop to fill front_predictors and back_predictorS
    for window_length in window_lengths:
        for L, luminosity in enumerate(luminosities):
            #On line below for rear_window, need to use max(,0) to prevent
            #array indexing from wrapping to the array end
            rear_window = luminosities[max(L - window_length - 1, 0):L]
            front_window = luminosities[L + 1:L + window_length + 1]
            if len(rear_window) > 0:
                front_predictor = np.mean(rear_window)
                assert 0 < L #we're not at the edge
            else:
                front_predictor = None #this means we are at the edge
                assert L == 0
            front_predictors[window_length][L] = front_predictor
            if len(front_window) > 0:
                back_predictor = np.mean(front_window)
                assert L < len(luminosities) - 1 #we're not at the edge
            else:
                back_predictor = None #this means we are at the edge
                assert L == len(luminosities) - 1
            back_predictors[window_length][L] = back_predictor
    logger.debug("stepfitting_library.chung_kennedy_filter: " +
                 "front_predictors = " + str(front_predictors))
    logger.debug("stepfitting_library.chung_kennedy_filter: " +
                 "back_predictors = " + str(back_predictors))
    #Prepare weights for front and back predictors.
    #
    #Weights are kept in dictionaries analagous to front_predictors and
    #back_predictors.
    front_weights = {w: [None for L, luminosity in enumerate(luminosities)]
                     for w in window_lengths}
    back_weights = {w: [None for L, luminosity in enumerate(luminosities)]
                    for w in window_lengths}
    #Loop to fill front_weights and back_weights
    for window_length in window_lengths:
        for L, luminosity in enumerate(luminosities):
            #Note that 'rear_window' and 'front_window' variables here differ
            #from the loop above.
            #
            #On lines below for rear_window and f_predictors, need to use
            #max(,0) to prevent array indexing from wrapping to the array end
            rear_window = luminosities[max(L - M + 1, 0):L + 1] #Hitchcock!
            f_predictors = \
                       front_predictors[window_length][max(L - M + 1, 0):L + 1]
            front_window = luminosities[L:L + M]
            b_predictors = back_predictors[window_length][L:L + M]
            #If we're at an edge frame, use only the appropriate front/back
            #predictor.
            if L == 0:
                #We are at the first frame; use back predictor only.
                front_weights[window_length][L] = 0
                back_weights[window_length][L] = 1
            elif L == len(luminosities) - 1:
                #We are at the last frame; use front predictor only.
                front_weights[window_length][L] = 1
                back_weights[window_length][L] = 0
            else:
                #We are at neither the first nor last frame. However, the
                #comparison window determined by M may include the edge frames
                #of the predictors, which were filled as 'None' above. In this
                #case, we need to truncate the arrays we will be using to
                #compute the weighting factors.
                if L - M < 0:
                    #This means that we are including the first frame in
                    #rear_window and f_predictors.
                    rear_window = rear_window[1:]
                    f_predictors = f_predictors[1:]
                if L + M >= len(luminosities) - 1:
                    #This means we are including the last frame in front_window
                    #and b_predictors.
                    front_window = front_window[:-1]
                    b_predictors = b_predictors[:-1]
                #compute sum of square differences
                b_diff = sum((np.array(rear_window) -
                              np.array(f_predictors))**2)
                f_diff = sum((np.array(front_window) -
                              np.array(b_predictors))**2)
                if b_diff != 0 and f_diff != 0:
                    front_weights[window_length][L] = b_diff**-p
                    back_weights[window_length][L] = f_diff**-p
                elif b_diff == 0 and f_diff != 0:
                    front_weights[window_length][L] = 1
                    back_weights[window_length][L] = 0
                elif b_diff != 0 and f_diff == 0:
                    front_weights[window_length][L] = 0
                    back_weights[window_length][L] = 1
                else:
                    front_weights[window_length][L] = 1
                    back_weights[window_length][L] = 0
    #Normalize front_weights and back_weights.
    total_weights = [(sum([weight if L == L2 else 0
                           for wL, weights in front_weights.iteritems()
                           for L2, weight in enumerate(weights)]) +
                      sum([weight if L == L2 else 0
                           for wL, weights in back_weights.iteritems()
                           for L2, weight in enumerate(weights)]))
                     for L, luminosity in enumerate(luminosities)]
    logger.debug("stepfitting_library.chung_kennedy_filter: total_weights = " +
                 str(total_weights))
    front_weights = {wL: [float(weight) / total_weights[L]
                          for L, weight in enumerate(weights)]
                     for wL, weights in front_weights.iteritems()}
    back_weights = {wL: [float(weight) / total_weights[L]
                         for L, weight in enumerate(weights)]
                    for wL, weights in back_weights.iteritems()}
    logger.debug("stepfitting_library.chung_kennedy_filter: front_weights = " +
                 str(front_weights))
    logger.debug("stepfitting_library.chung_kennedy_filter: back_weights = " +
                 str(back_weights))
    filtered_luminosities = [None for L, luminosity in enumerate(luminosities)]
    for L, luminosity in enumerate(luminosities):
        if L == 0:
            #For the first frame, we should only be using the back predictor.
            assert all(front_weights[window_length][L] == 0
                       for window_length in window_lengths)
            logger.debug("stepfitting_library.chung_kennedy_filter: L = " +
                         str(L))
            logger.debug("stepfitting_library.chung_kennedy_filter: " +
                         "[len(back_weights[wL]) for wL in window_lengths] " +
                         " = " + str([len(back_weights[wL])
                                      for wL in window_lengths]))
            logger.debug("stepfitting_library.chung_kennedy_filter: " +
                         "[len(back_predictors[wL]) " +
                         "for wL in window_lengths] = " +
                         str([len(back_predictors[wL])
                              for wL in window_lengths]))
            filtered_luminosities[L] = sum([back_weights[wL][L] *
                                            back_predictors[wL][L]
                                            for wL in window_lengths])
        elif L == len(luminosities) - 1:
            #For the last frame, we should only be using the front predictor.
            assert all(back_weights[window_length][L] == 0
                       for window_length in window_lengths)
            logger.debug("stepfitting_library.chung_kennedy_filter: L = " +
                         str(L))
            logger.debug("stepfitting_library.chung_kennedy_filter: " +
                         "[len(front_weights[wL]) for wL in window_lengths] "+
                         "= " + str([len(front_weights[wL])
                                     for wL in window_lengths]))
            logger.debug("stepfitting_library.chung_kennedy_filter: " +
                         "[len(front_predictors[wL]) " +
                         "for wL in window_lengths] = " +
                         str([len(front_predictors[wL])
                              for wL in window_lengths]))
            filtered_luminosities[L] = sum([front_weights[wL][L] *
                                            front_predictors[wL][L]
                                            for wL in window_lengths])
        else:
            filtered_luminosities[L] = \
                          sum([front_weights[wL][L] * front_predictors[wL][L] +
                               back_weights[wL][L] * back_predictors[wL][L]
                               for wL in window_lengths])
    return filtered_luminosities


def remove_blips(luminosities, plateaus, smoothing_stddev=0.8):
    """Remove temporarily upward blips in fitted step functions."""
    raise DeprecationWarning("This function was made quickly, and has some "
                             "fundamental logical errors. Use at own risk.")
    if len(plateaus) < 3:
        raise ValueError("plateaus must have at least three members for blip "
                         "smoothing to make sense.")
    filtered_plateaus = []
    for i, (a, b, c) in enumerate(_triplewise(plateaus)):
        a_start, a_stop, a_height = a
        b_start, b_stop, b_height = b
        c_start, c_stop, c_height = c
        if i > 0:
            #This if statement checks to see if the plateau called a now was
            #either added in or merged in the last round
            assert len(filtered_plateaus) > 0
            last_start, last_stop, last_height = filtered_plateaus[-1]
            if last_start <= a_start <= last_stop:
                assert last_start <= a_stop <= last_stop
                continue
        #compute smoothing distance threshold between a and c
        a_stddev = math.sqrt(_plateau_squared_residuals(luminosities, a))
        c_stddev = math.sqrt(_plateau_squared_residuals(luminosities, c))
        max_stddev = max(a_stddev, c_stddev)
        smoothing_threshold = smoothing_stddev * max_stddev
        if (abs(a_height - c_height) < smoothing_threshold and
            c_height - max(a_height, c_height) > 0):
            merged_plateau = a_start, c_stop, np.mean(a_height, c_height)
            filtered_plateaus.append(merged_plateau)
        else:
            filtered_plateaus.append(a)
    #The above triplewise-based loop may end up incorporating the third-before-
    #last plateau as a, and stop. Check to see if the last two plateaus in
    #plateaus were included.
    b_start, b_stop, b_height = b = plateaus[-2]
    c_start, c_stop, c_height = c = plateaus[-1]
    last_start, last_stop, last_height = filtered_plateaus[-1]
    if c_stop != last_stop:
        #This means the last two plateaus were not incorporated as parts of a
        #merged plateau.
        assert b_start == last_stop + 1
        filtered_plateaus.append(b)
        filtered_plateaus.append(c)
    return filtered_plateaus


def refit_plateaus(luminosities, plateaus):
    """Re-fit plateaus to luminosities without moving their boundaries."""
    return [_fit_plateau(luminosities, start, stop)
            for start, stop, height in plateaus]


def _t_test_filter_singlepass(luminosities, plateaus, p_threshold,
                              drop_sort=True, no_merge_start=0):
    """
    Utility function for t_test_filter. Will merge at least one step that does
    not satisfy p_threshold every time it is called. To guarantee all
    unsatisfactory steps are eliminated, this function must be applied at least
    as many times as there are plateaus, minus one.

    Arguments:
        luminosities: List of luminosities of a Spot through frames.
        plateaus: Plateaus to filter.
        p_threshold: Two plateaus whose luminosities' t-test p-value is greater
            than or equal to p_threshold are merged.
        drop_sort: Merge steps in the order of their mutual p-values, with
            pairs of plateaus with the largest mutual t-test p-value merged
            first.
        no_merge_start: Do not merge any plateaus which end before the
            no_merge_start'th frame.

    Returns:
        List of plateaus with at least one step that fails the t-test removed
        via plateau merging.
    """
    if len(plateaus) < 2:
        filtered_plateaus = plateaus
    elif not drop_sort:
        filtered_plateaus = []
        for i, (a, b) in enumerate(_pairwise(plateaus)):
            a_start, a_stop, a_height = a
            b_start, b_stop, b_height = b
            #Check to see if a was merged in the prior iteration.
            if len(filtered_plateaus) > 0:
                last_start, last_stop, last_height = filtered_plateaus[-1]
                if a_stop == last_stop:
                    continue
                else:
                    assert last_stop + 1 == a_start
            #Check to see if plateau a ends before no_merge_start; if so, skip.
            if a_stop < no_merge_start:
                filtered_plateaus.append(a)
                continue
            left_frames = luminosities[a_start:a_stop + 1]
            right_frames = luminosities[b_start:b_stop + 1]
            t, p = ttest_ind(left_frames, right_frames, equal_var=False)
            if p >= p_threshold:
                merged = _merge_plateaus(luminosities, a, b)
                filtered_plateaus.append(merged)
            else:
                filtered_plateaus.append(a)
        #If the last two plateaus were not merged, append the last b.
        last_start, last_stop, last_height = plateaus[-1]
        last_flt_start, last_flt_stop, last_flt_height = filtered_plateaus[-1]
        if last_stop != last_flt_stop:
            filtered_plateaus.append(plateaus[-1])
    #Honestly, this section here is sufficient to cover the non-drop_sort case
    #too, simply by skipping the sorting step. But, I implemented this in a
    #rush, so here we are.
    else:
        pair_drops = [[a, b, abs(a[2] - b[2]), r]
                      for r, (a, b) in enumerate(_pairwise(plateaus))]
        for i, (a, b, d, r) in enumerate(pair_drops):
            a_start, a_stop, a_height = a
            b_start, b_stop, b_height = b
            left_frames = luminosities[a_start:a_stop + 1]
            right_frames = luminosities[b_start:b_stop + 1]
            t, p = ttest_ind(left_frames, right_frames, equal_var=False)
            pair_drops[i][2] = p
        s_pair_drops = sorted(pair_drops, key=lambda x:x[2], reverse=True)
        merge_bools = [False for m in s_pair_drops]
        for i, (a, b, p, r) in enumerate(s_pair_drops):
            a_start, a_stop, a_height = a
            if p >= p_threshold and a_stop >= no_merge_start:
                merge_bools[i] = True
        for i, (a, b, d, r) in enumerate(s_pair_drops):
            if merge_bools[i]:
                for j, (a2, b2, d2, r2) in enumerate(s_pair_drops):
                    if j <= i:
                        continue
                    else:
                        assert a != a2 and b != b2
                    if a == b2:
                        assert b != a2
                        merge_bools[j] = False
                    elif b == a2:
                        merge_bools[j] = False
        filtered_plateaus = []
        for r, (a, b) in enumerate(_pairwise(plateaus)):
            a_start, a_stop, a_height = a
            b_start, b_stop, b_height = b
            #Check to see if a was merged in the prior iteration.
            if len(filtered_plateaus) > 0:
                last_start, last_stop, last_height = filtered_plateaus[-1]
                if a_stop == last_stop:
                    continue
                else:
                    assert last_stop + 1 == a_start
            for i, (a2, b2, d2, r2) in enumerate(s_pair_drops):
                if r == r2:
                    assert a == a2 and b == b2
                    if merge_bools[i]:
                        merged = _merge_plateaus(luminosities, a, b)
                        filtered_plateaus.append(merged)
                        break
            else:
                filtered_plateaus.append(a)
        #If the last two plateaus were not merged, append the last b.
        last_start, last_stop, last_height = plateaus[-1]
        last_flt_start, last_flt_stop, last_flt_height = filtered_plateaus[-1]
        if last_stop != last_flt_stop:
            filtered_plateaus.append(plateaus[-1])
    return filtered_plateaus


def t_test_filter(luminosities, plateaus, p_threshold, drop_sort=True,
                  no_merge_start=0):
    """
    Merges plateaus that do not pass Welch's t-test.

    Between every plateau, a t-test of the luminosities is performed to
    evaluate the probability they belong to the same normal distribution. If
    the p-value that they are from the distribution is sufficiently small, then
    the step is retained; otherwise, the step is merged.

    Note that this function does not do any kind of global fit optimization. It
    merely iteratively merges plateaus until no unsatisfactory steps are left.
    The algorithm immediately merges an unsatisfactory step as soon as it finds
    one. It does not check whether this new merge produces a new unsatisfactory
    step or not. Every successive merge to remove an unsatisfactory step may
    generate a new one. It is hence quite possible to feed in a sequence of
    plateaus that will be merged until there is only one left.

    Arguments:
        luminosities: List of luminosities of a Spot through frames.
        plateaus: Plateaus to filter.
        p_threshold: Two plateaus whose luminosities' t-test p-value is greater
            than or equal to p_threshold are merged.
        drop_sort: Merge steps in the order of their mutual p-values, with
            pairs of plateaus with the largest mutual t-test p-value merged
            first.

    Returns:
        List of plateaus merged such that no two adjacent plateau fail the
        Welch t-test with p_threshold.
    """
    filtered_plateaus = plateaus
    for n in range(len(plateaus) - 1):
        filtered_plateaus = _t_test_filter_singlepass(
                                                 luminosities,
                                                 filtered_plateaus,
                                                 p_threshold,
                                                 drop_sort=drop_sort,
                                                 no_merge_start=no_merge_start)
    return filtered_plateaus


def stepfit_r_squared(luminosities, plateaus):
    """
    Coefficient of determination for a given step fit over luminosities.
    luminosities outside of the plateau sequence are ignored.
    """
    first_plateau, last_plateau = plateaus[0], plateaus[-1]
    first_start, first_stop, first_height = first_plateau
    last_start, last_stop, last_height = last_plateau
    mean_plateau = _fit_plateau(luminosities, first_start, last_stop)
    if len(plateaus) == 1:
        #This if statement is for debugging only.
        ss_res = float(_plateaus_squared_residuals(luminosities, plateaus))
        ss_tot = _plateau_squared_residuals(luminosities, mean_plateau)
        r_2 = \
            1.0 - (float(_plateaus_squared_residuals(luminosities, plateaus)) /
                   _plateau_squared_residuals(luminosities, mean_plateau))
        logger = logging.getLogger()
        logger.debug("stepfitting_library.stepfit_r_squared: " +
                     "len(plateaus) = 1; locals = " + str(locals()))
    return 1.0 - (float(_plateaus_squared_residuals(luminosities, plateaus)) /
                  _plateau_squared_residuals(luminosities, mean_plateau))


def linear_fits(luminosities, plateaus, midpoint_fits=True):
    """
    Perform linear fits on luminosities between all pairwise combinations of
    plateaus.

    This is a useful method when trying to determine whether a sequence of
    luminosities is more step-like or more gradient-like. Once a step-fit is
    performed, we can try interpolating the luminosities across plateaus with a
    line fit, and then evaluating whether the lines fit better than steps.

    To that end, this function provides the best line-fits between every
    possible pair of plateaus. It can optionally fit a line between the two
    plateaus' endpoints (the first plateau's first frame, and the last
    plateau's last frame), or between the midpoints of the plateaus. The
    function then evaluates the coefficient of determination R^2 over the
    luminosities between the pair of plateaus, for both the linear
    interpolation step-fit of the plateaus themselves.

    Arguments:
        luminosities: List of luminosities of a Spot through frames.
        plateaus: Plateau fit to work with.
        midpoint_fits: If True, fit using the midponits of the plateaus.
            Otherwise, use the opposite endpoints of each plateau pair.

    Returns:
        Dictionary as below:

        {
         (plateau_1, plateau_2): (linear_r_2, stepfit_r_2),
         (plateau_1, plateau_3): (linear_r_2, stepfit_r_2),

         ...

         (plateau_n-1, plateau_n): (linear_r_2, stepfit_r_2)
        }

        The dictionary keys are all pairwise combinations of plateaus, with the
        earlier plateau of the pair being placed first. The values are tuples
        storing the coefficients of determination for each type of fit.
    """
    logger = logging.getLogger()
    r_2 = {}
    indexed_plateaus = [(i, plateau) for i, plateau in enumerate(plateaus)]
    for ((ia, plateau_a),
         (ib, plateau_b)) in itertools.combinations(indexed_plateaus, 2):
        a_start, a_stop, a_height = plateau_a
        b_start, b_stop, b_height = plateau_b
        if midpoint_fits:
            a_midpoint = int(np.around((a_stop - a_start) / 2.0) + a_start)
            b_midpoint = int(np.around((b_stop - b_start) / 2.0) + b_start)
            linear_to_fit = \
               [(L, lum) for L, lum in enumerate(luminosities)][a_midpoint:
                                                                b_midpoint + 1]
            truncated_a = a_midpoint, a_stop, a_height
            truncated_b = b_start, b_midpoint, b_height
            step_to_fit = [truncated_a] + plateaus[ia + 1:ib] + [truncated_b]
        else:
            linear_to_fit = \
                    [(L, lum)
                     for L, lum in enumerate(luminosities)][a_start:b_stop + 1]
            step_to_fit = plateaus[ia:ib + 1]
        linear_to_fit_L, to_fit_lum = zip(*linear_to_fit)
        (slope, intercept, r_val, p_val,
         stderr) = linregress(linear_to_fit_L, to_fit_lum)
        linear_r_2 = r_val**2
        stepfit_r_2 = stepfit_r_squared(luminosities, step_to_fit)
        logger.debug("stepfitting_library.linear_fits: linear_r_2 = " +
                     str(linear_r_2) + "; stepfit_r_2 = " + str(stepfit_r_2))
        r_2.setdefault((ia, ib), (linear_r_2, stepfit_r_2))
    return r_2


def best_linear_explainer(r_2, steepest=True, longest=False,
                          r2_ratio_threshold=1.0, plateaus=None,
                          track_index=None):
    """
    Finds the pair of plateaus with the best linear fit, as returned by
    linear_fits.

    This function takes the input from linear_fits and finds the pair of
    plateaus between which there is the best linear explainer returned. The
    definition of "best" in this context is some function of the ratio of the
    linear fit R^2 to the step fit R^2. One variant of this defition is to find
    the longest linear fit that has the ratio linear_fit_r2:step_fit_r2 above a
    threshold. Another variant is to find the linear fit that has the highest
    ratio, regardless of length. This function offers both options.

    Arguments:
        r_2: Dictionary as returned by linear_fits.
        steepest: Return the linear fit in r_2 that has the highest
            linear_r_2/stepfit_r_2 ratio.
        longest: Return the longest linear fit that has its
            linear_r_2/stepfit_r_2 above r2_ratio_threshold.
        r2_ratio_threshold: Used when finding the longest linear fit.
        plateaus: Needed only in the case that debugging is turned on.
        track_index: Needed only in the case that debugging is turned on.

    Returns:
        (ia, ib, ratio)

        ia and ib are the indices of the two plateaus such that r_2[(ia, ib)]'s
        linear_r_2/stepfit_r_2 ratio satisfies either the the steepest or
        longest criteria. ratio = linear_r_2/stepfit_r_2.
    """
    if (steepest and longest) or (not steepest and not longest):
        raise ValueError("Must select either steepest or longest as criteria.")
    logger = logging.getLogger()
    largest_linear_explainer = (None, None, None) #ia, ib, ratio (None if NaN)
    for (ia, ib), (linear_r_2, stepfit_r_2) in r_2.iteritems():
        LLa, LLb, LLr = largest_linear_explainer
        assert ((LLa is None and LLb is None) or
                (LLa is not None and LLb is not None and LLa < LLb))
        assert ib > ia
        if stepfit_r_2 == 0:
            continue
            if LLr == None:
                if LLa is None:
                    largest_linear_explainer = ia, ib, None
            else:
                assert LLa is not None and LLb is not None
                assert LLb > LLa
                if LLb - LLa < ib - ia:
                    largest_linear_explainer = ia, ib, None
        else:
            explainer_ratio = float(linear_r_2) / stepfit_r_2
            if plateaus is not None:
                p_a, p_b = plateaus[ia], plateaus[ib]
            else:
                p_a, p_b = None, None
            logger.debug("stepfitting_library.largest_linear_explainer: " +
                         "track_index = " + str(track_index) +
                         "; plateau_a, plateau_b = " + str((p_a, p_b)) +
                         #"; ia, ib = " + str((ia, ib)) +
                         "; linear_r_2 = " + str(linear_r_2) +
                         "; stepfit_r_2 = " + str(stepfit_r_2) +
                         "; explainer_ratio = " + str(explainer_ratio))
            if LLa is None and explainer_ratio > r2_ratio_threshold:
                logger.debug("stepfitting_library.largest_linear_explainer: "
                             "case 1.")
                largest_linear_explainer = ia, ib, explainer_ratio
            elif LLr is None:
                logger.debug("stepfitting_library.largest_linear_explainer: "
                             "case 2.")
                continue
            elif (longest and
                  LLb - LLa < ib - ia and
                  explainer_ratio > r2_ratio_threshold):
                logger.debug("stepfitting_library.largest_linear_explainer: "
                             "case 3.")
                largest_linear_explainer = ia, ib, explainer_ratio
            elif steepest and explainer_ratio > LLr:
                logger.debug("stepfitting_library.largest_linear_explainer: "
                             "case 4.")
            elif LLb - LLa == ib - ia and explainer_ratio > LLr:
                logger.debug("stepfitting_library.largest_linear_explainer: "
                             "case 5.")
                largest_linear_explainer = ia, ib, explainer_ratio
    return largest_linear_explainer


def best_t_test_split(luminosities, plateau_a, plateau_b, p_threshold,
                      split_range=None, find_best_p=True):
    """
    Relocate step between plateau_a and plateau_b to maximize the length of
    plateau_a as long as the step's t-test p-value is above p_threshold.

    Does not return different plateaus if p value for any other split never
    goes below p_threshold.
    """
    raise DeprecationWarning("This was used as a function for some algorithm "
                             "we were trying. Not really needed right now.")
    a_start, a_stop, a_height = plateau_a
    b_start, b_stop, b_height = plateau_b
    #The iterating s is the b_start.
    if split_range is None:
        #using 2 as split_start because below that, scipy.stats.ttest_ind fails
        split_start, split_stop = 2, a_stop
    else:
        split_start, split_stop = split_range
        if split_start < 2:
            raise ValueError("split_range[0] cannot be less than 2; "
                             "scipy.stats.ttest_ind will yield NaN.")
    best_p, longest_s = 1, b_start
    for s in range(split_start, split_stop + 1):
        left_frames = luminosities[a_start:s]
        right_frames = luminosities[s:split_stop]
        t, p = ttest_ind(left_frames, right_frames, equal_var=False)
        if find_best_p and p < best_p:
            if p < best_p:
                best_p, longest_s = p, s
        elif p < p_threshold and s > longest_s:
            best_p, longest_s = p, s
    refitted_a = _fit_plateau(luminosities, a_start, longest_s - 1)
    refitted_b = _fit_plateau(luminosities, longest_s, b_stop)
    return refitted_a, refitted_b


def mirror_photometries(photometries, mirror_size):
    """
    Prepend photometies with a mirror image of the first mirror_size
    photometries.

    In those cases when steps occur in the very first few frames, filters such
    as Chung-Kennedy have a hard time capturing the step. In these cases, it is
    useful to mirror the first few frames to convert the first very short and
    assymetric plateau into a slightly longer and symmetric plateau. Once
    fitting is performed, the fitted plateaus can be truncated back into the
    original photometry sequence.
    """
    if mirror_size < 0:
        raise ValueError("mirror_size must be greater than 0.")
    return ([x for x in reversed(photometries[:mirror_size])] +
            list(photometries))


def unmirror_photometries(photometries, mirror_size):
    """Undo mirror_photometries."""
    if mirror_size < 0:
        raise ValueError("mirror_size must be greater than 0.")
    return photometries[mirror_size:]


def unmirror_plateaus(plateaus, mirror_size):
    """
    If plateaus are fitted on mirrored photometries as produced by
    mirror_photometries, this function truncates the fit to the original
    photometry sequence.
    """
    if mirror_size < 0:
        raise ValueError("mirror_size must be greater than 0.")
    unmirrored_plateaus = []
    shifted_plateaus = [(a - mirror_size, o - mirror_size, h)
                        for a, o, h in plateaus]
    for a, o, h in shifted_plateaus:
        if a < 0 and o < 0:
            continue
        elif a < 0 and o >= 0:
            unmirrored_plateaus.append((0, o, h))
        else:
            unmirrored_plateaus.append((a, o, h))
    return unmirrored_plateaus


def plateau_starts(plateaus):
    """Get set of frame indices corresponding to plateau starts."""
    return set([pa for pa, po, ph in plateaus])
