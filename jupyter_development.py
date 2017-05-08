import glob
from scipy.misc import imread
import numpy as np
import MCsimlib
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
#.............................................................................
from plotly.graph_objs import *
init_notebook_mode()
import os
from os.path import join
from math import sqrt, log
from datetime import datetime
from scipy.ndimage.filters import gaussian_filter1d
import math
from scipy.signal import find_peaks_cwt, argrelextrema
from itertools import combinations_with_replacement, product, izip, tee, chain, combinations, chain
from cPickle import load, dump
from scipy.stats import norm, pearsonr, probplot, find_repeats, linregress, mode, skew, kendalltau
from operator import mul
import multiprocessing
from random import sample, choice
from scipy.stats.mstats import gmean
import shelve
from ast import literal_eval
import gc
import colorlover as cl
from IPython.display import HTML
import csv
from sklearn.mixture import GMM, DPGMM
import plotly.tools
from scipy.optimize import nnls
from scipy.spatial.distance import euclidean, cdist, canberra
import sys
from collections import namedtuple, defaultdict




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
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def grab_ON_OFFS(all_fit_info, allow_bad_fits=False, alpha_adjust=None):
    on_offs = {} #(cycle, field): [(iON, # fluors dropped)]
    for (channel, field, h, w, row,
         category, intensities,
         signal, is_zero,
         dye_sequence, lmii, total_score, per_frame_scores, starting_intensity) in all_fit_info:
        if not allow_bad_fits and dye_sequence is None:
            continue
        for i, (iON, iOFF) in enumerate(_pairwise(intensities)):
            if category[i] and not category[i + 1]:
                if not allow_bad_fits:
                    if alpha_adjust is not None:
                        on_offs.setdefault((i, field), []).append((iON, dye_sequence[i] - dye_sequence[i + 1]))
                    else:
                        on_offs.setdefault((i, field), []).append((iON - alpha_adjust, dye_sequence[i] - dye_sequence[i + 1]))
                else:
                    if alpha_adjust is not None:
                        on_offs.setdefault((i, field), []).append((iON - alpha_adjust, None))
                    else:
                        on_offs.setdefault((i, field), []).append((iON, None))
    on_offs = {(cycle, field): tuple(drops) for (cycle, field), drops in on_offs.iteritems()}
    return on_offs

def grab_ith_intensities(all_fit_info, i=1, grab_signal=None,
                         allow_nonzero=False, log_xform=True,
                         alpha_adjust=None, grab_category=None,
                         grab_well_sequenced=None, grab_poorly_sequenced=None,
                         grab_last_on=None):
    i -= 1
    intensities_by_field = {} #field: [first intensities]
    for (channel, field, h, w, row,
         category, intensities,
         signal, is_zero,
         dye_sequence, lmii, total_score, per_frame_scores, starting_intensity) in all_fit_info:
        if grab_signal is not None and (signal is None or grab_signal != signal):
            continue
        if grab_category is not None and category != grab_category:
            continue
        if not allow_nonzero and (is_zero is None or not is_zero):
            continue
        if grab_well_sequenced is not None and grab_well_sequenced and signal is None:
            continue
        if grab_poorly_sequenced is not None and grab_poorly_sequenced and signal is not None:
            continue
        if (grab_last_on is not None and grab_last_on and
            (i == len(intensities) - 1 or not (category[i] and not category[i + 1]))):
            continue
        target_intensity = intensities[i]
        if alpha_adjust is not None:
            target_intensity -= alpha_adjust
        if log_xform and target_intensity <= 0:
            continue
        target_intensity = log(target_intensity) if log_xform else target_intensity
        intensities_by_field.setdefault(field, []).append(target_intensity)
    intensities_by_field = {field: tuple(target_intensities)
                            for field, target_intensities
                            in intensities_by_field.iteritems()}
    return intensities_by_field

def generate_intensities(fluorosequence, beta, beta_sigma, number, quench_factors=None):
    if quench_factors is None:
        quench_factors = [0.0] * len(fluorosequence)
    category = tuple([False if seq == 0 else True for seq in fluorosequence])
    intensities = [np.random.lognormal(mean=log(beta) + log(seq) - quench_factors[seq - 1],
                                       sigma=beta_sigma,
                                       size=number)
                   if seq > 0
                   else [0.0] * number
                   for seq in fluorosequence]
    return category, tuple(zip(*intensities))

def fast_mode(array):
    array = np.asarray(array)
    values, counts = find_repeats(array)
    if len(counts) == 0:
        array.sort() # mimic first value behavior
        return array[0], 1.
    else:
        position = counts.argmax()
        return values[position], counts[position]
    
def grab_ith_jth_intensities(all_fit_info, i=1, j=5, grab_signal=None,
                             allow_nonzero=False, log_xform=True,
                             alpha_adjust=None, norm_scoring=None):
    i -= 1
    j -= 1
    intensity_pairs_by_field = {} #field: [(ith intensity, jth intensity)]
    for (channel, field, h, w, row,
         category, intensities,
         signal, is_zero,
         dye_sequence, lmii, total_score, per_frame_scores, starting_intensity) in all_fit_info:
        if signal is not None and signal != grab_signal:
            continue
        if not is_zero and not allow_nonzero:
            continue
        target_intensity_i, target_intensity_j = intensities[i], intensities[j]
        if alpha_adjust is not None:
            target_intensity_i -= alpha_adjust
            target_intensity_j -= alpha_adjust
        intensity_i = log(target_intensity_i) if log_xform else target_intensity_i
        intensity_j = log(target_intensity_j) if log_xform else target_intensity_j
        if norm_scoring is not None:
            mean_i, std_i, mean_j, std_j = norm_scoring
            intensity_i = float(intensity_i - mean_i) / std_i
            intensity_j = float(intensity_j - mean_j) / std_j
        intensity_pairs_by_field.setdefault(field, []).append((intensity_i, intensity_j))
    intensity_pairs_by_field = {field: tuple(intensity_pairs)
                                for field, intensity_pairs
                                in intensity_pairs_by_field.iteritems()}
    return intensity_pairs_by_field

def gmm_raw_photometries(raw_photometries):
    nested = [[p] for p in raw_photometries]
    g = GMM(n_components=1, n_init=100, n_iter=100, covariance_type='full')
    g.fit(nested)
    mean = float(g.means_[0])
    std = float(sqrt(g.covars_[0]))
    return g, mean, std

def qq(sample1, sample2, num_quantiles=101):
    sorted_sample1, sorted_sample2 = sorted(sample1), sorted(sample2)
    quantiles = np.linspace(0, 100, num_quantiles)
    qq_pairs = [(np.percentile(sorted_sample1, q), np.percentile(sorted_sample2, q))
                for q in quantiles]
    return tuple(qq_pairs)

def signal_to_sequence(signal, num_frames, starting_intensity=None):
    if starting_intensity is None:
        intensity = len(signal)
    else:
        intensity = starting_intensity
    drop_positions = set([pos for aa, pos in signal])
    drop_counts = {pos: len([p for aa, p in signal if p == pos])
                   for pos in list(drop_positions)}
    seq = []
    for frame in range(num_frames):
        if frame in drop_positions:
            intensity -= drop_counts[frame]
        seq.append(intensity)
    return tuple(seq)

def sequence_to_signal(seq):
    signal_TFn = [seq[f] - fc for f, fc in enumerate(seq[1:])]
    signal = []
    for i, tf in enumerate(signal_TFn):
        if tf > 0:
            signal += [('A', i + 1)] * tf
        elif tf < 0:
            signal = None
            break
    return tuple(signal)

def r_squared(data, fit):
    data, fit = np.array(data), np.array(fit)
    res = float(sum(np.reshape((data - fit)**2, -1)))
    tot = float(sum((np.reshape(data, -1) - np.mean(data))**2))
    return 1.0 - res / tot

def sequence_to_category(seq):
    return tuple([True if s > 0 else False for s in seq])

def make_histx(bins):
    return [np.mean([x1, x2]) for x1, x2 in _pairwise(tuple(bins))]

def split_heatmap(num_cycles, cycle):
    all_SD_signals = [(('A', c),) for c in range(1, num_cycles + 1)]
    all_DD_signals = [(('A', b), ('A', c))
                      for c in range(1, num_cycles + 1)
                      for b in range(1, c)]
    before_cycle = (
                    [(((aa, c),), True, 1)
                     for ((aa, c),) in all_SD_signals if c < cycle]
                    +
                    [(((aa1, b), (aa2, c)), True, 2)
                       for ((aa1, b), (aa2, c)) in all_DD_signals
                       if c < cycle]
                   )
    after_cycle = (
                    [(((aa, c),), True, 1)
                     for ((aa, c),) in all_SD_signals if c >= cycle]
                    +
                    [(((aa1, b), (aa2, c)), True, 2)
                       for ((aa1, b), (aa2, c)) in all_DD_signals
                       if c >= cycle]
                   )
    return tuple(before_cycle), tuple(after_cycle)

def unwind_photometries(photometries):
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                yield (channel, field, h, w, category, intensities, row)
                
def generate_sequences(max_possible, num_cycles, num_samples, category):
    return tuple(zip(*[[choice(range(1, max_possible + 1)) for x in range(num_samples)]
                       if category[cycle]
                       else [0] * num_samples
                       for cycle in range(num_cycles)]))

def ON_OFF_adjust_photometries(photometries, ON_OFFS, alpha):
    adjusted_photometries = {}
    last_beta_dict = {(cycle, field): np.median([iON for iON, ddiff in drops])
                      for (cycle, field), drops in ON_OFFS.iteritems()}
    last_beta_median = float(np.median(last_beta_dict.values()))
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                adjusted_intensities = [(intensity - alpha) * last_beta_median / last_beta_dict[(i, field)]
                                        if i < len(intensities) - 1 and (i, field) in last_beta_dict
                                        else intensity
                                        for i, intensity in enumerate(intensities)]
                adjusted_photometries.setdefault(channel, {}).setdefault(field, {}).setdefault((h, w),
                                                            (category, tuple(adjusted_intensities), row))
    return adjusted_photometries


def signal_correlation(observed_signals, fit_signals,
                       heatmap_only=True, zero_only=True,
                       metric='naive', normalize_counts=False,
                       matching_p=0.10, exclude_signals=None,
                       print_included_signals=False,
                       select_signals=None, heatmap_normalize_counts=False,
                       allow_multidrop=False, small_count_cutoff=None,
                       euclidean_weights=None,
                      ):
    paired_signal_counts = []
    for (s, z, si), observed_count in observed_signals.iteritems():
        if select_signals is not None and (s, z, si) not in select_signals:
            continue
        if zero_only and not z:
            continue
        if heatmap_only and len(s) not in (1, 2):
            continue
        if not allow_multidrop and len(set(s)) < len(s):
            continue
        if exclude_signals is not None and (s, z, si) in exclude_signals:
            continue
        fit_count = fit_signals.get((s, z, si), 0)
        if print_included_signals:
            print("Including signal " + str((s, z, si)))
        paired_signal_counts.append((observed_count, fit_count, (s, z, si)))
    #Take care of signals that are in fit_signals but not in observed_signals
    for (s, z, si), fit_count in fit_signals.iteritems():
        if (s, z, si) in observed_signals:
            continue
        if select_signals is not None and (s, z, si) not in select_signals:
            continue
        if zero_only and not z:
            continue
        if heatmap_only and len(s) not in (1, 2):
            continue
        if not allow_multidrop and len(set(s)) < len(s):
            continue
        if exclude_signals is not None and (s, z, si) in exclude_signals:
            continue
        observed_count = observed_signals.get((s, z, si), 0)
        if print_included_signals:
            print("Including signal " + str((s, z, si)))
        paired_signal_counts.append((observed_count, fit_count, (s, z, si)))
    if small_count_cutoff is not None:
        paired_signal_counts = [(observed_count, fit_count, (s, z, si))
                                for observed_count, fit_count, (s, z, si) in paired_signal_counts
                                if (observed_count >= small_count_cutoff
                                    and fit_count >= small_count_cutoff)]
    observed_counts = np.array([observed_count
                                for observed_count, fit_count, (s, z, si) in paired_signal_counts])
    fit_counts = np.array([fit_count
                           for observed_count, fit_count, (s, z, si) in paired_signal_counts])
    if normalize_counts and len(paired_signal_counts) > 0 and np.sum(fit_counts) > 0:
        normalization_factor = float(np.sum(observed_counts)) / np.sum(fit_counts)
    elif heatmap_normalize_counts:
        observed_heatmap_total, fit_heatmap_total = 0, 0
        for (s, z, si), observed_count in observed_signals.iteritems():
            if not z:
                continue
            if len(s) not in (1, 2):
                continue
            if len(set(s)) < len(s):
                continue
            observed_heatmap_total += observed_count
            fit_count = fit_signals.get((s, z, si), 0)
            fit_heatmap_total += fit_count
        #Take care of signals that are in fit_signals but not in observed_signals
        for (s, z, si), fit_count in fit_signals.iteritems():
            if (s, z, si) in observed_signals:
                continue
            if not z:
                continue
            if len(s) not in (1, 2):
                continue
            if len(set(s)) < len(s):
                continue
            fit_heatmap_total += fit_count
        normalization_factor = float(observed_heatmap_total) / float(fit_heatmap_total)
    else:
        normalization_factor = 1.0
    #Fit counts are normalized for metrics that use observed_counts and fit_counts directly
    fit_counts = fit_counts * float(normalization_factor)
    #paired_signal_counts are normalized for those metrics that use this array instead
    paired_signal_counts = [(observed_count, fit_count * float(normalization_factor), (s, z, si))
                            for observed_count, fit_count, (s, z, si) in paired_signal_counts]
    contributions = {}
    if len(paired_signal_counts) == 0:
        result = None
    elif metric == 'naive':
        contributions = {(s, z, si): observed_count * fit_count
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        result = sum(contributions.values())
    elif metric == 'pearson':
        raise NotImplementedError()
        result = pearsonr(observed_counts, fit_counts)[0]
    elif metric == 'euclidean':
        raise NotImplementedError()
        result = euclidean(observed_counts, fit_counts)
    elif metric == 'chebyshev':
        raise NotImplementedError()
        observed_counts = np.array([[x for x, y, s in paired_signal_counts]])
        fit_counts = np.array([[y for x, y, s in paired_signal_counts]])
        result = cdist(observed_counts, fit_counts, metric='chebyshev')
    elif metric == 'my_chebyshev':
        contributions = {(s, z, si): abs(observed_count - fit_count)
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        result = np.amax(contributions.values())
    elif metric == 'my_normalized_chebyshev':
        contributions = {(s, z, si): abs(observed_count - fit_count) / float(observed_count)
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts
                         if observed_count > 0}
        result = np.amax(contributions.values())
    elif metric == 'my_std_normalized_chebyshev':
        n = sum([observed_count
                 for (s, z, si), observed_count in observed_signals.iteritems()
                 if ((not zero_only or (zero_only and z))
                     and
                     (allow_multidrop or (not allow_multidrop and len(set(s)) == len(s))))])
        stds = {(s, z, si): sqrt(observed_count * (n - observed_count) / float(n))
                if observed_count > 0 else 1
                for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        contributions = {(s, z, si): abs(observed_count - fit_count) / float(stds[(s, z, si)])
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        result = np.amax(contributions.values())
    elif metric == 'canberra':
        raise NotImplementedError()
        observed_counts = np.array([[x for x, y, s in paired_signal_counts]])
        fit_counts = np.array([[y for x, y, s in paired_signal_counts]])
        result = cdist(observed_counts, fit_counts, metric='canberra')
    elif metric == 'matching':
        if matching_p is None:
            raise ValueError("If matching, matching_p cannot be None")
        contributions = {(s, z, si): True
                         if abs(observed_count - fit_count) / float(observed_count) <= matching_p
                         else False
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        result = sum([1 for m in contributions.itervalues() if m])
    elif metric == 'matching_10p':
        matching = [True
                    if abs(fit_counts[i] - v) / float(v) <= 0.10
                    else False
                    for i, v in enumerate(observed_counts)]
        result = sum([1 for m in matching if m])
    elif metric == 'kendalltau':
        raise NotImplementedError()
        pairs_by_observed = sorted(paired_signal_counts, key=lambda x:x[0])
        ranked_by_observed = [(i + 1, f) for i, (o, f, s) in enumerate(pairs_by_observed)]
        pairs_by_fit = sorted(ranked_by_observed, key=lambda x:x[1])
        ranked_by_fit = [(ro, i + 1) for i, (ro, f) in enumerate(pairs_by_fit)]
        fit_ranks = [x for x, y in ranked_by_fit]
        observed_ranks = [y for x, y in ranked_by_fit]
        result = kendalltau(observed_ranks, fit_ranks)[0]
    elif metric == 'my_euclidean':
        contributions = {(s, z, si): (fit_count - observed_count)**2
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        result = sqrt(sum(contributions.values()))
    elif metric == 'scipy_canberra':
        result = canberra(observed_counts, fit_counts)[0]
    elif metric == 'normalized_euclidean':
        contributions = {(s, z, si): (float(fit_count - observed_count)
                                      / observed_count)**2
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts
                         if observed_count > 0}
        result = sqrt(sum(contributions.values()))
    elif metric == 'my_std_normalized_euclidean':
        n = sum([observed_count
                 for (s, z, si), observed_count in observed_signals.iteritems()
                 if ((not zero_only or (zero_only and z))
                     and
                     (allow_multidrop or (not allow_multidrop and len(set(s)) == len(s))))])
        stds = {(s, z, si): sqrt(observed_count * (n - observed_count) / float(n))
                if observed_count > 0 else 1
                for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        contributions = {(s, z, si): (float(fit_count - observed_count) / stds[(s, z, si)])**2
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        result = sqrt(sum(contributions.values()))
    elif metric == 'my_sim_std_normalized_euclidean':
        #n = sum([fit_count
        #         for (s, z, si), observed_count in fit_signals.iteritems()
        #         if ((not zero_only or (zero_only and z))
        #             and
        #             (allow_multidrop or (not allow_multidrop and len(set(s)) == len(s))))])
        n = sum(fit_signals.itervalues())
        stds = {(s, z, si): sqrt(fit_count * (n - fit_count) / float(n))
                if fit_count > 0 else 1
                for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        contributions = {(s, z, si): (float(fit_count - observed_count) / stds[(s, z, si)])**2
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        result = sqrt(sum(contributions.values()))
    elif metric == 'my_weighted_std_normalized_euclidean':
        if euclidean_weights is None:
            raise ValueError("my_weighted_std_normalized_euclidean "
                             "requires euclidean_weights.")
        n = sum([observed_count
                 for (s, z, si), observed_count in observed_signals.iteritems()
                 if ((not zero_only or (zero_only and z))
                     and
                     (allow_multidrop or (not allow_multidrop and len(set(s)) == len(s))))])
        stds = {(s, z, si): sqrt(observed_count * (n - observed_count) / float(n))
                if observed_count > 0 else 1
                for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        filled_euclidean_weights = {(s, z, si): weight
                                    for (s, z, si), weight
                                    in euclidean_weights.iteritems()}
        for observed_count, fit_count, (s, z, si) in paired_signal_counts:
            if (s, z, si) not in filled_euclidean_weights:
                filled_euclidean_weights.setdefault((s, z, si), 0)
        contributions = {(s, z, si): (float(fit_count - observed_count)
                                      * filled_euclidean_weights[(s, z, si)]
                                      / stds[(s, z, si)])**2
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        result = sqrt(sum(contributions.values()))
    elif metric == 'log_rmsd':
        contributions = {(s, z, si): float(log(observed_count + 1) - log(fit_count + 1))**2
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts}
                         #if observed_count > 0 and fit_count > 0}
        if len(contributions) > 0:
            result = sqrt(sum(contributions.values()) / float(len(contributions)))
        else:
            result = None
    elif metric == 'my_canberra':
        contributions = {(s, z, si): float(abs(observed_count - fit_count))
                                     / (abs(observed_count) + abs(fit_count))
                         for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        result = sum(contributions.values())
    elif metric == 'my_pearson':
        diffs = {(s, z, si): (observed_count - fit_count, observed_count, fit_count)
                 for observed_count, fit_count, (s, z, si) in paired_signal_counts}
        observed_sigma = np.std([o for d, o, f in diffs.values()])
        fit_sigma = np.std([f for d, o, f in diffs.values()])
        observed_mean = np.mean([o for d, o, f in diffs.values()])
        fit_mean = np.mean([f for d, o, f in diffs.values()])
        contributions = {(s, z, si): (o - observed_mean) * (f - fit_mean)
                         for (s, z, si), (d, o, f) in diffs.iteritems()}
        n = len(contributions)
        result = sum(contributions.values()) / float(fit_sigma * observed_sigma * n)
    elif metric == 'my_kendalltau':
        contributions = {}
        for i, (observed_count_i, fit_count_i, (s_i, z_i, si_i)) in enumerate(paired_signal_counts):
            for j, (observed_count_j, fit_count_j, (s_j, z_j, si_j)) in enumerate(paired_signal_counts):
                #if j >= i:
                #    break
                #if i == 0:
                #    break
                if i == j:
                    continue
                diff_observed_count = observed_count_i - observed_count_j
                diff_fit_count = fit_count_i - fit_count_j
                if diff_observed_count == 0 or diff_fit_count == 0:
                    continue
                if diff_observed_count < 0:
                    observed_count_sign = -1
                else:
                    observed_count_sign = 1
                if diff_fit_count < 0:
                    fit_count_sign = -1
                else:
                    fit_count_sign = 1
                contributions.setdefault((s_i, z_i, si_i), 0)
                contributions[(s_i, z_i, si_i)] += observed_count_sign * fit_count_sign
                contributions.setdefault((s_j, z_j, si_j), 0)
                contributions[(s_j, z_j, si_j)] += observed_count_sign * fit_count_sign
        numerator = sum(contributions.values())
        denominator = len(paired_signal_counts) * (len(paired_signal_counts) - 1) / 2.0
        contributions_compensation = 4.0
        denominator *= contributions_compensation
        if denominator != 0:
            result =  numerator / denominator
        else:
            result = None
    elif metric == 'my_spearman_rho':
        observed_counts_rank = sorted([(i, observed_count_i, (s_i, z_i, si_i))
                                       for i, (observed_count_i, fit_count_i, (s_i, z_i, si_i))
                                       in enumerate(paired_signal_counts)],
                                      key=lambda x:x[1])
        fit_counts_rank = sorted([(i, fit_count_i, (s_i, z_i, si_i))
                                  for i, (observed_count_i, fit_count_i, (s_i, z_i, si_i))
                                  in enumerate(paired_signal_counts)],
                                 key=lambda x:x[1])
        observed_counts_mean_rank = fit_counts_mean_rank = (len(fit_counts_rank) - 1) / 2.0
        observed_counts_deltas = {(s_i, z_i, si_i): j - observed_counts_mean_rank
                                  for j, (i, observed_count_i, (s_i, z_i, si_i))
                                  in enumerate(observed_counts_rank)}
        fit_counts_deltas = {(s_i, z_i, si_i): j - fit_counts_mean_rank
                             for j, (i, fit_count_i, (s_i, z_i, si_i))
                             in enumerate(fit_counts_rank)}
        contributions = {(s_i, z_i, si_i): observed_delta * fit_counts_deltas[(s_i, z_i, si_i)]
                         for (s_i, z_i, si_i), observed_delta
                         in observed_counts_deltas.iteritems()}
        numerator = sum(contributions.values())
        observed_sum_of_squares = sum([v**2 for v in observed_counts_deltas.itervalues()])
        fit_sum_of_squares = sum([v**2 for v in fit_counts_deltas.itervalues()])
        denominator = sqrt(observed_sum_of_squares * fit_sum_of_squares)
        if denominator != 0:
            result = numerator / denominator
        else:
            result = None
    else:
        raise ValueError("Invalid metric chosen.")
    return result, (normalization_factor, contributions)




colors = {405:'GnBu', 488:'YIOrRd', 561:'YIOrRd', 647:'YIGnBu'}

def single_drops_heatmap_v2(signals, num_mocks, num_edmans, num_mocks_omitted,
                            peptide_string, wavelength, zmin, zmax, filepath,
                            plot_multidrops=False,
                            plot_remainders=False, transparent=True,
                            float_data=False, return_components=False):
    num_mocks -= num_mocks_omitted
    total_cycles = num_mocks + num_edmans
    if plot_remainders:
        heatmap_array_size = total_cycles + 1
    else:
        heatmap_array_size = total_cycles
    if float_data:
        heatmap_array = np.array([[0.0 for y in range(heatmap_array_size)]])
    else:
        heatmap_array = np.array([[0 for y in range(heatmap_array_size)]])
    for (signal, is_zero, starting_intensity), count in signals.iteritems():
        if starting_intensity > 1:
            continue
        if len(signal) != 1:
            continue
        if signal == (('A', 0),):
            if not plot_remainders:
                continue
            if is_zero:
                continue
            else:
                x, y = 0, heatmap_array_size - 1
        else:
            if not is_zero:
                continue
            else:
                x, y = 0, signal[0][1] - 1
        assert heatmap_array[x, y] == 0
        if float_data:
            count = round(float(count), 2)
        heatmap_array[x, y] = count
    color_channel = wavelength
    if color_channel not in colors:
        raise("Exception: Invalid wavelength.")
    cs = colors[color_channel] #cs = "color space"
    cycles_header = (["M" + str(i + 1 + num_mocks_omitted)
                      for i in range(num_mocks)] +
                     ["E" + str(i + 1) for i in range(num_edmans)] +
                     ["R"])
    annotations = []
    text_limit = np.amax(heatmap_array)
    for (y, x), count in np.ndenumerate(heatmap_array):
        annotations.append(dict(text=str(count),
                                x=cycles_header[x],
                                y="C",
                                font=dict(color=('white'
                                                 if count > (text_limit * 0.75)
                                                 else 'black')),
                                showarrow=False))
    layout = graph_objs.Layout(title=("Single Drops (" + str(color_channel) +
                                      " Channel) Total: " +
                                      str(np.sum(heatmap_array)) + " - " +
                                      peptide_string),
                               annotations=annotations,
                               titlefont=dict(size=16),
                               yaxis=dict(title="",
                                          titlefont=dict(size=14),
                                          ticks="",
                                          autorange='reversed'),
                               xaxis=dict(title="Drop Position",
                                          titlefont=dict(size=16),
                                          ticks="",
                                          side='top'),
                               margin=graph_objs.Margin(l=50, r=50, b=100,
                                                        t=150, pad=2),
                               width=700,
                               height=325,
                               autosize=False)
    if transparent:
        layout['plot_bgcolor'], layout['paper_bgcolor'] = 'rgba(0,0,0,0)', 'rgba(0,0,0,0)'
    data = [graph_objs.Heatmap(z=heatmap_array,
                               x=cycles_header,
                               y=["C", ""],
                               colorscale=cs,
                               reversescale=True,
                               zmin=(np.amin(heatmap_array)
                                     if zmin is None else zmin),
                               zmax=(np.amax(heatmap_array)
                                     if zmax is None else zmax))]
    if return_components:
        return (annotations, layout, data)
    fig = graph_objs.Figure(data=data, layout=layout)
    iplot(fig)
    
def double_drops_heatmap_v2(signals, num_mocks, num_edmans, num_mocks_omitted,
                            peptide_string, wavelength, zmin, zmax, filepath,
                            plot_multidrops=False, plot_remainders=True,
                            transparent=False, float_data=False, return_components=False):
    num_mocks -= num_mocks_omitted
    total_cycles = num_mocks + num_edmans
    if plot_remainders:
        heatmap_array_size_x = total_cycles
        heatmap_array_size_y = total_cycles + 1
    else:
        heatmap_array_size_y = heatmap_array_size_x = total_cycles
    if float_data:
        heatmap_array = np.array([[0.0 for y in range(heatmap_array_size_y)]
                                  for x in range(heatmap_array_size_x)])
    else:
        heatmap_array = np.array([[0 for y in range(heatmap_array_size_y)]
                                  for x in range(heatmap_array_size_x)])
    for (signal, is_zero, starting_intensity), count in signals.iteritems():
        if starting_intensity > 2:
            continue
        if len(signal) == 1:
            if signal == (('A', 0),):
                continue
            elif plot_remainders and not is_zero:
                x, y = signal[0][1] - 1, heatmap_array_size_y - 1
            else:
                continue
        elif len(signal) == 2:
            if not plot_multidrops and len(signal) > len(set(signal)):
                continue
            elif is_zero:
                x, y = signal[0][1] - 1, signal[1][1] - 1
            else:
                continue
        elif len(signal) > 2:
            continue
        assert heatmap_array[x, y] == 0
        if float_data:
            count = round(float(count), 2)
        heatmap_array[x, y] = count
    color_channel = wavelength
    if color_channel not in colors:
        raise("Exception: Invalid wavelength.")
    cs = colors[color_channel] #cs = "color space"
    y_cycles_header = (["M" + str(i + 1 + num_mocks_omitted)
                        for i in range(num_mocks)] +
                       ["E" + str(i + 1) for i in range(num_edmans)])
    if plot_remainders:
        x_cycles_header = y_cycles_header + ["R"]
    else:
        x_cycles_header = y_cycles_header
    annotations = []
    text_limit = np.amax(heatmap_array)
    for (y, x), count in np.ndenumerate(heatmap_array):
        annotations.append(dict(text=str(count),
                                x=x_cycles_header[x],
                                y=y_cycles_header[y],
                                font=dict(color=('white'
                                                 if count > (text_limit * 0.75)
                                                 else 'black')),
                                showarrow=False))
    layout = graph_objs.Layout(title=("Double Drops (" + str(color_channel) +
                                      " Channel) Total: " +
                                      str(np.sum(heatmap_array)) + " - " +
                                      peptide_string),
                               annotations=annotations,
                               titlefont=dict(size=16),
                               yaxis=dict(title="First Drop",
                                          titlefont=dict(size=16),
                                          ticks="",
                                          autorange='reversed'),
                               xaxis=dict(title="Second Drop",
                                          titlefont=dict(size=16),
                                          ticks="",
                                          side='top'),
                               margin=graph_objs.Margin(l=50, r=50, b=100,
                                                        t=150, pad=4),
                               width=700,
                               height=735,
                               autosize=False)
    if transparent:
        layout['plot_bgcolor'], layout['paper_bgcolor'] = 'rgba(0,0,0,0)', 'rgba(0,0,0,0)'
    data = [graph_objs.Heatmap(z=heatmap_array,
                               x=x_cycles_header,
                               y=y_cycles_header,
                               colorscale=cs,
                               reversescale=True,
                               zmin=(np.amin(heatmap_array)
                                     if zmin is None else zmin),
                               zmax=(np.amax(heatmap_array)
                                     if zmax is None else zmax))]
    if return_components:
        return (annotations, layout, data)
    fig = graph_objs.Figure(data=data, layout=layout)
    iplot(fig)

IncompatibilityKey = namedtuple('IncompatibilityKey',
                                [
                                 'metric',
                                 'reverse_order',
                                 'normalize_counts',
                                 'heatmap_normalize_counts',
                                 'heatmap_only',
                                 'zero_only',
                                 'allow_multidrop',
                                 'small_count_cutoff',
                                 'matching_p',
                                 'split_cycle',
                                 'incompatibility_threshold',
                                 'compute_incompatibility_scores',
                                ])

def match_diagnostic(
                     all_simulations,
                     observed_signals,
                     metric,
                     reverse_order,
                     normalize_counts,
                     heatmap_normalize_counts,
                     heatmap_only,
                     zero_only,
                     allow_multidrop,
                     small_count_cutoff,
                     matching_p,
                     split_cycle,
                     incompatibility_threshold,
                     compute_incompatibility_scores,
                     num_mocks,
                     num_mocks_omitted,
                     num_edmans,
                    ):
    num_cycles = num_mocks + num_mocks_omitted - num_edmans
    if normalize_counts == heatmap_normalize_counts:
        raise ValueError("normalize_counts == heatmap_normalize_counts")
    if heatmap_only:
        if not heatmap_normalize_counts or allow_multidrop:
            raise ValueError("If heatmap_only, then "
                             "heatmap_normalize_counts "
                             "and not allow_multidrop")
    if incompatibility_threshold is not None and not compute_incompatibility_scores:
        raise ValueError("If incompatibility_threshold is not None, "
                         "then compute_incompatibility_scores")
    incompatibility_key_tuple = IncompatibilityKey(
                                                   metric=metric,
                                                   reverse_order=reverse_order,
                                                   normalize_counts=normalize_counts,
                                                   heatmap_normalize_counts=heatmap_normalize_counts,
                                                   heatmap_only=heatmap_only,
                                                   zero_only=zero_only,
                                                   allow_multidrop=allow_multidrop,
                                                   small_count_cutoff=small_count_cutoff,
                                                   matching_p=matching_p,
                                                   split_cycle=split_cycle,
                                                   incompatibility_threshold=incompatibility_threshold,
                                                   compute_incompatibility_scores=
                                                                      compute_incompatibility_scores,
                                                  )
    print("incompatibility_key_tuple = " + str(incompatibility_key_tuple))
    incompatibility_key = str(incompatibility_key_tuple)
    should_be_empty, all_cycles = split_heatmap(num_cycles=num_cycles, cycle=0)
    num_combinations = len(list(combinations(all_cycles, 2)))
    if compute_incompatibility_scores and (incompatibility_key_tuple not in incompatibility_scores_cache):
        print("Starting incompatibility computation at " + str(datetime.now()))
        select_signal_distances = {}
        for i, (ss1, ss2) in enumerate(combinations(all_cycles, 2)):
            select_signals = set((ss1, ss2))
            assert (ss1, ss2) not in select_signal_distances
            assert (ss2, ss1) not in select_signal_distances
            if i % 100 == 0:
                print("Combination #" + str(i + 1) + " of " + str(num_combinations)
                      + " at " + str(datetime.now()))
            all_correlations = {(p, b, u): signal_correlation(
                                                              observed_signals=ADJ_SDL_signals,
                                                              fit_signals=signals,
                                                              heatmap_only=heatmap_only,
                                                              zero_only=zero_only,
                                                              normalize_counts=normalize_counts,
                                                              metric=metric,
                                                              exclude_signals=None,
                                                              matching_p=matching_p,
                                                              select_signals=select_signals,
                                                              print_included_signals=False,
                                                              heatmap_normalize_counts=
                                                                    heatmap_normalize_counts,
                                                              small_count_cutoff=small_count_cutoff,
                                                             )
                                for (p, b, u), (signals, molecular_signals) in all_simulations.iteritems()}
            optimal_pbu = None
            normalization_factor = None
            for i, ((p, b, u), (result, (nf, contrib))) in enumerate(sorted(all_correlations.iteritems(),
                                                                            key=lambda x:x[1][0],
                                                                            reverse=reverse_order)):
                if i == 0:
                    optimal_pbu = (p, b, u)
                    normalization_factor = nf
                    break        
            contributions = all_correlations[optimal_pbu][1][1]
            distances = contributions.get(ss1, None), contributions.get(ss2, None)
            select_signal_result = (optimal_pbu, distances, normalization_factor)
            #print(str((ss1, ss2)) + " " + str(select_signal_result) + " at " + str(datetime.now()))
            select_signal_distances.setdefault((ss1, ss2), select_signal_result)
        incompatibilities = {}
        for (ss1, ss2), (opbu, (d1, d2), nf) in select_signal_distances.iteritems():
            incompatibilities.setdefault(ss1, []).append(d1)
            incompatibilities.setdefault(ss2, []).append(d2)
        if reverse_order:
            max_incompatibilities = {(s, z, si): np.amin(incompatibilities_list)
                                     for (s, z, si), incompatibilities_list
                                     in incompatibilities.iteritems()}
        else:
            max_incompatibilities = {(s, z, si): np.amax(incompatibilities_list)
                                     for (s, z, si), incompatibilities_list
                                     in incompatibilities.iteritems()}
        max_incompatibilities = {(s, z, si): count
                                 for (s, z, si), count in max_incompatibilities.iteritems()
                                 if count is not None}
        print("Incompatibility computation done at " + str(datetime.now()))
        incompatibility_scores_cache.setdefault(incompatibility_key_tuple, max_incompatibilities)
        incompatibility_scores_shelf = shelve.open(incompatibility_scores_filename, 'c')
        assert incompatibility_key not in incompatibility_scores_shelf
        incompatibility_scores_shelf[incompatibility_key] = (incompatibility_key_tuple,
                                                             max_incompatibilities)
        incompatibility_scores_shelf.close()
    else:
        print("incompatibility_key is in incompatibility_scores_cache; no need to recompute.")

    #If not requesting incompatibility_scores, display all 0s
    if compute_incompatibility_scores:
        incompatibility_scores = incompatibility_scores_cache[incompatibility_key_tuple]
    else:
        incompatibility_scores = {}

    #Build set of excluded signals
    if incompatibility_threshold is not None:
        exclude_by_incompatibility = set([(s, z, si)
                                          for (s, z, si), mi in incompatibility_scores.iteritems()
                                          if mi > incompatibility_threshold])
    else:
        exclude_by_incompatibility = set()
 

    before_cycle, after_cycle = split_heatmap(num_cycles=num_cycles, cycle=split_cycle)
    exclude_signals = exclude_by_incompatibility | set(before_cycle)
    print("exclude_signals = " + str(exclude_signals) + "\n")


    #Find best-fitting simulation and prepare for plotting
    all_correlations = {(p, b, u): signal_correlation(
                                                      observed_signals=observed_signals,
                                                      fit_signals=signals,
                                                      heatmap_only=heatmap_only,
                                                      zero_only=zero_only,
                                                      normalize_counts=normalize_counts,
                                                              metric=metric,
                                                      exclude_signals=None,
                                                      matching_p=matching_p,
                                                      select_signals=None,
                                                      print_included_signals=False,
                                                      heatmap_normalize_counts=
                                                                    heatmap_normalize_counts,
                                                      small_count_cutoff=small_count_cutoff,
                                                     )
                        for (p, b, u), (signals, molecular_signals) in all_simulations.iteritems()}
    
    optimal_pbu = None
    normalization_factor = None
    optimal_contributions = None
    for i, ((p, b, u), (result, (nf, contrib))) in enumerate(sorted(all_correlations.iteritems(),
                                                                    key=lambda x:x[1][0],
                                                                    reverse=reverse_order)):
        if i == 0:
            optimal_pbu = (p, b, u)
            normalization_factor = nf
            optimal_contributions = contrib
            break
    plot_signals, plot_molecular_signals = all_simulations[optimal_pbu]
    normalized_plot_signals = {(s, z, si): int(round(count * normalization_factor))
                               for (s, z, si), count in plot_signals.iteritems()}
    normalized_plot_molecular_signals = {(s, z, si): int(round(count * normalization_factor))
                                         for (s, z, si), count in plot_molecular_signals.iteritems()}

    #Find %diff
    diff_plot_signals = {(s, z, si): float(observed_count - normalized_plot_signals[(s, z, si)])
                                     / observed_count
                         for (s, z, si), observed_count in observed_signals.iteritems()
                         if (s, z, si) in normalized_plot_signals and observed_count > 0}


    #Plots
    #(SINGLE DROP) [Data] [Fit] [%diff]
    #(DOUBLE DROP) [Data] [Fit] [%diff]
    #(SINGLE DROP) [Cntrb] [Mol] [incompatibility]
    #(DOUBLE DROP) [Cntrb] [Mol] [incompatibility]
    fig = plotly.tools.make_subplots(rows=4, cols=3,
                                     subplot_titles=('Data', 'Fit', '% Diff',
                                                     '', '', '',
                                                     'Contributions', 'Dye counts', 'Incompatibility',
                                                     '', '', ''),
                                     shared_yaxes=False,
                                     shared_xaxes=False)


    single_drop_counts, double_drop_counts = [], []
    for (s, z, si), count in chain(observed_signals.iteritems(),
                                   normalized_plot_signals.iteritems(),
                                   normalized_plot_molecular_signals.iteritems()):
        if not z:
            continue
        if len(s) > len(set(s)):
            continue
        if len(s) == 1:
            assert si == 1, "si != 1"
            single_drop_counts.append(count)
        elif len(s) == 2:
            assert si == 2
            double_drop_counts.append(count)
        else:
            assert si > 2, "si <= 2"

    single_drop_z_min, single_drop_z_max = 0, max(single_drop_counts)
    double_drop_z_min, double_drop_z_max = 0, max(double_drop_counts)

    incompatibility_scores_z_min, incompatibility_scores_z_max = (min(0,
                                                                      min(incompatibility_scores.values()
                                                                          + optimal_contributions.values())
                                                                     ),
                                                                  max(incompatibility_scores.values()
                                                                      + optimal_contributions.values()))

    diff_scores_z_min, diff_scores_z_max = min(diff_plot_signals.values()), max(diff_plot_signals.values())

    (annotations1, layout1, data1) = single_drops_heatmap_v2(signals=observed_signals,
                                                             num_mocks=num_mocks,
                                                             num_edmans=num_edmans,
                                                             num_mocks_omitted=num_mocks_omitted,
                                                             peptide_string="Observed",
                                                             wavelength=647,
                                                             zmin=single_drop_z_min,
                                                             zmax=single_drop_z_max,
                                                             filepath=None,
                                                             float_data=False,
                                                             plot_remainders=True, transparent=True,
                                                             return_components=True)
    for a, ann in enumerate(annotations1):
        annotations1[a]['xref'], annotations1[a]['yref'] = 'x1', 'y1'

    (annotations2, layout2, data2) = single_drops_heatmap_v2(signals=normalized_plot_signals,
                                                             num_mocks=num_mocks,
                                                             num_edmans=num_edmans,
                                                             num_mocks_omitted=num_mocks_omitted,
                                                             peptide_string="Observed",
                                                             wavelength=647,
                                                             zmin=single_drop_z_min,
                                                             zmax=single_drop_z_max,
                                                             filepath=None,
                                                             float_data=False,
                                                             plot_remainders=True, transparent=True,
                                                             return_components=True)
    for a, ann in enumerate(annotations2):
        annotations2[a]['xref'], annotations2[a]['yref'] = 'x2', 'y2'
    
    (annotations3, layout3, data3) = single_drops_heatmap_v2(signals=diff_plot_signals,
                                                             num_mocks=num_mocks,
                                                             num_edmans=num_edmans,
                                                             num_mocks_omitted=num_mocks_omitted,
                                                             peptide_string="Observed",
                                                             wavelength=647,
                                                             zmin=diff_scores_z_min,
                                                             zmax=diff_scores_z_max,
                                                             filepath=None,
                                                             float_data=True,
                                                             plot_remainders=True, transparent=True,
                                                             return_components=True)
    for a, ann in enumerate(annotations3):
        annotations3[a]['xref'], annotations3[a]['yref'] = 'x3', 'y3'
    
    
    (annotations4, layout4, data4) = double_drops_heatmap_v2(signals=observed_signals,
                                                             num_mocks=num_mocks,
                                                             num_edmans=num_edmans,
                                                             num_mocks_omitted=num_mocks_omitted,
                                                             peptide_string="Observed",
                                                             wavelength=647,
                                                             zmin=double_drop_z_min,
                                                             zmax=double_drop_z_max,
                                                             filepath=None,
                                                             float_data=False,
                                                             plot_remainders=True, transparent=True,
                                                             return_components=True)
    for a, ann in enumerate(annotations4):
        annotations4[a]['xref'], annotations4[a]['yref'] = 'x4', 'y4'
    
    
    (annotations5, layout5, data5) = double_drops_heatmap_v2(signals=normalized_plot_signals,
                                                             num_mocks=num_mocks,
                                                             num_edmans=num_edmans,
                                                             num_mocks_omitted=num_mocks_omitted,
                                                             peptide_string="Observed",
                                                             wavelength=647,
                                                             zmin=double_drop_z_min,
                                                             zmax=double_drop_z_max,
                                                             filepath=None,
                                                             float_data=False,
                                                             plot_remainders=True, transparent=True,
                                                             return_components=True)
    for a, ann in enumerate(annotations5):
        annotations5[a]['xref'], annotations5[a]['yref'] = 'x5', 'y5'
    
    
    (annotations6, layout6, data6) = double_drops_heatmap_v2(signals=diff_plot_signals,
                                                             num_mocks=num_mocks,
                                                             num_edmans=num_edmans,
                                                             num_mocks_omitted=num_mocks_omitted,
                                                             peptide_string="Observed",
                                                             wavelength=647,
                                                             zmin=diff_scores_z_min,
                                                             zmax=diff_scores_z_max,
                                                             filepath=None,
                                                             float_data=True,
                                                             plot_remainders=True, transparent=True,
                                                             return_components=True)
    for a, ann in enumerate(annotations6):
        annotations6[a]['xref'], annotations6[a]['yref'] = 'x6', 'y6'
    

    (annotations7, layout7, data7) = single_drops_heatmap_v2(signals=optimal_contributions,
                                                             num_mocks=num_mocks,
                                                             num_edmans=num_edmans,
                                                             num_mocks_omitted=num_mocks_omitted,
                                                             peptide_string="Observed",
                                                             wavelength=647,
                                                             zmin=incompatibility_scores_z_min,
                                                             zmax=incompatibility_scores_z_max,
                                                             filepath=None,
                                                             float_data=True,
                                                             plot_remainders=True, transparent=True,
                                                             return_components=True)
    for a, ann in enumerate(annotations7):
        annotations7[a]['xref'], annotations7[a]['yref'] = 'x7', 'y7'
    
    
    (annotations8, layout8, data8) = single_drops_heatmap_v2(signals=normalized_plot_molecular_signals,
                                                             num_mocks=num_mocks,
                                                             num_edmans=num_edmans,
                                                             num_mocks_omitted=num_mocks_omitted,
                                                             peptide_string="Observed",
                                                             wavelength=647,
                                                             zmin=single_drop_z_min,
                                                             zmax=single_drop_z_max,
                                                             filepath=None,
                                                             float_data=False,
                                                             plot_remainders=True, transparent=True,
                                                             return_components=True)
    for a, ann in enumerate(annotations8):
        annotations8[a]['xref'], annotations8[a]['yref'] = 'x8', 'y8'
    
    (annotations9, layout9, data9) = single_drops_heatmap_v2(signals=incompatibility_scores,
                                                             num_mocks=num_mocks,
                                                             num_edmans=num_edmans,
                                                             num_mocks_omitted=num_mocks_omitted,
                                                             peptide_string="Observed",
                                                             wavelength=647,
                                                             zmin=incompatibility_scores_z_min,
                                                             zmax=incompatibility_scores_z_max,
                                                             filepath=None,
                                                             float_data=True,
                                                             plot_remainders=True, transparent=True,
                                                             return_components=True)
    for a, ann in enumerate(annotations9):
        annotations9[a]['xref'], annotations9[a]['yref'] = 'x9', 'y9'
    
    
    (annotations10, layout10, data10) = double_drops_heatmap_v2(signals=optimal_contributions,
                                                                num_mocks=num_mocks,
                                                                num_edmans=num_edmans,
                                                                num_mocks_omitted=num_mocks_omitted,
                                                                peptide_string="Observed",
                                                                wavelength=647,
                                                                zmin=incompatibility_scores_z_min,
                                                                zmax=incompatibility_scores_z_max,
                                                                filepath=None,
                                                                float_data=True,
                                                                plot_remainders=True, transparent=True,
                                                                return_components=True)
    for a, ann in enumerate(annotations10):
        annotations10[a]['xref'], annotations10[a]['yref'] = 'x10', 'y10'
    
    
    (annotations11, layout11, data11) = double_drops_heatmap_v2(signals=normalized_plot_molecular_signals,
                                                                num_mocks=num_mocks,
                                                                num_edmans=num_edmans,
                                                                num_mocks_omitted=num_mocks_omitted,
                                                                peptide_string="Observed",
                                                                wavelength=647,
                                                                zmin=double_drop_z_min,
                                                                zmax=double_drop_z_max,
                                                                filepath=None,
                                                                float_data=False,
                                                                plot_remainders=True, transparent=True,
                                                                return_components=True)
    for a, ann in enumerate(annotations11):
        annotations11[a]['xref'], annotations11[a]['yref'] = 'x11', 'y11'
    
    (annotations12, layout12, data12) = double_drops_heatmap_v2(signals=incompatibility_scores,
                                                                num_mocks=num_mocks,
                                                                num_edmans=num_edmans,
                                                                num_mocks_omitted=num_mocks_omitted,
                                                                peptide_string="Observed",
                                                                wavelength=647,
                                                                zmin=incompatibility_scores_z_min,
                                                                zmax=incompatibility_scores_z_max,
                                                                filepath=None,
                                                                float_data=True,
                                                                plot_remainders=True, transparent=True,
                                                                return_components=True)
    for a, ann in enumerate(annotations12):
        annotations12[a]['xref'], annotations12[a]['yref'] = 'x12', 'y12'


    data1[0]['showscale'] = False
    data2[0]['showscale'] = False
    data3[0]['showscale'] = False
    data4[0]['showscale'] = False
    data5[0]['showscale'] = False
    data6[0]['showscale'] = False
    data7[0]['showscale'] = False
    data8[0]['showscale'] = False
    data9[0]['showscale'] = False
    data10[0]['showscale'] = False
    data11[0]['showscale'] = False
    data12[0]['showscale'] = False
    fig.append_trace(data1[0], 1, 1)
    fig.append_trace(data2[0], 1, 2)
    fig.append_trace(data3[0], 1, 3)
    fig.append_trace(data4[0], 2, 1)
    fig.append_trace(data5[0], 2, 2)
    fig.append_trace(data6[0], 2, 3)
    fig.append_trace(data7[0], 3, 1)
    fig.append_trace(data8[0], 3, 2)
    fig.append_trace(data9[0], 3, 3)
    fig.append_trace(data10[0], 4, 1)
    fig.append_trace(data11[0], 4, 2)
    fig.append_trace(data12[0], 4, 3)
    #fig['layout']['annotations'] = (annotations1
    #                                + annotations2
    #                                + annotations3
    #                                + annotations4
    #                                + annotations5
    #                                + annotations6
    #                                + annotations7
    #                                + annotations8
    #                                + annotations9
    #                                + annotations10
    #                                + annotations11
    #                                + annotations12)
    #fig['layout']['annotations'] = []
    fig['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
    fig['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
    fig['layout'].update(height=1000,
                         title="Data vs Simulation: Distance Metric Diagnostics, "
                               + str(metric) + "; (p, b, u) = " + str(optimal_pbu),
                         showlegend=False,
                        )
    fig['layout']['yaxis1']['domain'] = [0.91, 0.95]
    fig['layout']['yaxis2']['domain'] = [0.91, 0.95]
    fig['layout']['yaxis3']['domain'] = [0.91, 0.95]
    fig['layout']['yaxis4']['domain'] = [0.55, 0.80]
    fig['layout']['yaxis5']['domain'] = [0.55, 0.80]
    fig['layout']['yaxis6']['domain'] = [0.55, 0.80]
    fig['layout']['yaxis7']['domain'] = [0.36, 0.40]
    fig['layout']['yaxis8']['domain'] = [0.36, 0.40]
    fig['layout']['yaxis9']['domain'] = [0.36, 0.40]
    fig['layout']['yaxis10']['domain'] = [0.0, 0.25]
    fig['layout']['yaxis11']['domain'] = [0.0, 0.25]
    fig['layout']['yaxis12']['domain'] = [0.0, 0.25]
    fig['layout']['yaxis4']['autorange'] = 'reversed'
    fig['layout']['yaxis5']['autorange'] = 'reversed'
    fig['layout']['yaxis6']['autorange'] = 'reversed'
    fig['layout']['yaxis10']['autorange'] = 'reversed'
    fig['layout']['yaxis11']['autorange'] = 'reversed'
    fig['layout']['yaxis12']['autorange'] = 'reversed'


    iplot(fig)
    
    return normalized_plot_signals, optimal_contributions, incompatibility_scores


def fasta_to_dict(fasta_file, include_fragments=False):
    f = open(fasta_file)
    proteome_to_seq = {}
    frag_proteome_to_seq = {}
    label, sequence = None, None
    fragment_flag = False
    label_counter = 0
    for line in f:
        if line[0] == '>':
            if label is not None:
                if sequence is None:
                    raise Exception()
                if fragment_flag:
                    frag_proteome_to_seq.setdefault(label, sequence)
                else:
                    proteome_to_seq.setdefault(label, sequence)
                label, sequence = None, None
                if "Fragment" in line or line[:3] == ">tr":
                    fragment_flag = True
                else:
                    fragment_flag = False
            label = line.split('|')[1]
            label_counter += 1
        else:
            if sequence is None:
                sequence = line[:-1]
            else:
                sequence += line[:-1]
    if label is None or sequence is None:
        raise Exception()
    else:
        proteome_to_seq.setdefault(label, sequence)
    assert len(proteome_to_seq) + len(frag_proteome_to_seq) == label_counter
    assert set(proteome_to_seq.keys()).isdisjoint(
                                              set(frag_proteome_to_seq.keys()))
    f.close()
    if include_fragments:
        proteome_to_seq.update(frag_proteome_to_seq)
    return proteome_to_seq

def sig(peptides, acid='C'):
    signature = []
    for head, tail in peptides:
        if acid in head:
            s = head.split(acid)
            if s[-1] == acid:
                sigt = tuple([len(c) + 1 for c in s])
            else:
                sigt = tuple([len(c) + 1 for c in s][:-1])
            signature.append(sigt)
    return set(signature), signature

def signal_to_cumulative(signal):
    cs = [s + sum(signal[:i]) for i, s in enumerate(signal)]
    return tuple(cs)


def diff_signals(boc_signals, ac_signals, zero_only=True,
                 allow_multidrop=False, filter_negatives=True,
                 max_baseline_method=False, percent_change=False):
    filtered_boc_signals = {(s, z, si): count
                            for (s, z, si), count in boc_signals.iteritems()
                            if not (zero_only and not z)
                            and not (not allow_multidrop and len(s) < len(set(s)))}
    filtered_ac_signals = {(s, z, si): count
                           for (s, z, si), count in ac_signals.iteritems()
                           if not (zero_only and not z)
                           and not (not allow_multidrop and len(s) < len(set(s)))}
    if max_baseline_method:
        boc_to_ac_ratios = defaultdict(float)
        for (s, z, si), ac_count in filtered_ac_signals.iteritems():
            assert ac_count > 0
            boc_count = filtered_boc_signals.get((s, z, si), 0)
            boc_to_ac_ratio = float(boc_count) / ac_count
            boc_to_ac_ratios[(s, z, si)] = boc_to_ac_ratio
        lowest_boc_to_ac_ratio = min(boc_to_ac_ratios.itervalues())
        normalization_ratio = lowest_boc_to_ac_ratio
    else:
        total_filtered_boc = sum(filtered_boc_signals.values())
        total_filtered_ac = sum(filtered_ac_signals.values())
        normalization_ratio = float(total_filtered_boc) / total_filtered_ac
    diff = defaultdict(int)
    all_filtered_keys = chain(filtered_boc_signals.iterkeys(), filtered_ac_signals.iterkeys())
    for (s, z, si) in all_filtered_keys:
        boc_count = filtered_boc_signals.get((s, z, si), 0)
        ac_count = filtered_ac_signals.get((s, z, si), 0)
        diff[(s, z, si)] = int(round(boc_count - ac_count * normalization_ratio))
    if filter_negatives:
        diff = {(s, z, si): count for (s, z, si), count in diff.iteritems() if count > 0}
    if percent_change:
        pc = {}
        for (s, z, si), count in diff.iteritems():
            boc_count = filtered_boc_signals.get((s, z, si), 0)
            if boc_count != 0:
                pc.setdefault((s, z, si), float(count) / boc_count)
        diff = pc
    return diff
