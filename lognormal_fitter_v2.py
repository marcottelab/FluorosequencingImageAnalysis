#!/usr/bin/python


"""
Fit track photometries using the lognormal algorithm.
"""


import argparse
from sys import argv
import sys
sys.path.insert(0, '/home/proteanseq/pflib')
from time import time
from collections import defaultdict
from csv import writer
from os.path import abspath
from cPickle import dump
from MCsimlib import (
                      read_track_photometries_csv,
                      _get_m0Dm1,
                      _photometries_lognormal_fit_MP_v8,
                      last_drop_method_v2,
                     )
from pflib import _epoch_to_hash
from plotting import (
                      plot_histogram,
                      single_drops_heatmap_v2,
                      double_drops_heatmap_v2,
                     )
import jupyter_development as jd
from collections import defaultdict


#define and parse arguments; use custom MyFormatter to do both ArgumentDefault
#and RawDescription Formatters via multiple inheritence, this is a trick to
#preserve docstring formatting in --help output
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
                  argparse.RawDescriptionHelpFormatter):
    pass
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=MyFormatter)
tracks_helpstring = "track_photometries_??????.csv file to fit."
parser.add_argument('tracks', nargs=1, type=str, help=tracks_helpstring)
channel_helpstring = "Which channel to fit. Must be either 1 or 2."
parser.add_argument('-c', '--channel', type=int, default=1,
                    help=channel_helpstring)
wavelength_helpstring = \
    ("Specify the wavelength of the channel. Used for color-coding heatmaps. "
     "If 0, will map channel 1 to 561nm and channel 2 to 647nm. Only valid "
     "options are 405, 488, 561, 647.")
parser.add_argument('-w', '--wavelength', type=int, default=0,
                    help=wavelength_helpstring)
num_mocks_helpstring = "Number of mocks performed. Used for heatmaps."
parser.add_argument('-m', '--num_mocks', type=int, default=4,
                    help=num_mocks_helpstring)
num_omitted_mocks_helpstring = "Number of mocks not imaged. Used for heatmaps."
parser.add_argument('-o', '--num_mocks_omitted', type=int, default=1,
                    help=num_mocks_helpstring)
num_edmans_helpstring = "Number of Edmans performed. Used for heatmaps."
parser.add_argument('-e', '--num_edmans', type=int, default=8,
                    help=num_edmans_helpstring)
peptide_label_helpstring = "Peptide sequence as string. Used for heatmaps."
parser.add_argument('-s', '--sequence', type=str, default=None,
                    help=peptide_label_helpstring)
num_processors_helpstring = "Maximum number of processors to use."
parser.add_argument('-n', '--num_processors', type=int, default=None,
                    help=num_processors_helpstring)
max_possible_helpstring = \
     ("Maximum number of fluors to try to fit. Although the lognormal fitter "
      "has a good way to guess how many fluors are on a peptide and hence "
      "usually doesn't care if this number is high, theoretically this "
      "increases as a power law with the number of fluors, so nasty datasets "
      "may actually allow the program to run for a long time.")
parser.add_argument('--max_possible', type=int, default=5,
                    help=max_possible_helpstring)
max_deviation_helpstring = \
    ("Maximum standard deviations away from mean for fitting an intensity to "
     "a fluor.")
parser.add_argument('--max_deviation', type=int, default=3,
                    help=max_deviation_helpstring)
ddif_factor_helpstring = "Dye-dye interaction factor."
parser.add_argument('--ddif', type=float, default=0.30,
                    help=ddif_factor_helpstring)
beta_sigma_helpstring = "Lognormal shape parameter."
parser.add_argument('--beta_sigma', type=float, default=0.20,
                    help=beta_sigma_helpstring)
beta_helpstring = "Manually specify 1-fluor intensity"
parser.add_argument('--beta', type=float, default=None,
                    help=beta_helpstring)
no_adjustment_helpstring = \
    "Do not perform ON->OFF based per-image photometry adjustment."
parser.add_argument('--no_adjustment', action='store_true', default=False,
                    help=no_adjustment_helpstring)
no_multidrop_helpstring = "No drops greater than one dye allowed during fit."
parser.add_argument('--no_multidrop', action='store_true', default=False,
                    help=no_multidrop_helpstring)
onoff_truncation_helpstring = (
    "Ignore this number of cycles at the beginning when trying to guess the "
    "one fluor intensity."
                              )
parser.add_argument('--truncate', type=int, default=0,
                    help=onoff_truncation_helpstring)
args = parser.parse_args()

tracks_filepath = abspath(args.tracks[0])

if args.channel == 1:
    channel = 'ch1'
elif args.channel == 2:
    channel = 'ch2'

timestamp_epoch = round(time())
timestamp_hash = _epoch_to_hash(timestamp_epoch)
output_filepath_base = \
         tracks_filepath + "_" + str(timestamp_hash) + "_" + str(channel) + "_"

print("Using timestamp_hash " + str(timestamp_hash))

commandline_pkl_filepath = output_filepath_base + 'COMMANDLINE.pkl'
dump(argv, open(commandline_pkl_filepath, 'w'))

photometries, row_photometries = \
                            read_track_photometries_csv(
                                                        tracks_filepath,
                                                        head_truncate=0,
                                                        tail_truncate=0,
                                                        downstep_filtered=True,
                                                        channels=[channel],
                                                       )

raw_photometries = tuple([intensity
                          for channel, field, h, w, category, intensities, row
                              in jd.unwind_photometries(photometries)
                          for intensity in intensities])

alpha = _get_m0Dm1(raw_photometries=raw_photometries,
                   optimal_bin_number=None)[7]

alpha_adjusted_photometries = defaultdict(dict)
for (channel, field, h, w,
     category, intensities, row) in jd.unwind_photometries(photometries):
    alpha_adjusted_intensities = tuple([intensity - alpha
                                        for intensity in intensities])
    (alpha_adjusted_photometries.setdefault(channel, {})
     .setdefault(field, {})
     .setdefault((h, w), (category, alpha_adjusted_intensities, row)))

truncated_alpha_adjusted_photometries = defaultdict(dict)
for (
     channel,
     field,
     h, w,
     category,
     intensities,
     row,
    ) in jd.unwind_photometries(photometries):
    truncated_category = category[args.truncate:]
    truncated_intensities = intensities[args.truncate:]
    (truncated_alpha_adjusted_photometries.setdefault(channel, {})
     .setdefault(field, {})
     .setdefault((h, w), (truncated_category, truncated_intensities, row)))

original_beta, original_beta_sigma = \
        last_drop_method_v2(photometries=truncated_alpha_adjusted_photometries)

if args.beta is not None:
    original_beta = args.beta

allow_multidrop = not args.no_multidrop

ddif = tuple([0.0] + [args.ddif] * (args.max_possible + 1))
original_plf_results = \
         (original_signals,
          original_total_count,
          original_none_count,
          original_all_fit_info) = \
    _photometries_lognormal_fit_MP_v8(photometries=alpha_adjusted_photometries,
                                      beta=original_beta,
                                      beta_sigma=args.beta_sigma,
                                      max_possible=args.max_possible,
                                      allow_upsteps=False,
                                      allow_multidrop=allow_multidrop,
                                      max_deviation=3,
                                      quench_factor=0,
                                      quench_factors=ddif)

on_offs = jd.grab_ON_OFFS(original_all_fit_info, alpha_adjust=0)


if not args.no_adjustment:
    adj_photometries = jd.ON_OFF_adjust_photometries(photometries=photometries,
                                                     ON_OFFS=on_offs,
                                                     alpha=alpha)
else:
    adj_photometries = alpha_adjusted_photometries

adj_beta, adj_beta_sigma = last_drop_method_v2(photometries=adj_photometries)

if args.beta is not None:
    adj_beta = args.beta

plf_results = \
         (signals,
          total_count,
          none_count,
          all_fit_info) = \
             _photometries_lognormal_fit_MP_v8(photometries=adj_photometries,
                                               beta=adj_beta,
                                               beta_sigma=args.beta_sigma,
                                               max_possible=args.max_possible,
                                               allow_upsteps=False,
                                               allow_multidrop=allow_multidrop,
                                               max_deviation=3,
                                               quench_factor=0,
                                               quench_factors=ddif)

pkl_all_filepath = output_filepath_base + 'INTERMEDIATES_v2.pkl'
dump(((alpha, adj_beta, args.beta_sigma, ddif), plf_results, args),
     open(pkl_all_filepath, 'w'))

csv_output_filepath = output_filepath_base + 'CLUSTERED.csv'
csv_file = open(csv_output_filepath, 'w')
csv_writer = writer(csv_file)

csv_file.close()

pkl_output_filepath = output_filepath_base + 'SIGNALS.pkl'
dump(signals, open(pkl_output_filepath, 'w'))

print("")
print("Signals:")
for (signal, is_zero,
     s_i), count in sorted(signals.items(), key=lambda x:x[0]):
    print(str((signal, is_zero, s_i)) + "    " + str(count))
print("Total number of signals: " + str(sum(signals.values())))
print("Total number of signals that fall to 0: " +
      str(sum([count for (s, z, si), count in signals.iteritems() if z])))
print("")

rp_pkl_output_filepath = output_filepath_base + 'RAW_PHOTOMETRIES.pkl'
dump(raw_photometries, open(rp_pkl_output_filepath, 'w'))

try:
    histogram_filepath = output_filepath_base + 'HISTOGRAM.html'
    plot_histogram(plot_target=raw_photometries,
                   title="Spot intensity log histogram",
                   yaxis_title="log(counts)", xaxis_title="photometry",
                   log_yaxis=True, filepath=histogram_filepath)
except Exception as e:
    print("Error saving histogram using plotting.py functions. Exception: " +
          str(e))

try:
    single_drops_filepath = output_filepath_base + 'SINGLE_DROPS_HEATMAP.html'
    single_drops_heatmap_v2(signals=signals, num_mocks=args.num_mocks,
                            num_edmans=args.num_edmans,
                            num_mocks_omitted=args.num_mocks_omitted,
                            peptide_string=args.sequence,
                            wavelength=args.wavelength, zmin=None,
                            zmax=None,
                            filepath=single_drops_filepath,
                            plot_remainders=True)
except Exception as e:
    print("Error saving single drops heatmap using plotting.py functions."
          " Exception: " + str(e))

try:
    double_drops_filepath = output_filepath_base + 'DOUBLE_DROPS_HEATMAP.html'
    double_drops_heatmap_v2(signals=signals, num_mocks=args.num_mocks,
                            num_edmans=args.num_edmans,
                            num_mocks_omitted=args.num_mocks_omitted,
                            peptide_string=args.sequence,
                            wavelength=args.wavelength, zmin=None,
                            zmax=None,
                            filepath=double_drops_filepath,
                            plot_remainders=True)
except Exception as e:
    print("Error saving double drops heatmap using plotting.py functions."
          " Exception: " + str(e))
