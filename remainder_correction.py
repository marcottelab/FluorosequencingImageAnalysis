#!/usr/bin/python


"""
Adjust track photometries based on persistent spots.
"""


import argparse
import csv
import cPickle
import os.path
import numpy as np
import MCsimlib


#define and parse arguments; use custom MyFormatter to do both ArgumentDefault
#and RawDescription Formatters via multiple inheritence, this is a trick to
#preserve docstring formatting in --help output
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
                  argparse.RawDescriptionHelpFormatter):
    pass
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=MyFormatter)
#Script arguments
tracks_helpstring = "track_photometries_??????.csv file to adjust."
parser.add_argument('tracks', nargs=1, type=str, help=tracks_helpstring)
minimum_remainders_per_field_helpstring = \
              ("Discard fields without at least this many remainders in them.")
parser.add_argument('--min', type=int, default=5,
                    help=minimum_remainders_per_field_helpstring)
diff_median_helpstring = \
    ("Method 1: Whether to use remainder track median instead of mean as "
     "benchmark.")
parser.add_argument('--M1_diff_median', action='store_true', default=False,
                    help=diff_median_helpstring)
print_adjustments_helpstring = "Print adjustments to screen."
parser.add_argument('--print_adjustments', action='store_true', default=False,
                    help=print_adjustments_helpstring)
save_adjustments_helpstring = "Save adjustments used to pkl file."
parser.add_argument('--save_adjustments', action='store_true', default=False,
                    help=save_adjustments_helpstring)
method_helpstring = \
    ("Which method to use. NOTE: Only method 4 available. Others are "
     "nonsense.")
parser.add_argument('--method', type=int, default=4, help=method_helpstring)
args = parser.parse_args()

csv_path = os.path.abspath(args.tracks[0])

if args.method != 4:
    raise Exception("Older methods not supported.")

photometries, row_photometries = \
                  MCsimlib.read_track_photometries_csv(csv_path,
                                                       head_truncate=0,
                                                       tail_truncate=0,
                                                       downstep_filtered=False)


def method_1(photometries, minimum, num_frames, use_median):
    remainder_diffs = {}

    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            remainder_diffs.setdefault(channel, {}).setdefault(field,
                                               [[] for f in range(num_frames)])
            for (h, w), (category, intensities, row) in fdict.iteritems():
                if set(category) != set([True]):
                    continue
                else:
                    if use_median:
                        remainder_m = np.median(intensities)
                    else:
                        remainder_m = np.mean(intensities)
                    diffs = [intensity - remainder_m
                             for intensity in intensities]
                    for frame, diff in enumerate(diffs):
                        remainder_diffs[channel][field][frame].append(diff)
    remainder_medians = {}
    for channel, cdict in remainder_diffs.iteritems():
        for field, diff_lists in cdict.iteritems():
            if any([len(diffs) < minimum
                    for frame, diffs in enumerate(diff_lists)]):
                continue
            remainder_medians.setdefault(channel, {}).setdefault(field,
                                    [np.median(diffs) for diffs in diff_lists])
    adjusted_photometries = {}
    for channel, cdict in remainder_medians.iteritems():
        adjusted_photometries.setdefault(channel, {})
        for field, medians in cdict.iteritems():
            adjusted_photometries[channel].setdefault(field, {})
            fdict = photometries[channel][field]
            for (h, w), (category, intensities, row) in fdict.iteritems():
                adjusted_intensities = [intensity - medians[frame]
                                for frame, intensity in enumerate(intensities)]
                adjusted_photometries[channel][field].setdefault((h, w),
                                              (category, adjusted_intensities))
    return adjusted_photometries, remainder_medians


def method_2(photometries, minimum, num_frames):
    remainder_values = {}
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                if set(category) != set([True]):
                    continue
                remainder_values.setdefault(channel, {}).setdefault(field,
                                               [[] for f in range(num_frames)])
                for frame, intensity in enumerate(intensities):
                    remainder_values[channel][field][frame].append(intensity)
    remainder_adjustments = {}
    for channel, cdict in remainder_values.iteritems():
        for field, remainder_lists in cdict.iteritems():
            if len(remainder_lists[0]) < minimum:
                continue
            remainder_medians = [np.median(remainder_list)
                                 for remainder_list in remainder_lists]
            adjustments = [median - remainder_medians[0]
                           for median in remainder_medians]
            remainder_adjustments.setdefault(channel, {}).setdefault(field,
                                                                   adjustments)
    adjusted_photometries = {}
    for channel, cdict in remainder_adjustments.iteritems():
        adjusted_photometries.setdefault(channel, {})
        for field, adjustments in cdict.iteritems():
            adjusted_photometries[channel].setdefault(field, {})
            fdict = photometries[channel][field]
            for (h, w), (category, intensities, row) in fdict.iteritems():
                adjusted_intensities = [intensity - adjustments[frame]
                                for frame, intensity in enumerate(intensities)]
                adjusted_photometries[channel][field].setdefault((h, w),
                                              (category, adjusted_intensities))
    return adjusted_photometries, remainder_adjustments

def method_3(photometries, minimum, num_frames):
    remainder_values = {}
    for channel, cdict in photometries.iteritems():
        for field, fdict in cdict.iteritems():
            for (h, w), (category, intensities, row) in fdict.iteritems():
                if set(category) != set([True]):
                    continue
                remainder_values.setdefault(channel, {}).setdefault(field,
                                               [[] for f in range(num_frames)])
                for frame, intensity in enumerate(intensities):
                    remainder_values[channel][field][frame].append(intensity)
    remainder_adjustments = {}
    for channel, cdict in remainder_values.iteritems():
        for field, remainder_lists in cdict.iteritems():
            if len(remainder_lists[0]) < minimum:
                continue
            remainder_medians = [np.median(remainder_list)
                                 for remainder_list in remainder_lists]
            adjustments = [remainder_medians[0] / float(median)
                           for median in remainder_medians]
            remainder_adjustments.setdefault(channel, {}).setdefault(field,
                                                                   adjustments)
    adjusted_photometries = {}
    for channel, cdict in remainder_adjustments.iteritems():
        adjusted_photometries.setdefault(channel, {})
        for field, adjustments in cdict.iteritems():
            adjusted_photometries[channel].setdefault(field, {})
            fdict = photometries[channel][field]
            for (h, w), (category, intensities, row) in fdict.iteritems():
                adjusted_intensities = [intensity * adjustments[frame]
                                for frame, intensity in enumerate(intensities)]
                adjusted_photometries[channel][field].setdefault((h, w),
                                              (category, adjusted_intensities))
    return adjusted_photometries, remainder_adjustments


num_frames = len(row_photometries.popitem()[1][4])
#Deleting row_photometries because we have modified it by popping. Want to
#avoid bugs caused by assumption that this dictionary still has everything.
del row_photometries
if args.method == 1:
    adjusted_photometries, remainder_adjustments = method_1(photometries,
                                     args.min, num_frames, args.M1_diff_median)
elif args.method == 2:
    adjusted_photometries, remainder_adjustments = method_2(photometries,
                                                          args.min, num_frames)
elif args.method == 3:
    adjusted_photometries, remainder_adjustments = method_3(photometries,
                                                          args.min, num_frames)
elif args.method == 4:
    adjusted_photometries, adjustment_ratio_medians = \
        MCsimlib._remainder_adjust_2(photometries=photometries,
                                     num_frames=num_frames,
                                     minimum_r_per_field=args.min)
    remainder_adjustments = adjustment_ratio_medians
else:
    raise ValueError("Unknown method.")
if args.print_adjustments:
    print(remainder_adjustments)
output_filepath = csv_path + '_adjusted.csv'
if args.save_adjustments:
    adjustments_output_filepath = csv_path + '_adjustments.pkl'
    cPickle.dump(remainder_adjustments, open(adjustments_output_filepath, 'w'))
csv_writer = csv.writer(open(output_filepath, 'w'))
header_row = (["CHANNEL", "FIELD", "H", "W", "CATEGORY"] +
              ["FRAME " + str(frame) for frame in range(num_frames)])
csv_writer.writerow(header_row)
for channel, cdict in adjusted_photometries.iteritems():
    for field, fdict in cdict.iteritems():
        for (h, w), (parsed_category, adjusted_intensities, row) in fdict.iteritems():
            row = [str(channel), str(field), str(h), str(w),
                   str(parsed_category)]
            row += [str(intensity) for intensity in adjusted_intensities]
            csv_writer.writerow(row)
