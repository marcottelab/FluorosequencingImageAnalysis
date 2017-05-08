#!/home/boulgakov/anaconda2/bin/python

"""
Find the background in a sequencing experiment using averaged ac- datasets.


Most input and output files are pickeled signals dictionaries, as indicated
below. For these dictionaries, keys are (signal, is_zero, starting_intensity).
Example key: ((('A', 2), ('A', 5)), False, 3). Values are either counts at that
position, or some function of counts -- e.g. percent, average, standard
deviation, etc -- as specified below.

The individual ac- datasets averaged into the background are read from the
SIGNALS.pkl dictionaries produced by lognormal_fitter_v2. Their values are raw
(integer) counts of each observed signal. The user provides a csv file
specifying their location and experimental information in the following format:

Index    Filepath             Mocks    Edmans    Mocks omitted   Description
-----    -----------------    -----    ------    -------------   --------------
0        /dir1/signals.pkl    2        8         1               "PEG ac(2, 5)"
1        /dir2/signals.pkl    2        8         1               "AS ac(4, 7)"
2        /dir3/signals.pkl    4        8         0               "PEG ac(2, 5)"

An example csv file referencing actual data is provided as an example. "Index"
and "Filepath" are required entries used by the script to collate and access
the data. The other entries are for documentation purposes only, and may be
left blank and/or contain arbitrary information.

It is recommended that as ac- data is accumulated, it is added to a shared csv
file or sorted into separate files by number of mocks/Edmans/surface etc. For
each background correction run, the ac- data to use can be specified by passing
its index via the commandline as described in the options.

The number of mocks and Edman cycles performed in each ac- and boc- experiment
may vary. To background correct a boc- experiment using ac- experiments done
with a different number of cycles, it is necessary to instruct the program how
to reconcile these datasets. The background correction algorithm is agnostic
about whether particular cycles are mock or Edman: as long as the total number
of cycles in the boc- dataset and all ac- datasets matches, the correction will
be performed. Essentially, the boc- and/or ac- heatmaps need to be cropped such
that they are all identical in size. Two operations are available (and can be
applied to both boc- and ac- datasets in combination, if needed) to do this:
1. truncating a dataset's first few cycles and 2. Truncating a dataset to a
maximum number of cycles.

Truncating a dataset's first few cycles:
It is possible that either the boc- or ac- experiments have more mock cycles
than the other. It is possible to discard the first few cycles of the boc-
experiment, the first few cycles of the ac- experiments, or both. After
discarding, the first remaining cycle is renamed as cycle 1, etc.

Truncating a dataset to a maximum number of cycles:
Remove all signals that have one or more drops after some cycle. If performed
together with a head truncation, this operation is applied last.

Note that if the above two operations are applied to the ac- datasets, they are
uniformly applied to all ac- datasets at once, regardless of how many cycles
each ac- dataset has. For example, if two cycles are truncated from the
beginning of ac- datasets, one with 8 cycles and the other with 12 cycles, then
the number of cycles remaining in each ac- dataset will be 6 and 10,
respectively. In this scenario, performing an additonal crop to 6 cycles total
may resolve the mismatch. Otherwise, if an average is desired from ac- datasets
with differing total cycle numbers, they must be manually modified and re-saved
into new pickle files with a matching number of cycles. Variance in cycle
numbers between any datasets will likely lead to nonsense results.

Importantly, it is up to the user to ensure that the reconciled datasets make
sense: this includes issues of how to deal with the increased peptide/dye loss
in the first few cycles, etc. It is recommended to background correct using ac-
datasets that match the boc- experiment to begin with, if possible.

This program will output the following pickled signal dictionaries. The drop
positions in the keys are in the coordinates of the reconciled/cropped
heatmaps, not the original experiments.

1. average_background.pkl: Values are the average number of drops at each
    position, expressed as a percentage of the total.
2. std_background.pkl: Values are the standard deviation of drops at each
    position, expressed as a percent.
3. experiment_background.pkl: Background of the experiment in counts.
4. corrected_experiment.pkl: Background-corrected experiment in counts, i.e.
    the original (possibly cropped) data minus the experiment_background.
"""

import argparse
from os import getcwd, makedirs
from os.path import abspath, exists, join
from csv import reader
from cPickle import load, dump
from time import time
import sys
sys.path.insert(0, '/home/boulgakov/git2/proteanseq/pflib')
from MCsimlib import (head_truncate,
                      discard_late_signals,
                      average_signals,
                      signals_std,
                      counts_to_percent,
                      iterative_peak_finding,
                      iterative_peak_finding_v2,
                      iterative_peak_finding_v3,
                     )
from pflib import _epoch_to_hash

#define and parse arguments; use custom MyFormatter to do both ArgumentDefault
#and RawDescription Formatters via multiple inheritence, this is a trick to
#preserve docstring formatting in --help output
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
                  argparse.RawDescriptionHelpFormatter):
    pass
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=MyFormatter,
                                 epilog="Curiouser and curiouser!",
                                )
bocfile_helpstring = "boc- experiment signals pickle."
parser.add_argument('--boc_file', nargs=1, required=True,
                    help=bocfile_helpstring)
acfile_helpstring = (
    "CSV file containing list of filepaths of ac- experiment pickles to use "
    "as collective background."
                     )
parser.add_argument('--ac_file', nargs=1, required=True,
                    help=acfile_helpstring)
head_boc_helpstring = (
    "Truncate this many cycles from the beginning of the boc- experiment."
                      )
parser.add_argument('--head_boc', type=int, default=0,
                    help=head_boc_helpstring)
head_ac_helpstring = (
    "Truncate this many cycles from the beginning of the ac- experiments."
                         )
parser.add_argument('--head_ac', type=int, default=0, help=head_ac_helpstring)
boc_total_helpstring = "Truncate boc- experiment to this number of cycles."
parser.add_argument('--boc_total', type=int, default=None,
                    help=boc_total_helpstring)
ac_total_helpstring = "Truncate ac- experiment to this number of cycles."
parser.add_argument('--ac_total', type=int, default=None,
                    help=ac_total_helpstring)
num_cycles_helpstring = (
    "Total number of cycles after reconciling all datasets."
                        )
parser.add_argument('--num_cycles', type=int, required=True,
                    help=num_cycles_helpstring)
ac_use_helpstring = (
    "Use only ac- datasets with these indexes in the ac- CSV. If both "
    "--ac_use and --ac_omit are given, --ac_use takes precedence and "
    "--ac_omit is ignored completely. If neither argument is given, all ac- "
    "datasets in the CSV are used."
                    )
parser.add_argument('--ac_use', type=int, nargs='+', help=ac_use_helpstring)
ac_omit_helpstring = (
    "Use all ac- datasets in the ac- CSV except those with these indices."
                     )
parser.add_argument('--ac_omit', type=int, nargs='+', help=ac_omit_helpstring)
omit_multidrop_helpstring = "Omit signals with more than one drop per cycle."
parser.add_argument('--omit_multidrop', action='store_true', default=False,
                    help=omit_multidrop_helpstring)
sigma_helpstring = (
    "Iterate until background is within this many standard deviations from "
    "the ac- mean."
                   )
parser.add_argument('--sigma', type=float, default=2, help=sigma_helpstring)
output_directory_helpstring = (
    "Output files to this directory. The directory is created if it doesn't "
    "exist. Existing files in the target directory with the same output names "
    "as generated by this script will be overwritten, however as long as "
    "simulations are not started simultaneously, e.g. via shell forking, the "
    "timestamp hashes should prevent file overwrites."
                              )
parser.add_argument('--output_directory', nargs=1, default=[getcwd()],
                    help=output_directory_helpstring)
args = parser.parse_args()

timestamp_epoch = round(time())
timestamp_hash = _epoch_to_hash(timestamp_epoch)

include_multidrop = not args.omit_multidrop
#Until we figure out how to deal with remainders, we omit them
include_remainders = False

ac_use_set = set() if args.ac_use is None else set(args.ac_use)
ac_omit_set = (set()
               if len(ac_use_set) > 0 or args.ac_omit is None
               else set(args.ac_omit)
              )

ac_experiments = {}
with open(args.ac_file[0]) as ac_csv:
    csv_reader = reader(ac_csv)
    for r, row in enumerate(csv_reader):
        if r == 0:
            continue
        ac_index, ac_filepath = row[:2]
        ac_index = int(ac_index)
        if ac_index in ac_omit_set:
            continue
        elif len(ac_use_set) > 0 and ac_index not in ac_use_set:
            continue
        try:
            ac_signals = load(open(ac_filepath))
            if not include_remainders:
                ac_signals = {(s, z, si): count
                              for (s, z, si), count in ac_signals.iteritems()
                              if z}
            ac_experiments.setdefault(ac_index, ac_signals)
        except Exception as e:
            print("Could not load " + str(ac_filepath) + " due to " + str(e)
                  + "; omitting.")

if args.head_ac > 0:
    for ac_index in ac_experiments.keys():
        ac_signals = ac_experiments[ac_index]
        truncated_ac_signals = head_truncate(
                                             signals=ac_signals,
                                             num_cycles=args.head_ac,
                                            )
        ac_experiments[ac_index] = truncated_ac_signals
elif args.head_ac == 0:
    pass
else:
    raise ValueError("--head_ac must be a non-negative integer.")

if args.ac_total is None:
    pass
elif args.ac_total > 0:
    for ac_index in ac_experiments.keys():
        ac_signals = ac_experiments[ac_index]
        truncated_ac_signals = discard_late_signals(
                                                    signals=ac_signals,
                                                    max_cycle=args.ac_total,
                                                   )
        ac_experiments[ac_index] = truncated_ac_signals
else:
    raise ValueError("--ac_total must be a positive integer.")

#No need to catch anything here. If this fails, we can't do anything anyways.
boc_experiment = load(open(args.boc_file[0]))
if not include_remainders:
    boc_experiment = {(s, z, si): count
                      for (s, z, si), count in boc_experiment.iteritems()
                      if z}

if args.head_boc > 0:
    boc_experiment = head_truncate(
                                   signals=boc_experiment,
                                   num_cycles=args.head_boc,
                                  )
elif args.head_boc == 0:
    pass
else:
    raise ValueError("--head_boc must be a non-negative integer.")

if args.boc_total is None:
    pass
elif args.boc_total > 0:
    boc_experiment = discard_late_signals(
                                          signals=boc_experiment,
                                          max_cycle=args.boc_total,
                                         )
else:
    raise ValueError("--boc_total must be a positive integer.")

if args.omit_multidrop:
    boc_experiment = {(s, z, si): count
                      for (s, z, si), count in boc_experiment.iteritems()
                      if len(s) == len(set(s))}

averaged_ac = average_signals(
                              experiments=ac_experiments.values(),
                              include_remainders=include_remainders,
                              include_multidrop=include_multidrop,
                              max_cycle=None,
                             )
ac_stds = signals_std(
                      experiments=ac_experiments.values(),
                      include_remainders=include_remainders,
                      include_multidrop=include_multidrop,
                      max_cycle=None,
                     )
boc_percent = counts_to_percent(
                                signals=boc_experiment,
                                include_remainders=include_remainders,
                                include_multidrop=include_multidrop,
                                max_cycle=None,
                               )
peak_list, undefined_peaks, updated_boc_raw, updated_boc_percent = \
    iterative_peak_finding_v3(
                              boc_raw=boc_experiment,
                              boc_percent=boc_percent,
                              ac_average=averaged_ac,
                              ac_std=ac_stds,
                              num_cycles=args.num_cycles,
                              sigma_threshold=args.sigma,
                              include_multidrop=include_multidrop,
                             )
background_corrected_raw = {(s, z, si):
                            max(boc_experiment[(s, z, si)] - background_count,
                                0)
                            for (s, z, si), background_count
                            in updated_boc_raw.iteritems()}

output_directory = abspath(args.output_directory[0])
if not exists(output_directory):
    makedirs(output_directory)

print("Background iteration completed. Saving results using filename hash "
      + str(timestamp_hash))
output_average_filename = "average_background_" + str(timestamp_hash) + ".pkl"
output_average_filepath = join(output_directory, output_average_filename)
dump(averaged_ac, open(output_average_filepath, 'w'))
output_stds_filename = "std_background_" + str(timestamp_hash) + ".pkl"
output_stds_filepath = join(output_directory, output_stds_filename)
dump(ac_stds, open(output_stds_filepath, 'w'))
output_background_filename = (
                              "experiment_background_"
                              + str(timestamp_hash) + ".pkl"
                             )
output_background_filepath = join(output_directory, output_background_filename)
dump(updated_boc_raw, open(output_background_filepath, 'w'))
corrected_experiment_filename = (
                                 "corrected_experiment_"
                                 + str(timestamp_hash) + ".pkl"
                                )
corrected_experiment_filepath = join(output_directory,
                                     corrected_experiment_filename)
dump(background_corrected_raw, open(corrected_experiment_filepath, 'w'))
