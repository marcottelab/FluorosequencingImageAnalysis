#!/home/boulgakov/anaconda2/bin/python


"""
Simulate fluorosequencing for a peptide.

Output is a pkl file of a tuple. First entry is list of parameters used. Second
entry are the final resulting signals. Third entry are the molecular error
signals (i.e. the intermediate results of simulating the chemical errors only).
"""


import argparse
import sys
from os import getcwd, makedirs
from os.path import abspath, exists, join
#sys.path.insert(0, '/home/boulgakov/peptidesequencing/git/proteanseq/pflib')
sys.path.insert(0, '/home/boulgakov/git2/proteanseq/pflib')
import traceback
from time import time
from datetime import datetime
from math import log
from multiprocessing import cpu_count
from collections import defaultdict
from cPickle import dump
from MCsimlib import (
                      _photometries_lognormal_fit_MP_v8,
                      write_photometries_dict_to_csv,
                     )
from pflib import _epoch_to_hash
import peptide_simulator

#define and parse arguments; use custom MyFormatter to do both ArgumentDefault
#and RawDescription Formatters via multiple inheritence, this is a trick to
#preserve docstring formatting in --help output
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
                  argparse.RawDescriptionHelpFormatter):
    pass
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=MyFormatter)
sequence_helpstring = "The peptide as a string of amino acids."
parser.add_argument('sequence', nargs=1, type=str, help=sequence_helpstring)
labels_helpstring = \
    ("Letters indicating which amino acids will be labeled (currently only "
     "one supported).")
parser.add_argument('labels', nargs=1, type=str, help=labels_helpstring)
num_sims_helpstring = "Number of samples to simulate."
parser.add_argument('-N', '--num_sims', type=int, default=100000,
                    help=num_sims_helpstring)
num_mocks_helpstring = "Number of mocks performed."
parser.add_argument('-m', '--num_mocks', type=int, default=4,
                    help=num_mocks_helpstring)
num_omitted_mocks_helpstring = "Number of mocks not imaged."
parser.add_argument('-o', '--num_mocks_omitted', type=int, default=1,
                    help=num_mocks_helpstring)
num_edmans_helpstring = "Number of Edmans performed."
parser.add_argument('-e', '--num_edmans', type=int, default=8,
                    help=num_edmans_helpstring)
parser.add_argument('--edman_efficiency', type=float, default=0.90,
                    help="(default: %(default)s)")
dye_destruction_helpstring = \
    ("This is the rate of dye destruction per cycle. It is NOT the exponent "
     "b in e^-kb.")
parser.add_argument('--dye_destruction', type=float, default=0.1,
                    help=dye_destruction_helpstring)
parser.add_argument('--dud_dyes', type=float, default=0.50,
                    help="(default: %(default)s)")
parser.add_argument('--surface_degradation_1', type=float, default=0.30,
                    help="(default: %(default)s)")
surface_degradation_1_num_cycles_helpstring = \
    ("Simulate surface_degradation_1 for the first X cycles "
     "(doesn't matter if mock or Edman), then use surface_degradation_2.")
parser.add_argument('--surface_degradation_1_num_cycles', type=int, default=3,
                    help=surface_degradation_1_num_cycles_helpstring)
parser.add_argument('--surface_degradation_2', type=float, default=0.10,
                    help="(default: %(default)s)")
fluor_intensity_helpstring = "Intensity of one fluor."
parser.add_argument('--fluor_intensity', type=float, default=70000,
                    help=fluor_intensity_helpstring)
ddif_2_factor_helpstring = "Dye-dye interaction factor for second fluor."
parser.add_argument('--ddif_2', type=float, default=0.30,
                    help=ddif_2_factor_helpstring)
ddif_3_factor_helpstring = \
    "Dye-dye interaction factor for third and above fluors."
parser.add_argument('--ddif_3', type=float, default=0.30,
                    help=ddif_3_factor_helpstring)
beta_sigma_helpstring = "Lognormal shape parameter."
parser.add_argument('--beta_sigma', type=float, default=0.20,
                    help=beta_sigma_helpstring)
distance_ddifs_helpstring = (
    "List DDIFs based on distance between dyes in amino acid units. The first "
    "value is the DDIF for dyes adjacent to each other, the second value is "
    "the DDIF for dyes one amino-acid apart, etc. DDIF values for distances "
    "higher than those listed are default to 0. Note that the fitter does not "
    "use these DDIFs (it always uses the --ddif_2 and --ddif_3 factors); "
    "distance DDIFs are only applied during the simulation step."
                           )
parser.add_argument('--distance_ddifs', nargs='+', type=float,
                    help=distance_ddifs_helpstring)
num_processors_helpstring = "Maximum number of processors to use."
parser.add_argument('-n', '--num_processors', type=int, default=None,
                    help=num_processors_helpstring)
no_csv_output_helpstring = "Do not output csv file of simulated photometries."
parser.add_argument('--no_csv', action='store_true', default=False,
                    help=no_csv_output_helpstring)
output_directory_helpstring = (
    "Output files to this directory. The directory is created if it doesn't "
    "exist. Existing files in the target directory with the same output names "
    "as generated by this script will be overwritten, however as long as "
    "simulations are not started simultaneously, e.g. via shell forking, the "
    "timestamp hashes should prevent file overwrites."
                              )
parser.add_argument('--output_directory', nargs=1, default=[getcwd()],
                    help=output_directory_helpstring)
no_multidrop_helpstring = "No drops greater than one dye allowed during fit."
parser.add_argument('--no_multidrop', action='store_true', default=False,
                    help=no_multidrop_helpstring)
superdye_rate_helpstring = (
                            "Chance of a dye being a superdye. Must be a "
                            "float between 0 and 1."
                           )
parser.add_argument('--superdye_rate', type=float, default=0.0,
                    help=superdye_rate_helpstring)
superdye_factor_helpstring = (
                              "Superdyes are brighter than normal by "
                              "superdye_factor."
                             )
parser.add_argument('--superdye_factor', type=float, default=1.0,
                    help=superdye_factor_helpstring)
beta_sigma_helpstring = "Lognormal shape parameter."
args = parser.parse_args()

if args.num_processors is None:
    num_processors = cpu_count()
else:
    num_processors = args.num_processors

parameters = (
              sequence,
              labels,
              N,
              m,
              o,
              e,
              edman_efficiency,
              dye_destruction,
              dud_dyes,
              surface_degradation_1,
              surface_degradation_1_num_cycles,
              surface_degradation_2,
              fluor_intensity,
              ddif_2,
              ddif_3,
              beta_sigma,
              distance_ddifs,
              superdye_rate,
              superdye_factor,
             ) = (
                  args.sequence[0],
                  args.labels[0],
                  args.num_sims,
                  args.num_mocks,
                  args.num_mocks_omitted,
                  args.num_edmans,
                  args.edman_efficiency,
                  args.dye_destruction,
                  args.dud_dyes,
                  args.surface_degradation_1,
                  args.surface_degradation_1_num_cycles,
                  args.surface_degradation_2,
                  args.fluor_intensity,
                  args.ddif_2,
                  args.ddif_3,
                  args.beta_sigma,
                  args.distance_ddifs,
                  args.superdye_rate,
                  args.superdye_factor,
                 )

output_directory = abspath(args.output_directory[0])
if not exists(output_directory):
    makedirs(output_directory)

allow_multidrop = not args.no_multidrop

timestamp_epoch = round(time())
timestamp_hash = _epoch_to_hash(timestamp_epoch)
output_filename = (
                   "Simulated_"
                   #+ str(parameters)
                   #+ "_"
                   + str(timestamp_hash)
                   + ".pkl"
                  )
output_filepath = join(output_directory, output_filename)

max_possible = 5
ddif = [0, ddif_2] + [ddif_3] * 5

if distance_ddifs is not None:
    sequence_length = len(sequence)
    maximum_distance = sequence_length - 1
    padding_required = maximum_distance - len(distance_ddifs)
    if padding_required == 0:
        padded_distance_ddifs = distance_ddifs
    elif padding_required > 0:
        padded_distance_ddifs = distance_ddifs + [0.0] * padding_required
    else:
        padded_distance_ddifs = distance_ddifs[:padding_required]
    distance_ddifs = dict(zip(range(1, maximum_distance),
                              padded_distance_ddifs))

print("Parameters loaded. Starting simulation at " + str(datetime.now()))

results = \
    peptide_simulator.peptide_simulation(
                                         sequence=sequence,
                                         labels=labels,
                                         num_mocks=m - o,
                                         num_edmans=e,
                                         num_simulations=N,
                                         random_seed=None,
                                         num_processes=num_processors,
                                         reserved_character=None,
                                         p=edman_efficiency,
                                         b=-log(1.0 - dye_destruction),
                                         u=dud_dyes,
                                         s=surface_degradation_1,
                                         sc=surface_degradation_1_num_cycles,
                                         s2=surface_degradation_2,
                                         beta=fluor_intensity,
                                         beta_sigma=beta_sigma,
                                         ddif=ddif,
                                         distance_ddifs=distance_ddifs,
                                         superdye_rate=superdye_rate,
                                         superdye_factor=superdye_factor,
                                        )

results = peptide_simulator.convert_to_oldstyle(results)
molecular_error_signals = defaultdict(int)
photometries = {'ch1': {0: {}}}
t = 0
for dye_decrements, dye_counts, event_buffer, intensities_dict in results:
    for label, (category, (intensities,)) in intensities_dict.iteritems():
        photometries['ch1'][0].setdefault((t, t), (category, intensities, t))
        t += 1
    assert len(dye_counts) > 0
    if len(dye_counts) > 1:
        raise NotImplementedError("This part currently only works for one label.")
    label, seq = dye_counts.popitem()
    z = True if seq[-1] == 0 else False
    molecular_error_signals[(dye_decrements, z, seq[0])] += 1
molecular_error_signals = dict(molecular_error_signals)

if not args.no_csv:
    csv_filepath = output_filepath[:-4] + ".csv"
    try:
        row_counter = write_photometries_dict_to_csv(
                                                     photometries=photometries,
                                                     filepath=csv_filepath,
                                                    )
        print("Wrote " + str(row_counter) + " rows to " + str(csv_filepath))
    except Exception as e:
        print("Failed to write simulated photometries to " + str(csv_filepath)
              + " due to exception " + str(e))
        traceback.print_exc()

print("Simulation complete. Fitting simulated tracks at "
      + str(datetime.now()))

plf_results = \
         (signals,
          total_count,
          none_count,
          all_fit_info) = \
             _photometries_lognormal_fit_MP_v8(photometries=photometries,
                                               beta=fluor_intensity,
                                               beta_sigma=beta_sigma,
                                               max_possible=max_possible,
                                               num_processes=num_processors,
                                               allow_upsteps=False,
                                               allow_multidrop=allow_multidrop,
                                               max_deviation=3,
                                               quench_factor=0,
                                               quench_factors=ddif)
print("Fitting completed at " + str(datetime.now())
      + ". Saving results to " + str(output_filename))
dump((args, signals, molecular_error_signals),
     open(output_filepath, 'w'))
