#!/home/boulgakov/anaconda2/bin/python


"""
Basic script that analyses images from a multifield simple Edman sequencing
experiment, and summarizes basic sequence information from peptides.

A multifield simple Edman sequencing experiment tracks a chosen set of fields
through a series of (possibly mock) Edman reactions. For each field, there is
one image taken before the first Edman reaction, followed by one image after
each subsequent reaction. Currently, up to two channels are tracked for
sequencing through the experiment. Optionally, an additional sequence of frames
can be provided with fiduciary markers in a separate channel, so as to align
fields after stage drift.

For each image, the program tries to find its PSFs pkl file as saved by pflib
at the default location specified by pflib.save_psfs_pkl and
pflib._psfs_filename, and assume the images' spots are all identified there. If
there are multiple PSF pkl files for the image, the script orders them by their
epoch hashtag and uses the latest file. If no PSF pkl files are found, it will
preform a full peak fitting on the image via pflib.find_peptides. If desired,
peak fitting can be recomputed for an image even if it already has a PSF pkl
file.

For each field, the script finds all spots via pflib peak fitting, and tracks
them through the experiment. The states of tracked spots after each Edman
reaction are classified as either 'ON' or 'OFF', depending on if each is
present in each frame or not.

The final results of the script are saved to disk and printed out onscreen. For
documentation of the csv file output, see
flexlibrary.MultifieldMultichannelSequenceExperiment.track_photometries_as_csv.
For documentation of screen output, see
flexlibrary.MultifieldMultichannelSequenceExperiment.category_counts_as_string.
For sanity check images, see flexlibrary.Experiment.plot_traces.
"""


import sys
import time
import datetime
import argparse
import os.path
import flexlibrary
import logging
import glob
import pflib
import os
import cPickle
import scipy.misc
from PIL import Image
from skimage import exposure
import numpy as np
import multiprocessing
import ast


#get timestamp that will be used throughout this run
timestamp_epoch = int(round(time.time()))
timestamp_datetime = datetime.datetime.fromtimestamp(timestamp_epoch)
epoch_hash = pflib._epoch_to_hash(timestamp_epoch)


#define and parse arguments; use custom MyFormatter to do both ArgumentDefault
#and RawDescription Formatters via multiple inheritence, this is a trick to
#preserve docstring formatting in --help output
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
                  argparse.RawDescriptionHelpFormatter):
    pass
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=MyFormatter)
debug_help_string = ("Log debugging output.")
parser.add_argument('-D', '--debug', action='store_true',
                    help=debug_help_string, default=False)
num_processes_help_string = ("Number of processes to use. Default defined by "
                             "pflib.parallel_image_batch.")
parser.add_argument('-n', '--num_processes', type=int, nargs=1,
                    default=[multiprocessing.cpu_count()],
                    help=num_processes_help_string)
default_log_directory = '/home'
default_log_filename = ('basic_experiment_script_' + str(timestamp_datetime) +
                        '.log')
log_path_help_string = \
    ("Pathname for log. Default is "
     "/project/boulgakov/microscope/pf_log_files/basic_experiment_script_ + "
     "str(datetime.datetime.now())  + .log. If the log file already exists, "
     "further logging output follows default Python behavior, which is "
     "currently to append the file.")
default_log_path = os.path.join(default_log_directory, default_log_filename)
parser.add_argument('-L', '--log_path', nargs=1, default=[default_log_path],
                    help=log_path_help_string)
output_directory_help_string = \
    ("All output files are saved to this directory and to subdirectories "
     "specifically created for certain types of output. Default is the "
     "current working directory, except for trace image files, which are "
     "saved alongside their source image files as described in "
     "flexlibrary.Experiment.plot_traces. Caution: we recommend using a fresh "
     "directory as output_directory, because otherwise things may be "
     "overwritten.")
parser.add_argument('--output_directory', nargs=1, default=None,
                    help=output_directory_help_string)
recompute_help_string = \
    ("Recompute peak fitting on the images. Will save results using the "
     "default pflib.save_psfs_* filenames with a new timestamp. (CURRENTLY "
     "NOT IMPLEMENTED)")
parser.add_argument('-r', '--recompute', action='store_true',
                    help=recompute_help_string, default=False)
keep_invalid_helpstring = \
    ("Keep tracks where they leave the field of view or are too close to the "
     "edge to calculate the proper photometry with the method requested "
     "(specified via e.g. --photometry_parameters). Keeping these tracks is "
     "the default behavior of the program before June 3rd 2016. If this "
     "option is not selected, those tracks are discarded and the output "
     "track_photometries_hash.csv is saved as "
     "track_photometries_NO_NONES_hash.csv")
parser.add_argument('--keep_invalid', action='store_true',
                    help=keep_invalid_helpstring, default=False)
pkl_invalid_helpstring = \
    ("If keep_invalid is not selected, this will save the discarded tracks as "
     "a pkl file. Filesize is large, approx 10Gig/100 fields (one channel).")
parser.add_argument('--pkl_invalid', action='store_true',
                    help=pkl_invalid_helpstring, default=False)
no_self_align_help_string = \
    ("Do not use peptide_frames as alignment_frames when alignment_frames not "
     "provided. If alignment_frames is not provided and this option is "
     "chosen, no alignment is performed.")
parser.add_argument('-ns', '--no_self_align', action='store_true',
                    default=False, help=no_self_align_help_string)
no_sanity_check_images_helpstring = \
    ("Don't make sanity check images.")
parser.add_argument('--no_sanity_check_images', action='store_true',
                    help=no_sanity_check_images_helpstring, default=False)
peptide_extraction_number_help_string = \
    ("Number of peptide tracks to extract to separate images for each "
     "well-behaved pattern.")
parser.add_argument('-en', '--extraction_number', type=int, default=10,
                    help=peptide_extraction_number_help_string)
peptide_extraction_size_help_string = \
    ("When extracting peptide tracks to separate images, make the square "
     "images are this size on the side. Must be an odd number.")
parser.add_argument('-es', '--extraction_size', type=int, default=9,
                    help=peptide_extraction_size_help_string)
save_tracks_help_string = \
    ("Save tracks to Python pkl files. Caution: may eat up lots of space!")
parser.add_argument('--save_tracks', action='store_true', default=False,
                    help=save_tracks_help_string)
sextractor_help_string = ("Use sextractor photometry algorithm.")
parser.add_argument('--sextractor', action='store_true', default=False,
                    help=sextractor_help_string)
photometry_parameters_help_string = \
    ("Parameters for Spot's photometry function. Expects a Python dictionary "
     "in quotes. Example: --photometry_parameters=\"{'photometry_method': "
     "'mexican_hat', 'brim_size': 4, 'radius': 5}\". These parameters will be "
     "used throughout for all flexlibrary.Spot instances' photometry method "
     "calls. For the keyword arguments omitted, their defaults in flexlibrary "
     "are used.")
parser.add_argument('--photometry_parameters', type=str, nargs=1,
                    default=[None], help=photometry_parameters_help_string)
photometries_help_string = \
    ("Save tracks' photometries as csv file. Also see --not_all_photometries "
     "below.")
parser.add_argument('--save_photometries', action='store_true', default=True,
                    help=photometries_help_string)
not_all_photometries_help_string = \
    ("By default, saving tracks' photometries as csv file via "
     "--save_photometries will output every tracks' photometry in every "
     "frame, regardless of whether a spot was detected in that frame. If a "
     "spot was not detected, it uses interpolation between found spots across "
     "frames to find the appropriate areas in the image to compute photometry "
     "on, and saves all photometries except for those spots that would be "
     "outside their frame due to stage drift. If --not_all_photometies is "
     "given, only the photometries of identified spots are taken into "
     "account, no interpolation to measure photometries in intervening frames "
     "is done, and only the average of a track's found spots is saved. See "
     "flexlibrary.MultifieldMultichannelSequenceExperiment."
     "track_photometries_as_csv docstring for more details.")
parser.add_argument('--not_all_photometries', action='store_true', default=False,
                    help=not_all_photometries_help_string)
collate_fields_help_string = \
    ("Collate data by fields in CSV and string output.")
parser.add_argument('--collate_fields', action='store_true', default=False,
                    help=collate_fields_help_string)
all_categories_help_string = \
    ("If True, will print out all category combinations to the screen as part "
     "of the output. If False, will only print out one-drop categories (e.g. "
     "will print [ON][ON][OFF][OFF] but not [ON][OFF][ON][ON]). [NOT YET "
     "IMPLEMENTED.]")
parser.add_argument('--all_categories', action='store_true', default=False,
                    help=all_categories_help_string)
alignment_frames_help_string = \
    ("Pathnames to images in the alignment channel. Images are first grouped "
     "between experimental cycles by directory: each directory corresponds to "
     "images taken during one experimental cycle. The alphanumeric order of "
     "directory names matches their cycles' chronological order. For every "
     "field of view in the experiment, each directory contains one image. The "
     "order of filenames for fields of view is consistent across directories. "
     "If provided, there must be one alignment frame for each peptide channel "
     "frame, and they are matched to each other via matching their "
     "counterparts in the grouping & sorting scheme described.")
parser.add_argument('--alignment_files', nargs='+', type=str,
                    help=alignment_frames_help_string, default=None,
                    required=False)
peptide_frames_help_string = \
    ("Pathnames to images in the peptide channel. Images are first grouped "
     "between experimental cycles by directory: each directory corresponds to "
     "images taken during one experimental cycle. The alphanumeric order of "
     "directory names matches their cycles' chronological order. For every "
     "field of view in the experiment, each directory contains one image. The "
     "order of filenames for fields of view is consistent across directories.")
parser.add_argument('--peptide_files', nargs='+', type=str,
                    help=peptide_frames_help_string, required=True)
second_channel_help_string = \
    ("Pathnames to images of the second peptide channel. Images are grouped "
     "as for peptide_frames. If second_channel is provided, then there must "
     "be one image for each frame in peptide_frames. Alignment is based on "
     "peptide_files unles alignment_files is provided.")
parser.add_argument('--second_channel', nargs='+', type=str,
                    help=second_channel_help_string, default=None) 
args = parser.parse_args()

#set up logging
if args.debug:
    logging.basicConfig(filename=args.log_path[0], level=logging.DEBUG)
else:
    logging.basicConfig(filename=args.log_path[0], level=logging.INFO)
logger = logging.getLogger()
logger.info("basic_experiment_script starting at " + str(timestamp_datetime))
logger.info("args = " + str(args))

#recompute is not implemented
if args.recompute:
    raise NotImplementedError("--recompute option not currently implemented.")
if args.all_categories:
    raise NotImplementedError("--all_categories option not currently "
                              "implemented.")

#ensure absolute paths
peptide_files = [os.path.abspath(f) for f in args.peptide_files]

#fit unfitted files
need_fitting = []
need_fitting_map = {}
for f, fullpath in enumerate(peptide_files):
    psf_pkl_files = sorted(glob.glob(fullpath + '*_psfs_*.pkl'))
    if len(psf_pkl_files) == 0:
        need_fitting.append(fullpath)
        need_fitting_map.setdefault(fullpath, f)
logger.info("Could not find PSF pkl files for these images; they will be " +
            "submitted to pflib: " + str(need_fitting))
processed_images = \
                pflib.parallel_image_batch(image_paths=need_fitting,
                                           find_peptides_parameters=None,
                                           timestamp_epoch=timestamp_epoch,
                                           num_processes=args.num_processes[0])
for original_path, (converted_path, psfs_pkl_path, psfs_csv_path,
                    psfs_png_path) in processed_images.iteritems():
    peptide_files[need_fitting_map[original_path]] = converted_path

#group & sort images by field and chronological order
directory_indexed_peptide_files = {} #used to check that every directory (i.e.
                                     #experimental cycle) has the same number
                                     #of images
for f in peptide_files:
    head, tail = os.path.split(f)
    directory_indexed_peptide_files.setdefault(head, []).append(tail)
if len(set([len(heads)
            for tail, heads
            in directory_indexed_peptide_files.iteritems()])) != 1:
    raise Exception("For peptide_files, each directory must have the same "
                    "number of files specified.")
frame_indexed_peptide_files, field_indexed_peptide_files = \
                  flexlibrary.Experiment.easy_sort_target_images(peptide_files)
if args.alignment_files is not None:
    alignment_files = [os.path.abspath(f) for f in args.alignment_files]
elif not args.no_self_align:
    alignment_files = [os.path.abspath(f) for f in args.peptide_files]
else:
    alignment_files = []
frame_indexed_alignment_files, field_indexed_alignment_files = \
                flexlibrary.Experiment.easy_sort_target_images(alignment_files)
if (args.alignment_files is not None
        and
    (
        set(frame_indexed_peptide_files.keys()) !=
        set(frame_indexed_alignment_files.keys())
            or
        not all([len(files) == len(frame_indexed_alignment_files[d])
                 for d, files in frame_indexed_peptide_files.iteritems()])
    )
   ):
    raise Exception("Alignment files given, but not every peptide image file "
                    "has one.")
if args.second_channel is not None:
    second_channel_files = [os.path.abspath(f)
                            for f in args.second_channel]
    need_fitting = []
    need_fitting_map = {}
    for f, fullpath in enumerate(second_channel_files):
        psf_pkl_files = sorted(glob.glob(fullpath + '*_psfs_*.pkl'))
        if len(psf_pkl_files) == 0:
            need_fitting.append(fullpath)
            need_fitting_map.setdefault(fullpath, f)
    logger.info("Could not find PSF pkl files for these images; they will be " +
                "submitted to pflib: " + str(need_fitting))
    processed_images = \
                pflib.parallel_image_batch(image_paths=need_fitting,
                                           find_peptides_parameters=None,
                                           timestamp_epoch=timestamp_epoch,
                                           num_processes=args.num_processes[0])
    for original_path, (converted_path, psfs_pkl_path, psfs_csv_path,
                        psfs_png_path) in processed_images.iteritems():
        second_channel_files[need_fitting_map[original_path]] = converted_path
else:
    second_channel_files = []
frame_indexed_second_channel_files, field_indexed_second_channel_files = \
           flexlibrary.Experiment.easy_sort_target_images(second_channel_files)
if (args.second_channel is not None
        and
    (
        set(frame_indexed_second_channel_files.keys()) !=
        set(frame_indexed_second_channel_files.keys())
            or
        not all([len(files) == len(frame_indexed_second_channel_files[d])
               for d, files in frame_indexed_second_channel_files.iteritems()])
    )
   ):
    raise Exception("Second channel files given, but not every peptide image "
                    "file has one.")

logger.info("frame_indexed_peptide_files " + str(frame_indexed_peptide_files))
logger.info("field_indexed_peptide_files " + str(field_indexed_peptide_files))
logger.info("frame_indexed_alignment_files " +
            str(frame_indexed_alignment_files))
logger.info("field_indexed_alignment_files " +
            str(field_indexed_alignment_files))
logger.info("frame_indexed_second_channel_files " +
            str(frame_indexed_second_channel_files))
logger.info("field_indexed_second_channel_files " +
            str(field_indexed_second_channel_files))

#make sure output directory exists
output_directory = os.path.abspath(args.output_directory[0])
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#find all images that do not have a corresponding psfs pkl file and peakfit
#need_fitting = []
#for index, filepaths in frame_indexed_peptide_files.iteritems():
#    for f, fullpath in enumerate(filepaths):
#        psf_pkl_files = sorted(glob.glob(fullpath + '_psfs_*.pkl'))
#        if len(psf_pkl_files) == 0:
#            need_fitting.append((0, index, f, fullpath))
#for index, filepaths in frame_indexed_second_channel_files.iteritems():
#    for f, fullpath in enumerate(filepaths):
#        psf_pkl_files = sorted(glob.glob(fullpath + '_psfs_*.pkl'))
#        if len(psf_pkl_files) == 0:
#            need_fitting.append((1, index, f, fullpath))
#NO NEED TO PEAKFIT ALIGNMENT FRAMES! IMAGE REGISTRATION BASED ON CORR
#for index, filepaths in frame_indexed_alignment_files.iteritems():
#    for fullpath in filepaths:
#        psf_pkl_files = sorted(glob.glob(fullpath + '_psfs_*.pkl'))
#        if len(psf_pkl_files) == 0:
#            need_fitting.append(fullpath)
#replace unconverted/unfitted paths in frame_indexed_peptide_files and
#frame_indexed_second_channel_files with the converted versions from
#processed_images
#for original_path, (converted_path, psfs_pkl_path, psfs_csv_path,
#                    psfs_png_path) in processed_images.iteritems():
#    for t, i, f, p in need_fitting:
#        if original_path == p:
#            if t == 0:
#                frame_indexed_peptide_files[i][f] = converted_path
#            elif t == 1:
#                frame_indexed_second_channel_files[i][f] = converted_path

#load image and PSFs pkl files
peptide_fields = {}
for field, files in field_indexed_peptide_files.iteritems():
    peptide_fields.setdefault(field, [])
    for f in files:
        image_object, discarded_spots = \
                            flexlibrary.Experiment.easy_load_processed_image(f)
        if discarded_spots > 0:
            logger.info("For file " + str(f) + " in peptide_fields, " +
                        "discarded " + str(discarded_spots) + " Spots.")
        peptide_fields[field].append(image_object)
#peptide_fields = {field: [flexlibrary.Experiment.easy_load_processed_image(f)
#                          for f in files]
#                  for field, files in field_indexed_peptide_files.iteritems()}
alignment_fields = {}
for field, files in field_indexed_alignment_files.iteritems():
    alignment_fields.setdefault(field, [])
    for f in files:
        image_object, discarded_spots = \
           flexlibrary.Experiment.easy_load_processed_image(f, load_psfs=False)
        if discarded_spots > 0:
            logger.info("For file " + str(f) + " in alignment_fields, " +
                        "discarded " + str(discarded_spots) + " Spots.")
        alignment_fields[field].append(image_object)
#alignment_fields = \
#  {field: [flexlibrary.Experiment.easy_load_processed_image(f, load_psfs=False)
#           for f in files]
#   for field, files in field_indexed_alignment_files.iteritems()}
second_channel_fields = {}
for field, files in field_indexed_second_channel_files.iteritems():
    second_channel_fields.setdefault(field, [])
    for f in files:
        image_object, discarded_spots = \
                            flexlibrary.Experiment.easy_load_processed_image(f)
        if discarded_spots > 0:
            logger.info("For file " + str(f) + " in second_channel_fields, " +
                        "discarded " + str(discarded_spots) + " Spots.")
        second_channel_fields[field].append(image_object)

#second_channel_fields = \
#           {field: [flexlibrary.Experiment.easy_load_processed_image(f)
#                    for f in files]
#            for field, files in field_indexed_second_channel_files.iteritems()}

#build SequenceExperiment's, run alignment & traces
single_field_experiments = []
second_channel_experiments =[]
combined_channel_experiments = []
for field, frames in peptide_fields.iteritems():
    if len(alignment_fields) > 0:
        alignment_frames = alignment_fields[field]
    else:
        alignment_frames = None
    ex = flexlibrary.SequenceExperiment(peptide_frames=frames,
                                      alignment_frames=alignment_fields[field])
    ex.offsets_from_frames()
    single_field_experiments.append(ex)
    if len(second_channel_fields) == 0:
        combined_channel_dict = {'ch1': ex}
    else:
        s_field = second_channel_fields[field]
        ex2 = flexlibrary.SequenceExperiment(peptide_frames=s_field,
                                      alignment_frames=alignment_fields[field])
        ex2.offsets_from_frames()
        second_channel_experiments.append(ex2)
        combined_channel_dict = {'ch1': ex, 'ch2': ex2}
    c_ex = flexlibrary.MultichannelSequenceExperiment(combined_channel_dict)
    combined_channel_experiments.append(c_ex)

#combine in MultifieldMultichannelSequenceExperiment
mfmc_experiment = \
    flexlibrary.MultifieldMultichannelSequenceExperiment(experimental_fields=
                                                  combined_channel_experiments)
#get photometry arguments
if args.photometry_parameters[0] is not None:
    p_params = ast.literal_eval(args.photometry_parameters[0])
else:
    if args.sextractor:
        p_params = {'photometry_method': 'sextractor'}
    else:
        p_params = {}


#parse output_directory option
if args.output_directory is None:
    output_directory = os.getcwd()
    trace_directory = None
else:
    output_directory = args.output_directory[0]
    trace_directory = os.path.join(output_directory, 'sanity_check_pngs_' +
                                                     epoch_hash)
    if not os.path.exists(trace_directory) and not args.no_sanity_check_images:
        os.makedirs(trace_directory)

#Trace spots and discrd invalid if requested
mfmc_experiment.trace_existing_spots()
discard_invalid = not args.keep_invalid
if not args.keep_invalid:
    invalid_traces = mfmc_experiment.discard_invalid_traces(**p_params)
    invalid_traces_pkl_path = os.path.join(output_directory, 'discarded_traces_' + str(epoch_hash) + '.pkl')
    if args.pkl_invalid:
        cPickle.dump(invalid_traces, open(invalid_traces_pkl_path, 'w'))

#save sanity check images
if not args.no_sanity_check_images:
    mfmc_experiment.plot_traces(timestamp_epoch=timestamp_epoch,
                                trace_directory=trace_directory)

#get stats
category_stats, categories = mfmc_experiment.count_binary_trace_categories()
filtered_stats = mfmc_experiment.filtered_binary_trace_category_counts(
                                                 include_first_frame_only=True)

#The following pkl file is very large (easily 500MB+ for ~120 images)
#cPickle.dump(categories,
#             open(os.path.join(output_directory, 
#                               'categories_' + str(epoch_hash) + '.pkl'),
#                  'w'))

#Instead of the above one very large file, save per category
if args.save_tracks:
    category_output_directory = os.path.join(output_directory,
                                             'category_pkls_' + epoch_hash)
    if not os.path.exists(category_output_directory):
        os.makedirs(category_output_directory)
    for category, traces in categories.iteritems():
        output_filepath = os.path.join(category_output_directory,
                                       'category_' + str(category) + '.pkl')
        cPickle.dump(traces, open(output_filepath, 'w'))

#save category stats
cPickle.dump(category_stats,
             open(os.path.join(output_directory,
                               'category_stats_' + str(epoch_hash) + '.pkl'),
                  'w'))
cPickle.dump(filtered_stats,
             open(os.path.join(output_directory,
                               'filtered_stats_' + str(epoch_hash) + '.pkl'),
                  'w'))

#Automated Matplotlib plots are now deprecated.
#mfmc_experiment.plot_filtered_binary_trace_counts(
#     os.path.join(output_directory, 'basic_plot_' + str(epoch_hash) + '.png'))

mfmc_experiment.category_counts_as_csv(os.path.join(output_directory,
                                                    'category_counts_' +
                                                    str(epoch_hash) + '.csv'),
                                       collate_fields=args.collate_fields)


#Extract peptide tracks and save as PNGs
if args.save_tracks:
    track_output_directory = os.path.join(output_directory, 'track_pngs_' +
                                                            epoch_hash)
    if not os.path.exists(track_output_directory):
        os.makedirs(track_output_directory)
    num_frames = len(frame_indexed_peptide_files)
    for drop in range(1, num_frames + 1):
        pattern = tuple([True] * drop + [False] * (num_frames - drop))
        if args.extraction_size % 2 == 0:
            raise ValueError("extraction_size must be an odd number.")
        radius = int((args.extraction_size - 1) / 2)
        tracks = mfmc_experiment.extract_tracks(trace_category=pattern,
                                                radius=radius,
                                                number=args.extraction_number)
        #get uniform normalization for saving tracks
        #track_norm = np.amax([np.amax(parent_Image.image)
        #track_norm = np.amax([np.mean(parent_Image.image) +
        #                      np.std(parent_Image.image)
        #                    for c, c_tracks in tracks.iteritems()
        #                    for t, ((h, w), track) in enumerate(c_tracks)
        #                    for f, (frame, parent_Image) in enumerate(track)])
        #track_min_norm = np.amin([np.amin(parent_Image.image)
        #track_min_norm = np.amin([np.mean(parent_Image.image) -
        #                          np.std(parent_Image.image)
        #                         for c, c_tracks in tracks.iteritems()
        #                         for t, ((h, w), track) in enumerate(c_tracks)
        #                         for f, (frame, parent_Image)
        #                              in enumerate(track)])
        logger.debug("basic_experiment_script: drop = " + str(drop))
        for c, c_tracks in tracks.iteritems():
            logger.debug("basic_experiment_script: c_tracks = " +
                         str(c_tracks))
            for t, ((h, w), track) in enumerate(c_tracks):
                logger.debug("basic_experiment_script: track = " +
                             str(track))
                #get max of all traces to normalize saving images
                frame_norm = np.amax([np.amax(frame)
                              for f, (frame, parent_Image) in enumerate(track)
                              if frame.shape[0] != 0 and frame.shape[1] != 0])
                frame_min_norm = np.amin([np.amin(frame)
                                          for f, (frame, parent_Image)
                                              in enumerate(track)
                                          if (frame.shape[0] != 0 and
                                              frame.shape[1] != 0)])
                for f, (frame, parent_Image) in enumerate(track):
                    output_filepath = os.path.join(track_output_directory,
                                                   'track_drop_' + str(drop) +
                                                   '_hw_' + str((h, w)) +
                                                   '_channel_' + str(c) +
                                                   '_track_' + str(t) +
                                                   '_frame_' + str(f) +
                                                   '.png')
                    logger.debug("basic_experiment_script: (h, w) = " +
                                 str((h, w)))
                    logger.debug("basic_experiment_script: frame = " +
                                 str(frame))
                    logger.debug("basic_experiment_script: frame shape = " +
                                 str(frame.shape))
                    if frame.shape[0] == 0 or frame.shape[1] == 0:
                        #this happens because spots that fall off frame when
                        #no longer present but still tracked occur
                        logger.debug("Skipping this frame due to shape "
                                     "dimension 0. Frame f = " + str(f))
                        continue
                    #r_frame = pflib._intensity_scaling(frame)
                    r_frame = exposure.rescale_intensity(
                                         frame,
    #                                    in_range=(track_min_norm, track_norm),
                                         in_range=(frame_min_norm, frame_norm),
    #                                     in_range=(20000, 26000),
                                         out_range=np.uint8).astype(np.uint8)
                    #                     in_range=np.uint32,
                    #                     out_range=np.uint8).astype(np.uint8)
                    pillow_image = Image.fromarray(r_frame, mode="L")
                    pillow_image.save(output_filepath)
                    #logger.debug("basic_experiment_script: frame.dtype = " +
                    #             str(frame.dtype))
                    #scipy.misc.toimage(frame,
                    #                   cmin=track_min_norm,
                    #                   cmax=track_norm).save(output_filepath)
                    #scipy.misc.imsave(output_filepath, frame)
    #TODO: SCALING TRACKING IMAGES!!!!

#save photometries
if args.save_photometries:
    if args.keep_invalid:
        csv_filename = os.path.join(output_directory,
                                    'track_photometries_' + str(epoch_hash) +
                                    '.csv')
    else:
        csv_filename = os.path.join(output_directory,
                             'track_photometries_NO_NONES_' + str(epoch_hash) +
                             '.csv')
    mfmc_experiment.track_photometries_as_csv(
                                       filepath=csv_filename,
                                       save_averages=args.not_all_photometries,
                                       discard_invalid=False,
                                       **p_params)


print("")
print("")
print("Summary stats")
print("-------------")
print("Stage drift offsets:")
print(mfmc_experiment.offsets_as_string())
mfmc_experiment.save_offsets_as_dict(os.path.join(output_directory,
                                   'offsets_dict_' + str(epoch_hash) + '.pkl'))
print("Total spots found in all peptide frames: " +
      str(mfmc_experiment.spot_count()))
print("Number of spots discarded due to stage drift: " +
      str(mfmc_experiment.count_discarded_spots()))
print("Total number of traced spots: " + str(mfmc_experiment.trace_count()))
print("Singleton count: " + str(mfmc_experiment.singleton_count()))
print("Basic track breakdown:")
print(mfmc_experiment.category_counts_as_string(
                                           filtered=(not args.all_categories),
                                           collate_fields=args.collate_fields))
