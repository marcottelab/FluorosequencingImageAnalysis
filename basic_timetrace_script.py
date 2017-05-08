#!/home/boulgakov/anaconda2/bin/python


"""
Basic script for analyzing a single-field time trace.
"""


#making sure to import flexlibrary first because it has a matplotlib.use('agg')
#that won't work if pflib is imported first
import flexlibrary
import time
import datetime
import argparse
import logging
import os
import os.path
import glob
import pflib
import cPickle
import errno
import cPickle
import ast


#get timestamp that will be used throughout this run
timestamp_epoch = int(round(time.time()))
timestamp_datetime = datetime.datetime.fromtimestamp(timestamp_epoch)
epoch_hash = pflib._epoch_to_hash(timestamp_epoch)


#Define nd parse arguments; use custom MyFormatter to do both ArgumentDefault
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
default_log_directory = '/project/boulgakov/microscope/pf_log_files'
default_log_filename = ('basic_timetrace_script_' + str(timestamp_datetime) +
                        '.log')
log_path_help_string = \
    ("Pathname for log. If the log file already exists, further logging "
     "output follows default Python behavior, which is currently to append "
     "the file.")
default_log_path = os.path.join(default_log_directory, default_log_filename)
parser.add_argument('-L', '--log_path', nargs=1, default=[default_log_path],
                    help=log_path_help_string)
output_directory_help_string = \
    ("All output files are saved to this directory and to subdirectories "
     "specifically created for certain types of output. Caution: we recommend "
     "using a fresh directory as output_directory, because otherwise things "
     "may be overwritten.")
parser.add_argument('--output_directory', nargs=1, default=[os.getcwd()],
                    help=output_directory_help_string)
sanity_check_images_helpstring = \
    ("Don't make sanity check images.")
parser.add_argument('--no_sanity_check_images', action='store_true',
                    help=sanity_check_images_helpstring, default=False)
save_traces_pkl_helpstring = ("Save the found traces to a pickle file.")
parser.add_argument('--save_traces_pkl', action='store_true',
                    help=save_traces_pkl_helpstring, default=False)
sextractor_help_string = ("Use sextractor photometry algorithm.")
parser.add_argument('--sextractor', action='store_true', default=False,
                    help=sextractor_help_string)
timetrace_frames_helpstring = \
    ("Sequence of frames comprising the time trace. The chronological order "
     "of the frames is based on the order in which they are provided to this "
     "script.")
photometry_parameters_help_string = \
    ("Parameters for Spot's photometry function. Expects a Python dictionary "
     "in quotes. Example: --photometry_parameters=\"{'method': 'mexican_hat', "
     "'brim_size': 4, 'radius': 5}\". These parameters will be used throughout "
     "for all flexlibrary.Spot instances' photometry method calls. If any of "
     "the keyword arguments are omitted, the defaults in flexlibrary are "
     "used.")
parser.add_argument('--photometry_parameters', type=str, nargs=1,
                    default=[None], help=photometry_parameters_help_string)
#min_step_magnitude_helpstring = \
#    ("Ignore steps smaller than this value when using chi_squared method.")
#parser.add_argument('--min_step_magnitude', type=float, nargs=1,
#                    default=[0.0], help=min_step_magnitude_helpstring)
#min_step_noise_ratio_helpstring = \
#    ("Minimum ratio of step size to noise allowed.")
#parser.add_argument('--min_step_noise_ratio', type=float, nargs=1,
#                    default=[0.0], help=min_step_noise_ratio_helpstring)
photometry_minimum_helpstring = \
    ("If given, will round all photometries below this value up to it, e.g. "
     "if it is 0, then all photometries below 0 will become 0.")
parser.add_argument('--photometry_minimum', type=float, nargs=1,
                    default=[None], help=photometry_minimum_helpstring)
#method_helpstring = ("Method to use for stepfitting. Available are 't_test' "
#                     "and 'chi_squared'")
#parser.add_argument('--method', nargs=1, default=['chi_squared'],
#                    help=method_helpstring)
#remove_blips_helpstring = ("Remove temporary upward blips in fitted steps.")
#parser.add_argument('--remove_blips', action='store_true',
#                    help=remove_blips_helpstring, default=False)
#smoothing_stddev_helpstring = \
#    ("If removing blips, use this smoothing coefficient.")
#parser.add_argument('--smoothing_stddev', type=float, nargs=1, default=[0.8],
#                    help=smoothing_stddev_helpstring)
p_threshold_helpstring = \
    ("Use this p_threshold for the t-test to decide if a step exists.")
parser.add_argument('--p_threshold', type=float, nargs=1, default=[0.01],
                    help=p_threshold_helpstring)
linear_fit_threshold_helpstring = \
    ("If linear_fit_r_2 / step_fit_r_2 is above this value, discard the "
     "trace.")
parser.add_argument('--linear_fit_threshold', type=float, nargs=1,
                    default=[1.0], help=linear_fit_threshold_helpstring)
#downsteps_only_helpstring = \
#    ("If using t_test, fit downsteps only.")
#parser.add_argument('--downsteps_only', action='store_true',
#                    help=downsteps_only_helpstring, default=False)
chung_kennedy_helpstring = \
    ("Number of times to apply Chung-Kennedy filter to photometries prior to "
     "fitting steps.")
parser.add_argument('--chung_kennedy', type=int, nargs=1,
                    default=[0], help=chung_kennedy_helpstring)
#min_step_length = \
#    ("Minimum step length in frames for chi_squared.")
#parser.add_argument('--min_step_length', type=int, nargs=1, default=[2],
#                    help=min_step_length)
#window_radius_helpstring = ("Radius of window to use for t-test.")
#parser.add_argument('--window_radius', type=int, nargs=1, default=[5],
#                    help=window_radius_helpstring)
#median_filter_helpstring = \
#    ("Median filter kernel size to use. If 0, not applied.")
#parser.add_argument('--median_filter', type=int, nargs=1, default=[0],
#                    help=median_filter_helpstring)
#double_t_helpstring = \
#    ("Use a second ttest after applying Chung-Kennedy filter to filter for "
#     "final steps. Use this value for the p_threshold. If 0, no steps will be "
#     "filtered.")
#parser.add_argument('--double_t', type=float, nargs=1, default=[1.0],
#                    help=double_t_helpstring)
#num_steps_helpstring = ("Max number of steps to fit when doing chi-squared.")
#parser.add_argument('--num_steps', type=int, nargs=1, default=[10],
#                    help=num_steps_helpstring)
#magic_start_helpstring = \
#    ("Number of first frames to use for magic start (median).")
#parser.add_argument('--magic_start', type=int, nargs=1, default=[0],
#                    help=magic_start_helpstring)
mirror_start_helpstring = ("Number of first frames to mirror.")
parser.add_argument('--mirror_start', type=int, nargs=1, default=[0],
                    help=mirror_start_helpstring)
parser.add_argument('timetrace_frames', nargs='+', type=str,
                    help=timetrace_frames_helpstring)
args = parser.parse_args()


#set up logging
if args.debug:
    logging.basicConfig(filename=args.log_path[0], level=logging.DEBUG)
else:
    logging.basicConfig(filename=args.log_path[0], level=logging.INFO)
logger = logging.getLogger()
logger.info("basic_timetrace_script starting at " + str(timestamp_datetime))
logger.info("args = " + str(args))


#ensure absolute paths
timetrace_frames = [os.path.abspath(f) for f in args.timetrace_frames]


#Make sure the output directory exists.
try:
    os.makedirs(args.output_directory[0])
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise


#load frames as Numpy arrays and convert into Image objects
timetrace_image_arrays = [pflib.read_image(f) for f in timetrace_frames]
frame_Images = [flexlibrary.Image(image=timetrace_image_arrays[f][1],
                                  metadata={'filepath': frame},
                                  spots=None)
                for f, frame in enumerate(timetrace_frames)]


#get psfs for initial frame
initial_frame_psfs_pkls = glob.glob(timetrace_frames[0] + '*_psfs_*.pkl')
if len(initial_frame_psfs_pkls) == 0:
    logger.info("Could not find PSF pkl files for " + timetrace_frames[0] +
                "; it will be submitted to pflib.")
    processed_initial_frame = \
                   pflib.parallel_image_batch(image_paths=timetrace_frames[:1],
                                              find_peptides_parameters=None,
                                              timestamp_epoch=timestamp_epoch)
    (converted_path,
     psfs_pkl_path,
     psfs_csv_path,
     psfs_png_path) = processed_initial_frame[timetrace_frames[0]]
else:
    psfs_pkl_path = initial_frame_psfs_pkls[0]
initial_psfs = cPickle.load(open(psfs_pkl_path))
initial_Spots = [flexlibrary.Spot(parent_Image=frame_Images[0],
                                  h=h_0, w=w_0,
                                  size=gaussian_fit[7].shape[0],
                                  gaussian_fit=gaussian_fit)
                 for (h_0, w_0), gaussian_fit in initial_psfs.iteritems()]


#add Spots to initial frame
frame_Images[0].spots = initial_Spots



#what we're all here for
#spot_traces = flexlibrary.Experiment.luminosity_centroid_particle_tracking(
#    frames=frame_Images,
#    initial_spots=initial_Spots,
#    search_radius=3,
#    s_n_cutoff=3.0,
#    offsets=None)


#frame_spots = [initial_Spots] + [None] * (len(frame_Images) - 1)
#tte = flexlibrary.TimetraceExperiment(frames=frame_Images,
#                                      frame_spots=frame_spots,
#                                      spot_traces=None)

tte = flexlibrary.TimetraceExperiment(frames=frame_Images, spot_traces=None,
                                      step_fits=None,
                                      step_fit_intermediates=None)

tte.lc_create_traces()
#make sanity check images
if not args.no_sanity_check_images:
    tte.wildcolor_plot_tracks(filepath_prefix=
                              os.path.join(args.output_directory[0], 'test_'))

#make csvs
#tte.save_stepfits_as_csv(output_path=os.path.join(args.output_directory[0],
#                                                  'test.csv'),
#                         min_step_magnitude=args.min_step_magnitude[0],
#                         method=args.method[0],
#                         photometry_min=args.photometry_minimum[0],
#                         remove_blips=args.remove_blips,
#                         chung_kennedy=args.chung_kennedy[0],
#                         smoothing_stddev=args.smoothing_stddev[0],
#                         downsteps_only=args.downsteps_only,
#                         p_threshold=args.p_threshold[0],
#                         min_step_noise_ratio=args.min_step_noise_ratio[0],
#                         double_t=args.double_t[0],
#                         linear_fit_threshold=args.linear_fit_threshold[0],
#                         min_step_length=args.min_step_length[0],
#                         median_filter=args.median_filter[0],
#                         num_steps=args.num_steps[0],
#                         magic_start=args.magic_start[0],
#                         mirror_start=args.mirror_start[0])

#get photometry arguments for saving photometries
if args.photometry_parameters[0] is not None:
    p_params = ast.literal_eval(args.photometry_parameters[0])
else:
    if args.sextractor:
        p_params = {'photometry_method': 'sextractor'}
    else:
        p_params = {}

step_fits, step_fit_intermediates = tte.stepfit_tracks(
                                     photometry_min=args.photometry_minimum[0],
                                     mirror_start=args.mirror_start[0],
                                     chung_kennedy=args.chung_kennedy[0],
                                     p_threshold=args.p_threshold[0],
                                     **p_params)

pkl_filepath = os.path.join(args.output_directory[0], 'test.pkl')
cPickle.dump((step_fits, step_fit_intermediates), open(pkl_filepath, 'w'))
csv_filepath = os.path.join(args.output_directory[0], 'test.csv')
tte.save_experiment_as_csv(output_path=csv_filepath, include_step_fits=True,
                           include_intermediates=True, **p_params)
if args.save_traces_pkl:
    tte.save_traces_pkl(path=os.path.join(args.output_directory[0],
                                          'traces.pkl'))
