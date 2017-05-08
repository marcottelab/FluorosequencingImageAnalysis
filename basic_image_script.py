#!/home/boulgakov/anaconda2/bin/python

"""
Find fluorescent spots in all images.

Will traverse all target_directories and process all found *.tif files through
pflib.parallel_image_batch. For each image, a png version will be created if it
is not found.

For each image, spot finding results will be output in three files: a Python
pickle file, a png file, and a csv file. See pflib.py's save_psfs_pkl,
save_psfs_png, and save_psfs_csv documentation for detailed description of each
file output.

See pflib.py for more documentation about the algorithms used.
"""

import pflib
import logging
import os
import os.path
import datetime
import argparse
import time
import ast


#get timestamp that will be used throughout this run
timestamp_epoch = time.time()
timestamp_datetime = datetime.datetime.fromtimestamp(timestamp_epoch)

#define and parse arguments; use custom MyFormatter to do both ArgumentDefault
#and RawDescription Formatters via multiple inheritence, this is a trick to
#preserve docstring formatting in --help output
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter,
                  argparse.RawDescriptionHelpFormatter):
    pass
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=MyFormatter)

#old parser, now using MyFormatter above
#script_description_string = ("Will traverse all target_directories and "
#                             "process all found *.tif files through "
#                             "pflib.parallel_image_batch.")
##parser = argparse.ArgumentParser(description=script_description_string)

parameters_help_string = ("Parameters for pflib's find_peptides function. "
                          "Expects a Python dictionary in quotes. Example: "
                          "--parameters=\"{'median_filter_size': 6, "
                          "'c_std': 3}\" . These parameters will override "
                          "pflib.find_peptides defaults; anything not "
                          "specified will not be affected.")
parser.add_argument('--parameters', type=str, nargs=1, default=[None],
                    help=parameters_help_string)
monte_carlo_help_string = \
    ("Use Monte Carlo method to peakfit (instead of default Levernbert-"
     "Marquardt Gaussian). Note that if 'fit_type' is passed as a parameter "
     "to pflib.find_peptides, it overrides this option.")
parser.add_argument('-mc', '--monte_carlo', action='store_true', default=False,
                    help=monte_carlo_help_string)
N_iter_help_string = ("Number of samples to use if using --monte_carlo.")
parser.add_argument('--N_iter', type=int, nargs=1, default=[10**3],
                    help=N_iter_help_string)
num_processes_help_string = ("Number of processes to use. Default defined by "
                             "pflib.parallel_image_batch.")
parser.add_argument('-n', '--num_processes', type=int, nargs=1, default=[None],
                    help=num_processes_help_string)
default_log_directory = '/home'
default_log_filename = 'basic_image_script_' + str(timestamp_datetime) + '.log'
log_path_help_string = \
                   ("Pathname for log. Default is "
                    "/project/boulgakov/microscope/pf_log_files/"
                    "basic_image_script_ + str(datetime.datetime.now()) + .log"
                    ". If the log file already exists, further logging output "
                    "follows default Python behavior, which is currently to "
                    "append the file.")
default_log_path = os.path.join(default_log_directory, default_log_filename)
parser.add_argument('-L', '--log_path', nargs=1, default=[default_log_path],
                    help=log_path_help_string)
target_directories_help_string = ("Directories to process. At least one must "
                                  "be specified.")
parser.add_argument('target_directories', nargs='+',
                    help=target_directories_help_string)
args = parser.parse_args()

#normalize all target directories to absolute paths
target_directories = [os.path.abspath(d) for d in args.target_directories]

#setup logging for debug and error statements
logging.basicConfig(filename=args.log_path[0], level=logging.DEBUG)
logger = logging.getLogger()
logger.info("basic_image_script starting at " + str(timestamp_datetime))

#parse find_peptides parameter dictionary if given
if args.parameters[0] is not None:
    fp_parameters = ast.literal_eval(args.parameters[0])
else:
    fp_parameters = None

#Insert fit type into fp_parameters dictionary.
if args.monte_carlo:
    if fp_parameters is None:
        fp_parameters = {}
    fp_parameters.setdefault('fit_type', 'monte_carlo')
    fp_parameters.setdefault('N_iter', args.N_iter[0])

#find all tif files in directories
target_images = []
for target_dir in target_directories:
    for root, subfolders, files in os.walk(target_dir):
        for f in files:
            if f[-4:] == '.tif':
                target_images.append(os.path.join(root, f))

#confirm to log what files will be processed
logger.info("Scanned target directories\n"  + '\n'.join(target_directories))
logger.info("Will process target images\n" + '\n'.join(target_images))

#this does all the work!
processed_images = \
             pflib.parallel_image_batch(target_images,
                                        find_peptides_parameters=fp_parameters,
                                        timestamp_epoch=timestamp_epoch,
                                        num_processes=args.num_processes[0])

#summarize current run
logger.info("Pathnames of images processed: " +
            str('\n'.join(processed_images.keys())))
logger.info("basic_image_scipt finished at " + str(datetime.datetime.now()))
