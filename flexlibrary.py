#!/home/boulgakov/anaconda2/bin/python


"""
Fluorosequencing Experiment Library (flexlibrary) manages fluorosequencing
image data, and facilitates combining this data to extract conclusions from
experiments.

flexlibrary implements three core classes: Image, Spot, and Experiment.

Image is the fundamental class in flexlibrary. Each fluorosequencing image is
an Image object. This object contains the image itself, and associated metadata
-- such as filepath, date, experimental conditions, etc.

Fluorosequencing images are analyzed to find and analyze labeled peptides.
Labeled peptides are observed as point sources under the microscope, each
observable as a spot with a characteristic point spread function (PSF). Each
spot is a Spot object identified by its source image, and its spatial
coordinates (h, w) in that image.

Information about the state of a peptide is to be inferred by analyzing its
spot's PSF. Comparing a peptide's PSFs across a series of images may yield
further information about its state, or -- if the images are temporally
separated -- changes in its state.

Defining a relationship between a set of images inherently defines a
relationship between co-localized spots in those images. For example: if ten
images are snapshots of a single field, each taken after a successive Edman
reaction, then the co-localized spots in those images are the same labeled
peptides observed through the experiment. Another example: if three images are
three color channels of the same field taken at one time, then the co-localized
spots reflect the state of orthogonal labels on each peptide. Such a
relationship is defined by an Experiment.

The Experiment class is a parent class from which various Experiment subclasses
are derived. Each subclass represents a specific, predefined relationship
between a set of images. For example, there is a TimetraceExperiment that is
used to look at continuous time traces of a field. Each subclass of Experiment
comes with its own useful methods that perform analysis on the Experiment's set
of Images and their associated Spots.
"""


import sys
sys.path.insert(0, '/home/proteanseq/pflib')
import pflib
from scipy.misc import imread
import os.path
import tempfile
import os
import numpy as np
from scipy.spatial.distance import euclidean
import glob
import cPickle
from phase_correlate import phase_correlate
import math
import time
import logging
import random
import csv
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import medfilt
import photutils
import multiprocessing
import stepfitting_library


#if the calling application does not have logging setup, flexlibrary will log
#to NullHandler by default
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Spot(object):
    """
    A Spot is a square of pixels in an image. It may contain a luminescent spot
    caused by a point source.

    The square must be an odd number of pixels on each side in size; there must
    be a unique center pixel.

    A Spot object stores only the coordinates and size of its square. To access
    the actual pixels, use Spot.image_slice().

    Attributes:
        parent_Image: Image object this Spot exists inside of.
        h: The horizontal coordinate of the center pixel of the square area.
        w: The vertical coordinate of the center pixel of the square area.
        size: Length of Spot square side in pixels.
        gaussian_fit: Tuple characterizing the Gaussian fit of the Spot's PSF.
            The tuple definition is the same as that of the tuples stored as
            values in the dictionary by returned pflib.find_peptides.
            gaussian_fit can remain None. For convenience, here is the tuple
            order from pflib.find_peptides: (h_0, w_0, H, A, sigma_h, sigma_w,
            theta, sub_img, fit_img, rmse, r_2, s_n). Note that sub_img is not
            dynamically updated if parent_Image changes.
    """
    def __init__(self, parent_Image, h, w, size, gaussian_fit=None):
        self.parent_Image = parent_Image
        if size % 2 == 0:
            raise AttributeError("Spot.size must be odd.")
        self.size = size
        if not (0 <= h  - (size - 1) / 2 and
                h + (size - 1) / 2 < parent_Image.image.shape[0] and
                0 <= w  - (size - 1) / 2 and
                w + (size - 1) / 2 < parent_Image.image.shape[1]):
            if (gaussian_fit is None or not
                ((size - 1) / 2 <=
                  gaussian_fit[0] <
                  parent_Image.image.shape[0] - (size - 1) / 2) and
                ((size - 1) / 2 <=
                  gaussian_fit[1] <
                  parent_Image.image.shape[1] - (size - 1) / 2)):
                raise AttributeError("Spot area of size " + str(size) +
                                     " at " + str((h, w)) + " with " +
                                     "gaussian_fit " + str(gaussian_fit) +
                                     " does not fit into " +
                                     "parent_Image.image.shape of " +
                                     str(parent_Image.image.shape))
        self.h, self.w = h, w
        self.gaussian_fit = gaussian_fit

    def image_slice(self, radius=None):
        """
        Get the Spot's square of pixels. This is sliced from parent_Image at
        the time the method is called; changes in parent_Image will reflect
        themselves in the returned slice.

        Optionally, this function can also return a larger slice of
        parent_Image. However, the function does not guarantee that a square of
        the desired size will be returned because it is possible that the
        desired slice exceeds image bounds.

        Arguments:
            radius: If not None, returns a square slice of the desired radius.

        Returns:
            Spot's square of pixels as a Numpy array.
        """
        if radius is None:
            radius = (self.size - 1) / 2
        return self.parent_Image.image[max(0, self.h - radius):
                                       min(self.parent_Image.image.shape[0],
                                           self.h + radius + 1),
                                       max(0, self.w - radius):
                                       min(self.parent_Image.image.shape[1],
                                           self.w + radius + 1)]

    def valid_slice(self, radius=None):
        """Is the slice of requested radius contained within parent image."""
        if radius is None:
            radius = (self.size - 1) / 2
        wanted_slice_size = 2 * radius + 1
        slice = self.image_slice(radius=radius)
        if slice.shape[0] == slice.shape[1] == wanted_slice_size:
            return True
        else:
            return False

    def simple_photometry_metric(self, return_invalid=True):
        """
        Photometry is measurement of the electromagnetic flux of an object.

        Returns:
            The sum of all pixel values in this Spot.
        """
        if not return_invalid and not self.valid_slice():
            return None
        else:
            return np.sum(self.image_slice())

    def mexican_hat_photometry_metric(self, brim_size=6, radius=9,
                                      return_invalid=True):
        """
        Similar to simple_photometry_metric, but adjusts for background
        intensity around a peptide.

        Arguments:
            brim_size: Size of brim to use as background adjustment.
            radius: Total radius of square to use as the hat. If this remains
                None, (self.size - 1) / 2 is used as the natural radius. If the
                radius is overridden to be greater than self.size, this
                function does not guarantee that all pixels within that radius
                are within the image boundary, and hence the area used for
                photometry may be truncated.
        """
        logger = logging.getLogger()
        logger.debug("Spot.mexican_hat_photometry_metric: locals() = " +
                     str(locals()))
        if radius is None:
            radius = (self.size - 1) / 2
        if not return_invalid and not self.valid_slice(radius=radius):
            photometry = None
        else:
            diameter = 2 * radius + 1
            crown_pixels = []
            brim_pixels = []
            for (h, w), p in np.ndenumerate(self.image_slice(radius=radius)):
                if (brim_size <= h < diameter - brim_size and
                    brim_size <= w < diameter - brim_size):
                    crown_pixels.append(p)
                else:
                    brim_pixels.append(p)
            logger.debug("Spot.mexican_hat_photometry_metric: "
                         "crown_pixels = " + str(crown_pixels))
            logger.debug("Spot.mexican_hat_photometry_metric: brim_pixels = " +
                         str(brim_pixels))
            photometry = (sum(crown_pixels) -
                          len(crown_pixels) * np.median(brim_pixels))
        return photometry

    def gaussian_volume_photometry_metric(self, scaling=10**6, default=0,
                                          return_invalid=True):
        """
        An alternative photometry metric based on the gaussian_fit.

        Returns:
            The volume of the two dimensional gaussian. In pflib.find_peptides
            terms: A * sigma_h * sigma_w
        """
        if not return_invalid and not self.valid_slice():
            photometry = None
        elif self.gaussian_fit is None:
            #raise AttributeError("Cannot give gaussian_volume_metric without "
            #                     "a self.gaussian_fit")
            photometry = default
        else:
            photometry = (float(scaling) * self.gaussian_fit[3] *
                          self.gaussian_fit[4] * self.gaussian_fit[5])
        return photometry

    def gaussian_sigmas_photometry_metric(self, scaling=10**6,
                                          return_invalid=True):
        if not return_invalid and not self.valid_slice():
            photometry = None
        if self.gaussian_fit is None:
            photometry = -10**9
        else:
            photometry = (float(scaling) * self.gaussian_fit[4] *
                          self.gaussian_fit[5])
        return photometry

    def sextractor_photometry_metric(self, radius=3, box_size=10,
                                     filter_size=10, return_invalid=True,
                                     **kwargs):
        """
        Photutils' sextractor photometry.
        """
        if not return_invalid and not self.valid_slice(radius=radius):
            photometry = None
        else:
            background = self.parent_Image.get_photometry_background(
                                                       box_size=box_size,
                                                       filter_size=filter_size,
                                                       method='sextractor')
            aperture = self.parent_Image.get_photometry_aperture(spot=self,
                                                                 radius=radius)
            background_array = background.background
            photometry = float(photutils.aperture_photometry(
                                   self.parent_Image.image - background_array,
                                                     aperture)['aperture_sum'])
        return photometry

    def maximum_photometry_metric(self, radius=5, top=1,
                                  background_adjust='none',
                                  return_invalid=True):
        if not return_invalid and not self.valid_slice(radius=radius):
            photometry = None
        else:
            r = np.sort(np.ravel(self.image_slice(radius=radius)))
            if background_adjust == 'none':
                pass
            elif background_adjust == 'additive':
                median = np.median(self.parent_Image.image)
                r = r - median
            elif background_adjust == 'multiplicative':
                raise NotImplementedError("Not sure what to do if median is 0."
                                          " This may be a poor metric.")
                median = np.median(self.parent_Image.image)
            else:
                raise ValueError(str(background_adjust) + " is not a valid "
                                 "option.")
            photometry = float(np.sum(r[-top:]))
        return photometry

    def photometry(self, method='mexican_hat', photometry_method=None,
                   return_invalid=True, **kwargs):
        """
        If return_invalid is True, return photometry even if it cannot be
        calculated correctly. If return_invalid is False, return None instead
        of photometry.
        """
        logger = logging.getLogger()
        if photometry_method is not None:
            method = photometry_method
        if method == 'mexican_hat':
            photometry = self.mexican_hat_photometry_metric(
                                       return_invalid=return_invalid, **kwargs)
        elif method == 'gaussian_volume':
            photometry = self.gaussian_volume_photometry_metric(
                                       return_invalid=return_invalid, **kwargs)
        elif method == 'simple':
            photometry = self.simple_photometry_metric(
                                       return_invalid=return_invalid, **kwargs)
        elif method == 'sextractor':
            photometry = self.sextractor_photometry_metric(
                                       return_invalid=return_invalid, **kwargs)
        elif method == 'maximum':
            photometry = self.maximum_photometry_metric(
                                       return_invalid=return_invalid, **kwargs)
        elif method == 'sigmas':
            photometry  = self.gaussian_sigmas_photometry_metric(
                                       return_invalid=return_invalid, **kwargs)
        else:
            raise ValueError("Uknown method specified.")
        logger.debug("Spot.photometry_metric: photometry = " + str(photometry))
        return photometry

    def illumina_s_n(self):
        return pflib.illumina_s_n(self.image_slice())


class Image(object):
    """
    A fluorosequencing image, its metadata, and its Spots.

    The image is represented as a two-dimensional Numpy array, with dimensions
    height x width. Pixel coordinates are indexed by positive integer pairs
    [h, w]. The coordinate system origin is in the upper left-hand corner, is
    0-indexed, and with h(eight) and w(idth) the vertical and horizontal
    indeces from the origin respectively. For subpixel resolution, the
    coordinate system is extended to floating point pairs [h, w]. Pixel values
    are unsigned integers, with black (i.e. no photons detected) assigned 0.
    Bit depth is assumed to be at most 64 bits.

    Image metadata is a dictionary that can contain arbitrary key:value pairs
    as needed to document the image. Image does not inherently enforce any
    rules about what this metadata can be, except for reserved keys as
    follows:
        1. 'filepath' keyword: value indicates where the image is stored as an
            image file. Currently, only those files that ImageMagick 'convert'
            can convert into PNG are supported. Files with a '.png' or '.PNG'
            suffix are assumed to be PNG files that can be read directly,
            otherwise conversion is attempted. The intermediate PNG is not
            saved. Absolute paths are strongly recommended; use relative paths
            at own risk, as flexlibrary functions will write to filenames based
            on this metadata entry!

    Spots in the image are stored as a list of Spot objects; see their class
    docstring for further documentation.

    Attributes:
        image: Numpy array representation of the image. Can be passed directly
            at instatiation. If not passed, 'filepath' must be passed in
            metadata and a valid image be read from it. If both image and the
            'filepath' metadatum are passed, image is taken as the matrix. No
            safety check is performed to ensure their image array matches.
        metadata: Dictionary containing image metadata. Can have any content
            valid in Python. Reserved keywords as described above are assumed
            to be used as indicated.
        spots: List of Spot instances. See their class documentation. Can be
            passed at initialization or added later, either by appending the
            list or calling Spot-generating functions for an Image.
        photometry_background: A dictionary of photutils.Background objects
            used to cache this image's background information. They are
            computed once to allow rapid photutils-based photometry
            computations for each Spot. Stored as a dictionary to allow caching
            of backgrounds computed based on different parameters. The three
            parameters are: (box_size, filter_size, method). They are passed to
            photutils.background.Background(box_shape=(box_size, box_size),
                                            filter_shape=(filter_size,
                                                          filter_size)
                                            method=method).
            Tuples of parameters as above are the keys in this dictionary. The
            resulting Background objects are the values.
        photometry_apertures: Dictionary of lists of
            photutils.CircularApertures, one for each Spot.
            photometry_apertures is designed to memoize apertures of different
            radii. Each key in the dictionary is a radius. Its corresponding
            value is a list of photutils.CircularApertures, each of that
            radius. The order of apertures corresponds to self.spots, such that
            the i'th member of the list is the aperture for the i'th spot in
            self.spots. This is computed once to allow rapid photutils-based
            photometry computations for each Spot.
    """
    def __init__(self, image=None, metadata=None, spots=None,
                 photometry_background=None, photometry_apertures=None):
        self.metadata = {}
        if metadata is not None:
            self.metadata = metadata
        self.image = None
        if image is not None:
            self.image = image
        elif 'filepath' in metadata:
            try:
                tmppath = None #used to clean up tempfiles in finally block
                if os.path.splitext(metadata['filepath'])[1] in ('png', 'PNG'):
                    self.image = imread(metadata['filepath'])
                else:
                    tmpfile, tmppath = tempfile.mkstemp()
                    tmpfile.close() #close before ImageMagick writes to it
                    pflib.convert_image(metadata['filepath'],
                                        output_path=tmppath)
                    self.image = imread(tmppath)
            finally:
                #make sure tmppath is cleaned up regardless whether exception
                #is raised or not
                if os.path.isfile(tmppath):
                    os.remove(tmppath)
        else:
            raise AttributeError("Image.image must be defined: it was neither "
                                 "passed at initialization nor given a "
                                 "filepath to be read from.")
        self.spots = []
        if spots is not None:
            self.spots = spots
        if photometry_background is not None:
            self.photometry_background = photometry_background
        else:
            self.photometry_background = {}
        if photometry_apertures is not None:
            self.photometry_apertures = photometry_apertures
        else:
            self.photometry_apertures = {}

    def find_gaussian_psfs(self, pflib_args=None, spots_append=True):
        """
        Apply pflib.find_peptides to self.image.

        Arguments:
            pflib_args: Arguments to be passed to pflib.find_peptides. Useful
                to modify its default search parameters.
            spots_append: If True, will append any found PSFs as Spots to
                self.spots. Does not check if existing Spots are identical. If
                not True, will overwrite self.spots to contain only those Spots
                found by this method call.

        Returns:
            Number of Spots found by pflib.find_peptides during this run. If
            spots_append is True, this means that number of members at the end
            of self.spots were appended.
        """
        #using empty dictionaries/lists as a default argument is a bug factory
        #because if they are modified, these modifications persist as the
        #defaults for all subsequent calls. Better to use None and case it.
        if pflib_args is None:
            pflib_args = {}
        new_fits = pflib.find_peptides(self.image, **pflib_args)
        if not spots_append:
            self.spots = []
        for (h, w), new_fit in new_fits.iteritems():
            int_h, int_w = int(round(h)), int(round(w))
            spot = Spot(self, int_h, int_w, 5, gaussian_fit=new_fit)
            self.spots.append(spot)
        return len(new_fits)

    def get_photometry_background(self, box_size=10, filter_size=10,
                                  method='sextractor'):
        """Returns self.photometry_background. Computes it if it's None."""
        if (box_size, filter_size, method) not in self.photometry_background:
            box_shape = (box_size, box_size)
            filter_shape = (filter_size, filter_size)
            bkg = photutils.background.Background(data=self.image,
                                                  box_shape=box_shape,
                                                  filter_shape=filter_shape,
                                                  method=method)
            self.photometry_background.setdefault((box_size,
                                                   filter_size,
                                                   method), bkg)
        return self.photometry_background[(box_size, filter_size, method)]

    def get_photometry_aperture(self, spot, radius=3):
        """
        Returns aperture corresponding to spot. Creates all apertures for the
        requested radius if they've not yet been made.
        """
        if radius not in self.photometry_apertures:
            #Remember in photutils the coordinates are swapped.
            self.photometry_apertures.setdefault(radius,
                              [photutils.CircularAperture([s.w, s.h], r=radius)
                               for s in self.spots])
        elif len(self.photometry_apertures[radius]) != len(self.spots):
            #Must re-compute if self.spots has been changed.
            self.photometry_apertures[radius] = \
                              [photutils.CircularAperture([s.w, s.h], r=radius)
                               for s in self.spots]
        return_aperture = None
        for i, s in enumerate(self.spots):
            if spot is s:
                return_aperture = self.photometry_apertures[radius][i]
                break
        else:
            raise Exception("argument spot not in self.spots; locals() = " +
                            str(locals()))
        assert return_aperture is not None
        return return_aperture

    def append_photometry_apertures(self):
        """
        A Spot was just appended to self.spots. Append to all radius entries in
        self.photometry_aptertures to match.
        """
        h, w = self.spots[-1].h, self.spots[-1].w
        for radius, apertures in self.photometry_apertures.iteritems():
            apertures.append(photutils.CircularAperture([w, h], r=radius))
        return self.photometry_apertures


class Experiment(object):
    """
    The base class for all Experiments. It does not define any kind of
    experiment itself, and hence is never instantiated. It is useful as a
    repository of methods useful for all Experiment subclasses.
    """
    @staticmethod
    def easy_load_processed_image(image_filepath, psf_pkl_filepath=None,
                                  load_psfs=True):
        """
        Utility function to load a pre-processed image and its PSF pkl files as
        produced by pflib into Image and Spot objects.

        Arguments:
            image_filepath: Path to PNG image file.
            psf_pkl_filepath: Path to PSF pkl file for the image, as produced
                by pflib. If None, follows pflib._psfs_filename and uses glob
                to try to find the pkl at image_filepath + '_psfs_*.pkl'.
            load_psfs: If False, does not load PSF pkl files. The resulting
                Image objects have no Spots.

        Returns:
            Creates and returns the Image object with all its Spot objects to
            describe the image and the PSFs contained in the pkl. Also returns
            number of Spots not loaded from the pkl file due to errors.
        """
        image = imread(image_filepath)
        image_object = Image(image=image,
                             metadata={'filepath': image_filepath}, spots=None)
        discarded_spots = 0
        if load_psfs:
            if psf_pkl_filepath is None:
                pkl_files = sorted(glob.glob(image_filepath + '*_psfs_*.pkl'))
                if len(pkl_files) == 0:
                    raise ValueError("For image_filepath = " + image_filepath +
                                     " psf_pkl_filepath passed as None when " +
                                     "no pkl files available.")
                psf_pkl_filepath = pkl_files[-1]
            psfs = cPickle.load(open(psf_pkl_filepath))
            spot_objects = []
            for (h, w), gaussian_fit in psfs.iteritems():
                (h_0, w_0, H, A, sigma_h, sigma_w, theta, sub_img, fit_img,
                 rmse, r_2, s_n) = gaussian_fit
                try:
                    int_h, int_w = int(round(h)), int(round(w))
                    new_spot = Spot(parent_Image=image_object,
                                    h=int_h, w=int_w,
                              size=fit_img.shape[0], gaussian_fit=gaussian_fit)
                    spot_objects.append(new_spot)
                except Exception as e:
                    logger.info("flexlibrary.easy_load_processed_image: "
                               "Ignoring Spot due to Spot.__init__ exception.")
                    logger.exception(e, exc_info=True)
                    discarded_spots += 1
            image_object.spots = spot_objects
        return image_object, discarded_spots

    @staticmethod
    def accumulate_offsets(offsets):
        """
        Given a sequence of image offsets with respect to the directly
        preceding image, convert to offsets with respect to the first frame.

        Arguments:
            offsets: List of offsets of successive images, each with respect
                to the directly preceeding one.

        Returns:
            List of offsets for those with respect to the first image.
        """
        logger = logging.getLogger()
        if offsets[0] != (0, 0):
            raise ValueError("The first image's offset must be (0, 0) by "
                             "definiton.")
        cumulative_offsets = []
        for f, offset in enumerate(offsets):
            cumulative_h_offset = sum([offset[0]
                                       for offset in offsets[:f + 1]])
            cumulative_w_offset = sum([offset[1]
                                       for offset in offsets[:f + 1]])
            cumulative_offsets.append((cumulative_h_offset,
                                       cumulative_w_offset))
        logger.debug("accumulate_offsets: offsets = " + str(offsets))
        logger.debug("accumulate_offsets: cumulative_offsets = " +
                     str(cumulative_offsets))
        return cumulative_offsets

    @staticmethod
    def get_cumulative_offset(offsets, f, g=0):
        """Get cumulative offset for frame f, with respect to frame g."""
        cf = Experiment.accumulate_offsets(offsets)[f]
        cg = Experiment.accumulate_offsets(offsets)[g]
        return (cf[0] - cg[0], cf[1] - cg[1])

    @staticmethod
    def round_coordinates(h, w):
        return int(round(h)), int(round(w))

    @staticmethod
    def apply_offset(coordinates, offset):
        h, w = coordinates
        d_h, d_w = offset
        return h + d_h, w + d_w

    @staticmethod
    def unapply_offset(offset_coordinates, offset):
        h, w = offset_coordinates
        d_h, d_w = offset
        return h - d_h, w - d_w

    @staticmethod
    def offset_frame_coordinates(offsets, coordinate, f, g):
        """Given coordinate in frame g, find its coordinate in frame f."""
        gf_offset = Experiment.get_cumulative_offset(offsets=offsets, f=f, g=g)
        return Experiment.apply_offset(coordinate, gf_offset)

    @staticmethod
    def discard_dropouts(spots, spot_cumulative_offsets,
                         frame_cumulative_offsets, image_shape, spot_radius=0):
        """
        Discard Spots that align to outside of a sequence of frames.

        When tracking spots through frames, stage drift may place some of them
        (or their former position, if they are no longer visible) outside of
        the field of view at some point. We want to ignore them. This function
        filters out spots whose position would end up outside of the field of
        view of a sequence of frames. Because some functions use coordinates
        rounded to the nearest whole number, discard_dropouts also eliminates
        any points whose offset coordinates would round up past the size of the
        image (i.e. its upper cutoff is 0.5 below the image boundary).

        Arguments:
            psfs: List of Spot objects.
            spot_cumulative_offsets: List of offsets corresponding to each Spot's
                frame, with respect to the primary field of view.
            frame_cumulative_offsets: List of cumulative offsets for all frames
                through which the Spot's position will be monitored.
            image_shape: Shape of the frame's Numpy array.
            spot_radius: Discard spots within this distance of the edge.

        Returns:
            The original list of Spots, with those that would be out of frame
                at any point removed, and the number of Spots discarded.
        """
        logger = logging.getLogger()
        filtered_spots = []
        number_discarded = 0
        for i, spot in enumerate(spots):
            #oh, ow are the coordinates of the spot in frame 0's coordinate sys
            oh, ow = Experiment.apply_offset((spot.h, spot.w),
                                             spot_cumulative_offsets[i])
            logger.debug("discard_dropouts: oh, ow = " + str((oh, ow)))
            #track spot position through each frame, check to see if it is ever
            #outside the field of view
            for offset in frame_cumulative_offsets:
                gh, gw = Experiment.unapply_offset((oh, ow), offset)
                logger.debug("discard_dropouts: gh, gw = " + str((gh, gw)))
                if not (spot_radius <= gh <
                                           image_shape[0] - 0.5 - spot_radius
                        and
                        spot_radius <= gw <
                                           image_shape[1] - 0.5 - spot_radius):
                    number_discarded += 1
                    logger.debug("discard_dropouts: discarded")
                    break
            else:
                logger.debug("discard_dropouts: NOT discarded")
                filtered_spots.append(spot)
        return filtered_spots, number_discarded

    @staticmethod
    def greedy_particle_tracking(frame_spots, frame_shape, candidate_radius=2,
                                 offsets=None, spot_radius=0):
        """
        Track Spots across frames.

        One of the basic problems flexlibrary needs to solve is associating
        spots that are co-localized across frames. Co-localization across
        frames implies persistence through time and/or across multiple
        channels. Hence, this is essentially the act of tracking spots.

        Tracking spots across frames is a particle tracking problem under the
        simplifying condition that the particles cannot move further than a
        very small distance. Therefore, a very simple and very greedy algorithm
        should be sufficient for our needs.

        Each nonempty frame has a set of Spots, each uniquely defined by their
        (h, w) coordinates. For this algorithm, all that matters are the Spots'
        coordinates.

        For each Spot, the goal of the algorithm is to associate it to a Spot
        in the subsequent frame, such that it is plausible that it is the same
        particle. Two complicating factors arise: (1) a Spot may not be present
        in a succeeding frame, and (2) multiple Spots in the first frame may be
        potentially assigned to one Spot in the succeeding frame.

        The algorithm approaches this task as follows, comparing two
        consecutive frames at a time:
            1. For each Spot, all Spots in the following frame that are within
                a given radius are considered potential candidates. For
                convenience, we call Spots in the first frame "ancestors", and
                Spots in the second frame "descendants".
            2. For each pair of ancestor-descendant, Eucledian distance is
                computed.
            3. All ancestor-descendant pairs are sorted by their Eucledian
                distances in ascending order.
            4. Shorter distance indicates a better pair. The sorted list of
                ancertor-descendant pairs is traversed. If a pair where neither
                the ancestor nor the descendant have been assigned are
                encountered, the assignment is made and all subsequent pairs
                that involve the ancestor or the descendant are ignored.
            5. Any ancestors that do not have a descendat assigned to them
                because either (a) no descendants are found within the given
                radius or (b) all potential descendants were paired with nearer
                ancestors, are considered not to have any descendants in that
                frame. Their coordinates are merged with ancestor PSFs for
                repeating the algorithm for the next frame and its successor.
            6. For each Spot, its succeeding chain of pairwise ancestor-
                candidate assignments is its path across the frames. This is
                true whether the Spot skips one or more frames.

        Due to stage drift, captured image fields may be offset. Providing a
        sequence of image alignments allows Spot tracking in this scenario.
        However, any Spots that would align to positions outside of any frames
        in the sequence are ignored and not tracked.

        Arguments:
            frame_spots: A iterable of iterables of Spots:

              [[frame0_Spot1, ..., frame0_SpotI, ...],   #frame 1
               ...,
               [frameN_Spot1, ..., frameN_SpotJ, ...],   #frame N
               ... ]

                Each frame for which Spots are to be tracked is summarized as
                an iterable of Spot objects (containing their coordinates
                (h, w)). In the above list notation of these iterables, frameN
                indicates it's the N'th frame and SpotI indicates that is the
                I'th Spot in the frame. Spot objects for each frame are not
                assumed to have any particular order. The sequence of frames,
                however, is assumed to follow the order in which the Spots will
                be tracked. It is assumed that there are no Spots within two
                pixels of each other in any one frame.
            frame_shape: This method assumes all frames have the same shape.
                frame_shape should be the frame's shape tuple (h, w) as given
                by one of the parent_Image.image.shape.
            candidate_radius: Radius within which candidates in the following
                frame are paired with precursors. The default is set to 2
                because our peptides shouldn't move more than that. Change only
                after thinking about it really hard.
            offsets: Specifies offsets (delta_h, delta_w) between frames.
            spot_radius: Discard spots within this distance of any frame's
                edge.

        Returns:
            A tuple of the following items:

            1. The traces themselves as a list of lists of Spot objects:

                [[Spot1_frame0, Spot1_frame2, ..., Spot1_frameI, ...],  #Spot 1
                 ...,
                 [SpotN_frame0, SpotN_frame2, ..., SpotN_frameJ, ...],  #Spot N
                 ... ]

                Each Spot's existence in each frame is represented by the Spot
                object in that frame. Hence, for each spot being tracked across
                frames, its existence is a sequence of Spot objects. Each of
                these sequences is represented as a list, and a list of these
                lists is the object returned. In those cases where the spot is
                not present in a frame, None is used as a placeholder in the
                list. Thus, all of the Spot lists are the same length, matching
                the number of frames that were to be processed.

            2. Number of Spots discarded and not incorporated into any traces
                due to stage drift.
        """
        logger = logging.getLogger()
        if offsets is None:
            offsets = [(0, 0) for f in len(frame_spots)]
        #cumulative offsets are useful in some contexts
        cumulative_offsets = Experiment.accumulate_offsets(offsets)
        #This algorithm speeds things up by binning all Spot objects into the
        #Mumpy array frame_bins by rounding their (h, w) to the nearest whole
        #number. The behavior of the Python round function is the reason this
        # method requires all Spots to be at least two pixels apart.
        #
        #frame_bins is the central data structure used within this function.
        #For each frame, frame bins has a corresponding, identically-shaped
        #numpy array. Each pixel in this array contains the dictionary
        #
        #{'spt': spot,
        # 's_L': (own_frame, own_h, own_w),
        # 'a_L': (prior_frame, prior_h, prior_w),
        # 'd_L': (next_frame, next_h, next_w)}
        #
        #spt is the Spot object whose (h, w) coordinates round to this pixel.
        #If there is no Spot in this pixel, then all four items -- spt, and all
        #three tuples -- are set to None. The three tuples in the pixel perform
        #the functions of a doubly-linked linked list that connects Spots
        #across frames. The first tuple indicates the coordinates of the spot
        #itself: its frame #, and its rounded (h, w) coordinates. The second
        #tuple indicates the coordinates of the ancestor spot in frame_bins.
        #The third tuple does the same for the spots descendant. If there is no
        #ancestor or descendant, then the corresponding tuple is remains a None
        #object.
        #The objective of this method is then to essentially fill in the
        #ancestor and descendant links such that the PSFs are tracked across
        #frames.
        frame_bins = [np.full(frame_shape,
                              {'spt': None, 's_L': None,
                               'a_L': None, 'd_L': None},
                              dtype=np.object)
                      for frame in frame_spots]
        #ancestor_cache stores those Spots who do not have descendants
        ancestor_cache = np.full(frame_shape,
                                 {'spt': None, 's_L': None,
                                  'a_L': None, 'd_L': None},
                                 dtype=np.object)
        #ignore spots that would be outside of the first frame when aligned;
        #using filtered_spots as temporary variable so as to not modify
        #frame_spots that could be being used elsewhere
        filtered_spots = [[] for frame in frame_spots]
        total_discarded_spots = 0
        for f, frame in enumerate(frame_spots):
            filtered_spots[f], number_discarded = \
               Experiment.discard_dropouts(spots=frame,
                            spot_cumulative_offsets=[cumulative_offsets[f]
                                                     for x in frame_spots[f]],
                                  frame_cumulative_offsets=cumulative_offsets,
                                                      image_shape=frame_shape,
                                                       spot_radius=spot_radius)
            total_discarded_spots += number_discarded
        frame_spots = filtered_spots    #stop using filtered_spots as temp var
        #Populate frame_bins with all Spots.
        for f, frame in enumerate(frame_spots):
            for spt in frame:
                h, w = Experiment.apply_offset((spt.h, spt.w),
                                               cumulative_offsets[f])
                rh, rw = int(round(h)), int(round(w))
                assert all([frame_bins[f][rh, rw]['spt'] is None,
                            frame_bins[f][rh, rw]['s_L'] is None,
                            frame_bins[f][rh, rw]['a_L'] is None,
                            frame_bins[f][rh, rw]['d_L'] is None]), \
                    (str((rh, rw)) + " is already filled in frame_bins[" +
                     str(f) + "]")
                frame_bins[f][rh, rw] = {'spt': spt, 's_L': (f, rh, rw),
                                         'a_L': None, 'd_L': None}

        #Main loop: go through all the frames, and make the best connections
        for f, frame in enumerate(frame_bins):
            if f == 0:
                continue #skip first frame; it won't have any ancestor links
            else:
                #Merge Spots from prior frame into ancestor_cache
                for (rh, rw), fbin in  np.ndenumerate(frame_bins[f - 1]):
                    (spt, s_L, a_L, d_L) = (fbin['spt'], fbin['s_L'],
                                            fbin['a_L'], fbin['d_L'])
                    if spt is None:
                        continue
                    assert rh == s_L[1] and rw == s_L[2], \
                        "s_L and (rh, rw) mismatch"
                    o_h, o_w = Experiment.apply_offset((spt.h, spt.w),
                                                     cumulative_offsets[f - 1])
                    logger.debug("greedy_particle_tracking: (spt.h, spt.w) " +
                                 str((spt.h, spt.w)) + " at frame f = " + str(f))
                    logger.debug("greedy_particle_tracking: " +
                                 "cumulative_offsets[f] = " +
                                 str(cumulative_offsets[f - 1]) +
                                 " at frame f = " + str(f))
                    logger.debug("greedy_particle_tracking: (o_h, o_w) = " +
                                 str((o_h, o_w)) + " at frame f = " + str(f))
                    ro_h, ro_w = int(round(o_h)), int(round(o_w))
                    logger.debug("greedy_particle_tracking: (ro_h, ro_w) = " +
                                 str((ro_h, ro_w)) + " at frame f = " + str(f))
                    assert (rh, rw) == (ro_h, ro_w), \
                        (str(rh) + ", " + str(rw) + " != " + str(ro_h) + ", " +
                         str(ro_w))
                    logging.debug("greedy_particle_tracking: " +
                                  "ancestor_cache[rh, rw] = " +
                                  str(ancestor_cache[rh, rw]) +
                                  " at frame f = " + str(f))
                    #I am removing this assertion because I came across cases
                    #where there are two very closely competing ancestors from
                    #different frames
                    #assert all([ancestor_cache[rh, rw]['spt'] is None,
                    #            ancestor_cache[rh, rw]['s_L'] is None,
                    #            ancestor_cache[rh, rw]['a_L'] is None,
                    #            ancestor_cache[rh, rw]['d_L'] is None]), \
                    #   str((rh, rw)) + " already filled in ancestor_cache " + \
                    #   " (s_L, a_L, d_L) = " + str((s_L, a_L, d_L)) + \
                    #   " at f = " + str(f)
                    ancestor_cache[rh, rw] = {'spt': spt,
                                              's_L': (f - 1, rh, rw),
                                              'a_L': None, 'd_L': None}
                    logger.debug("greedy_particle_tracking: Setting " +
                                 "ancestor_cache[" + str(rh) + ", " + str(rw) +
                                 "] to s_L = " + str((f - 1, rh, rw)))
            #used to sort all ancestor-descendant pairs by Eucledian distance
            ancestor_descendant_pairs = []
            for (ah, aw), abin in  np.ndenumerate(ancestor_cache):
                (a_spt, as_L, aa_L, ad_L) = (abin['spt'], abin['s_L'],
                                             abin['a_L'], abin['d_L'])
                if a_spt is None:
                    continue
                assert ad_L is None, "ancestor_cache shouldnt have descendants"
                assert ah == as_L[1] and aw == as_L[2], \
                    "as_L and (ah, aw) mismatch"
                aaf = as_L[0]
                frame_slice = frame[max(ah - candidate_radius - 2, 0):
                                    ah + candidate_radius + 3,
                                    max(aw - candidate_radius - 2, 0):
                                    aw + candidate_radius + 3]
                for (dh, dw), dbin in  np.ndenumerate(frame_slice):
                    (d_spt, ds_L, da_L, dd_L) = (dbin['spt'], dbin['s_L'],
                                                 dbin['a_L'], dbin['d_L'])
                    if d_spt is None:
                        assert ds_L is None and da_L is None and dd_L is None,\
                            "spt is None, but has links!"
                        continue
                    if ah - candidate_radius - 2 > 0:#sliced close to the edge?
                        dh += ah - candidate_radius - 2
                    if aw - candidate_radius - 2 > 0:#sliced close to the edge?
                        dw += aw - candidate_radius - 2
                    assert dh == ds_L[1] and dw == ds_L[2], \
                        "ds_L and (dh, dw) mismatch"
                    ddf = ds_L[0]
                    distance = euclidean(
                          Experiment.apply_offset((a_spt.h, a_spt.w),
                                                  cumulative_offsets[as_L[0]]),
                          Experiment.apply_offset((d_spt.h, d_spt.w),
                                                  cumulative_offsets[f]))
                    if distance < candidate_radius:
                        ancestor_descendant_pairs.append((a_spt, aaf, ah, aw,
                                                          d_spt, ddf, dh, dw,
                                                          distance))
            ancestor_descendant_pairs = sorted(ancestor_descendant_pairs,
                                               key=lambda x:x[8])
            for (a_spt, aaf, ah, aw,
                 d_spt, ddf, dh, dw,
                 distance) in ancestor_descendant_pairs:
                if ancestor_cache[ah, aw] == {'spt': None, 's_L': None,
                                              'a_L': None, 'd_L': None}:
                    assert frame_bins[aaf][ah, aw]['d_L'] is not None, \
                        "Unpaired ancestor was removed from ancestor_cache."
                    logger.debug("greedy_particle_tracking: frame[" + str(dh) +
                                 ", " + str(dw) + "]" + " skipped because " +
                                 "ancestor has been paired. Frame f = " +
                                 str(f) + ", distance = " + str(distance))
                    continue #This indicates that ancestor has been paired
                elif frame[dh, dw]['a_L'] is not None:
                    logger.debug("greedy_particle_tracking: frame[" + str(dh) +
                                 ", " + str(dw) + "]" + " skipped because " +
                                 "descendant has been paired. Frame f = " +
                                 str(f) + ", distance = " + str(distance))
                    continue #This indicates that descendant has been paired
                else:
                    assert frame[dh, dw]['a_L'] is None, \
                        "Descendant being paired more than once."
                    frame[dh, dw]['a_L'] = (aaf, ah, aw)
                    logger.debug("greedy_particle_tracking: frame[" + str(dh) +
                                 ", " + str(dw) + "][a_L] = " +
                                 str((aaf, ah, aw)) + ", frame = " + str(f) +
                                 ", distance = " + str(distance))
                    assert frame_bins[aaf][ah, aw]['d_L'] is None, \
                        "Ancestor being paired more than once."
                    frame_bins[aaf][ah, aw]['d_L'] = (ddf, dh, dw)
                    logger.debug("greedy_particle_tracking: frame[" + str(ah) +
                                 ", " + str(aw) + "][d_L] = " +
                                 str((ddf, dh, dw)) + ", frame = " + str(f) +
                                 ", distance = " + str(distance))
                    ancestor_cache[ah, aw] = {'spt': None, 's_L': None,
                                              'a_L': None, 'd_L': None}
                    logger.debug("greedy_particle_tracking: " +
                                 "ancestor_cache[" + str(ah) + ", " + str(aw) +
                                 "] set to None. Frame f = " + str(f) +
                                 ", distance = " + str(distance))
        #Should now have all Spots linked. Extract them by following links.
        #traces will be the links of Spots that this method returns
        traces = []
        #first find all head Spots throughout all frame_bins
        heads = []
        for f, frame in enumerate(frame_bins):
            for (h, w), fbin in np.ndenumerate(frame):
                (spt, s_L, a_L, d_L) = (fbin['spt'], fbin['s_L'],
                                        fbin['a_L'], fbin['d_L'])
                if spt is not None and a_L is None:
                    heads.append((spt, s_L, a_L, d_L))
        #now follow the segments from all heads
        for spt, s_L, a_L, d_L in heads:
            assert s_L is not None, \
              "s_L cannot be None for a head spot " + str((spt, s_L, a_L, d_L))
            #prepend Nones if psf is not in the first frame
            trace = [None for x in range(s_L[0])]
            trace += [spt]
            if d_L is None:
                trace += [None for x in range(len(frame_spots) - s_L[0] - 1)]
                traces.append(trace)
                continue
            df, dh, dw = d_L
            while True:
                dbin = frame_bins[df][dh, dw]
                (d_spt, ds_L, da_L, dd_L) = (dbin['spt'], dbin['s_L'],
                                             dbin['a_L'], dbin['d_L'])
                assert da_L == s_L, \
                       "Ancestor link does not match ancestor's self-link " + \
                       str((s_L, a_L, d_L, ds_L, da_L, dd_L))
                #prepend Nones to fill frames where PSF is not present
                trace += [None for x in range(ds_L[0] - s_L[0] - 1)]
                trace.append(d_spt)
                if dd_L is None:
                    #reached end of this Spot
                    break
                else:
                    s_L = ds_L
                    df, dh, dw = dd_L
            #now suffix Nones to get to the end of frames
            trace += [None for x in range(len(frame_spots) - ds_L[0] - 1)]
            traces.append(trace)
        return traces, total_discarded_spots

    @staticmethod
    def plot_traces(traces, output_filepaths):
        """
        Generate images to visually inspect traces.

        For a series of traces, images are saved with the tracked Spots
        highlighted. Tracking information is color-coded into the highlighting:

        Red: A Spot has no ancestor in the prior frame and no descendant in the
             following frame.
        Green: A Spot has no ancestor in the prior frame, but has a descendant
             in the following frame.
        Blue: A Spot has an ancestor and a descendant.
        Light blue: A spot only has an ancestor.

        Arguments:
            traces: traces as returned by greedy_particle_tracking. It is
                assumed all Spots in traces have the same parent_Image for
                each corresponding frame.
            output_filepaths: List of filepaths to save images to. The order of
                filepaths should follow chronological order. The filename
                suffix for all filepaths given must be '.png' (only saving to
                PNG for now). Any files at these paths will be written to.

        Returns:
            List of paths to saved image files.
        """
        #first split up traces bewteen frames with a color assignment
        #yellow = start, blue = going, lightblue = end, red = single frame
        framewise_traces = [[] for f in range(len(output_filepaths))]
        for trace in traces:
            for f, spot in enumerate(trace):
                if spot is None:
                    continue
                #determine color
                color = None
                am_i_first = (f == 0 or trace[f - 1] is None)
                am_i_last = (f == len(trace) - 1 or trace[f + 1] is None)
                am_i_middle = (f != 0 and
                               f != len(trace) - 1 and
                               trace[f - 1] is not None and
                               trace[f + 1] is not None)
                assert am_i_first or am_i_last or am_i_middle, "Who am I?"
                if am_i_first and am_i_last:
                    color = 'red'
                elif am_i_first and not am_i_last:
                    color = 'yellow'
                elif not am_i_first and am_i_last:
                    color = 'lightblue'
                elif am_i_middle:
                    color = 'blue'
                else:
                    raise AssertionError("Who am I?")
                framewise_traces[f].append((spot, color))
        output_paths = []
        for f, frame in enumerate(framewise_traces):
            #get image_path from the first spot
            if len(frame) == 0 or len(frame[0]) == 0:
                continue
            image_path = frame[0][0].parent_Image.metadata['filepath']
            spots = {(int(round(spot.gaussian_fit[0])),
                      int(round(spot.gaussian_fit[1]))): spot.gaussian_fit
                     for spot, color in frame}
            square_colors = {(int(round(spot.gaussian_fit[0])),
                              int(round(spot.gaussian_fit[1]))): color
                             for spot, color in frame}
            output_path = output_filepaths[f]
            if output_filepaths[f][-4:] != '.png':
                raise ValueError("output_filepaths must be .png files only.")
            pflib.save_psfs_png(psfs=spots, image_path=image_path,
                                output_path=output_filepaths[f],
                                square_color='purple',
                                square_colors=square_colors)
            output_paths.append(output_path)
        return output_paths

    @staticmethod
    def easy_sort_target_images(filepath_list):
        """
        Sorts a list of image files to be processed by field of view and order
        taken.

        Currently, when performing a MultifieldEdmanSequenceExperiment, our
        setup saves all fields of view taken together at one experimental cycle
        into one directory. Directory names follow chronological order. Each
        directory in the experiment has the same fields of view, and their
        filepaths share the same filename order across all directories.

        Arguments:
            filepath_list: List of image filepaths.

        Returns:
            Two different sorts are returned: (frame_indexed, field_indexed)

            frame_indexed: For each field of view, its experimental cycle
                frames are 0-indexed chronologically. frame_indexed is a
                dictionary with frame indeces as keys, and lists of fields of
                view for that experimental cycle as values. The order of fields
                of view in each list is identical. All directory and filenames
                are converted into absolute paths.

            field_indexed: Each field of view is indexed according to its order
                in frame_indexed. field_indexed is a dictionary with field of
                view indeces as keys, and lists of the field's corresponding
                frames as values. The order of frames in the lists are
                chronological.
        """
        logger = logging.getLogger()
        grouped_filepaths = {}
        for fpath in filepath_list:
            d, f = os.path.split(os.path.abspath(fpath))
            grouped_filepaths.setdefault(d, []).append(f)
        grouped_filepaths = {d: sorted(flist)
                             for d, flist in grouped_filepaths.iteritems()}
        logger.debug("easy_sort_target_images: grouped_filepaths = " +
                     str(grouped_filepaths))
        frame_indexed = {}
        for index, d in enumerate(sorted(grouped_filepaths.keys())):
            for filepath in grouped_filepaths[d]:
                frame_indexed.setdefault(index, []).\
                                              append(os.path.join(d, filepath))
        field_indexed = {}
        for frame, fields in frame_indexed.iteritems():
            for f, field in enumerate(fields):
                field_indexed.setdefault(f, []).append(field)
        return frame_indexed, field_indexed

    @staticmethod
    def trace_to_binary(trace):
        return [True if spot is not None else False for spot in trace]

    @staticmethod
    def truefalse_to_onoff(pattern):
        return ' '.join(['[ON] ' if p else '[OFF]' for p in pattern])

    @staticmethod
    def trace_to_photometry(trace, method='mexican_hat', return_invalid=True,
                            **kwargs):
        return [(spot.h, spot.w, spot.photometry(method=method,
                                                 return_invalid=return_invalid,
                                                 **kwargs))
                if spot is not None else (None, None, None) for spot in trace]

    @staticmethod
    def next_frame_spot_by_luminosity_centroid(spot, next_frame, offset=(0, 0),
        search_radius=3, s_n_cutoff=3.0):
        """
        Find putative position of a Spot in the succeeding frame based on the
        centroid of pixel luminosities of the Spot's vicinity in that frame.

        Given a Spot in an imaged field of view, we may need to rapidly find
        the Spot in the succeeding frame. One approach is to search in the
        immediate vicinity of the Spot's position in the original frame, and
        find in this region the centroid of luminosity. In terms of pixel
        values, the centroid is the center of mass of pixel values.

        The algorithm proceeds as follows:
            1. The Spot's coordinates in the original frame are found in the
                next frame via image alignment. (In this function, this
                alignment is passed as argument 'offset').
            2. Round these coordinates to the nearest whole pixel.
            3. Grab the square of pixels within a specified radius around this
                coordinate.
            4. Find the centroid of pixel values inside this square.
            5. Compute Illumina signal-to-noise metric for this Spot. If it is
                not above a threshold, then the Spot does not persist into this
                frame.
            6. Otherwise, the Spot is the same radius as in the preceeding frame,
                centered about this centroid.

        Arguments:
            spot: The Spot in the preceeding frame.
            next_frame: The Image in which we want to find this Spot.
            offset: The offset between the images based on alignment.
            search_radius: How large to make the search radius for the
                centroid. Beware of making this value too large: you may
                include an adjacent, unrelated Spot into the centroid.
            s_n_cutoff: Illumina signal-to-noise metric of the Spot must be
                above this threshold to be a Spot. Otherwise, the function
                assumes that the Spot is not present in the frame. Default is
                based on emperical shufti that should not discard more than
                about 1% of valid Spots.

        Returns:
            Spot in the new frame if (1) fully inside the boundaries of the
            image and (2) has a sufficiently high Illumina signal-to-noise
            metric. If it is outisde the image boundary, then None is returned.
            If it is inside the image boundary but has an insufficient Illumina
            signal-to-noise, then the Spot is the area at the same coordinates
            as in the preceeding frame (adjusted for offset).
        """
        logger = logging.getLogger()
        #get Spot coordinates in next_frame's context
        o_h, o_w = Experiment.unapply_offset((spot.h, spot.w), offset)
        #get image slice from next_frame
        image_slice = \
                  next_frame.image[o_h - search_radius:o_h + search_radius + 1,
                                   o_w - search_radius:o_w + search_radius + 1]
        #check if too close to the edge
        if image_slice.shape != (1 + 2 * search_radius, 1 + 2 * search_radius):
            next_spot = None
        else:
            #Uses scipy.ndimage.measurements.center_of_mass to compute centroid
            c_h, c_w = center_of_mass(image_slice)
            r_c_h, r_c_w = (int(round(c_h + o_h - search_radius)),
                            int(round(c_w + o_w - search_radius)))
            #use Spot's built in error checks to see if this can be a valid
            #Spot that's not outside the image boundary
            try:
                next_spot = Spot(next_frame, r_c_h, r_c_w, spot.size,
                                 gaussian_fit=None)
            except AttributeError as e:
                logger.debug(
                        "Experiment.spot_descendant_by_luminosity_centroid: " +
                        "cannot initialize next_spot due to exception " +
                        str(e) + "; setting next_spot as None.")
                next_spot = None
            else:
                if next_spot.illumina_s_n() < s_n_cutoff:
                    try:
                        next_spot = Spot(next_frame,
                                         int(round(spot.h)),
                                         int(round(spot.w)), spot.size,
                                         gaussian_fit=None)
                    except AttributeError as e:
                        logger.debug(
                         "Experiment.spot_descendant_by_luminosity_centroid:" +
                         " cannot initialize next_spot due to exception " +
                         str(e) + "; setting next_spot as None.")
                        next_spot = None
        return next_spot

    @staticmethod
    def luminosity_centroid_particle_tracking(frames, initial_spots,
                                              search_radius=3, s_n_cutoff=3.0,
                                              offsets=None):
        """
        Computationally fast method of tracking particles in a field of view
        across multiple frames. This is especially useful for time traces that
        have many frames and for which peak-fitting all of them is expensive.

        Given an initial frame and a set of Spots in that frame, this function
        uses next_frame_spot_by_luminosity_centroid to track Spots through each
        succeeding frame.

        The algorithm proceeds as follows:
            1. Start with initial_spots, that are all in the initial frame (in
                this context, frames[0]).
            2. For each Spot, find its descendant via next_frame_spot_by_
                luminosity_centroid()
            3. If any Spot in the prior frame is a None (i.e. not found by
                next_frame_spot_by_luminosity_centroid), then the last non-None
                Spot is used to find descendants.
            3. Use this set of spots to proceed through the sequence of frames.

        Arguments:
            frames: Iterable of Images through which to track the particles.
                They must all be the same shape. The initial frame is the first
                Image returned by the iterable.
            initial_spots: The Spots in the initial Image that will be tracked.
            search_radius: Passed to next_frame_spot_by_luminosity_centroid.
            s_n_cutoff: Passed to next_frame_spot_by_luminosity_centroid.
            offsets: Specifies offsets (delta_h, delta_w) between frames.

        Returns:
            Same as the first item returned by greedy_particle_tracking.
        """
        #verify that all initial_spots are from frames[0]
        if not all([True if spot.parent_Image is frames[0] else False
                    for spot in initial_spots]):
            raise ValueError("All initial_spots must be in frames[0].")
        spot_tracks = []
        for spot in initial_spots:
            spot_tracks.append([spot])
            prior_frame_spot = spot
            for f, frame in enumerate(frames):
                if f == 0:
                    continue
                offset = offsets[f] if offsets is not None else (0, 0)
                next_spot = Experiment.next_frame_spot_by_luminosity_centroid(
                                                   spot=prior_frame_spot,
                                                   next_frame=frame,
                                                   offset=offset,
                                                   search_radius=search_radius,
                                                   s_n_cutoff=s_n_cutoff)
                spot_tracks[-1].append(next_spot)
                if next_spot is not None:
                    prior_frame_spot = next_spot
        return spot_tracks


class Trace(object):
    """
    A Trace is a sequence of a Spot through multiple Images.

    Once an Experiment is defined, it establishes a relationship between its
    constituent Spots and Images. One important class of relationships is a
    timetrace, i.e. continuously tracking a Spot through multiple frames.

    Trace encapsulates the trace of one Spot through a number of frames. Frames
    are 0-indexed. A Trace does not have to be a series of Spots: it can be a
    sequence of data points corresponding to Spots. The objective of Trace and
    its subclasses it to organize working with the sequential data resulting
    from tracing a Spot.

    Attributes:
        This serves as a base class for different types of Traces. All Trace
        subclasses must contain the following attributes and functions.

        trace: The actual trace. There is no uniform format for what this is,
            because it differs between the different Trace subclasses.
        h, w: The (h, w) coordinates of a Trace are taken to be its position in
            the first frame.
        frame_num: Number of frames in the trace.
        photometry(frame): Photometry of the Spot in frame.
    """
    def photometry(self, **kwargs):
        raise AttributeError("Every Trace subclass must implement its own "
                             "photometry() method")

    def photometries(self, photometry_min=None,
                     photometry_method='mexican_hat', **kwargs):
        """
        Get the full sequence of photometries for this Trace.

        If a member in the sequence is None and thus does not implement the
        photometry() method, its photometry is returned as 0.

        Arguments:
            photometry_min: If not None, all photometries below this value are
                rounded up to it.
            photometry_method: Method to calculate the photometry. Must be an
                available option in Spot.photometry.
            **kwargs: All other arguments are passed as parameters to
                Spot.photometry.

        Returns:
            Sequence of photometries as a tuple.
        """
        logger = logging.getLogger()
        logger.debug("flexlibrary.Trace.photometries: locals() = " +
                     str(locals()))
        return_photometries = \
                           [spot.photometry(method=photometry_method, **kwargs)
                            if spot is not None else 0
                            for f, spot in enumerate(self.trace)]
        if photometry_min is not None:
            return_photometries = [max(photometry_min, rp)
                                   for rp in return_photometries]
        return tuple(return_photometries)

    def stepfit_photometries(self, h, w, mirror_start=0, chung_kennedy=0,
                             p_threshold=0.01, photometry_min=None,
                             photometry_method='mexican_hat', **kwargs):
        """
        Stepfit this Trace's sequence of photometries.

        Below is a summary of the steps used. Read documentation in
        stepfitting_library for more details.

        1. Mirror the first mirror_start frames. This is useful if looking for
            very short steps at the very beginning. Furthermore, if
            mirror_start used (i.e. it is > 0), then t-test filtering is not
            applied to plateaus within the first mirror_start frames.
        2. Apply Chung-Kennedy filter -- possibly multiple number of times --
            to smooth the data into plateaus.
        3. Convert the smoothed data into plateaus.
        4. Perform Welch's t-test between all plateaus, and merge plateaus that
            do not fail the null hypothesis that they may come from different
            normal distributions (i.e. different number of fluors turned on).

        Arguments:
            h, w: (h, w) coordinates of the trace to be fitted.
            mirror_start: Number of frames to mirror.
            chung_kennedy: Number of times to apply Chung-Kennedy.
            p_threshold: Threshold to use when deciding whether or not to merge
                plateaus.
            photometry_min, photometry_method, and **kwargs are passed to
                self.photometries.

        Returns:
            Stepfit as defined by stepfitting_library and all processing
            intermediates:

            (photometries, ck_filtered_photometries, plateaus,
             t_filtered_plateaus)

            where photometries are the original given photometries as a
            PhotometryTrace, ck_filtered_photometries are photometries after
            Chung-Kennedy filtering as a SimpleTrace, plateaus are stepfits
            based on the CK-filtered data as a PlateauTrace, and
            t_filtered_plateaus are these plateaus filtered via t-test as a
            PlateauTrace.

            t_filtered_plateaus are the final step fit.
        """
        photometries = self.photometries(photometry_min=photometry_min,
                                         photometry_method=photometry_method,
                                         **kwargs)
        mirrored_photometries = \
              stepfitting_library.mirror_photometries(photometries,
                                                      mirror_size=mirror_start)
        ck_filtered_photometries = mirrored_photometries
        for c in range(chung_kennedy):
            ck_filtered_photometries = \
                                      stepfitting_library.chung_kennedy_filter(
                                          luminosities=mirrored_photometries,
                                          window_lengths=(2, 4, 8, 16))
        plateaus = stepfitting_library.sliding_t_fitter(
                                  luminosity_sequence=ck_filtered_photometries,
                                  window_radius=6,
                                  p_threshold=p_threshold,
                                  median_filter_size=None,
                                  downsteps_only=False,
                                  min_step_magnitude=None)
        plateaus = stepfitting_library.refit_plateaus(mirrored_photometries,
                                                      plateaus)
        t_filtered_plateaus = stepfitting_library.t_test_filter(
                                            luminosities=mirrored_photometries,
                                            plateaus=plateaus,
                                            p_threshold=p_threshold,
                                            drop_sort=True,
                                            no_merge_start=mirror_start)
        unmirrored_ck_filtered_photometries = \
            stepfitting_library.unmirror_photometries(ck_filtered_photometries,
                                                      mirror_size=mirror_start)
        unmirrored_plateaus = stepfitting_library.unmirror_plateaus(
                                                      plateaus,
                                                      mirror_size=mirror_start)
        unmirrored_t_filtered_plateaus = stepfitting_library.unmirror_plateaus(
                                                      t_filtered_plateaus,
                                                      mirror_size=mirror_start)
        #Convert to Trace instances
        photometries = PhotometryTrace(photometries, h, w)
        unmirrored_ck_filtered_photometries = \
                     PhotometryTrace(unmirrored_ck_filtered_photometries, h, w)
        unmirrored_plateaus = PlateauTrace(unmirrored_plateaus, h, w)
        unmirrored_t_filtered_plateaus = \
                             PlateauTrace(unmirrored_t_filtered_plateaus, h, w)
        return (photometries, unmirrored_ck_filtered_photometries,
                unmirrored_plateaus, unmirrored_t_filtered_plateaus)

    def frame_output(self, frame, **kwargs):
        """Outputs frame's photometry by default."""
        return self.photometry(frame, **kwargs)

    @staticmethod
    def trace_comparison_rss(trace_A, trace_B, photometry_method='mexican_hat',
                             **kwargs):
        """
        Calculates residual sum of squares between two Traces' photometries.
        """
        if trace_A.num_frames != trace_B.num_frames:
            raise Exception("trace_A and trace_B must cover an identical "
                            "number of frames for comparison to be valid.")
        return sum([(trace_A.photometry(frame=f,
                                        photometry_method=photometry_method,
                                        **kwargs) -
                     trace_B.photometry(frame=f,
                                        photometry_method=photometry_method,
                                        **kwargs))**2
                    for f in range(trace_A.num_frames)])

    def total_sum_squares(self, photometry_method='mexican_hat', **kwargs):
        """Total sum of squares for this Trace's photometries."""
        photometries = self.photometries(photometry_min=None,
                                         photometry_method=photometry_method,
                                         **kwargs)
        photometry_mean = float(np.mean(photometries))
        return sum((p - photometry_mean)**2 for p in photometries)

    @staticmethod
    def coefficient_of_determination(trace_A, trace_B,
                                     photometry_method='mexican_hat',
                                     **kwargs):
        """
        Coefficient of determination for how well trace_B photometries fit
        trace_A photometries.
        """
        rss = float(Trace.trace_comparison_rss(trace_A, trace_B,
                                           photometry_method=photometry_method,
                                               **kwargs))
        tss = float(trace_A.total_sum_squares(
                                           photometry_method=photometry_method,
                                              **kwargs))
        return 1.0 - rss / tss

        


class SimpleTrace(Trace):
    """
    A simple trace represented as a sequence of Spots in a list.

    Attributes:
        trace: trace as documented in Experiment.greedy_particle_tracking. Must
            have at least one Spot in the sequence; cannot be all None's.
        h, w: The (h, w) coordinate of the first non-None Spot in this Trace.
    """
    def _trace_hw(self):
        """
        Get (h, w) coordinate of the first non-None Spot in this Trace.
        """
        h, w = None, None
        for spot in self.trace:
            if spot is not None:
                h, w = spot.h, spot.w
                break
        else:
            raise Exception("flexlibrary.Trace.trace_hw: this Trace is " +
                            "composed entirely of None's.")
        assert h is not None and w is not None
        return h, w

    def __init__(self, trace):
        self.trace = trace
        self.h, self.w = self._trace_hw()
        self.num_frames = len(trace)

    def photometry(self, frame, photometry_method='mexican_hat', **kwargs):
        """
        Get the photometry of the Spot at frame.

        Arguments:
            frame: Index of the frame to get the photometry from.
            photometry_method: Method to calculate the photometry. Must be an
                available option in Spot.photometry.
            **kwargs: All other arguments are passed as parameters to
                Spot.photometry.

        Returns:
            Photometry of the Spot in that frame. 0 if Spot is a None in that
            frame.
        """
        spot = self.trace[frame]
        if spot is None:
            photometry = 0
        else:
            photometry = spot.photometry(method=photometry_method, **kwargs)
        return photometry

    def coordinates(self, frame):
        """Get (h, w) coordinate for this Trace in frame."""
        if self.trace[frame] is not None:
            h, w = self.trace[frame].h, self.trace[frame].w
        else:
            h, w = None, None
        return h, w

    def plateau_starts(self):
        """
        Useful for compatability with Trace subclasses that are plateau-like.

        plateau_starts is a way to indicate when a new plateau starts. This is
        useful to avoid re-computing information from the Trace that doesn't
        change within a constant plateau region (unless a new plateau is
        encountered). plateau_starts must return a start if there is a
        possibility that ANY function will return something different for a new
        frame: ALL Trace properties within a plateau must be constant.
        Otherwise, we risk not correctly obtaining the Trace's property for a
        frame. For this Trace subclass, there are no assumed plateaus, so every
        frame is considered a plateau_start.
        """
        return set(range(self.num_frames))


class PhotometryTrace(Trace):
    """
    A Trace containing only the Spot photometries as a sequence.

    Attributes:
        trace: Sequence of photometries.
    """
    def __init__(self, trace, h, w):
        self.trace = trace
        self.h, self.w = h, w
        self.num_frames = len(trace)

    def photometry(self, frame, **kwargs):
        return self.trace[frame]

    def plateau_starts(self):
        """
        Useful for compatability with Trace subclasses that are plateau-like.

        plateau_starts is a way to indicate when a new plateau starts. This is
        useful to avoid re-computing information from the Trace that doesn't
        change within a constant plateau region (unless a new plateau is
        encountered). plateau_starts must return a start if there is a
        possibility that ANY function will return something different for a new
        frame: ALL Trace properties within a plateau must be constant.
        Otherwise, we risk not correctly obtaining the Trace's property for a
        frame. For this Trace subclass, there are no assumed plateaus, so every
        frame is considered a plateau_start.
        """
        return set(range(self.num_frames))


class PlateauTrace(Trace):
    """
    A trace represented by a sequence of plateaus, as described in
    stepfitting_library.

    Attributes:
        trace: Trace represented by a series of plateaus, as documented in
            stepfitting_library.
        h, w: The (h, w) coordinate of the first non-None Spot in this Trace.
    """
    def __init__(self, trace, h, w):
        """Must pass (h, w); plateaus do not store that information."""
        self.trace = trace
        self.h, self.w = h, w
        self.num_frames = trace[-1][1] + 1 if len(trace) > 0 else 0

    def photometry(self, frame, **kwargs):
        """
        Get the photometry of the Spot at frame, as defined by plateau height.
        """
        return stepfitting_library.plateau_value(self.trace, frame)

    def last_step_info(self, frame):
        """Get information about the last step that took place before frame."""
        return stepfitting_library.last_step_info(self.trace, frame)

    def frame_plateau(self, frame):
        """
        Get plateau containing frame, and the plateau's index in self.trace.
        """
        return stepfitting_library.frame_plateau(self.trace, frame)

    def plateau_starts(self):
        """Get list of frame indices corresponding to plateau starts."""
        return stepfitting_library.plateau_starts(self.trace)


class SequenceExperiment(Experiment):
    """
    Tracks a single field of peptides across a sequence of experimental cycles.
    The first frame preceeds any experimental cycles. One full (possibly mock)
    Edman degradation takes place between all successive frames. Only one color
    channel is used for tracking peptide labels; i.e. this is for single-label
    experiments.

    Optionally, an additional channel containing orthogonally-labeled fiduciary
    markers may be provided for frame alignment to guard against stage drift.
    Alignment will be based on maximizing the cross-image correlation between
    successive alignment frames.

    It is also possible to provide alignment information directly as a sequence
    of coordinate displacements.

    All images are assumed to be the same shape.

    Attributes:
        peptide_frames: Sequence of Images of the field in the peptide label
            channel.
        alignment_frames: Sequence of Images of the field in the alignment
            label channel. If a peptide frame does not have a corresponding
            alignment frame, a None must be substituted as a placeholder in the
            sequence. Thus, if provided, alignment_frames must be the same
            length as peptide_frames.
        offsets: Sequence of coordinate displacements explicitly defining
            inter-frame alignments: [(dh_1, dw_1), (dh_2, dw_2), ...], where
            tuple (dh_i, dw_i) is the h and w displacement from the prior
            frame. This means that every position at coordinate (h, w) in frame
            i - 1 is at position (h + dh_i, w + dw_i) in frame i. No
            verification is performed to check if alignment_frames and
            alignments are consistent. If provided, alignments must have the
            same length as peptide_frames. This overrides alignment_frames as
            the alignment source for the experiment.
        spot_traces: Traces of spots in the format as returned by
            Experiment.greedy_particle_tracking().
        num_discarded_spots: Number of Spots discarded from traces by
            Experiment.greedy_particle_tracking().
        photometry_adjustments: Cached per-frame photometry adjustments. A
            photometry adjustment for a SequenceExperiment is a list or tuple
            of floats, each float's index in the list corresponding to its
            respective frame. The float is meant to be applied to all
            photometries in the entire frame. photometry_adjustments does not
            pre-define the way to apply the adjustments; this is determined by
            the adjustment function used. photometry_adjustments can cache
            multiple photometry adjustment variants: it is a dictionary with
            the key being an arbitrary user-defined label for that adjustment,
            and the value being the adjustment sequence. Example:
            {'adjustment A': (3, 4, 3, 3, 3), 'adjustment B': (10, 9, 5, 5, 5)}
            By default, photometry_adjustments is set to None. It can be
            generated using functions in the class.
    """
    def offsets_from_frames(self, upsample_factor=20):
        """
        Use DFT image cross-correlation to find frame alignments.

        Uses port of Manuel Guizar's code phase_correlation.py

        Arguments:
            upsample_factor: Align images at resolution of 1/upsample_factor
                pixels.

        Returns:
            Updated self.offsets
        """
        if self.alignment_frames is None:
            raise AttributeError("Calling offsets_from_frames without "
                                 "alignment_frames defined.")
        offsets = [(0, 0) for frame in self.alignment_frames]
        for f, frame in enumerate(self.alignment_frames[1:]):
            d_h, d_w, err, diffphase = \
                      phase_correlate(ref_image=self.alignment_frames[f].image,
                                      reg_image=frame.image,
                                      upsample_factor=upsample_factor)
            offsets[f + 1] = (d_h, d_w)
        self.offsets = offsets
        return self.offsets

    def __init__(self, peptide_frames, alignment_frames=None, offsets=None,
                 spot_traces=None, num_discarded_spots=0,
                 photometry_adjustments=None):
        self.peptide_frames = peptide_frames
        self.alignment_frames = [None for f in peptide_frames]
        if offsets is not None:
            if len(offsets) != len(peptide_frames):
                raise AttributeError("If provided, offsets must have the "
                                     "same number of items as peptide_frames.")
            self.offsets = offsets
        elif alignment_frames is not None:
            if len(alignment_frames) != len(peptide_frames):
                raise AttributeError("If provided, alignment_frames must have "
                                     "the same number of items as "
                                     "peptide_frames.")
            self.alignment_frames = alignment_frames
            self.offsets = self.offsets_from_frames()
        self.offsets = [(0, 0) for f in peptide_frames]
        self.spot_traces = spot_traces
        self.num_discarded_spots = num_discarded_spots
        if photometry_adjustments is not None:
            if not all([len(adjustments) == len(peptide_frames)
                        for adj_tag, adjustments
                        in photometry_adjustments.iteritems()]):
                raise AttributeError("All photometry adjustment lists must be "
                                     "the same length as peptide_frames.")
        self.photometry_adjustments = photometry_adjustments

    def trace_existing_spots(self, spot_radius=None):
        """
        Trace the fates of existing Spots across this experiment's frames using
        greedy_particle_tracking.

        No new peptides are searched for, and no new Spots are created.

        Arguments:
            spot_radius: Discard spots within this distance of frame edges,
                overriding spot radius as determined by Spot instances in this
                experiment. If this is not given, and there are no Spots in any
                of this experiment's frames, then spot_radius is set to 0. This
                default should not affect the outcome of analysis because there
                are no spots to discard to begin with. CURRENTLY NOT
                IMPLEMENTED.

        Returns:
            Traces as returned by Experiment.greedy_particle_tracking().
            Updates self.spot_traces and self.num_discarded_spots based on
            this.
        """
        if spot_radius is not None:
            raise NotImplementedError("spot_radius currently not implemented")
        if spot_radius is None:
            all_spots = [spot
                         for image in self.peptide_frames
                         for spot in image.spots]
            if len(all_spots) == 0:
                spot_radius = 0
            else:
                spot_size = all_spots[0].size
                spot_radius = (spot_size - 1) / 2
        self.spot_traces, self.num_discarded_spots = \
            Experiment.greedy_particle_tracking(
                    frame_spots=[image.spots for image in self.peptide_frames],
                    frame_shape=self.peptide_frames[0].image.shape,
                    offsets=self.offsets,
                    #spot_radius=spot_radius)
                    spot_radius=0) #This feature is turned off for now.
        return self.spot_traces

    def binary_trace_categories(self):
        """
        Basic approach to reduce spot traces to sequencing information.

        For each spot, it is either present or not present in a frame as
        defined by pflib.find_peaks's ability to find & fit it. This function
        reduces each spot trace to a chronological sequences of True/False
        booleans, and groups the traces based on their binary sequence.

        This function assumes that self.spot_traces is up to date via e.g.
        self.trace_existing_spots().

        Note that the number of categories explodes exponentially with
        increasing numbers of Edman cycles.

        Returns:
            Dictionary with binary trace sequences as keys and a list of all
            traces in the experiment that reduce to that sequence as values.
        """
        logger = logging.getLogger()
        #Code in case we want to generate all categories at once. Uses memory!
        #trace_categories = {sequence: [] for sequence in
        #     itertools.product(*(itertools.repeat([False, True],
        #                                          len(self.peptide_frames))))}
        trace_categories = {}
        for trace in self.spot_traces:
            trace_categories.setdefault(
                    tuple(Experiment.trace_to_binary(trace)), []).append(trace)
        return trace_categories

    def interpolate_spots(self,
                          (start_spot, start_frame),
                          (stop_spot, stop_frame)):
        """
        Given two spots in two frames of this experiment, generate Spot objects
        by interpolating the two Spot's positions across the intervening
        frames.

        If either of the Spot objects given as the endpoints to the
        interpolation are None, the interpolation simply uses the other Spot's
        position across all frames.

        If neither start_spot nor stop_spot are None, at least one frame must
        exist between start_frame and stop_frame.

        The self.offsets as it exists when this method is call is used. Make
        sure they are correctly set before calling this method.

        Arguments:
            start_spot: Spot object in the first bookending frame. If None,
                will use stop_spot to interpolate. start_spot and stop_spot
                cannot both be None.
            start_frame: Integer index of the frame in this
                SequenceExperiment's self.peptide_frames. start_frame must be
                less than stop_frame.
            stop_spot: Spot object in the last bookending frame. If None,
                will use start_spot to interpolate. start_spot and stop_spot
                cannot both be None.
            stop_frame: Integer index of the frame in this
                SequenceExperiment's self.peptide_frames. start_frame must be
                less than stop_frame.

        Returns:
            A list of Spots in the same order as the frames. The first and last
            Spots will be new objects identical to start_spot and stop_spot. If
            any of the interpolated Spots are outside of their parent image's
            frame, None is used as a placeholder.
        """
        #Some sanity checks.
        if not start_frame < stop_frame:
            raise ValueError("start_frame must come before stop_frame")
        if (not (start_spot is None or stop_spot is None) and
            not start_frame + 1 < stop_frame):
            raise ValueError("If neither start_spot or stop_spot are None, "
                             "stop_frame must have at least one frame between "
                             "it and start_frame.")
        if start_spot is None and stop_spot is None:
            raise ValueError("Both start_spot and stop_spot are None.")
        #Set up accumulated offsets.
        if self.offsets is None:
            use_offsets = [(0, 0) for i, f in enumerate(self.peptide_frames)]
        else:
            use_offsets = self.offsets
        #Set up Spot starting and stopping coordinates in terms of the starting
        #frame.
        if start_spot is not None:
            start_h, start_w = start_spot.h, start_spot.w
        else:
            start_h, start_w = \
                  Experiment.offset_frame_coordinates(offsets=use_offsets,
                                                      coordinate=(stop_spot.h,
                                                                  stop_spot.w),
                                                      f=start_frame,
                                                      g=stop_frame)
        if stop_spot is not None:
            stop_h, stop_w = \
                  Experiment.offset_frame_coordinates(offsets=use_offsets,
                                                      coordinate=(stop_spot.h,
                                                                  stop_spot.w),
                                                      f=start_frame,
                                                      g=stop_frame)
        else:
            stop_h, stop_w = start_spot.h, start_spot.w
        #Create list of interpolated coordinates.
        num_frames = stop_frame - start_frame
        h_increment = float(stop_h - start_h) / num_frames
        w_increment = float(stop_w - start_w) / num_frames
        h_coordinates = [start_h + h_increment * i
                         for i in range(num_frames + 1)]
        w_coordinates = [start_w + w_increment * i
                         for i in range(num_frames + 1)]
        #Note to self: if using a direct h_coordinates[-1] == stop_h assertion,
        #it may easily fail if the coordinates are off in e.g. the tenth
        #decimal place due to floating point rounding behaviour
        assert abs(h_coordinates[-1] - stop_h) < 0.01, str(locals())
        assert abs(w_coordinates[-1] - stop_w) < 0.01, str(locals())
        coordinates = zip(h_coordinates, w_coordinates)
        #Apply offsets to interpolated coordinates
        offset_coordinates = [None for c in coordinates]
        for i, (h, w) in enumerate(coordinates):
            offset = Experiment.get_cumulative_offset(offsets=use_offsets,
                                                      f=i + start_frame,
                                                      g=start_frame)
            offset_coordinates[i] = Experiment.apply_offset((h, w), offset)
        assert all(c is not None for c in offset_coordinates)
        #Create Spots at the interpolated positions.
        #This is where the interpolated spots will be stored. This is what the
        #function returns.
        interpolated_spots = []
        #Determine what Spot size to use.
        if start_spot is not None and stop_spot is None:
            spot_size = start_spot.size
        elif start_spot is None and stop_spot is not None:
            spot_size = stop_spot.size
        elif start_spot is not None and stop_spot is not None:
            if start_spot.size != stop_spot.size:
                raise ValueError("start_spot.size != stop_spot.size")
            else:
                spot_size = start_spot.size
        spot_radius = (spot_size - 1) / 2
        #Iterate over offset_coordinates to create the spots.
        for i, (h, w) in enumerate(offset_coordinates):
            target_index = start_frame + i
            frame = self.peptide_frames[target_index]
            #fh, fw = Experiment.offset_frame_coordinates(offsets=use_offsets,
            #                                             coordinate=(h, w),
            #                                             f=target_index,
            #                                             g=start_frame)
            frame_shape_h, frame_shape_w = frame.image.shape
            int_h, int_w = int(round(h)), int(round(w))
            #if (0 <= fh < frame_shape_h and 0 <= fw < frame_shape_w):
            #    new_spot = Spot(parent_Image=frame, h=fh, w=fw, size=spot_size,
            if (spot_radius <= int_h < frame_shape_h - spot_radius and
                spot_radius <= int_w < frame_shape_w - spot_radius):
                new_spot = Spot(parent_Image=frame,
                                h=int_h, w=int_w, size=spot_size,
                                gaussian_fit=None)
                frame.spots.append(new_spot)
                frame.append_photometry_apertures()
            else:
                new_spot = None
            interpolated_spots.append(new_spot)
        return interpolated_spots

    def fill_in_trace(self, trace):
        """
        Given a trace in this SequenceExperiment that has no Spots in some
        frames and thus contains None's as placeholders in their stead, fill in
        the trace with Spots generated by SequenceExperiment.interpolate_spots.
        """
        #First, get mask of where the holes are.
        holes = [True if s is None else False for s in trace]
        #Use mask to find borders of each hole. Each border is a tuple
        #((b1, i1), (b2, i2)) where b1 is the last non-None Spot before the
        #hole, and b2 is the first non-None Spot after the hole (special case:
        #holes extending to beginning/end), and i1 & i2 are the indexes of the
        #frames containing the correspoding spots.
        hole_borders = []
        border_start = None
        border_index_map = {}
        border_index_j = 0
        for i, (h1, h2) in enumerate(stepfitting_library._pairwise(holes)):
            #Reminder to self: index for s1 is i, index for s2 is i + 1.
            s1, s2 = trace[i], trace[i + 1]
            border_index_map.setdefault(i, (len(hole_borders), border_index_j))
            if h1 and h2:
                border_index_j += 1
            elif h1 and not h2:
                if border_start is None:
                    #This means the hole is from the beginning of the trace.
                    hole_borders.append(((s1, 0), (s2, i + 1)))
                else:
                    hole_borders.append((border_start, (s2, i + 1)))
                    border_start = None
            elif not h1 and h2:
                border_start = s1, i
                #Initialize to 1 here because the first interpolated Spot
                #returned by interpolate_spots in this scenario (not h1 and h2)
                #is the first border spot; this is not true for interpolation
                #when the first spot is a None.
                border_index_j = 1 
            elif not h1 and not h2:
                assert border_start is None
        #Special case: check if hole extends to the end.
        if border_start is not None:
            border_index_map.setdefault(i + 1, (len(hole_borders), -1))
            hole_borders.append((border_start, (s2, len(holes) - 1)))
        #Using hole borders, call interpolate_spots.
        interpolated_spots = [self.interpolate_spots((s1, i1), (s2, i2))
                              for (s1, i1), (s2, i2) in hole_borders]
        #Now merge the interpolated Spots and trace into a new list.
        merged_trace = []
        for i, s in enumerate(trace):
            if s is not None:
                merged_trace.append(s)
            else:
                #Need to find the corresponding Spot in interpolated_spots.
                border_index, j = border_index_map[i]
                interpolated_spot = interpolated_spots[border_index][j]
                merged_trace.append(interpolated_spot)
        return merged_trace

    def discard_invalid_traces(self, **pparams):
        """
        Discard any traces that will have spots with invalid photometries from
        self.spot_traces.

        Arguments:
            pparams: Parameters for spot photometries. If a spot's photometry
                cannot be calculated based on these parameters, the entire
                trace containing the spot is discarded.

        Returns:
            The discarded traces.
        """
        valid_traces = []
        invalid_traces = []
        for trace in self.spot_traces:
            filled_trace = self.fill_in_trace(trace)
            if None in filled_trace:
                invalid_traces.append(filled_trace)
                continue
            p = Experiment.trace_to_photometry(filled_trace,
                                               return_invalid=False,
                                               **pparams)
            if None in [ph for h, w, ph in p]:
                invalid_traces.append(filled_trace)
                continue
            else:
                valid_traces.append(trace)
        self.spot_traces = valid_traces
        return invalid_traces

    def binary_trace_categories_photometry(self, method='mexican_hat',
                                           interpolate=False,
                                           discard_invalid=False,
                                           adjustment_function=None, **kwargs):
        """
        Obtains photometry information via Experiment.trace_to_photometry for
        all traces in binary_trace_categories.

        Useful for looking at photometry distributions for categories of Spot
        behaviors.

        Arguments:
            method: Use this photometry method; passed to
                Experiment.trace_to_photometry.
            interpolate: Whether to interpolate Spots via self.fill_in_trace
                even if Spots were not detected in some frames, and use their
                photometries instead of None placeholders. If this option is
                chosen, then the only None entries in the list of photometries
                will be for those cases where the interpolated Spot would be
                outside its frame.
            discard_invalid: If True, will discard traces that have any None
                members due to e.g. being out of frame or too close to the edge
                to obtain a valid photometry. DEPRECATED
            adjustment_function: Apply this adjustment function to each
                photometry. Must take arguments (photometry, frame,
                adjustments); see mdma_adjustment for example.
            **kwargs: Use these additional parameters for photometry
                calculations; passed to Experiment.trace_to_photometry.

        Returns:
            Same dictionary as binary_trace_categories, except each trace is
            replaced by the sequence of its Spot's (h, w) coordinates and their
            photometry measurements. If a Spot is not detected in a frame, then
            None is used as a placeholder in the sequence.
        """
        logger = logging.getLogger()
        if discard_invalid:
            raise DeprecationWarning("discard_invalid is deprecated. Use "
                                     "discard_invalid_traces() functions")
        btc = self.binary_trace_categories()
        btc_photometries = {}
        for category, traces in btc.iteritems():
            for trace in traces:
                if interpolate:
                    use_trace = self.fill_in_trace(trace)
                else:
                    use_trace = trace
                if discard_invalid and None in use_trace:
                    continue
                p = Experiment.trace_to_photometry(use_trace,
                                          method=method,
                                          return_invalid=(not discard_invalid),
                                          **kwargs)
                if discard_invalid and None in [ph for h, w, ph in p]:
                    continue
                if adjustment_function is not None:
                    p = [(h, w,
                          adjustment_function(photometry=ph, frame=frame,
                                      adjustments=self.photometry_adjustments))
                         for frame, (h, w, ph) in enumerate(p)]
                logging.debug("SequenceExperiment." +
                              "binary_trace_categories_photometry: p = " +
                              str(p))
                btc_photometries.setdefault(category, []).append(p)
        return btc_photometries

    def multiplicative_delta_median_adjustments(self, tag='mdma',
                                                method='mexican_hat',
                                                **kwargs):
        """
        Create an 'mdma' entry in self.photometry_adjustments using the
        multiplicative delta median method. If there are no remainders in the
        field, the adjustments are all set to 0.

        An existing 'mdma' entry in self.photometry_adjustments will be
        overwritten. Supply an alternate tag to create a new entry.

        MDMA adjustments are derived as follows:
        1. Find all persistent remainders (traces with spots that are on across
            all frames).
        2. Find the median value of each remainder across all frames.
        3. For each remainder, take its values across all of the frames and
            subtract the median from them.
        4. Take the ratios of the differences to the median. In each frame,
            this ratio is called that remainder's adjustment factor for that
            frame.
        5. For each frame, take the median of all adjustment factors. This is
            now the per-frame adjustment factor Af (where f is the frame
            index). The sequence of Af's for this SequenceExperiment is what's
            stored (as a tuple) in self.photometry_adjustments['mdma'].
        6. To use MDMA adjustments to adjust intensities in frames, update all
            spot intensities I in frames to I_adjusted = I * (1.0 - Af). See
            function mdma_adjustment below to perform this.

        If this SequenceExperiment contains no persistent remainders, all
        adjustment Af's will be 0.0

        Arguments:
            tag: Use tag as the key for this method call's results in
                self.photometry_adjustments.
            method: Use this photometry method. Passed to
                binary_trace_categories_photometry.
            **kwargs: Additional parameters for photometry calculations; passed
                to binary_trace_categories_photometry.

        Returns:
            self.photometry_adjustments['mdma']
        """
        num_frames = len(self.peptide_frames)
        btc_photometries = self.binary_trace_categories_photometry(
           method=method, interpolate=False, discard_invalid=False, **kwargs)
        assert len([1 for category in btc_photometries.keys()
                    if set(category) == set([True])]) <= 1
        all_on_category = tuple([True] * len(self.peptide_frames))
        if all_on_category in btc_photometries:
            all_on_photometries = btc_photometries[all_on_category]
        else:
            all_on_photometries = []
        all_on_photometries = [photometry_trace
                               for photometry_trace in all_on_photometries
                               if all([ph is not None
                                       for h, w, ph in photometry_trace])]
        adjustment_ratios = [[] for n in self.peptide_frames]
        for photometry_trace in all_on_photometries:
            m = np.median([ph for (h, w, ph) in photometry_trace])
            for i, (h, w, ph) in enumerate(photometry_trace):
                r = float(ph - m) / m
                adjustment_ratios[i].append(r)
        adjustment_ratio_medians = [np.median(ratios) if len(ratios) > 0
                                    else 0.0
                                    for ratios in adjustment_ratios]
        if self.photometry_adjustments is None:
            self.photometry_adjustments = {}
        self.photometry_adjustments.setdefault('mdma', [])
        self.photometry_adjustments['mdma'] = tuple(adjustment_ratio_medians)
        return self.photometry_adjustments['mdma']

    @staticmethod
    def mdma_adjustment(photometry, frame, adjustments):
        """
        Apply an mdma adjustment to photometry in frame
        (c.f. multiplicative_delta_median_adjustments, which must be run before
        using this function).

        Arguments:
            photometry: Photometry as a floating point value.
            frame: Frame in which the Spot is located.
            adjustments: Dictionary as described for this class's
                photometry_adjustments attribute, containing an 'mdma' entry.

        Returns:
            MDMA adjusted photometry.
        """
        if 'mdma' in adjustments:
            return photometry * (1.0 - adjustments['mdma'][frame])
        else:
            return photometry

    def count_remainders(self):
        """Returns number of Spots that persist across all frames."""
        btc = self.binary_trace_categories()
        all_on_category = tuple([True] * len(self.peptide_frames))
        if all_on_category not in btc:
            num_remainders = 0
        else:
            num_remainders = len(btc[all_on_category])
        return num_remainders

    def plot_traces(self, timestamp_epoch=None, trace_directory=None,
                    prefix=''):
        """
        Plot all traces of this experiment.

        If trace_directory is not provided, then the traces are saved at
        each frame's Image.metadata['filepath'] + '_traces_' +
        pflib._epoch_to_hash(timestamp_epoch) + '.png'. If trace_directory is
        provided, then the frames for the trace are saved in the directory with
        filenames prefix + '_frame_' + framenumber + '_traces_' + epoch_hash +
        '.png'. Any files at these locations are overwritten.

        Honestly, if you are not using absolute filepaths in Image metadata,
        I leave it up to you to figure out what this function will do.

        Arguments:
            timestamp_epoch: Use this Unix epoch to generate the timestamp_hash
                from pflib._epoch_to_hash(). If None, then plot_traces will
                poll the current Unix epoch at runtime.
            trace_directory: If given, will save the traces here. Strongly
                recommended to be given as an absolute path.
            prefix: Useful to distinguish files written to the same
                trace_directory. May be used, for example, to index them by
                field of view.

        Returns:
            List of paths to saved image files.
        """
        if timestamp_epoch is None:
            timestamp_epoch = round(time.time())
        epoch_hash = pflib._epoch_to_hash(timestamp_epoch)
        #generate list of output filepaths
        output_filepaths = []
        if trace_directory is not None:
            #below assumes all traces must be same length
            if not os.path.exists(trace_directory):
                os.makedirs(trace_directory)
            frame_zfill = int(np.ceil(math.log10(len(self.peptide_frames))))
            for f, p in enumerate(self.peptide_frames):
                output_filepath = os.path.join(trace_directory, prefix +
                                               '_frame_' +
                                               str(f).zfill(frame_zfill) +
                                               '_' + epoch_hash + '.png')
                output_filepaths.append(output_filepath)
        else:
            for frame in self.peptide_frames:
                output_filepath = (frame.metadata['filepath'] + '_traces_' +
                                   epoch_hash + '.png')
                output_filepaths.append(output_filepath)
        return Experiment.plot_traces(self.spot_traces,
                                      output_filepaths=output_filepaths)

    def spot_count(self):
        """
        Count total number of Spots in all frames of this SequenceExperiment.

        Returns:
            Number of spots in self.peptide_fields, number of spots in
            self.alignment_fields.
        """
        return sum([len(frame.spots) for frame in self.peptide_frames])

    def singleton_count(self):
        """
        Count number of traces that are Spots that appear for only one frame.
        """
        return sum([1
                    for trace in self.spot_traces
                    if len([t for t in trace if t is not None]) == 1])

    def extract_tracks(self, trace_category, radius=4, number=5):
        """
        Return a visual sample of peptide tracks that follow a trace pattern.

        trace_category is the pattern of True/False as returned by
        Experiment.trace_to_binary, applied to self.spot_traces, indicating
        whether the spot is present in a frame or not as this experiment is
        run.

        Traces in this SequenceExperiment.spot_traces that match trace_category
        are randomly sampled, and then output as sequences of Numpy image
        arrays containing the immediate vicinity of the traced spot.

        Arguments:
            trace_category: Sample self.spot_traces that match this pattern.
            radius: Returned image arrays are squares centered about the
                tracked spot, of sides (2 * radius + 1) in size.
            number: Number of tracks to return. If this is larger than the
                number of tracks in this Experiment, return all of them.

        Returns:
            Each sampled track will be a list of Numpy image arrays tracking a
            spot across this SequenceExperiment's frames, with its
            parent_Image:

            track = [(spot_area_in_frame1, parent_Image1),
                     (spot_area_in_frame2, parent_Image1), ... ]

            This method returns a list of such tracks, with its coordinates in
            the first frame of the experiment, and its parent Image:

            [((h, w), track), ((h2, w2), track2), ... ]
        """
        logger = logging.getLogger()
        binary_trace_categories = self.binary_trace_categories()
        image_sequences = []
        if trace_category in binary_trace_categories:
            traces = binary_trace_categories[trace_category]
            sample = random.sample(traces, min(number, len(traces)))
            for trace in sample:
                image_sequence = []
                #used to get coordinates for frames where trace is None
                nonnull_traces = [(f, frame)
                                  for f, frame in enumerate(trace)
                                  if frame is not None]
                if len(nonnull_traces) == 0:
                    raise Exception("This trace has no non-None frames.")
                else:
                    n, nspot = nonnull_traces[0]
                    ndh, ndw = Experiment.get_cumulative_offset(self.offsets,
                                                                n)
                    nh, nw = Experiment.apply_offset((nspot.h, nspot.w),
                                                     (ndh, ndw))
                for f, frame in enumerate(trace):
                    if frame is None:
                        img = self.peptide_frames[f].image
                        oh, ow = \
                              Experiment.get_cumulative_offset(self.offsets, f)
                        gh, gw = Experiment.unapply_offset((nh, nw), (oh, ow))
                        rgh, rgw = Experiment.round_coordinates(gh, gw)
                        logger.debug("SequenceExperiment.extract_tracks: " +
                                     "frame is None; " +
                                     "(oh, ow) = " + str((oh, ow)) +
                                     "(gh, gw) = " + str((gh, gw)) +
                                     "(rgh, rgw) = " + str((rgh, rgw)))
                    else:
                        img = frame.parent_Image.image
                        rgh, rgw = Experiment.round_coordinates(frame.h,
                                                                frame.w)
                        logger.debug("SequenceExperiment.extract_tracks: " +
                                    "frame is not None; " +
                                    "(rgh, rgw) = " + str((rgh, rgw)))
                    subimg = img[max(0, rgh - radius):
                                 min(rgh + radius + 1, img.shape[0]),
                                 max(0, rgw - radius):
                                 min(rgw + radius + 1, img.shape[1])]
                    image_sequence.append((subimg, self.peptide_frames[f]))
                image_sequences.append(((nh, nw), image_sequence))
        return image_sequences


class MultifieldSequenceExperiment(Experiment):
    """
    Simultaneous SequenceExperiment's carried out over multiple fields.

    Used to summarize observations of a multifield Edman sequencing experiment.
    Most methods are self-explanatory upon reading documentation for their
    constituent SequenceExperiment methods.

    NOTE: Because this is just a special case of a
        MultifieldMultichannelSequenceExperiment, it will not be updated. Use
        the broader class.

    Attributes:
        experimental_fields: List of SequenceExperiment's.
    """
    def __init__(self, experimental_fields):
        self.experimental_fields = experimental_fields
        raise DeprecationWarning("This class is no longer maintained. Use "
                                 "MultifieldMultichannelSequenceExperiment "
                                 "instead.")

    def trace_existing_spots(self):
        for ex in self.experimental_fields:
            ex.trace_existing_spots()

    def plot_traces(self, timestamp_epoch=None, trace_directory=None):
        for e, ex in enumerate(self.experimental_fields):
            ex.plot_traces(timestamp_epoch=timestamp_epoch,
                           trace_directory=trace_directory,
                           prefix=str(e))

    def binary_trace_categories(self):
        merged = {}
        for ex in self.experimental_fields:
            to_merge = ex.binary_trace_categories()
            for k, v in to_merge.iteritems():
                merged.setdefault(k, [])
                merged[k] += v
        return merged

    def count_binary_trace_categories(self):
        merged = self.binary_trace_categories()
        counts = {k: len(v) for k, v in merged.iteritems()}
        return counts, merged

    def filtered_binary_trace_category_counts(self):
        """
        Return counts of traces where only one ON->OFF transition occurs,
        excluding those that are on only in the first frame.
        """
        counts, merged = self.count_binary_trace_categories()
        return {bt: count
                for bt, count in counts.iteritems()
                if tuple(sorted(bt, reverse=True)) == bt and bt[1]}

    def plot_filtered_binary_trace_counts(self, output_filepath):
        raise DeprecationWarning("Deprecating for now in favor of outputting "
                                 "CSV files. Assume this function is no "
                                 "longer maintained.")
        filtered = self.filtered_binary_trace_category_counts()
        to_plot = sorted(list(filtered.iteritems()), key=lambda x:x[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ind = np.arange(len(to_plot))
        width = 0.7
        rects = ax.bar(ind, [c for s, c in to_plot])
        ax.set_xlim(-width, len(ind) + width)
        ax.set_ylabel('Counts')
        xTickMarks = [str(len([x for x in s if x])) for s, c in to_plot]
        ax.set_xticks(ind + width)
        xTickNames = ax.set_xticklabels(xTickMarks)
        plt.setp(xTickNames, fontsize=10)
        plt.savefig(output_filepath)

    def count_discarded_spots(self):
        return sum([ex.num_discarded_spots for ex in self.experimental_fields])

    def spot_count(self):
        return sum([ex.spot_count() for ex in self.experimental_fields])

    def trace_count(self):
        return sum([len(ex.spot_traces) for ex in self.experimental_fields])

    def singleton_count(self):
        return sum([ex.singleton_count() for ex in self.experimental_fields])


class MultichannelSequenceExperiment(SequenceExperiment):
    """
    Extends SequenceExperiment across multiple color channels.

    This class combines multiple SequenceExperiments, each representing a
    separate color channel for the same field of view. By default, it is
    assumed that all channels are aligned because switching the excitation
    laser should not perturb the stage. However, if alignment is necessary,
    then all that needs to be done is that the alignment channel in all
    constituent SequenceExperiments needs to have their alignment_frames set to
    the same sequence of frames. This class is agnostic to the alignment_frames
    of its constituent SequenceExperiments; it's up to the user to control
    this.

    This class assumes that all SequenceExperiments have the same number of
    peptide_frames and (if present) alignment_frames.

    Attributes:
        experiment_channels: Dictionary of SequenceExperiments. Keys are
            channel names or numbers (i.e. either strings or numbers can be
            used: it's up to the user to keep track of these names). Values are
            the SequenceExperiments representing each channel.
    """
    def __init__(self, channels):
        if not (len(set([len(chan.peptide_frames)
                         for chan in channels.values()])) ==
                len(set([len(chan.alignment_frames)
                     for chan in channels.values()]))
                and
                len(set([len(chan.alignment_frames)
                     for chan in channels.values()])) == 1):
            raise AttributeError("Number of peptide_frames and alignment_frames"
                                 "does not match across channels.")
        self.channels = channels

    def trace_existing_spots(self):
        logger = logging.getLogger()
        logger.debug("MultichannelSequenceExperiment.trace_existing_spots " +
                     "self.channels = "+ str(self.channels))
        for chan in self.channels.itervalues():
            chan.trace_existing_spots()

    def plot_traces(self, timestamp_epoch=None, trace_directory=None,
                    prefix=''):
        for c, chan in self.channels.iteritems():
            chan.plot_traces(timestamp_epoch=timestamp_epoch,
                             trace_directory=trace_directory,
                             prefix=prefix + '_channel_' + str(c))

    def binary_trace_categories(self):
        logger = logging.getLogger()
        merged = {}
        for c, chan in self.channels.iteritems():
            to_merge = chan.binary_trace_categories()
            logger.debug("MultichannelSequenceExperiment." +
                         "binary_trace_categories: to_merge " + str(to_merge))
            merged.setdefault(c, to_merge)
        return merged

    def binary_trace_categories_photometry(self, method='mexican_hat',
                                           interpolate=False,
                                           discard_invalid=False,
                                           adjustment_function=None, **kwargs):
        if discard_invalid:
            raise DeprecationWarning("discard_invalid is deprecated. Use "
                                     "discard_invalid_traces() functions")
        merged = {}
        for c, chan in self.channels.iteritems():
            to_merge = \
               chan.binary_trace_categories_photometry(
                                       method=method,
                                       interpolate=interpolate,
                                       discard_invalid=discard_invalid,
                                       adjustment_function=adjustment_function,
                                       **kwargs)
            merged.setdefault(c, to_merge)
        return merged

    def count_binary_trace_categories(self):
        merged = self.binary_trace_categories()
        counts = {c: {k: len(v) for k, v in chan.iteritems()}
                  for c, chan in merged.iteritems()}
        return counts, merged

    def filtered_binary_trace_category_counts(self):
        counts, merged = self.count_binary_trace_categories()
        return {c: {bt: count
                    for bt, count in chan.iteritems()
                    if tuple(sorted(bt, reverse=True)) == bt and bt[1]}
                for c, chan in counts.iteritems()}

    def plot_filtered_binary_trace_counts(self, output_filepaths):
        raise DeprecationWarning("Deprecating for now in favor of outputting "
                                 "CSV files. Assume this function is no "
                                 "longer maintained.")
        filtered_channels = self.filtered_binary_trace_category_counts()
        for c, filtered in filtered_channels.iteritems():
            to_plot = sorted(list(filtered.iteritems()), key=lambda x:x[0])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ind = np.arange(len(to_plot))
            width = 0.7
            rects = ax.bar(ind, [c for s, c in to_plot])
            ax.set_xlim(-width, len(ind) + width)
            ax.set_ylabel('Counts')
            xTickMarks = [str(len([x for x in s if x])) for s, c in to_plot]
            ax.set_xticks(ind + width)
            xTickNames = ax.set_xticklabels(xTickMarks)
            plt.setp(xTickNames, fontsize=10)
            plt.savefig(output_filepaths[c])

    def count_discarded_spots(self):
        return {c: channel.num_discarded_spots
                for c, channel in self.channels.iteritems()}

    def spot_count(self):
        return {c: channel.spot_count()
                for c, channel in self.channels.iteritems()}

    def trace_count(self):
        return {c: len(chan.spot_traces)
                for c, chan in self.channels.iteritems()}

    def singleton_count(self):
        return {c: chan.singleton_count()
                for c, chan in self.channels.iteritems()}

    def extract_tracks(self, trace_category, radius=4, number=5):
        return {c: chan.extract_tracks(trace_category, radius=radius,
                                       number=number)
                for c, chan in self.channels.iteritems()}

    def get_offsets(self):
        return {c: chan.offsets for c, chan in self.channels.iteritems()}

    def discard_invalid_traces(self, **pparams):
        return {c: chan.discard_invalid_traces(**pparams)
                for c, chan in self.channels.iteritems()}

    def multiplicative_delta_median_adjustments(self, tag='mdma',
                                                method='mexican_hat',
                                                channels=None,
                                                **kwargs):
        if channels is None:
            return {c: chan.multiplicative_delta_median_adjustments(
                                                                 tag=tag,
                                                                 method=method,
                                                                 **kwargs)
                    for c, chan in self.channels.iteritems()}
        else:
            return {c: chan.multiplicative_delta_median_adjustments(
                                                                 tag=tag,
                                                                 method=method,
                                                                 **kwargs)
                    for c, chan in self.channels.iteritems() if c in channels}

    def count_remainders(self):
        return {c: chan.count_remainders()
                for c, chan in self.channels.iteritems()}


class MultifieldMultichannelSequenceExperiment(MultifieldSequenceExperiment):
    """
    Extends a MultichannelSequenceExperiment over multiple fields.

    Used to summarize observations of a multifield Edman sequencing experiment
    carried out over multiple channels. Most methods are simple combinations of
    MultichannelSequenceExperiment methods, which in turn are composed of
    SequenceExperiment methods.

    This class assumes that all MultichannelSequenceExperiments have the same
    number of peptide_frames and (if present) alignment_frames.

    Attributes:
        experimental_fields: List of MultichannelSequenceExperiment's.
        invalid_fields_mask: List of True/False that corresponds to
            self.experimental_fields, allowing functions such as
            remainder_threshold_fields to declare fields invalid. Fields whose
            corresponding mask value is True are considered valid.
    """
    def __init__(self, experimental_fields, invalid_fields_mask=None):
        if not (len(set([len(chan.peptide_frames)
                         for ex in experimental_fields
                         for chan in ex.channels.values()])) ==
                len(set([len(chan.alignment_frames)
                         for ex in experimental_fields
                         for chan in ex.channels.values()]))
                and
                len(set([len(chan.alignment_frames)
                         for ex in experimental_fields
                         for chan in ex.channels.values()])) == 1):
            raise AttributeError("Number of peptide_frames and alignment_frames"
                                 "does not match across fields and channels.")
        self.experimental_fields = experimental_fields
        if invalid_fields_mask is not None:
            if len(invalid_fields_mask) != len(self.experimental_fields):
                raise AttributeError("invalid_fields_mask must be the same "
                                     "length as experimental_fields.")
            self.invalid_fields_mask = invalid_fields_mask
        else:
            self.invalid_fields_mask = [True] * len(self.experimental_fields)

    def trace_existing_spots(self, parallel=False,
                             ignore_invalid_fields=False):
        if parallel:
            raise NotImplementedError("Classes in multiple processes do not "
                                      "share state, therefore if we want to "
                                      "parallelize this function, we will "
                                      "need to shuttle information between "
                                      "instances.")
            num_processes = min(multiprocessing.cpu_count(),
                                len(self.experimental_fields))
            pool = multiprocessing.Pool(processes=num_processes,
                                        maxtasksperchild=None)
            processes = []
            for ex in self.experimental_fields:
                processes.append(pool.apply_async(ex.trace_existing_spots))
            pool.close()
            pool.join()
        else:
            for e, ex in enumerate(self.experimental_fields):
                if ignore_invalid_fields and not self.invalid_fields_mask[e]:
                    continue
                ex.trace_existing_spots()

    def plot_traces(self, timestamp_epoch=None, trace_directory=None,
                    prefix='', ignore_invalid_fields=False):
        for e, ex in enumerate(self.experimental_fields):
            if ignore_invalid_fields and not self.invalid_fields_mask[e]:
                continue
            ex.plot_traces(timestamp_epoch=timestamp_epoch,
                           trace_directory=trace_directory,
                           prefix=prefix + '_field_' + str(e))

    def binary_trace_categories(self, ignore_invalid_fields=False):
        logger = logging.getLogger()
        merged = {}
        for e, ex in enumerate(self.experimental_fields):
            if ignore_invalid_fields and not self.invalid_fields_mask[e]:
                continue
            to_merge = ex.binary_trace_categories()
            logger.debug("MultifieldMultichannelSequenceExperiment." +
                         "binary_trace_categories: to_merge " + str(to_merge))
            for c, chan in to_merge.iteritems():
                merged.setdefault(c, {})
                merged[c].setdefault(e, {})
                for k, v in chan.iteritems():
                    merged[c][e].setdefault(k, [])
                    merged[c][e][k] += v
        return merged

    def binary_trace_categories_photometry(self, method='mexican_hat',
                                           interpolate=False,
                                           discard_invalid=False,
                                           adjustment_function=None,
                                           ignore_invalid_fields=False,
                                           **kwargs):
        if discard_invalid:
            raise DeprecationWarning("discard_invalid is deprecated. Use "
                                     "discard_invalid_traces() functions")
        merged = {}
        for e, ex in enumerate(self.experimental_fields):
            if ignore_invalid_fields and not self.invalid_fields_mask[e]:
                continue
            to_merge = \
                 ex.binary_trace_categories_photometry(
                                       method=method,
                                       interpolate=interpolate,
                                       discard_invalid=discard_invalid,
                                       adjustment_function=adjustment_function,
                                       **kwargs)
            for c, chan in to_merge.iteritems():
                merged.setdefault(c, {})
                merged[c].setdefault(e, {})
                for k, v in chan.iteritems():
                    merged[c][e].setdefault(k, [])
                    merged[c][e][k] += v
        return merged

    def all_raw_photometries(self, method='mexican_hat', interpolate=False):
        """
        """
        pass

    def track_photometries_as_csv(self, filepath, dialect='excel',
                                  photometry_method='mexican_hat',
                                  save_averages=True, discard_invalid=False,
                                  ignore_invalid_fields=False,
                                  adjustment_function=None, **kwargs):
        """
        Saves average intensities of each track's photometry in csv format.

        By default, saved file has the following data:

        CHANNEL    FIELD    H      W      CATEGORY         AVERAGE_INTENSITY
        -------    -----    ---    ---    -------------    -----------------
        2          3        206    142    [ON][ON][OFF]    908
        ...        ...      ...    ...    ...              ...


        The average intensity is computed by binary_trace_categories_photometry
        as specified by 'method', with '**kwargs' passed with it. The average
        intensity does not take into account the intensity of the Spot in those
        frames where it was not detected.

        Alternatively, if save_averages is set to False, the saved file has
        photometries for every track saved as follows:

        CHANNEL    FIELD    H      W      CATEGORY         FRAME_0    FRAME_1
        -------    -----    ---    ---    -------------    -------    -------
        2          3        206    142    [ON][ON][OFF]    908        760
        ...        ...      ...    ...    ...              ...        ...


        FRAME_2  ...
        -------
        830      ...
        ...      ...

        H and W are the coordinates of the track's Spot in the first frame. If
        using save_averages=True and the Spot is not present in the first
        frame, will use the Spot's first instance -- i.e. at the earliest frame
        it's detected and categorized as '[ON]' -- and use the coordinates in
        that frame as H and W. If using save_averages=False, and the Spot is
        not in the first frame, then H and W are the coordinates of the Spot in
        the first frame based on adjusting the coordinates of the Spot in the
        first frame it appears to the first frame in the trace, based on
        offsets. There is a corner case where if the spot doesn't appear in the
        first frame, and offsetting it from the first frame it appears in to
        the first frame places it outside of frame boundaries, then the
        coordinates used are the first frame found where offsetting the spot
        puts it inside frame boundaries.

        In this case, the photometric intensity for every frame is saved for
        each spot in its row. For those frames where a Spot was not detected,
        interpolation is used to get the photometry.

        Arguments:
            filepath: Path to csv output file. Anything at filepath will be
                overwritten.
            dialect: Format of csv to write; passed to Python csv module.
            photometry_method: What photometry method to use; passed to
                binary_trace_categories_photometry.
            save_averages: If True, use the first format above and save only
                the average intensity across all non-None Spots in the track.
                If False, use the second format and save intensity for all
                Spots, including those that are None. Those Spots that are None
                are created using SequenceExperiment.fill_in_trace and
                SequenceExperiment.interpolate_spots.
            discard_invalid: Discard tracks that have one or more invalid
                photometries due to e.g. not being in frame or being too close
                to frame edge. DEPRECATED: Use discard_invalid_traces()
                functions.
            ignore_invalid_fields: If True, will omit fields with a False value
                in self.invalid_fields_mask.
            adjustment_function: Adjust photometries using this function.
            **kwargs: Additional photometry parameters passed to
                binary_trace_categories_photometry.

        Returns:
            Total number of rows written.
        """
        logger.debug("flexlibrary.MultifieldMultichannelSequenceExperiment." +
                     "track_photometries_as_csv: locals() = " + str(locals()))
        if discard_invalid:
            raise DeprecationWarning("discard_invalid is deprecated. Use "
                                     "discard_invalid_traces() functions")
        if save_averages:
            btcp = \
              self.binary_trace_categories_photometry(
                                   method=photometry_method,
                                   interpolate=False,
                                   discard_invalid=discard_invalid,
                                   ignore_invalid_fields=ignore_invalid_fields,
                                   adjustment_function=adjustment_function,
                                   **kwargs)
        else:
            btcp = \
              self.binary_trace_categories_photometry(
                                   method=photometry_method,
                                   interpolate=True,
                                   discard_invalid=discard_invalid,
                                   ignore_invalid_fields=ignore_invalid_fields,
                                   adjustment_function=adjustment_function,
                                   **kwargs)
        output_writer = csv.writer(open(filepath, 'w'), dialect=dialect)
        if save_averages:
            output_writer.writerow(['CHANNEL', 'FIELD', 'H', 'W', 'CATEGORY',
                                    'AVERAGE_INTENSITY'])
        else:
            #number_of_frames = len(btcp.values()[0].values()[0].keys()[0])
            number_of_frames = len(self.experimental_fields[0].
                                   channels.values()[0].peptide_frames)
            output_writer.writerow(['CHANNEL', 'FIELD', 'H', 'W', 'CATEGORY'] +
                                   ['FRAME ' + str(i)
                                    for i in range(number_of_frames)])
        row_counter = 0
        for chan, categories in btcp.iteritems():
            for e, ex in categories.iteritems():
                for category, trace_photometries in ex.iteritems():
                    for photometry in trace_photometries:
                        h, w = [fp[:2]
                                for fp in photometry
                                if fp is not (None, None, None)][0]
                        if save_averages:
                            photometry_mean = np.mean([fp[2]
                                                       for fp in photometry
                                                       if fp[2] is not None])
                            output_writer.writerow([str(chan), str(e),
                                                    str(h), str(w),
                                                    str(category),
                                                    str(photometry_mean)])
                        else:
                            all_photometries = [str(fp[2])
                                                if fp[2] is not None else '0'
                                                for fp in photometry]
                            output_writer.writerow([str(chan), str(e),
                                                    str(h), str(w),
                                                    str(category)] +
                                                   all_photometries)
                        row_counter += 1
        return row_counter

    def count_binary_trace_categories(self, ignore_invalid_fields=False):
        merged = self.binary_trace_categories(ignore_invalid_fields=
                                              ignore_invalid_fields)
        if ignore_invalid_fields:
            merged_fields = set([e
                                 for c, chan in merged.iteritems()
                                 for e, ex in chan.iteritems()])
            invalid_fields = \
                           set([e
                                for e, v in enumerate(self.invalid_fields_mask)
                                if not v])
            assert len(merged_fields & invalid_fields) == 0
        counts = {c: {e: {k: len(v) for k, v in ex.iteritems()}
                      for e, ex in chan.iteritems()}
                  for c, chan in merged.iteritems()}
        return counts, merged

    def filtered_binary_trace_category_counts(self,
                                              include_first_frame_only=True,
                                              ignore_invalid_fields=False):
        """
        Return counts of traces where only one ON->OFF transition occurs.

        Args:
            include_first_frame_only: If false, exclude those that are ON only
                in the first frame.
            ignore_invalid_fields: If True, ignore fields per
                self.invalid_fields_mask.
        """
        counts, merged = \
                      self.count_binary_trace_categories(ignore_invalid_fields=
                                                         ignore_invalid_fields)
        if ignore_invalid_fields:
            counts_fields = set([e
                                 for c, chan in counts.iteritems()
                                 for e, ex in chan.iteritems()])
            invalid_fields = \
                           set([e
                                for e, v in enumerate(self.invalid_fields_mask)
                                if not v])
            assert len(counts_fields & invalid_fields) == 0
        if include_first_frame_only:
            return {c: {e: {bt: count
                            for bt, count in ex.iteritems()
                            if tuple(sorted(bt, reverse=True)) == bt}
                        for e, ex in chan.iteritems()}
                    for c, chan in counts.iteritems()}
        else:
            return {c: {e: {bt: count
                            for bt, count in ex.iteritems()
                            if tuple(sorted(bt, reverse=True)) == bt and bt[1]}
                        for e, ex in chan.iteritems()}
                    for c, chan in counts.iteritems()}

    def category_counts_as_csv(self, filepath, filtered=True,
                               collate_fields=False, dialect='excel',
                               ignore_invalid_fields=False):
        """
        Save output from count_binary_trace_categories or
        filtered_binary_trace_category_counts as CSV file.

        Arguments:
            filepath: Path to CSV file. Will overwrite any file present.
            filtered: If True, use filtered_binary_trace_category_counts. Use
                count_binary_trace_categories otherwise.
            collate_fields: If True, include a column indicating field of view
                #.
            dialect: Which dialect for Python's CSV module to use.
            ignore_invalid_fields: If True, ignore fields per
                self.invalid_fields_mask.

        Returns:
            Path to saved CSV file.
        """
        logger = logging.getLogger()
        if filtered:
            to_save = \
              self.filtered_binary_trace_category_counts(ignore_invalid_fields=
                                                         ignore_invalid_fields)
        else:
            to_save = self.count_binary_trace_categories(ignore_invalid_fields=
                                                         ignore_invalid_fields)
        if ignore_invalid_fields:
            to_save_fields = set([e
                                  for chan, fields in to_save.iteritems()
                                  for e, patterns in fields.iteritems()])
            invalid_fields = \
                           set([e
                                for e, v in enumerate(self.invalid_fields_mask)
                                if not v])
            assert len(to_save_fields & invalid_fields) == 0
        to_save_channels = sorted(to_save.keys())
        if collate_fields:
            to_save_header = (["Pattern", "Field", "Channel", "Count"])
            #                  [str(chan) for chan in to_save_channels])
        else:
            to_save_header = (["Pattern", "Channel", "Count"])
            #                  [str(chan) for chan in to_save_channels])
        logger.debug("category_counts_as_csv: to_save = " + str(to_save))
        to_save_patterns = set([pattern
                                for chan, fields in to_save.iteritems()
                                for e, patterns in fields.iteritems()
                                for pattern, count in patterns.iteritems()])
        to_save_patterns = sorted(list(to_save_patterns))
        with open(filepath, 'w') as output_file:
            output_writer = csv.writer(output_file, dialect=dialect)
            output_writer.writerow(to_save_header)
            for pattern in to_save_patterns:
                pattern_row_base = [Experiment.truefalse_to_onoff(pattern)]
                for chan in to_save_channels:
                    if collate_fields:
                        for e, ex in to_save[chan].iteritems():
                            pattern_row = pattern_row_base + [str(e),
                                                              str(chan)]
                            if pattern in ex:
                                pattern_row += [str(ex[pattern])]
                            else:
                                pattern_row += ['0']
                            output_writer.writerow(pattern_row)
                    else:
                        pattern_row = pattern_row_base + [str(chan)]
                        count = 0
                        for e, ex in to_save[chan].iteritems():
                            if pattern in ex:
                                count += ex[pattern]
                        if count > 0:
                            pattern_row += [str(count)]
                        else:
                            pattern_row += ['0']
                        output_writer.writerow(pattern_row)
        return filepath

    def category_counts_as_string(self, filtered=True, collate_fields=False,
                                  ignore_invalid_fields=False):
        """
        Make a simple multiline string for the output from
        count_binary_trace_categories or filtered_binary_trace_category_counts.

        Arguments:
            filtered: If True, use filtered_binary_trace_category_counts. Use
                count_binary_trace_categories otherwise.
            collate_fields: If True, include a column indicating field of view
                #.
            ignore_invalid_fields: If True, ignore fields per
                self.invalid_fields_mask.

        Returns:
            Multiline string representing the data.
        """
        if filtered:
            to_string = \
              self.filtered_binary_trace_category_counts(ignore_invalid_fields=
                                                         ignore_invalid_fields)
        else:
            raise NotImplementedError("filtered=False not yet implemented.")
            to_string = \
                      self.count_binary_trace_categories(ignore_invalid_fields=
                                                         ignore_invalid_fields)
        output_string = ''
        for chan, ex in sorted(to_string.items(), key=lambda x:x[0]):
            if collate_fields:
                for e, patterns in ex.iteritems():
                    output_string += (" Channel " + str(chan) +
                                      " Frame " + str(e) + "\n")
                    for pattern, count in sorted(patterns.items(),
                                                 key=lambda x:x[0]):
                        output_string += \
                                 ("    " +
                                  str(Experiment.truefalse_to_onoff(pattern)) +
                                  "    " + str(count) + "\n")
            else:
                merged_patterns = {}
                for e, patterns in ex.iteritems():
                    for pattern, count in patterns.iteritems():
                        merged_patterns.setdefault(pattern, 0)
                        merged_patterns[pattern] += count
                output_string += str(chan) + "\n"
                for pattern, count in sorted(merged_patterns.items(),
                                             key=lambda x:x[0]):
                    output_string += \
                             ("    " +
                              str(Experiment.truefalse_to_onoff(pattern)) +
                              "    " + str(count) + "\n")
        return output_string

    def count_discarded_spots(self, ignore_invalid_fields=False):
        count = {}
        for e, ex in enumerate(self.experimental_fields):
            if ignore_invalid_fields and not self.invalid_fields_mask[e]:
                continue
            subcount = ex.count_discarded_spots()
            for c, num in subcount.iteritems():
                count.setdefault(c, 0)
                count[c] += num
        return count

    def spot_count(self, ignore_invalid_fields=False):
        count = {}
        for e, ex in enumerate(self.experimental_fields):
            if ignore_invalid_fields and not self.invalid_fields_mask[e]:
                continue
            subcount = ex.spot_count()
            for c, num in subcount.iteritems():
                count.setdefault(c, 0)
                count[c] += num
        return count

    def trace_count(self, ignore_invalid_fields=False):
        count = {}
        for e, ex in enumerate(self.experimental_fields):
            if ignore_invalid_fields and not self.invalid_fields_mask[e]:
                continue
            subcount = ex.trace_count()
            for c, num in subcount.iteritems():
                count.setdefault(c, 0)
                count[c] += num
        return count

    def singleton_count(self, ignore_invalid_fields=False):
        count = {}
        for e, ex in enumerate(self.experimental_fields):
            if ignore_invalid_fields and not self.invalid_fields_mask[e]:
                continue
            subcount = ex.singleton_count()
            for c, num in subcount.iteritems():
                count.setdefault(c, 0)
                count[c] += num
        return count

    def extract_tracks(self, trace_category, radius=4, number=5,
                       ignore_invalid_fields=False):
        tracks = {}
        for e, ex in enumerate(self.experimental_fields):
            if ignore_invalid_fields and not self.invalid_fields_mask[e]:
                continue
            subtracks = ex.extract_tracks(trace_category=trace_category,
                                          radius=radius, number=number)
            for sc, subtrack in subtracks.iteritems():
                tracks.setdefault(sc, [])
                tracks[sc] += subtrack
        return tracks

    def get_offsets(self, ignore_invalid_fields=False):
        return {e: ex.get_offsets()
                for e, ex in enumerate(self.experimental_fields)
                if not (ignore_invalid_fields and
                        not self.invalid_fields_mask[e])}

    def get_offsets_by_frame(self, ignore_invalid_fields=False):
        """Categorizes get_offsets output by frame."""
        logger = logging.getLogger()
        all_offsets = self.get_offsets()
        logger.debug("get_offsets_by_frame: all_offsets = " + str(all_offsets))
        by_frame = {}
        for e, ex_offsets in all_offsets.iteritems():
            if ignore_invalid_fields and not self.invalid_fields_mask[e]:
                continue
            for c, chan_offsets in ex_offsets.iteritems():
                for f, frame_offset in enumerate(chan_offsets):
                    by_frame.setdefault(f, {})
                    by_frame[f].setdefault(e, {})
                    by_frame[f][e].setdefault(c, (frame_offset[0],
                                                  frame_offset[1]))
        return by_frame

    def save_offsets_as_dict(self, filename, ignore_invalid_fields=False):
        """
        Dictionary structure is
        {frame # (i.e. experimental cycle): {field #: {channel #: (h, w)}}}
        """
        cPickle.dump(self.get_offsets_by_frame(ignore_invalid_fields=
                                               ignore_invalid_fields),
                     open(filename, 'w'))

    def offsets_as_string(self, ignore_invalid_fields=False):
        """Converts output from get_offsets_by_frame to a string."""
        to_string = self.get_offsets_by_frame(ignore_invalid_fields=
                                              ignore_invalid_fields)
        output_string = ''
        for f, frame_offsets in sorted(to_string.items(), key=lambda x:x[0]):
            output_string += "Frame " + str(f) + "\n"
            for e, ex_offsets in sorted(frame_offsets.items(),
                                        key=lambda x:x[0]):
                output_string += "    Field " + str(e) + "\n"
                for c, (h, w) in sorted(ex_offsets.items(), key=lambda x:x[0]):
                    output_string += ("        Channel " + str(c) + " " +
                                      str((h, w)) + "\n")
                all_h = [h for h, w in ex_offsets.values()]
                all_w = [w for h, w in ex_offsets.values()]
                mean_h, mean_w = np.mean(all_h), np.mean(all_w)
                std_h, std_w = np.std(all_h), np.std(all_w)
                output_string += \
                                ("        Mean Offsets for Field " +
                                 str(e) + " = " + str((mean_h, mean_w)) + "\n")
                output_string += \
                                ("        Std.Dev. Offsets for Field " +
                                 str(e) + " = " + str((std_h, std_w)) + "\n")
            all_h = [h for ex_offsets in frame_offsets.values()
                       for h, w in ex_offsets.values()]
            all_w = [w for ex_offsets in frame_offsets.values()
                       for h, w in ex_offsets.values()]
            mean_h, mean_w = np.mean(all_h), np.mean(all_w)
            std_h, std_w = np.std(all_h), np.std(all_w)
            output_string += ("    Mean Offsets for Frame " + str(f) +
                              str((mean_h, mean_w)) + "\n")
            output_string += ("        Std.Dev. Offsets for Field " +
                              str(f) + " = " + str((std_h, std_w)) + "\n")
        return output_string
    
    def discard_invalid_traces(self, ignore_invalid_fields=False, **pparams):
        return [ex.discard_invalid_traces(**pparams)
                if not (ignore_invalid_fields and
                        not self.invalid_fields_mask[e])
                else False
                for e, ex in enumerate(self.experimental_fields)]

    def multiplicative_delta_median_adjustments(self, tag='mdma',
                                                method='mexican_hat',
                                                channels=None,
                                                ignore_invalid_fields=False,
                                                **kwargs):
        return [ex.multiplicative_delta_median_adjustments(tag=tag,
                                                           method=method,
                                                           channels=None,
                                                           **kwargs)
                if not (ignore_invalid_fields and
                        not self.invalid_fields_mask[e])
                else False
                for e, ex in enumerate(self.experimental_fields)]

    def count_remainders(self, ignore_invalid_fields=False):
        return [ex.count_remainders()
                if not (ignore_invalid_fields and
                        not self.invalid_fields_mask[e])
                else False
                for e, ex in enumerate(self.experimental_fields)]

    def remainder_threshold_fields(self, channels=None, min_remainders=5):
        """
        For MultichannelSequenceExperiments that have less than num_remainders
        remainders in any of the specified channels, set
        self.invalid_fields_mask to False.

        Arguments:
            channels: If given, consider only remainders in this list/tuple of
                channels. This must be a list of channel names as used in
                MultichannelSequenceExperiment. Any channels not specified do
                not affect thresholding. If None, the channel with the minimum
                number of remainders is used as the threshold.
            min_remainders: Smallest number of remainders required in the
                channels.

        Returns:
            self.invalid_fields_mask
        """
        remainder_counts = self.count_remainders(ignore_invalid_fields=True)
        for e, ex_remainder_counts in enumerate(remainder_counts):
            if ex_remainder_counts is False:
                continue
            if channels is None:
                if any([channel_remainder_counts < min_remainders
                        for c, channel_remainder_counts
                        in ex_remainder_counts.iteritems()]):
                    self.invalid_fields_mask[e] = False
            else:
                if any([channel_remainder_counts < min_remainders
                        for c, channel_remainder_counts
                        in ex_remainder_counts.iteritems() if c in channels]):
                    self.invalid_fields_mask[e] = False
        return self.invalid_fields_mask
        

class TimetraceExperiment(Experiment):
    """
    Continuously films a single field of view in a single channel. Usually used
    to study Spot behavior under constant excitation.

    Currently does not implement offsets.

    Stepfitting is a significant component of this Experiment subclass. A
    significant portion of the documentation is made in the context of, and
    using the vocabulary of, stepfitting_library.

    Attributes:
        frames: Filmed sequence of Images of the field. These Images may
            contain Spots. All frames are assumed to be of the same shape.
        spot_traces: A list of SimpleTraces.
        step_fits: Step fits for spot_traces. This is a dictionary with keys
            that are the rounded integer pixel coordinates (h, w) of the first
            Spot for a trace, and values are PlateauTraces.
        step_fit_intermediates: Dictionary containing intermediate data
            produced during step fitting. The keys and values in this
            dictionary are arbitrary; this is meant to be a flexible place to
            store information about step fitting. Some class methods below may
            add or modify some of these entries. They will use pre-programmed
            keys and data structures as values to store their information. In
            general, they will store information on a per-track basis: the
            track's first Spot's (h, w) coordinates will be the key in
            step_fit_intermediates. The value under this key will be another
            dictionary, with standardized keys and values that store something
            useful. Read the method docstrings for details. The class user may
            also directly access the dictionary. It is up to the user to keep
            things organizized. This class often assumes that all traces have
            an identical set of intermediates.
    """
    def __init__(self, frames, spot_traces=None, step_fits=None,
                 step_fit_intermediates=None):
        self.frames = frames
        self.spot_traces = spot_traces
        self.step_fits = step_fits
        if step_fit_intermediates is None:
            self.step_fit_intermediates = {}
        else:
            self.step_fit_intermediates = step_fit_intermediates

    def lc_create_traces(self, initial_spots=None, search_radius=3.0,
                         s_n_cutoff=3.0):
        """
        Create traces for a set of initial Spots through this experiment's
        frames using Experiment.luminosity_centroid_particle_tracking.

        Will use Spots in the first frame unless overridden via initial_spots.
        If initial_spots' frame is not the first in self.frames, the traces
        will start in their parent_Image; traces will not be padded if
        initial_spots are not in the first frame.

        Arguments:
            initial_spots: If specified, these Spots -- given as a list -- will
                be the ones tracked. Otherwise will use Spots in the first
                frame by default. All initial_spots must be in the same frame.
            search_radius: Parameter used by
                Experiment.luminosity_centroid_particle_tracking.
            s_n_cutoff: Parameter used by
                Experiment.luminosity_centroid_particle_tracking.

        Returns:
            The updated self.spot_traces.
        """
        #The first step is to find what frame the trace starts in. It is the
        #first frame by default if initial_spots is not specified. Using None
        #as a placeholder if the frame is not found.
        first_frame, first_frame_index = None, None
        if initial_spots is not None:
            if initial_spots[0].parent_Image is None:
                raise ValueError("All initial_spots must have the same "
                                 "parent_image, and it must be one of the "
                                 "frames in this experiment.")
            else:
                initial_parent_Image = initial_spots[0].parent_Image
                #Search through self.frames to find this parent_Image.
                for f, frame in enumerate(self.frames):
                    if frame is initial_parent_Image:
                        first_frame, first_frame_index = frame, f
                        break
                else:
                    raise ValueError("All initial_spots must have the same "
                                     "parent_image, and it must be one of the "
                                     "frames in this experiment.")
        elif initial_spots is None and self.frames[0].spots is not None:
            initial_spots = self.frames[0].spots
            first_frame, first_frame_index = self.frames[0], 0
        elif initial_spots is None and self.frames[0].spots is None:
            raise ValueError("Cannot create traces unless either the first "
                             "frame does has Spots, or initial_spots are "
                             "specified via argument.")
        assert (first_frame is not None and first_frame_index is not None and
                initial_spots is not None)
        #Check that all spots have the same parent_Image
        if not all([s.parent_Image is first_frame for s in initial_spots]):
            raise ValueError("All initial_spots must have the same "
                             "parent_image, and it must be one of the frames "
                             "in this experiment.")
        raw_spot_traces = \
         Experiment.luminosity_centroid_particle_tracking(
                                        frames=self.frames[first_frame_index:],
                                        initial_spots=initial_spots,
                                        search_radius=search_radius,
                                        s_n_cutoff=s_n_cutoff,
                                        offsets=None)
        #Incorporate new spots into their parent_Images.
        for trace in raw_spot_traces:
            for spot in trace:
                if spot is None:
                    continue
                if spot.parent_Image.spots is None:
                    spot.parent_Image.spots = []
                spot.parent_Image.spots.append(spot)
        self.spot_traces = [SimpleTrace(trace) for trace in raw_spot_traces]
        return self.spot_traces

    def wildcolor_plot_tracks(self, filepath_prefix,
                              color_list = ('red', 'blue', 'yellow', 'purple',
                                            'orange', 'pink', 'lightblue',
                                            'green'), num_colors=8):
        """
        Plots self.spot_traces for visual sanity check.

        Arguments:
            filpath_prefix: Uniform prefix to save images with. Image filepaths
                will be of the form

                filepath_prefix + str(f).zfill(frame_zfill) + '.png'

                where f is the frame index.
            color_list: List of colors to highlight Spots with.
            num_colors: Number of colors to use out of color_list.

        Returns:
            Tuple of filepaths to saved images.
        """
        saved_filepaths = []
        logger = logging.getLogger()
        if self.spot_traces is not None:
            #assign initial spots colors from color list
            color_assignment = {t: random.choice(color_list[:num_colors])
                                for t, track in enumerate(self.spot_traces)}
            for f, frame in enumerate(self.frames):
                frame_zfill = int(np.ceil(math.log(len(self.frames), 10)))
                output_path = (filepath_prefix + str(f).zfill(frame_zfill) +
                               '.png')
                #we need to send fake psfs to pflib.save_psfs_png; the only
                #things that matter are the (h, w) keys, and the tuple as the
                #value is just going to be a filler of 0's
                psf_filler = tuple([0] * 12)
                filler_psfs = {}
                square_colors = {}
                for t, track in enumerate(self.spot_traces):
                    color = color_assignment[t]
                    photometry = track.photometry(f)
                    h, w = track.coordinates(f)
                    if not ((h is None and w is None) or
                            (h is not None and w is not None)):
                        raise Exception("h and w must either both be None " +
                                        "or both be not None; cannot be " +
                                        "(h, w) = " + str((h, w)))
                    if h is None or w is None:
                        continue
                    filler_psfs.setdefault(track.coordinates(f),
                                           psf_filler)
                    square_colors.setdefault(track.coordinates(f),
                                             color)
                    logger.debug("wildcolor_plot_tracks: frame " + str(f) +
                                 ", track " + str(t) + " (h, w) = " +
                                 str(track.coordinates(f)))
                saved_filepath = \
                     pflib.save_psfs_png(psfs=filler_psfs,
                                         image_path=frame.metadata['filepath'],
                                         timestamp_epoch=None,
                                         output_path=output_path,
                                         square_size=9,
                                         square_color=None,
                                         square_colors=square_colors)
                saved_filepaths.append(saved_filepath)
        return tuple(saved_filepaths)

    def stepfit_tracks(self, photometry_min=None,
                       photometry_method='mexican_hat', mirror_start=0,
                       chung_kennedy=0, p_threshold=0.01, **kwargs):
        """
        Stepfit the photometries of all self.spot_traces.

        Arguments are passed to TimetraceExperiment.trace_photometries and
        TimetraceExperiment.stepfit_photometries: read their docstrings.
        **kwargs is passed to TimetraceExperiment.trace_photometries.

        Updates self.step_fits with the results.

        Updates self.step_fit_intermediates with photometries,
        ck_filtered_photometries, plateaus, and t_filtered_plateaus as returned
        by stepfit_photometries. The data is stored in
        self.step_fit_intermediates in a per-track format: each track's first
        Spot's (h, w) coordinates act as keys. The values of the dictionary are
        themselves, in turn, another dictionary. In this per-track dictionary,
        the keys updated are the names of the data returned, i.e.
        'photometries', 'ck_filtered_photometries', and 'plateaus'. The values
        are the corresponding data structures (Trace instances) themselves,
        unaltered. Here is an illustration (showing only an entry in
        self.step_fit_intermediates updated by this function; other entries may
        be present from elsewhere):

        {
         (6, 7): {'photometries': PhotometryTrace instance,
                  'ck_filtered_photometries': PhotometryTrace instance,
                  'plateaus': PlateauTrace instance,
                  't_filtered_plateaus': PlateauTrace instance}
        }

        Updates to these dictionary will overwrite existing entries as
        necessary to make the updates.

        Returns the updated self.step_fits and self.step_fit_intermediates.
        """
        logger = logging.getLogger()
        #Not updating the instance dictionaries live, in case we hit an
        #exception before completing. Hence, declaring these temporary
        #containers.
        step_fits = {}
        step_fit_intermediates = {}
        for t, trace in enumerate(self.spot_traces):
            #Find the first non-None spot and use its (h, w) coordinates as the
            #key in step_fits.
            h, w = trace.h, trace.w
            if (h, w) in step_fits or (h, w) in step_fit_intermediates:
                raise Exception("Two tracks have initial Spots with identical "
                                "(h, w).")
            (photometries,
             ck_filtered_photometries,
             plateaus,
             t_filtered_plateaus) = trace.stepfit_photometries(
                                           h, w,
                                           mirror_start=mirror_start,
                                           chung_kennedy=chung_kennedy,
                                           p_threshold=p_threshold,
                                           photometry_method=photometry_method,
                                           **kwargs)
            step_fits.setdefault((h, w), t_filtered_plateaus)
            step_fit_intermediates.setdefault((h, w),
                         {'photometries': photometries,
                          'ck_filtered_photometries': ck_filtered_photometries,
                          'plateaus': plateaus,
                          't_filtered_plateaus': t_filtered_plateaus})
        self.step_fits = step_fits
        for (h, w), intermediates in step_fit_intermediates.iteritems():
            #If self.step_fit_intermediates[(h, w)] does not exist, this will
            #initialize it with an empty dictionary. If it already exists, this
            #will do nothing.
            self.step_fit_intermediates.setdefault((h, w), {})
            #It is possible that self.step_fit_intermediates[(h, w)] already
            #existed and was not a dictionary. Try updating it. If this fails,
            #overwrite whatever was there with a dictionary.
            try:
                self.step_fit_intermediates[(h, w)].update(intermediates)
            except Exception as e:
                logger.exception("flexlibrary.TimetraceExperiment." +
                                 "stepfit_tracks: updating self." +
                                 "step_fit_intermediates failed for " +
                                 "(h, w) = " + str((h, w)) + " with " +
                                 "Exception " + str(e) + "; overwriting " +
                                 "this key with a fresh dictionary and " +
                                 "updating with intermediates.", e,
                                 exc_info=True)
                self.step_fit_intermediates[(h, w)] = intermediates
        return self.step_fits, self.step_fit_intermediates

    def _get_all_intermediates(self):
        """Get all step fit intermediates for all traces."""
        #Check that all traces have identical intermediates.
        intermediate_key_sets = {(h, w): set(i_dict.keys())
                                 for (h, w), i_dict
                                 in self.step_fit_intermediates.iteritems()}
        test_hw, test_set = intermediate_key_sets.popitem()
        if not all(test_set == s
                   for (h, w), s in intermediate_key_sets.iteritems()):
            raise Exception("All traces must have identical intermediates.")
        return test_set

    def save_experiment_as_csv(self, output_path, dialect='excel',
                               include_step_fits=False,
                               photometry_method='mexican_hat',
                               include_intermediates=None, **kwargs):
        """
        Saves the current status of this experiment, i.e. its attributes
        self.spot_traces, self.step_fits, and self.step_fit_intermediates to a
        CSV.

        The CSV columns for self.spot_traces are always included, as follows:


        Trace #    Hcoord    Wcoord    Frame #    Photometry
        -------    ------    ------    -------    ----------


        Traces are written to the file in order they are stored in
        self.spot_traces. Each of their Spots is written in the order they are
        stored in the trace. One row is written per Spot. Trace # stores the
        index of the Trace, based on its order in self.spot_traces (0-indexed).
        Hcoord and Wcoord are the h and w coordinates of the Trace. Frame # is
        the frame # of the Spot being written. Photometry is the photometry of
        the Spot.

        If self.step_fits are included, the following additional columns are
        added for each Spot:


        Step #    Plateau Height     Step Size    Plateau Length
        ------    --------------     ---------    --------------

        Overall Fit R^2
        ---------------


        Step # is the number of steps preceeding that row's frame. It is set to
        0 if there were no prior steps yet, i.e. that we're still on the first
        plateau. Plateau Height is the height of the current photometry
        plateau. Step size is the size of the last step. Plateau Length is the
        length of the current plateau, measured in the total number of frames.
        Overall Fit R^2 is the coefficient of determination R^2 measure of the
        quality of fit for this entire fit (not just the current plateau).

        Optionally, information stored in step_fit_intermediates can be added as
        columns. For each entry in self.step_fit_intermediates requested, one
        additional column will be added to the CSV file. This column will use
        for its header the string version of its self.step_fit_intermediates
        dictionary key. This function assumes that each member of
        self.step_fit_intermediates that is requested for inclusion implements
        a frame_output(frame) method. This method's output is converted to a
        string and fills the cell.

        To accelerate processing, this method assumes that all Trace classes it
        uses utilize a plateau_starts method that returns a set of frame
        indexes when plateau starts occur.

        Arguments:
            output_path: Save file at output_path. Overwrites existing file if
                present.
            dialect: Dialect for Python's CSV library to use.
            include_step_fits: If True, include self.step_fits information.
            photometry_method: Method to calculate the photometry. Must be an
                available option in Spot.photometry.
            include_intermediates: List of keys from
                self.step_fit_intermediates, indicating which information to
                include in the CSV file. If just set to value 'True', will
                include all intermediates that have a frame_output() method.
                Note that the order of parameters output in the CSV depends on
                the order of the include_intermediates list. If True, Python's
                sorted() is used as the order.
            **kwargs: All other arguments are passed as parameters to
                frame_output().

        Returns:
            Number of rows written, including the header.
        """
        logger = logging.getLogger()
        rows_written = 0
        with open(output_path, 'w') as writer_file:
            writer = csv.writer(writer_file, dialect=dialect)
            header = ['Trace #', 'Hcoord', 'Wcoord', 'Frame #', 'Photometry']
            if include_step_fits:
                header += ['Step #', 'Plateau Height', 'Step Size',
                           'Plateau Length', 'Overall Fit R^2']
            if include_intermediates is True:
                include_intermediates = list(self._get_all_intermediates())
            if include_intermediates is not None:
                include_intermediates = sorted(include_intermediates)
                header += [str(intermediate)
                           for intermediate in include_intermediates]
            writer.writerow(header)
            rows_written += 1
            logger.debug("flexlibrary.TimetraceExperiment." +
                         "save_experiment_as_csv: include_intermediates = " +
                         str(include_intermediates))
            for t, trace in enumerate(self.spot_traces):
                #           Trace # Hcoord        Wcoord
                row_base = [str(t), str(trace.h), str(trace.w)]
                trace_intermediates = \
                                self.step_fit_intermediates[(trace.h, trace.w)]
                #Speed up plateau-related columns for step-fit by making a data
                #structure that keeps track of where plateau starts are.
                #Initialize column data with first plateau.
                if include_step_fits:
                    sf = self.step_fits[(trace.h, trace.w)]
                    sf_plateau_starts = sf.plateau_starts()
                    #Last step #, last step position, last step size
                    ls_num, ls_pos, ls_mag = sf.last_step_info(0)
                    #plateau start, stop, height
                    (pa, po, ph), pi = sf.frame_plateau(0)
                    plateau_length = po - pa + 1
                    #R^2. This stays constant for the entire trace, so no need
                    #to recompute per-frame.
                    r_2 = Trace.coefficient_of_determination(trace, sf,
                                           photometry_method=photometry_method,
                                                             **kwargs)
                #Speed up plateau-related intermediates by caching using
                #plateau starts.
                if include_intermediates is not None:
                    intermediates_plateau_starts = \
                            {intermediate:
                             trace_intermediates[intermediate].plateau_starts()
                             for intermediate in include_intermediates}
                    assert all(0 in ps
                               for i, ps
                               in intermediates_plateau_starts.iteritems())
                    #Initialize using Nones. Frame 0 and all subsequent starts
                    #will get update inside loop below.
                    intermediates_cache = {intermediate: None
                                     for intermediate in include_intermediates}
                for f in range(trace.num_frames):
                    row = row_base + [str(f)]
                    row += [trace.photometry(f,
                                photometry_method=photometry_method, **kwargs)]
                    #Update plateau-specific data if we're starting a new
                    #plateau.
                    if include_step_fits and f in sf_plateau_starts:
                        #Last step #, last step position, last step size
                        ls_num, ls_pos, ls_mag = sf.last_step_info(f)
                        #plateau start, stop, height
                        (pa, po, ph), pi = sf.frame_plateau(f)
                        plateau_length = po - pa + 1
                    if include_step_fits:
                        row += [str(ls_num), str(ph), str(ls_mag),
                                str(plateau_length), str(r_2)]
                    if include_intermediates is not None:
                        #Update chache if we're at a new plateau.
                        for (intermediate,
                           starts) in intermediates_plateau_starts.iteritems():
                            if f in starts:
                                intermediates_cache[intermediate] = \
                                            (trace_intermediates[intermediate].
                                                               frame_output(f))
                        row += [str(intermediates_cache[intermediate])
                                for intermediate in include_intermediates]
                    logger.debug("flexlibrary.TimetraceExperiment." +
                                 "save_experiment_as_csv: row = " + str(row))
                    writer.writerow(row)
                    rows_written += 1
        return rows_written

    def save_traces_pkl(self, path):
        """Save self.spot_traces to a pkl file at path."""
        cPickle.dump(self.spot_traces, open(path, 'w'))

    def save_stepfits_as_csv(self, output_path, min_step_magnitude=0.0,
                             method='t_test', photometry_min=None,
                             remove_blips=False, chung_kennedy=0,
                             smoothing_stddev=0.8, downsteps_only=False,
                             p_threshold=0.01, min_step_noise_ratio=0.0,
                             window_radius=10, double_t=1.0, drop_sort=True,
                             linear_fit_threshold=1.0, min_step_length=2,
                             median_filter=0, num_steps=10, magic_start=0,
                             mirror_start=0):
        raise DeprecationWarning("This is a giant hairball.")
        logger = logging.getLogger()
        self.lc_create_tracks()
        output_writer = csv.writer(open(output_path, 'w'), dialect='excel')
        output_writer.writerow(['Spot #',
                                'Frame #',
                                'Measured Photometry',
                                'Filtered Photometry',
                                'Unfiltered Plateaus',
                                'Downstep+MinStep Filtered Step Fits',
                                'Step Fit Photometry',
                                'Downstep Filtered Step Fits',
                                'Last Step #',
                                'Last Step Position (Frame)',
                                'Last Step Magnitude', 'Plateau #',
                                'First Frame H Coordinate',
                                'First Frame W Coordinate',
                                #'Plateau R^2',
                                'Total R^2',
                                'Linear Explainer'])
        for t, track in enumerate(self.spot_traces):
            photometries = [spot.photometry(method='mexican_hat',
                                                   brim_size=4,
                                                   radius=5)
                            if spot is not None else 0
                            for spot in track]
            if photometry_min is not None:
                photometries = [max(photometry_min, p) for p in photometries]
            unfiltered_photometries = photometries
            if mirror_start > 0:
                photometries = [x for x in reversed(photometries[:mirror_start])] + photometries
                unfiltered_photometries = [x for x in reversed(unfiltered_photometries[:mirror_start])] + unfiltered_photometries
            for c in range(chung_kennedy):
                photometries = \
                         stepfitting_library.chung_kennedy_filter(luminosities=photometries, window_lengths=(2, 4, 8, 16))
            if median_filter > 0:
                photometries = medfilt(photometries, kernel_size=median_filter)
            #photometries = Experiment.mean_filter(photometries, rank=5)
            if method == 'chi_squared':
                if mirror_start > 0:
                    raise NotImplementedError("chi_squared not supported with mirror_start because I'm trying to get this thing to work asap.")
                plateaus = \
                    stepfitting_library.chi_squared_step_fitter(
                                         luminosity_sequence=photometries,
                                         num_steps=num_steps,
                                         min_step_length=min_step_length,
                                         min_step_magnitude=min_step_magnitude,
                                         ignore_counterfits=False)
                if magic_start > 0:
                    magic_frames = unfiltered_photometries[:magic_start]
                    magic_frames_diff = np.diff(magic_frames)
                    largest_magic_frame_diff = np.argmax(magic_frames_diff) #largest diff is from unfiltered_photometries[largest_magic_frame_diff] to the next frame
                    first_start, first_stop, first_height = first_plateau = plateaus[0]
                    if first_stop > largest_magic_frame_diff and first_stop - largest_magic_frame_diff > 3:
##                        stepfitting_library._split_plateau(unfiltered_photometries, first_plateau, forbidden_splits=None)
                        p_a = first_start, largest_magic_frame_diff, 0 #ok to use 0 because refitting happens next
                        p_b = largest_magic_frame_diff + 1, first_stop, 0
                        plateaus = [p_a, p_b] + plateaus[1:]
                plateaus = stepfitting_library.refit_plateaus(unfiltered_photometries, plateaus)
                unfiltered_plateaus = plateaus
                if downsteps_only:
                    downstep_filtered_plateaus = \
                        stepfitting_library.filter_upsteps(photometries,
                                                            plateaus)
                    for n in range(len(downstep_filtered_plateaus)):
                        downstep_filtered_plateaus = \
                            stepfitting_library.filter_upsteps(photometries,
                                                    downstep_filtered_plateaus)
                else:
                    downstep_filtered_plateaus = plateaus
                if min_step_magnitude is not None:
                    downstep_minstep_filtered_plateaus = \
                        stepfitting_library.filter_small_steps(
                                     photometries,
                                     downstep_filtered_plateaus,
                                     min_magnitude=min_step_magnitude,
                                     min_noise_ratio=min_noise_ratio)
                    for n in range(len(downstep_minstep_filtered_plateaus)):
                        downstep_minstep_filtered_plateaus = \
                          stepfitting_library.filter_small_steps(photometries,
                                                     downstep_minstep_filtered_plateaus,
                                                  min_magnitude=min_step_magnitude,
                                     min_noise_ratio=min_step_noise_ratio)
                else:
                    downstep_minstep_filtered_plateaus = downstep_filtered_plateaus
                if double_t < 1:
                    magic_start_nofilter, first_plateau = False, None
                    #magic_start_nofilter = True
                    if magic_start > 0:
                        first_start, first_stop, first_height = first_plateau = downstep_minstep_filtered_plateaus[0]
                        if first_stop < magic_start:
                            magic_start_nofilter = True
                    #if len(downstep_minstep_filtered_plateaus) > 1:
                    #    plateau_a = downstep_minstep_filtered_plateaus[0]
                    #    plateau_b = downstep_minstep_filtered_plateaus[1]
                    #    a_start, a_stop, a_height = plateau_a
                    #    if a_start < 6:
                    #        refitted_a, refitted_b = stepfitting_library.best_t_test_split(unfiltered_photometries, plateau_a, plateau_b, p_threshold, split_range=None, find_best_p=False)
                    #        downstep_minstep_filtered_plateaus[0] = refitted_a
                    #        downstep_minstep_filtered_plateaus[1] = refitted_b
                    for n in range(len(downstep_minstep_filtered_plateaus)):
                        downstep_minstep_filtered_plateaus = stepfitting_library.t_test_filter(unfiltered_photometries, downstep_minstep_filtered_plateaus, drop_sort=drop_sort, p_threshold=double_t, magic_start_nofilter=magic_start_nofilter)
#                    if magic_start_nofilter:
#                        downstep_minstep_filtered_plateaus[0] = first_plateau
            elif method == 't_test':
                plateaus = \
                    stepfitting_library.sliding_t_fitter(
                                         luminosity_sequence=photometries,
                                         window_radius=window_radius,
                                         p_threshold=p_threshold,
                                         median_filter_size=None,
                                         downsteps_only=False,
                                         min_step_magnitude=None)
                if magic_start > 0:
                    magic_frames = unfiltered_photometries[:magic_start]
                    magic_frames_diff = np.diff(magic_frames)
                    largest_magic_frame_diff = np.argmax(magic_frames_diff) #largest diff is from unfiltered_photometries[largest_magic_frame_diff] to the next frame
                    first_start, first_stop, first_height = first_plateau = plateaus[0]
                    if first_stop > largest_magic_frame_diff and first_stop - largest_magic_frame_diff > 3:
##                        stepfitting_library._split_plateau(unfiltered_photometries, first_plateau, forbidden_splits=None)
                        p_a = first_start, largest_magic_frame_diff, 0 #ok to use 0 because refitting happens next
                        p_b = largest_magic_frame_diff + 1, first_stop, 0
                        plateaus = [p_a, p_b] + plateaus[1:]
                plateaus = stepfitting_library.refit_plateaus(unfiltered_photometries, plateaus)
                unfiltered_plateaus = plateaus
                if downsteps_only:
                    downstep_filtered_plateaus = \
                        stepfitting_library.filter_upsteps(photometries,
                                                            plateaus)
                    for n in range(len(downstep_filtered_plateaus)):
                        downstep_filtered_plateaus = \
                            stepfitting_library.filter_upsteps(photometries,
                                                    downstep_filtered_plateaus)
                else:
                    downstep_filtered_plateaus = plateaus
                if min_step_magnitude is not None:
                    downstep_minstep_filtered_plateaus = \
                        stepfitting_library.filter_small_steps(photometries,
                                                          downstep_filtered_plateaus,
                                                        min_magnitude=min_step_magnitude,
                                     min_noise_ratio=min_step_noise_ratio)
                    for n in range(len(downstep_minstep_filtered_plateaus)):
                        downstep_minstep_filtered_plateaus = \
                          stepfitting_library.filter_small_steps(photometries,
                                                     downstep_minstep_filtered_plateaus,
                                                  min_magnitude=min_step_magnitude,
                                     min_noise_ratio=min_step_noise_ratio)
                else:
                    downstep_minstep_filtered_plateaus = downstep_filtered_plateaus
                no_double_t_test_filtered_plateaus = downstep_minstep_filtered_plateaus
                logger.debug("flexlibrary.save_stepfits_as_csv: track t = " + str(t) + "; no_double_t_test_filtered_plateaus at assignment = " + str(no_double_t_test_filtered_plateaus))
                if double_t < 1:
                    magic_start_nofilter, first_plateau = False, None
                    #magic_start_nofilter = True
                    if magic_start > 0:
                        first_start, first_stop, first_height = first_plateau = downstep_minstep_filtered_plateaus[0]
                        if first_stop < magic_start:
                            magic_start_nofilter = True
                    #if len(downstep_minstep_filtered_plateaus) > 1:
                    #    plateau_a = downstep_minstep_filtered_plateaus[0]
                    #    plateau_b = downstep_minstep_filtered_plateaus[1]
                    #    a_start, a_stop, a_height = plateau_a
                    #    if a_start < 6:
                    #        refitted_a, refitted_b = stepfitting_library.best_t_test_split(unfiltered_photometries, plateau_a, plateau_b, p_threshold, split_range=None, find_best_p=False)
                    #        downstep_minstep_filtered_plateaus[0] = refitted_a
                    #        downstep_minstep_filtered_plateaus[1] = refitted_b
                    for n in range(len(downstep_minstep_filtered_plateaus)):
                        downstep_minstep_filtered_plateaus = stepfitting_library.t_test_filter(unfiltered_photometries, downstep_minstep_filtered_plateaus, drop_sort=drop_sort, p_threshold=double_t, magic_start_nofilter=magic_start_nofilter)
#                    if magic_start_nofilter:
#                        downstep_minstep_filtered_plateaus[0] = first_plateau
            else:
                ValueError("Invalid method selected.")
            #if remove_blips and len(plateaus) > 3:
            #    plateaus = \
            #        stepfitting_library.remove_blips(luminosities=photometries,
            #                                         plateaus=plateaus,
            #                                 smoothing_stddev=smoothing_stddev)
            if mirror_start > 0:
                photometries = photometries[mirror_start:]
                unfiltered_photometries = unfiltered_photometries[mirror_start:]
                plateaus = [(a - mirror_start, o - mirror_start, h) for a, o, h in plateaus]
                tmp_plateaus = []
                for a, o, h in plateaus:
                    if a < 0 and o < 0:
                        continue
                    elif a < 0 and o >= 0:
                        tmp_plateaus.append((0, o, h))
                    else:
                        tmp_plateaus.append((a, o, h))
                plateaus = tmp_plateaus
                tmp_plateaus = []
                downstep_minstep_filtered_plateaus = [(a - mirror_start, o - mirror_start, h) for a, o, h in downstep_minstep_filtered_plateaus]
                for a, o, h in downstep_minstep_filtered_plateaus:
                    if a < 0 and o < 0:
                        continue
                    elif a < 0 and o >= 0:
                        tmp_plateaus.append((0, o, h))
                    else:
                        tmp_plateaus.append((a, o, h))
                downstep_minstep_filtered_plateaus = tmp_plateaus
                tmp_plateaus = []
                downstep_filtered_plateaus = [(a - mirror_start, o - mirror_start, h) for a, o, h in downstep_filtered_plateaus]
                for a, o, h in downstep_filtered_plateaus:
                    if a < 0 and o < 0:
                        continue
                    elif a < 0 and o >= 0:
                        tmp_plateaus.append((0, o, h))
                    else:
                        tmp_plateaus.append((a, o, h))
                downstep_filtered_plateaus = tmp_plateaus
                tmp_plateaus = []
                unfiltered_plateaus = [(a - mirror_start, o - mirror_start, h) for a, o, h in unfiltered_plateaus]
                for a, o, h in unfiltered_plateaus:
                    if a < 0 and o < 0:
                        continue
                    elif a < 0 and o >= 0:
                        tmp_plateaus.append((0, o, h))
                    else:
                        tmp_plateaus.append((a, o, h))
                unfiltered_plateaus = tmp_plateaus
                tmp_plateaus = []
                no_double_t_test_filtered_plateaus = [(a - mirror_start, o - mirror_start, h) for a, o, h in no_double_t_test_filtered_plateaus]
                for a, o, h in no_double_t_test_filtered_plateaus:
                    if a < 0 and o < 0:
                        continue
                    elif a < 0 and o >= 0:
                        tmp_plateaus.append((0, o, h))
                    else:
                        tmp_plateaus.append((a, o, h))
                no_double_t_test_filtered_plateaus = tmp_plateaus
            if mirror_start > 0 and len(no_double_t_test_filtered_plateaus) > 1:
                #no_double_t_test_filtered_plateaus = downstep_minstep_filtered_plateaus
                logger.debug("flexlibrary.save_stepfits_as_csv: track t = " + str(t) + "; downstep_minstep_filtered_plateaus[0] = " + str(downstep_minstep_filtered_plateaus[0]))
                logger.debug("flexlibrary.save_stepfits_as_csv: track t = " + str(t) + "; no_double_t_test_filtered_plateaus[:2] = " + str(no_double_t_test_filtered_plateaus[:2]))
                pa_start, pa_stop, pa_height = downstep_minstep_filtered_plateaus[0]
                #pb_start, pb_stop, pb_height = downstep_minstep_filtered_plateaus[1]
                pc_start, pc_stop, pc_height = no_double_t_test_filtered_plateaus[0]
                pd_start, pd_stop, pd_height = no_double_t_test_filtered_plateaus[1]
                if pc_stop < mirror_start and pc_stop < pa_stop:
                    downstep_minstep_filtered_plateaus = [stepfitting_library._fit_plateau(unfiltered_photometries, pc_start, pc_stop), stepfitting_library._fit_plateau(unfiltered_photometries, pc_stop + 1, pa_stop)] + downstep_minstep_filtered_plateaus[1:]
                logger.debug("flexlibrary.save_stepfits_as_csv: track t = " + str(t) + "; downstep_minstep_filtered_plateaus[:2] after split = " + str(downstep_minstep_filtered_plateaus[:2]))
            steps = stepfitting_library.plateaus_to_steps(plateaus)
            logger.debug("TimetraceExperiment.save_stepfits_as_csv: for " +
                         "track t = " + str(t) + ", steps = " + str(plateaus))
            first_frame_h, first_frame_w = track[0].h, track[0].w
            r_2 = stepfitting_library.stepfit_r_squared(unfiltered_photometries, downstep_minstep_filtered_plateaus)
            linear_fits = stepfitting_library.linear_fits(unfiltered_photometries,
                                                          downstep_minstep_filtered_plateaus)
            LLa, LLb, LLr = largest_linear_explainer = stepfitting_library.best_linear_explainer(linear_fits, plateaus=downstep_minstep_filtered_plateaus, track_index=t, linear_fit_threshold=linear_fit_threshold)
            #logger.debug("flexlibrary.TimetraceExperiment.save_stepfits_as_csv: " +
            #             "track t = " + str(t) + "; LLa, LLb, LLr = " + str((LLa, LLb, LLr)))
            if (LLa is not None) and (LLr is None or LLr > linear_fit_threshold):
                continue
            for frame, spot in enumerate(track):
                photometry = photometries[frame]
                unfiltered_photometry = unfiltered_photometries[frame]
                plateau_height = stepfitting_library.plateau_value(plateaus,
                                                                   frame)
                downstep_filtered_plateau_height = stepfitting_library.plateau_value(downstep_filtered_plateaus,
                                                                   frame)
                downstep_minstep_filtered_plateau_height = stepfitting_library.plateau_value(downstep_minstep_filtered_plateaus,
                                                                   frame)
                plateau_num = stepfitting_library.frame_plateau(plateaus,
                                                                      frame)[1]
                last_step = stepfitting_library.last_step_info(steps, frame)
                if last_step is not None:
                    last_step_num, last_step_pos, last_step_mag = last_step
                else:
                    last_step_num, last_step_pos, last_step_mag = ('NA', 'NA',
                                                                   'NA')
                frame_plateau = stepfitting_library.frame_plateau(downstep_minstep_filtered_plateaus, frame)[0]
                #pr_2 = stepfitting_library.stepfit_r_squared(unfiltered_photometries, [frame_plateau])
                LLoutput = str(None)
                if LLa is not None:
                    assert LLb is not None
                    p_start, p_stop = (downstep_minstep_filtered_plateaus[LLa],
                                       downstep_minstep_filtered_plateaus[LLb])
                    p_start_mid = int(np.around((p_start[1] - p_start[0]) / 2.0) + p_start[0])
                    p_stop_mid = int(np.around((p_stop[1] - p_stop[0]) / 2.0) + p_stop[0])
                    #f_start, f_stop = (downstep_minstep_filtered_plateaus[LLa][0],
                    #                   downstep_minstep_filtered_plateaus[LLb][1])
                    f_start, f_stop = p_start_mid, p_stop_mid
                    if f_start <= frame <= f_stop:
                        if LLr is None:
                            LLoutput = 'NaN'
                        else:
                            LLoutput = str(LLr)
                unfiltered_plateau_height = stepfitting_library.plateau_value(unfiltered_plateaus, frame)
                output_writer.writerow([t, frame, unfiltered_photometry,
                                        photometry,
                                        unfiltered_plateau_height,
                                        downstep_minstep_filtered_plateau_height,
                                        plateau_height,
                                        downstep_filtered_plateau_height,
                                        last_step_num, last_step_pos,
                                        last_step_mag, plateau_num,
                                        first_frame_h, first_frame_w, r_2,
                                        LLoutput])
