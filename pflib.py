#!/home/boulgakov/anaconda2/bin/python


"""
Routines to identify labeled peptides in fluorosequencing TIRF images
and characterize their point spread functions, and associated utility
functions.
Marcotte Lab & Zack Simpson
"""


import numpy as np
import subprocess
import logging
import os
import sys
from scipy.ndimage.filters import median_filter
from scipy.signal import  correlate
import itertools
#import agpy#importing gaussfitter directly from corrected agpy version instead
#sys.path.append('./agpy')
sys.path.insert(0, './agpy')
import gaussfitter
import math
from scipy.misc import imread
import os.path
import multiprocessing
import cPickle
import time
import csv
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from skimage import exposure
import scipy

#if the calling application does not have logging setup, pflib will log
#to NullHandler by default
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#Peak fitting is expensive. Matrix correlation efficiently identifies
#areas of the image where peaks may be present. This is the default
#correlation matrix, obtained empirically by sampling images for peptide
#point spread functions. See find_peptides for algorithm details and
#other default values.
default_correlation_matrix = np.array([[-5935, -5935, -5935, -5935, -5935],
                                       [-5935,  8027,  8027,  8027, -5935],
                                       [-5935,  8027, 30742,  8027, -5935],
                                       [-5935,  8027,  8027,  8027, -5935],
                                       [-5935, -5935, -5935, -5935, -5935]])


def convert_image(input_path, output_path=None, output_format='png',
                  convert_command='convert'):
    """
    A wrapper to convert an image into the desired format by calling an
    external utility via the shell. This wrapper essentially performs
    subprocess.Popen([convert_command, input_path, output_path])
    Arguments:
        input_path: Path to source image.
        output_path: If not None, indicates path to write the converted
            image. If None, defaults to image_path + output_format
            suffix. Will overwrite existing file at the output path, if
            present.
        output_format: Desired output image format. Defaults to PNG.
        convert_command: Command used to convert image. Defaults to
            ImageMagick 'convert'.
    Returns:
        Path to converted image if successful, None otherwise.
    """
    logger = logging.getLogger()
    #if output_path not specified, use original path + format extension
    if output_path is None:
        output_path = '.'.join((input_path, output_format))
    try:
        p = subprocess.Popen([convert_command, input_path, output_path],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #log convert_commands output
        stdout, stderr = p.communicate()
        if stdout:
            logger.debug(stdout)
        if stderr:
            logger.debug(stderr)
        p.wait()
    except Exception as e:
        logger.exception(e, exc_info=True)
        output_path = None
    return output_path


def _2d_gaussian_function(H,A,h_0,w_0,sigma_h,sigma_w,theta,h,w):
    '''
	A*exp( -(a(h - h_0)^2 + 2b(h - h_0)(w - w_0) + c(w - w_0)^2) )
		    + H

		with the coefficients
		    a = (cos(theta))^2/(2*sigma_h^2) +
			(sin(theta))^2/(2*sigma_w^2)
		    b = sin(2*theta)/(4*sigma_h^2) -
			sin(2*theta)/(4*sigma_w^2)
		    c = (sin(theta))^2/(2*sigma_h^2) +
			(cos(theta))^2/(2*sigma_w^2)     
    '''
    
    #a = np.divide(((np.cos(theta))**2),(2*sigma_h**2)) +np.divide((np.sin(theta))**2,(2*sigma_w**2))
    #b = np.divide(np.sin(2*theta),(4*sigma_h**2)) -np.divide(np.sin(2*theta),(4*sigma_w**2))
    #c = np.divide((np.sin(theta))**2,(2*sigma_h**2)) +np.divide((np.cos(theta))**2,(2*sigma_w**2))
    
    #return A*np.exp( -(np.multiply(a,(h - h_0)**2) + 2*np.multiply(b,np.multiply((h - h_0),(w - w_0))) + np.multiply(c,(w - w_0)**2)) )+H
     
    a=(h-h_0)**2
    b=(w-w_0)**2
    return A*np.exp(-np.divide(a+b,2*sigma_h**2))+H

def _fit_2d_gaussian_monte_carlo(subimage, N_iter=10**3):
    assert subimage.shape[0] == 5 and subimage.shape[1] == 5
    #use moment method to calculate center of spot
    y=np.array([np.arange(5) for i in np.arange(5)])
    x=y.T
    #w0mean=np.sum(x*subimage)/np.sum(subimage)   
    #h0mean=np.sum(y*subimage)/np.sum(subimage)
   
    max_sub=np.max(subimage) 
    min_sub=np.max(subimage)
    (idx,idy)=np.where(np.max(subimage)==subimage)
    h0mean=idx[0]
    h0std=0.3
    w0mean=idy[0]
    w0std=0.3
    #N_iter=1

    sigmah_mean=1.2
    sigmah_std=0.3
    sigmaw_mean=1.0
    sigmaw_std=0.3


    RMS=10000000*25*np.max(subimage) #or whatever, just big
    RMS_current=RMS
    
        
    #print len(step_width_grid)
    #print step

    RMS=10000000*25*np.max(subimage)
    RMS_current=RMS
    #fit the 2d gaussian function. could be sped up by splitting up to multiple processors. each loop is independent of the other
    for i in range(N_iter):
        H=np.abs(scipy.random.normal(0.0,0.1))
        A=np.abs(scipy.random.normal(1.0,0.2))
        h_0=np.clip(scipy.random.normal(h0mean,h0std),0.01,4.99)
        w_0=np.clip(scipy.random.normal(w0mean,w0std),0.01,4.99)
        sigmah=np.abs(scipy.random.normal(sigmah_mean,sigmah_std))
        sigmaw=np.abs(scipy.random.normal(sigmaw_mean,sigmaw_std))
        theta=np.clip(0*scipy.random.exponential(),0,2*np.pi)
        
        gauss=_2d_gaussian_function(H,A,h_0,w_0,sigmah,sigmaw,theta,x,y)

        gauss=gauss/np.max(gauss)

        RMS=np.sqrt(np.sum((subimage-gauss)**2))
        if RMS<RMS_current:
            RMS_current=RMS
            A_current=A
            sigma_h_current=sigmah
            sigma_w_current=sigmaw
            theta_current=theta
            H_current=H
            h_0_current=h_0
            w_0_current=w_0
    #print RMS_current/np.sum(subimage)
    #print gauss
    #print subimage
    return (h_0_current, w_0_current, H_current, A_current, sigma_h_current, sigma_w_current\
        , theta_current,gauss)


def _fit_2d_gaussian(subimage, implementation='agpy'):
    """
    Fit 2D Gaussian to a 5x5 pixel area. This is a function internal to
    pflib that abstracts away the specific methods used to fit.
    See fit_psf for the description of the 2D Gaussian model,
    parameters, and return values.
    Arguments:
        subimage: 5x5 Numpy array of pixels to fit.
        algorithm: The name of the fitter to use. Currently, only agpy
            is supported.
    Returns:
        (h_0, w_0, H, A, sigma_h, sigma_w, theta, fit_img)
    """
    assert subimage.shape[0] == 5 and subimage.shape[1] == 5
    if implementation != 'agpy':
        raise NotImplementedError("Currently, only agpy is supported.")
    #agpy.gaussfit parameters --
    #(H(eight), A(mplitude), h_0, w_0, sigma_h, sigma_w, theta)
    #initial fitting parameters based on emperical eyeball shufti
    ((H, A, h_0, w_0, sigma_h, sigma_w, theta), fit_img) = \
         gaussfitter.gaussfit(subimage,
                              params=(np.median(subimage), np.amax(subimage),
                                      2.5, 2.5, 1, 1, 0),
                              return_all=False,
                              limitedmin=[True] * 7,
                              limitedmax=[False, False, True, True, True, True,
                                          True],
                              minpars=np.array([0.00,
                                                (np.amax(subimage) -
                                                 np.mean(subimage)) / 3.0,
                                                2.00, 2.00, 0.75, 0.75, 0.00]),
                              maxpars=np.array([0.00, 0.00, 3.00, 3.00, 2.00,
                                                2.00, 360.00]),
                              returnfitimage=True)
    return (h_0, w_0, H, A, sigma_h, sigma_w, theta, fit_img)


def _psf_candidates(image, median_filter_size=5,
                    correlation_matrix=default_correlation_matrix, c_std=2,
                    **kwargs):
    """
    Performs the first two steps of the algorithm in find_peptides on an
    image to find candidate pixels for PSF fitting: (1) applies a median
    filter and (2) correlates with a matrix. Algorithm details and
    argument definitions described in find_pepetides.
    This function was isolated from find_peptides because it is also
    used to load-balance multiprocessor image batching.
    Arguments:
        See find_peptides. Any arguments passed via kwargs are ignored
        unless they are one of the explicitly defined arguments in this
        function.
    Returns:
        Pixels at which to attempt to fit a PSF as a list:
        [(pixel_1_h, pixel_1_w), (pixel_2_h, pixel_2_w), ... ]
    """
    #check that matrix is square and has odd numbers of rows and columns
    if (correlation_matrix.shape[0] != correlation_matrix.shape[1] or
        correlation_matrix.shape[0] % 2 == 0):
        raise ValueError("correlation_matrix must be square, with an odd "
                         "number of rows and columns")
    #copy array as int64
    image = image.astype(np.int64)
    #median filter
    image_mf = \
        np.subtract(image, np.minimum(median_filter(image, median_filter_size),
                                      image))
    #apply correlation matrix for candidate peaks
    image_cm = np.maximum(correlate(image_mf, correlation_matrix, mode='same'),
                          np.zeros_like(image_mf)).astype(np.int64)
    #candidate correlation threshold
    correlation_threshold = np.mean(image_cm) + c_std * np.std(image_cm)
    candidate_pixels = []
    for h, w in itertools.product(range(2, image.shape[0] - 2),
                                  range(2, image.shape[1] - 2)):
        if image_cm[h, w] < correlation_threshold:
            continue
        else:
            candidate_pixels.append((h, w))
    return candidate_pixels


def illumina_s_n(sub_img):
    """
    Compute Illumina's signal to noise metric for sub_img, defined as

        s_n = (max(sub_img) - mean(img_edge)) / stddev(img_edge)

    where img_edge is the one-pixel thick boundary area around sub_img.

    Arguments:
        sub_img: A square Numpy matrix.

    Returns:
        The signal to noise ratio as defined above.
    """
    if not (len(sub_img.shape) == 2 and sub_img.shape[0] == sub_img.shape[1]):
        raise ValueError("sub_img must be square, but has shape " +
                         str(sub_img))
    op = \
     ([sub_img[h, w] for h in [0, -1] for w in range(sub_img.shape[1])] +
      [sub_img[h, w] for h in range(1, sub_img.shape[0] - 1) for w in [0, -1]])
    return (np.amax(sub_img) - np.mean(op)) / np.std(op)


def find_peptides(image, median_filter_size=5,
                  correlation_matrix=default_correlation_matrix,
                  candidate_pixels=None, c_std=2, r_2_threshold=0.7,
                  consolidation_radius=4, fit_type='gauss', N_iter=10**3):
    """
    Find labeled peptides in a TIRF image and characterize their point
    spread functions.
    The image is represented as a two-dimensional Numpy array, with
    dimensions height x width. Pixel coordinates are indexed by positive
    integer pairs [h, w]. The coordinate system origin is in the upper
    left-hand corner, is 0-indexed, and with h(eight) and w(idth) the
    vertical and horizontal indeces from the origin respectively. For
    subpixel resolution, the coordinate system is extended to floating
    point pairs [h, w]. Pixel values are unsigned integers, with black
    (i.e. no photons detected) assigned 0. Bit depth is assumed to be at
    most 64 bits.
    The following algorithm is applied to the image matrix to find
    labeled peptides:
    1. Variation in background is removed via a median filter. The
        median filter footprint is a square of a given size.
    2. The background-adjusted image is correlated with a square matrix
        approximating a labeled peptide point spread function (PSF).
        This yields a correlated image whose pixel values are the
        correlation of the background-adjusted image and the matrix
        centered on that pixel. Only those pixels whose correlation
        values are significantly higher than their peers are considered
        candidate peak locations; a threshold is defined based on the
        aggregate correlation values of the image, and pixels whose
        correlation values are beneath it are discarded. The threshold
        is mean(img_cm) + c_std * stdev(img_cm), where mean(img_cm) is
        the mean of the correlatred image pixel values, c_std is a
        coefficient, and stdev(img_cm) is the standard deviation of the
        correlated image pixel values.
    3. All candidate locations are fitted at subpixel resolution with a
        2D Gaussian PSF. For each candidate pixel, the 5x5 pixel area
        centered about it is used for fitting. Candidates within two
        pixels of the image edge are not sufficient for fitting and are
        discarded. The fitted PSFs are generalized two-dimensional
        elliptical Gaussian functions on [h, w] of the form
        A*exp( -(a(h - h_0)^2 + 2b(h - h_0)(w - w_0) + c(w - w_0)^2) )
            + H
        with the coefficients
            a = (cos(theta))^2/(2*sigma_h^2) +
                (sin(theta))^2/(2*sigma_w^2)
            b = sin(2*theta)/(4*sigma_h^2) -
                sin(2*theta)/(4*sigma_w^2)
            c = (sin(theta))^2/(2*sigma_h^2) +
                (cos(theta))^2/(2*sigma_w^2)
        The PSFs are thus characterized via the parameters: the center
        [h_0, w_0], background level H(eight), A(mplitude) above
        background, sigma_h and sigma_w for bell curve width along the
        ellipse's major and minor axes, and theta as the angle of the
        ellipse's rotation about the coordinate axes. Currently, the
        fitting is performed via Levenberg-Marquardt. The initial
        parameters fed into Levenberg-Marquardt are currently based on
        empirical shufti.
    4. Once a PSF is fitted over the 5x5 pixel area, its values at the
        pixel coordinates within the 5x5 fitting area are computed.
        These fitted values are then compared to the corresponding pixel
        values in the original image to evaluate the quality of fit. Fit
        quality is measured via the coefficient of determination
        computed over the 5x5 area:
        R^2 = 1 -
            sum((img[h,w] - fit[h,w])^2) / sum((img[h,w] - mean(img))^2)
        img[h,w] and fit[h,w] are the true image's pixel value and the
        fitted Gaussian's pixel value, respectively, at pixel [h,w]. The
        sums are over all 25 pixels in the 5x5 area used for fitting.
        mean(img) is the mean pixel value in the true image over the
        fitting area. Fits with R^2 below a threshold are discarded as
        poor.
    5. PSFs adjacent to each other are assumed to be competing for the
        same peptide. Their R^2 quality of fit is compared and only the
        best-fitting PSF is kept. More specifically, the PSFs are
        traversed in raster scan order -- i.e. starting from the top
        left corner (coordinate origin), rows are scanned one at a time
        from top to bottom, with pixels in each row scanned
        left-to-right. For each PSF under consideration, if any rival
        PSFs exist within a given radius, their R^2 values are compared.
        If the PSF under consideration has an R^2 value inferior or
        equal to any of its rivals, it is discarded.
    Arguments:
        image: Image data as a two-dimensional Numpy array. The array is
            copied inside the function, and hence the original is
            unaffected.
        median_filter_size: The median filter is applied as a square,
            its side length indicated by median_filter_size pixels.
        correlation_matrix: Correlate image with this matrix -- a
            two-dimensional Numpy array -- to identify candidate peaks.
            The matrix must be square, with an odd number of rows and
            columns.
        candidate_pixels: Not yet implemented.
        c_std: The threshold for correlation values for each pixel is
            evaluated as
            [mean value of corrlated image pixels] +
            c_std * [standard deviation of correlated image pixels]
            Any pixels whose correlation values are below this threshold
            are discarded as candidates.
        r_2_threshold: Candidate peaks are fitted using a 2D Gaussian.
            The coefficient of determination R^2 is the metric for
            quality of fit. Discard any candidate peaks whose quality of
            fit R^2 is below r_2_threshold.
        consolidation_radius: Multiple well-fitting peaks may be found
            adjacent to each other. The algorithm raster scans across
            the image and for each peak, looks for other peaks within a
            Eucledian distance consolidation_radius. If peaks are found,
            their qualities of fit R^2 are compared and if the peak
            under consideration is inferior, it is discarded. This
            radius must be at least 2.
        fit_type: Can be either 'gauss' for Levenberg-Marquardt, or
            'monte_carlo' for a Monte Carlo strategy.
        N_iter: Number of Monte-Carlo samples to use if fit_type=monte_carlo.
    Returns:
        Dictionary of found peptide PSFs. Each PSF is stored in the
        dictionary with the peak center's pixel location as the key and
        a tuple containing PSF characteristics as the value:
        {
         #KEY
         (h_0 rounded to nearest integer (i.e. to nearest pixel),
          w_0 rounded to nearest integer (i.e. to nearest pixel)):
         #VALUE
         (h_0, w_0, H, A, sigma_h, sigma_w, theta, sub_img, fit_img,
          rmse, r_2, s_n),
         ...
        }
        h_0, w_0, H, A, sigma_h, sigma_w, and theta are defined as in
        the algorithm description above. sub_img is a Numpy array copy
        of the 5x5 pixel area in the image used to fit the PSF. fit_img
        is a Numpy array copy of the same 5x5 pixel area, however with
        pixel values replaced by the value of the PSF at the pixel
        coordinates. Three metrics are of fit quality are computed from
        pixel values in sub_img and fit_img: root mean squared error
        (rmse), coefficient of determination (r_2), and signal to noise
        ratio (s_n). Only R^2 is used to judge quality of fit within the
        algorithm itself. The metrics are defined as follows:
            rmse = sqrt(sum((sub_img[h,w] - fit_img[h,w])^2) / 25)
            r_2 = 1 -
                        sum((sub_img[h,w] - fit_img[h,w])^2) /
                        sum((sub_img[h,w] - mean(sub_img))^2)
            s_n = (max(sub_img) - mean(img_edge)) / stddev(img_edge)
        sub_img[h,w] and fit_img[h,w] are the true image's pixel value
        and the fitted Gaussian's pixel value, respectively, at pixel
        [h,w]. All sums are over the 25 pixels in the 5x5 fitting area.
        img_edge is the set 16 of pixels along the edge of the fitting
        area. mean(img_edge) and stddev(img_edge) are the mean and
        standard deviations, respectively, of these pixel's values. 
    """
    #check if consolidation_radius is >= 2
    if consolidation_radius < 2:
        raise ValueError("consolidation_radius must be at least 2")
    #obtain candidate pixels by applying median filter and matrix correlation
    candidate_pixels = _psf_candidates(image,
                                       median_filter_size=median_filter_size,
                                       correlation_matrix=correlation_matrix,
                                       c_std=c_std)
    #dictionary to store found psfs; {(h, w): psf tuple}
    pixel_bins = {}
    #attempt 2D Gaussian PSF fit on candidate pixels
    for h, w in candidate_pixels:
        #define the 5x5 pixel area to fit & make a 64 bit copy
        sub_img = image[h - 2:h + 3, w - 2:w + 3].astype(np.int64)
	if fit_type=='monte_carlo':
		
		sub_img=(sub_img-np.min(sub_img))
        	sub_img=sub_img/float(np.max(sub_img))

        	(h_0, w_0, H, A, sigma_h, sigma_w, theta, fit_img) = \
                               _fit_2d_gaussian_monte_carlo(sub_img, N_iter)
	else:
				
        	(h_0, w_0, H, A, sigma_h, sigma_w, theta, fit_img) = \
                               _fit_2d_gaussian(sub_img, implementation='agpy')

	#h_0 and w_0 as returned by gaussian fitter are in the 5x5
        #fitting area coordinate system, i.e. their values are between 0
        #and 5, and not in the image's coordinate system; map h_0 and
        #w_0 to the image's coordinate system; pixel [h, w] in the image
        #is at position [2.5, 2.5] in the 5x5 fitting area
        h_0, w_0 = h_0 + h - 2.5, w_0 + w - 2.5
        #coefficient of determination R^2
        r_2 = (1.0 -
                          sum(np.reshape((sub_img - fit_img)**2, -1)) /
                          sum((np.reshape(sub_img, -1) - np.mean(sub_img))**2))
        if r_2 < r_2_threshold:
            #discard poor fit
            continue
        #rmse
        rmse = math.sqrt(sum([(sub_img[x,y] - fit_img[x,y])**2
                              for x,y in itertools.product(range(5),range(5))])
                         / 25.0)
        s_n = illumina_s_n(sub_img)
        #store found psf
        psf = (h_0, w_0, H, A, sigma_h, sigma_w, theta, sub_img, fit_img, rmse,
               r_2, s_n)
        pixel_bins.setdefault((h, w), psf)
    #loop through all bins and remove competing psfs
    for (h, w), psf in pixel_bins.items():
        #skip pixels that have had their psfs deleted
        if (h, w) not in pixel_bins:
            continue
        #search for all peaks that could be within consolidation_radius
        #by looking through all adjacent pixels
        #
        #first, define range of pixels to search for competing psfs in.
        #range is consolidation_radius plus an additional two pixels. it
        #is possible that a psf fitted to the 5x5 area centered on
        #(h, w) ends up with its peak in an adjacent pixel -- hence the
        #additional two pixel buffer.
        h_range = range(max(0, h - consolidation_radius - 2),
                        min(h + consolidation_radius + 3, image.shape[0]))
        w_range = range(max(0, w - consolidation_radius - 2),
                        min(w + consolidation_radius + 3, image.shape[1]))
        for (h_d, w_d) in itertools.product(h_range, w_range):
            #skip checking against self
            if h_d == h and w_d == w:
                continue
            #skip pixels if no psfs
            if (h_d, w_d) not in pixel_bins:
                continue
            #check if within consolidation radius
            h_0, w_0 = pixel_bins[(h, w)][:2]
            h_0_d, w_0_d = pixel_bins[(h_d, w_d)][:2]
            if (h_0 - h_0_d)**2 + (w_0 - w_0_d)**2 > consolidation_radius**2:
                continue
            #compare r_2 and delete inferior psfs
            if pixel_bins[(h, w)][10] > pixel_bins[(h_d, w_d)][10]:
                del pixel_bins[(h_d, w_d)]
            else:
                del pixel_bins[(h, w)]
                break
    #upon return, pixel_bins' dictionary keys must be the rounded h_0 and w_0
    for (h, w), psf in pixel_bins.items():
        h_0_r, w_0_r = int(round(psf[0])), int(round(psf[1]))
        if h_0_r != h or w_0_r != w:
            del pixel_bins[(h, w)]
            assert (h_0_r, w_0_r) not in pixel_bins#b/c consolidation_radius>=2
            pixel_bins.setdefault((h_0_r, w_0_r), psf)
    return pixel_bins


def _epoch_to_hash(epoch):
    """
    Generate an alphanumeric hash from a Unix epoch. Unix epoch is
    rounded to the nearest second before hashing.
    Arguments:
        epoch: Unix epoch time. Must be positive.
    Returns:
        Alphanumeric hash of the Unix epoch time.
    Cribbed from Scott W Harden's website
    http://www.swharden.com/blog/2014-04-19-epoch-timestamp-hashing/
    """
    if epoch <= 0:
        raise ValueError("epoch must be positive.")
    epoch = round(epoch)
    hashchars = '0123456789abcdefghijklmnopqrstuvwxyz'
    #hashchars = '01' #for binary
    epoch_hash = ''
    while epoch > 0:
        epoch_hash = hashchars[int(epoch % len(hashchars))] + epoch_hash
        epoch = int(epoch / len(hashchars))
    return epoch_hash


def _hash_to_epoch(epoch_hash):
    """
    Invert hashing function _epoch_to_hash.
    Arguments:
        epoch_hash: Alphanumeric hash of Unix epoch time as returned by
            _epoch_to_hash.
    Returns:
        epoch: Unix epoch time corresponding to epoch_hash.
    Cribbed from Scott W Harden's website
    http://www.swharden.com/blog/2014-04-19-epoch-timestamp-hashing/
    """
    hashchars = '0123456789abcdefghijklmnopqrstuvwxyz'
    #hashchars = '01' #for binary
    #reverse character order
    epoch_hash = epoch_hash[::-1]
    epoch = 0
    for i, c in enumerate(epoch_hash):
        if c not in hashchars:
            raise ValueError("epoch_hash contains unrecognized character(s).")
        epoch += hashchars.find(c)*(len(hashchars)**i)
    return epoch


def _psfs_filename(image_path, timestamp_epoch, format_suffix):
    """
    Defines standard filenames for PSF results files. Currently defined
    as image_path + '_psfs_' + timestamp_hash + format_suffix.
    image_path is always normalized to its absolute pathname.
    Arguments:
        image_path: Path to processed image file.
        timestamp_epoch: Filesnames for PSF results files incorporate a
            a hash based on a Unix epoch time as implemented by
            _epoch_to_hash. If this argument is None, then
            _psfs_filename will poll the current Unix epoch during
            runtime. If a Unix epoch time is provided via this argument,
            then that is used as input for the hash function.
        format_suffix: Filename extension indicating filetype.
    Returns:
        (os.path.abspath(image_path) + '_psfs_' + timestamp_hash +
         format_suffix)
    """
    if timestamp_epoch is None:
        #get Unix epoch rounded to nearest second
        timestamp_epoch = round(time.time())
    return (os.path.abspath(image_path) + '_psfs_' +
            _epoch_to_hash(timestamp_epoch) + format_suffix)


def save_psfs_pkl(psfs, image_path=None, timestamp_epoch=None,
                  output_path=None):
    """
    Save the dictionary of PSFs returned for an image by find_peptides
    into a Python pickle file. See find_pepetides for the dictionary
    structure.
    The pickle file path is constructed by providing either image_path
    or output_path. In the former case, the path used is defined by
    _psfs_filename(image_path, timestamp_epoch, '.pkl'). timestamp_epoch
    is the Unix epoch time either provided as an argument or, if None,
    polled by this function at runtime. In the latter case, output_path
    will be used without modification.
    Any existing file at the resulting filepath will be overwritten.
    Arguments:
        psfs: Dictionary of PSFs as returned by find_peptides.
        timestamp_epoch: If output_path is not provided, the pickle
            filepath written by save_psfs_pkl incorporates a hash based
            on a Unix epoch time as implemented by _epoch_to_hash. If
            this argument is None, then save_psfs_pkl will poll the
            current Unix epoch during runtime. If a Unix epoch time is
            provided via this argument, then that is given as input for
            the hash function.
        image_path: Path to image that was processed by find_peptides.
            Used to construct output filename. If None, output_path must
            be provided.
        output_path: Overrides output path of the file; image_path and
            timestamp_epoch are ignored. If None, image_path must be
            provided.
    Returns:
        Path to saved pickle file.
    """
    if image_path is None and output_path is None:
        raise ValueError("Either image_path or output_path must be provided.")
    #normalize image path to absolute path
    if image_path is not None:
        image_path = os.path.abspath(image_path)
    if output_path is None:
        if timestamp_epoch is None:
            #get Unix epoch rounded to nearest second
            timestamp_epoch = round(time.time())
        output_path = _psfs_filename(image_path, timestamp_epoch, '.pkl')
    cPickle.dump(psfs, open(output_path, 'w'))
    return output_path


def save_psfs_csv(psfs, image_path=None, timestamp_epoch=None,
                  output_path=None):
    """
    Save the PSFs returned for an image by find_peptides into a tab-
    delimited CSV file.
    The CSV file has the header row ['Absolute image path', 'PSF center
    (h) coordinate', 'PSF center (w) coordinate', 'PSF base (H)eight',
    'PSF (A)mplitude', 'PSF width (sigma_h)', 'PSF width (sigma_w)',
    'PSF (theta)', 'PSF (rmse)', 'PSF (r_2)', 'PSF (s_n)']. Each
    subsequent row contains the corresponding data for each PSF returned
    by find_peptides.
    The CSV file path is constructed by providing either image_path or
    output_path. In the former case, the path used is defined by
    _psfs_filename(image_path, timestamp_epoch, '.csv'). timestamp_epoch
    is the Unix epoch time either provided as an argument or, if None,
    polled by this function at runtime. In the latter case, output_path
    will be used without modification.
    Any existing file at the resulting filepath will be overwritten.
    Arguments:
        psfs: Dictionary of PSFs as returned by find_peptides.
        image_path: Path to image that was processed by find_peptides.
            Used to construct output filename. If None, output_path must
            be provided.
        timestamp_epoch: If output_path is not provided, the CSV
            filepath written by save_psfs_csv incorporates a hash based
            on a Unix epoch time as implemented by _epoch_to_hash. If
            this argument is None, then save_psfs_pkl will poll the
            current Unix epoch during runtime. If a Unix epoch time is
            provided via this argument, then that is given as input for
            the hash function.
        output_path: Overrides output path of the file; image_path and
            timestamp_epoch are ignored. If None, image_path must be
            provided.
    Returns:
        Path to saved CSV file.
    """
    if image_path is None and output_path is None:
        raise ValueError("Either image_path or output_path must be provided.")
    #normalize image path to absolute path
    if image_path is not None:
        image_path = os.path.abspath(image_path)
    if output_path is None:
        if timestamp_epoch is None:
            #get Unix epoch rounded to nearest second
            timestamp_epoch = round(time.time())
        output_path = _psfs_filename(image_path, timestamp_epoch, '.csv')
    with open(output_path, 'w') as output_file:
        output_writer = csv.writer(output_file, dialect='excel-tab')
        csv_header = ['Absolute image path',
                      'PSF center (h) coordinate',
                      'PSF center (w) coordinate',
                      'PSF base (H)eight',
                      'PSF (A)mplitude',
                      'PSF width (sigma_h)',
                      'PSF width (sigma_w)',
                      'PSF (theta)',
                      'PSF (rmse)',
                      'PSF (r_2)',
                      'PSF (s_n)']
        output_writer.writerow(csv_header)
        for (
             #KEY -- pixel coordinate
             (h, w),
             #VALUE -- psf data
             (h_0, w_0, H, A, sigma_h, sigma_w, theta, sub_img, fit_img, rmse,
              r_2, s_n)
             ###
            ) in psfs.iteritems():
            csv_row = [image_path, str(h_0), str(w_0), str(H), str(A),
                       str(sigma_h), str(sigma_w), str(theta), str(rmse),
                       str(r_2), str(s_n)]
            output_writer.writerow(csv_row)
    return output_path


def read_image(image_path):
    """
    Read an image into a Numpy array.
    The current pipeline can only read in PNG files, and relies on an
    external conversion software via convert_image to convert any other
    image formats into PNG. It assumes that an image is a PNG if and
    only if its suffix is '.png'. For target images that are not PNGs,
    image_batch checks if image_path + '.png' exists. If it exists, then
    it assumes this is the PNG version of the target image and uses it
    as input. If it does not exist, conversion is performed and the
    resulting PNGs saved at the default path defined by convert_image.
    Note that convert_image overwrites any files existing at its default
    outputpath.
    Arguments:
        image_path: Path to image.
    Returns:
        The path to the PNG version of the image (unchanged from
        original path if it was already a PNG), and the image as a Numpy
        array: (converted_path, image).
    """
    logger = logging.getLogger()
    converted_path = image_path = os.path.abspath(image_path)
    if image_path[-4:] != '.png':
        if os.path.exists(image_path + '.png'):
            converted_path += '.png'
        else:
            try:
                converted_path = convert_image(image_path)
            except Exception as e:
                logger.exception(e, exc_info=True)
                raise
    image = imread(converted_path)
    return converted_path, image


def _histogram_equalization(image, **kwargs):
    """
    Utility function to pass to save_psfs_png as contrast_filter. Performs
    histogram equalization on image using skimage.exposure.equalize_hist and
    rescales to 8 bits.
    Arguments:
        image: Image data as a two-dimensional Numpy array. The array is
            copied inside the function, and hence the original is
            unaffected.
        **kwargs: For compatibility with save_psfs_png. Not used.
    Returns: A copy of the image that has been histogram equalized and then
        rescaled to 8 bits via scikit-image. Returned Numpy array dtype is
        np.uint8.
    """
    return exposure.rescale_intensity(exposure.equalize_hist(image),
                                      out_range=np.uint8).astype(np.uint8)


def _intensity_scaling(image, **kwargs):
    """
    Utility function to pass to save_psfs_png as contrast_filter. Performs
    skimage.exposure.rescale_intensity to rescale the image into 8 bits.
    Arguments:
        image: Image data as a two-dimensional Numpy array. The array is
            copied inside the function, and hence the original is
            unaffected.
        **kwargs: For compatibility with save_psfs_png. Not used.
    Returns: A copy of the image that has been rescaled to 8 bits via
        scikit-image. Returned Numpy array dtype is np.uint8.
    """
    return exposure.rescale_intensity(image,
                                      out_range=np.uint8).astype(np.uint8)


def save_psfs_png(psfs, image_path, timestamp_epoch=None, output_path=None,
                  square_size=9, square_color='lightblue', square_colors=None,
                  contrast_filter=_intensity_scaling,
                  contrast_filter_args=None):
    """
    Highlight found PSFs in image via squares and save result as a PNG.
    The PNG file path is constructed by providing either image_path or
    output_path. In the former case, the path used is defined by
    _psfs_filename(image_path, timestamp_epoch, '.png'). timestamp_epoch
    is the Unix epoch time either provided as an argument or, if None,
    polled by this function at runtime. In the latter case, output_path
    will be used without modification.
    Any existing file at the resulting filepath will be overwritten.
    Arguments:
        psfs: Dictionary of PSFs as returned by find_peptides.
        image_path: Path to image that was processed by find_peptides.
        timestamp_epoch: If output_path is not provided, the PNG
            filepath written by save_psfs_png incorporates a hash based
            on a Unix epoch time as implemented by _epoch_to_hash. If
            this argument is None, then save_psfs_pkl will poll the
            current Unix epoch during runtime. If a Unix epoch time is
            provided via this argument, then that is given as input for
            the hash function.
        output_path: Overrides output path of the file; image_path and
            timestamp_epoch are ignored.
        square_size: The squares highlighting the PSFs are square_size
            pixels on each side. Must be an odd integer greater than or
            equal to 3.
        square_color: Square color string. Must use names available in
            Pillow (PIL fork). All PSF squares will use this color unless
            overridden by an entry in square_colors.
        square_colors: An optional dictionary that allows individual color
            assignment for each PSF. For each PSF in psfs, if its key (h, w) is
            in square_colors, then save_psfs_png expects the value to be a PIL
            color string. If an entry is not present, then square_color is used
            by default.
        contrast_filter: Fluorosequencing TIRF images are often hard to
            visually inspect on a screen without improving their contrast.
            Furthermore, the PNG produced by this function as currently
            implemented is 8 bit and the TIRF image values must be mapped to
            this interval. Therefore, it is useful to normalize this mapping to
            improve its contrast. This argument allows the user to pass a
            filter function that performs this purpose. save_psfs_png will then
            apply this function to the image first before saving it. Note that
            this function needs to return an image mapped to 8 bits for the PNG
            to appear sane. contrast_filter_args may be used to pass arguments
            to the given function as **kwargs. The filter will be applied by
            making the function call
            contrast_filter(image, **contrast_filter_args)
            The default filter is currently set to the pflib built-in
            _intensity_scaling. _histogram_equalization is another pflib
            built-in available.
         contrast_filter_args: Dictionary of keyword arguments that will be
            passed to contrast_filter.
    Returns:
        Path to saved PNG file.
    """
    logger = logging.getLogger()
    #normalize image path to absolute path
    image_path = os.path.abspath(image_path)
    if output_path is None:
        if timestamp_epoch is None:
            #get Unix epoch rounded to nearest second
            timestamp_epoch = round(time.time())
        output_path = _psfs_filename(image_path, timestamp_epoch, '.png')
    #read original image
    converted_path, image = read_image(image_path)
    #create colorized output image using Pillow hack
    #
    #first apply the contrast filter function and fit the image into 8 bits
    if contrast_filter_args is None:
        contrast_filter_args = {}
    filtered_image = contrast_filter(image, **contrast_filter_args)
    # convert to Pillow image format
    pillow_image = Image.fromarray(filtered_image, mode="L")
    #add color channels
    highlighted_image = ImageOps.colorize(pillow_image, (0,0,0), (255,255,255))
    #ensure square size is acceptable
    if square_size % 2 == 0 or square_size < 3:
        raise ValueError("square_size must be an odd integer >= 3")
    for (
         #KEY -- pixel coordinate
         (h, w),
         #VALUE -- psf data
         (h_0, w_0, H, A, sigma_h, sigma_w, theta, sub_img, fit_img, rmse, r_2,
          s_n)
         ###
        ) in psfs.iteritems():
        #create verteces for square
        radius = (square_size - 1) / 2
        square = ((w - radius, h - radius), (w + radius, h + radius))
        draw = ImageDraw.Draw(highlighted_image)
        if square_colors is None or (h, w) not in square_colors:
            draw.rectangle(square, fill=None, outline=square_color)
        else:
            draw.rectangle(square, fill=None, outline=square_colors[(h, w)])
    highlighted_image.save(output_path)
    return output_path


def image_batch(image_paths, find_peptides_parameters=None,
                timestamp_epoch=None):
    """
    Finds peptide PSFs via find_peptides in multiple images, and stores
    results as pickles, CSVs, and PNGs via save_psfs_pkl, save_psfs_csv,
    and save_psfs_png respectively.
    Target images are passed as an iterable of pathnames. For each
    image_path, the resulting pickle, CSV, and PNG files are saved at
    paths defined by _psfs_filename. The timestamp hash is generated
    based on a Unix epoch time either polled by this function at runtime
    or provided as an argument. Any existing files at these paths are
    overwritten.
    The current pipeline can only read in PNG files, and relies on an
    external conversion software via convert_image to convert any other
    image formats into PNG. It assumes that an image is a PNG if and
    only if its suffix is '.png'. For target images that are not PNGs,
    image_batch checks if image_path + '.png' exists. If it exists, then
    it assumes this is the PNG version of the target image and uses it
    as input. If it does not exist, it attempts to read in the image via
    read_image, which first converts it to PNG via convert_image. The
    resulting PNGs are saved at the default paths set by read_image.
    Beware that any existing files at these paths are overwritten by
    convert_image. Note also that the output filesnames will generated
    by _psfs_filename wil be based on this converted PNG as the
    image_path.
    Consult save_psfs_pkl, save_psfs_csv, and save_psfs_png
    documentation to interpret the output.
    Arguments:
        image_paths: Iterable returning paths to images to be processed.
            If any image paths appear more than once in the iterable,
            the duplicates are ignored; all image paths are processed
            only once.
        find_peptides_parameters: The peak finding parameters to be
            passed to find_peptides as a dictionary, i.e.
            {'median_filter_size': mfs_val, 'correlation_matrix':
             cm_val, 'candidate_pixels': cp_val, 'c_std': cs_val,
             'r_2_threshold': r2t_val, 'consolidation_radius': cr_val}
            where *_val represents the value passed. If any of these
            parameters is not specified, then find_peptides uses its
            defaults.
        timestamp_epoch: Pickle, CSV, and PNG files written by
            image_batch incorporate a hash based on a Unix epoch time as
            implemented by _epoch_to_hash. If this argument is None,
            then image_batch will poll the current Unix epoch during
            runtime and use that for all files generated during that
            invocation. If a Unix epoch time is provided via this
            argument, then that is used as input for the hash function.
    Returns:
        Dictionary of all images processed and resulting files:
        {
         #KEY
         image_path: #note KEY is always the original (unconverted) path
         #VALUE
         (converted_image_path, psfs_pkl_path, psfs_csv_path,
          psfs_png_path),
         ...
        }
    """
    logger = logging.getLogger()
    logger.debug("image_batch entered, os.getpid() = " + str(os.getpid()) +
                 ", os.getppid() = " + str(os.getppid()))
    if timestamp_epoch is None:
        timestamp_epoch = round(time.time())
    #normalize all paths to be absolute
    image_paths = [os.path.abspath(path) for path in image_paths]
    #remove duplicate image paths
    image_paths = list(set(image_paths))
    #using find_peptides_parameters={} as default in function definition
    #is a bad habit. Instead, use None as default and set to empty
    #dictionary here
    if find_peptides_parameters is None:
        find_peptides_parameters = {}
    #the dictionary of processed images to return
    processed_images = {}
    for image_path in image_paths:
        #(converted_image_path, psfs_pkl_path, psfs_csv_path, psfs_png_path)
        output_tuple = [None, None, None, None]
        try:
            converted_path, image = read_image(image_path)
        except Exception as e:
            logger.exception(e, exc_info=True)
            continue
        output_tuple[0] = converted_path
        #find PSFs
        try:
            psfs = find_peptides(image, **find_peptides_parameters)
        except Exception as e:
            logger.exception(e, exc_info=True)
            continue
        #save psfs as pickle
        try:
            pkl_path = save_psfs_pkl(psfs, image_path=converted_path,
                                     timestamp_epoch=timestamp_epoch)
        except Exception as e:
            logger.exception(e, exc_info=True)
            continue
        output_tuple[1] = pkl_path
        #save psfs as csv
        try:
            csv_path = save_psfs_csv(psfs, image_path=converted_path,
                                     timestamp_epoch=timestamp_epoch)
        except Exception as e:
            logger.exception(e, exc_info=True)
            continue
        output_tuple[2] = csv_path
        #save psfs as png
        try:
            png_path = save_psfs_png(psfs, image_path=converted_path,
                                     timestamp_epoch=timestamp_epoch)
        except Exception as e:
            logger.exception(e, exc_info=True)
            continue
        output_tuple[3] = png_path
        processed_images.setdefault(image_path, tuple(output_tuple))
    return processed_images


def parallel_image_batch(image_paths, find_peptides_parameters=None,
                         timestamp_epoch=None, num_processes=None):
    """
    Execute image_batch in parallel. All arguments and the returned
    dictionary are identical to image_batch, except for num_processes.
    See image_batch for documentation.
    PSF fitting is a computationally expensive component of
    fluorosequencing image analysis. When invoked with more than one
    parallel process requested, image_batch attempts to load-balance the
    processes by counting the number of candidate PSFs to be fitted in
    each image, and then allocating images to the parallel processes
    such that the PSF fits are partitioned across them as symmetrically
    as possible. Note that find_peptides_parameters is used to estimate
    the number of PSFs per image, and hence affects load balancing.
    Arguments:
        num_processes: Indicates how many independent processes to use.
            Must be an integer greater than or equal to 1. If None, will
            use the number returned by multiprocessing.cpu_count().
        For all other arguments, see image_batch for documentation.
    Returns:
        See image_batch for documentation.
    """
    logger = logging.getLogger()
    #special case: if num_processes or images is 1, just use image_batch
    if num_processes == 1 or len(image_paths) == 1:
        return image_batch(image_paths,
                           find_peptides_parameters=find_peptides_parameters,
                           timestamp_epoch=timestamp_epoch)
    #if num_processes not specified, use number of processors available
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    #generate timestamp_epoch
    if timestamp_epoch is None:
        timestamp_epoch = round(time.time())
    #normalize all paths to be absolute
    image_paths = [os.path.abspath(path) for path in image_paths]
    #remove duplicate image paths
    image_paths = list(set(image_paths))
    #using find_peptides_parameters={} as default in function definition
    #is a bad habit. Instead, use None as default and set to empty
    #dictionary here
    if find_peptides_parameters is None:
        find_peptides_parameters = {}
    #read in all images and count candidate PSFs; resulting images list format
    #images = [(path, image Numpy array, PSF candidate count), ...]
    images = []
    for path in image_paths:
        try:
            converted_path, image = read_image(path)
        except Exception as e:
            logger.exception(e, exc_info=True)
            continue
        candidate_count = \
                        len(_psf_candidates(image, **find_peptides_parameters))
        images.append((converted_path, image, candidate_count))
    #sort by decreasing candidate counts
    images = sorted(images, key=lambda x:x[2], reverse=True)
    #total number of PSFs to be fitted
    total_candidate_count = sum([count for path, image, count in images])
    if num_processes < 1 or round(num_processes) != num_processes:
        raise ValueError("Number of processes must be an integer >= 1")
    #partition images across processes so that PSF candidates are as evenly
    #distributed as possible
    partitioned_images = [[] for x in range(num_processes)]
    while images:
        path, image, candidate_count = images.pop()
        #find the partition with the least number of members
        emptiest_partition = sorted(partitioned_images,
                                    key=lambda p:sum([c for ip, c in p]))[0]
        emptiest_partition.append((path, candidate_count))
    logger.debug("parallel_image_batch: total_candidate_count = " +
                 str(total_candidate_count) +
                 "; len(partitioned_images) = " +
                 str(len(partitioned_images)) +
                 "; partitioned_images images per partition: " +
                 str([len(p) for p in partitioned_images]) +
                 "; candidate psfs per partition: " +
                 str([sum([c for ip, c in p]) for p in partitioned_images]))
    assert len(partitioned_images) <= num_processes
    #perform parallel processing
    #
    #initialize worker pool
    pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=1)
    #keep track of all worker processes here
    image_processes = []
    #start all processes
    for partition in partitioned_images:
        image_paths = [path for path, count in partition]
        try:
            p = pool.apply_async(image_batch,
                                 args=(image_paths, find_peptides_parameters,
                                       timestamp_epoch))
        except Exception as e:
            logger.exception(e, exc_info=True)
            continue
        else:
            image_processes.append(p)
    #wait for results to finish
    pool.close()
    pool.join()
    #merge results and return
    processed_images = {}
    for p in image_processes:
        try:
            results = p.get().items()
        except Exception as e:
            logger.exception(e, exc_info=True)
        else:
            logger.debug("parallel_image_batch: results = " + str(results))
            for k, v in results:
                processed_images.setdefault(k, v)
    return processed_images
