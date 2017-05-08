#!/home/boulgakov/anaconda2/bin/python
# -*- coding: utf-8 -*-
"""
Port of Manuel Guizar's code from:
http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation
"""

import numpy as np


def phase_correlate(ref_image, reg_image, upsample_factor=1):
    """
    Efficient subpixel image registration by crosscorrelation.

    This code gives the same precision as the FFT upsampled cross correlation
    in a small fraction of the computation time and with reduced memory
    requirements. It obtains an initial estimate of the cross-correlation peak
    by an FFT and then refines the shift estimation by upsampling the DFT only
    in a small neighborhood of that estimate by means of a matrix-multiply DFT.
    With this procedure all the image points are used to compute the upsampled
    cross-correlation.

    Parameters
    ----------
    ref_image : 2D array_like
        Reference image (real space).
    reg_image : 2D array_like
        Image to register (real space).
    upsample_factor : int
        Upsampling factor. Images will be registered to
        within 1 / ``upsample_factor`` of a pixel. For example
        ``upsample_factor`` = 20 means the images will be registered
        within 1/20 of a pixel. (default = 1)

    Returns
    -------
    row_shift : float
        row (Y)-shift (in pixels) required to shift image
        into registration with reference_image.
    col_shift : float
        column (X)-shift (in pixels) required to shift image
        into registration with reference_image.
    error : float
        Translation invariant normalized RMS error between f and g.
    diffphase : float
        Global phase difference between the two images (should be
        zero if images are non-negative).

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
        "Efficient subpixel image registration algorithms," Optics Letters 33,
        156-158 (2008).
    """
    # images must be the same shape
    if ref_image.shape != reg_image.shape:
        raise ValueError("Error: images must be same size for phase_correlate")

    # only 2D data makes sense right now
    if len(ref_image.shape) != 2:
        raise ValueError("Error: phase_correlate only supports 2D images")

    ref_image = np.array(ref_image, dtype=np.float64, copy=False)
    reg_image = np.array(reg_image, dtype=np.float64, copy=False)
    ref_image_freq = np.fft.fft2(ref_image)
    reg_image_freq = np.fft.fft2(reg_image)

    # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the
    # peak
    rows, cols = ref_image_freq.shape[:2]
    cross_correlation = np.fft.ifft2(ref_image_freq * reg_image_freq.conj())
    # Locate maximum
    row_max, col_max = np.unravel_index(
        np.argmax(cross_correlation), cross_correlation.shape)[:2]
    mid_row = np.fix(rows / 2)
    mid_col = np.fix(cols / 2)
    if row_max > mid_row:
        row_shift = row_max - rows
    else:
        row_shift = row_max
    if col_max > mid_col:
        col_shift = col_max - cols
    else:
        col_shift = col_max
    if upsample_factor == 1:
        rfzero = np.sum(np.abs(ref_image_freq) ** 2) / (rows * cols)
        rgzero = np.sum(np.abs(reg_image_freq) ** 2) / (rows * cols)
        CCmax = cross_correlation.max()
        error = 1.0 - CCmax * CCmax.conj() / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(CCmax.imag, CCmax.real)
        return row_shift, col_shift, error, diffphase
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        row_shift = np.round(row_shift * upsample_factor) / upsample_factor
        col_shift = np.round(col_shift * upsample_factor) / upsample_factor
        upsampled_pixels = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_pixels / 2)
        # Matrix multiply DFT around the current shift estimate
        cross_correlation = _dftups(reg_image_freq * ref_image_freq.conj(),
                                    upsampled_pixels,
                                    upsampled_pixels,
                                    upsample_factor,
                                    dftshift - row_shift * upsample_factor,
                                    dftshift - col_shift * upsample_factor).conj() / \
            (mid_row * mid_col * upsample_factor ** 2)
        # Locate maximum and map back to original pixel grid
        row_max, col_max = np.unravel_index(
            np.argmax(cross_correlation), cross_correlation.shape)
        row_max -= dftshift
        col_max -= dftshift
        row_shift = row_shift + row_max / upsample_factor
        col_shift = col_shift + col_max / upsample_factor
        CCmax = cross_correlation.max()
        rg00 = _dftups(ref_image_freq * ref_image_freq.conj(), 1, 1, upsample_factor) / \
            (mid_row * mid_col * upsample_factor ** 2)
        rf00 = _dftups(reg_image_freq * reg_image_freq.conj(), 1, 1, upsample_factor) / \
            (mid_row * mid_col * upsample_factor ** 2)
        error = 1.0 - CCmax * CCmax.conj() / (rg00 * rf00)
        error = np.sqrt(np.abs(error))[0, 0]
        diffphase = np.arctan2(CCmax.imag, CCmax.real)

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    if mid_row == 1:
        row_shift = 0
    if mid_col == 1:
        col_shift = 0
    # the result is the shift necessary to apply to "image" to bring it into
    # registration with "reference_image".  It will be opposite in sign to any
    # shift that was applied to "reference_image" to create "image".
    return row_shift, col_shift, error, diffphase


def _dftups(data, upsampled_rows=None, upsampled_cols=None,
            upsample_factor=1, row_offset=0, col_offset=0):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an [``upsampled_rows``, ``upsampled_cols``] region of the
          result, starting with the [``row_offset``+1 ``col_offset``+1]
          element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the
    zero-padded FFT approach if (``upsampled_rows``, ``upsampled_cols``) are
    much smaller than (``rows`` * ``upsample_factor``,
    ``cols`` *``upsample_factor``).

    Parameters
    ----------
    data : 2D array_like, real
        The input data array (DFT of original data) to upsample.
    upsampled_rows : integer or None
        The row size of the region to be sampled.
    upsampled_cols : integer or None
        The column size of the region to be sampled.
    upsample_factor : integer
        The upsampling factor.
    row_offset : int, optional, default is image center
        The row offset of the region to be sampled.
    col_offset : int, optional, default is image center
        The column offset of the region to be sampled.

    Returns
    -------
    output: 2D ndarray
            The upsampled DFT of the specified region.
    """
    rows, cols = data.shape
    if upsampled_rows is None:
        upsampled_rows = rows
    if upsampled_cols is None:
        upsampled_cols = cols
    # Compute kernels and obtain DFT by matrix products
    col_kernel = np.exp(
        (-1j * 2 * np.pi / (cols * upsample_factor)) *
        (np.fft.ifftshift(np.arange(cols))[:, np.newaxis] -
         np.floor(cols / 2)).dot(
             np.arange(upsampled_cols)[np.newaxis, :] - col_offset)
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (rows * upsample_factor)) *
        (np.arange(upsampled_rows)[:, np.newaxis] - row_offset).dot(
            np.fft.ifftshift(np.arange(rows))[np.newaxis, :] -
            np.floor(rows / 2))
    )
    return row_kernel.dot(data).dot(col_kernel)
