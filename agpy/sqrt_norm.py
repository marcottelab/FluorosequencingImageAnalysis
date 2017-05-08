"""
------------------------------
Square Root Colorbar Normalization
------------------------------

For use with, e.g., imshow - 
imshow(myimage, norm=SqrtNorm())

Some of the ideas used are from `aplpy <aplpy.github.com>`_


"""
from matplotlib.colors import Normalize
from matplotlib.cm import cbook
from numpy import ma
import numpy as np

class SqrtNorm(Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False, nthroot=2):
        """
        nthroot allows cube roots, fourth roots, etc.
        """
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip
        self.nthroot = nthroot

    def __call__(self, value, clip=None, midpoint=None):


        if clip is None:
            clip = self.clip

        if cbook.iterable(value):
            vtype = 'array'
            val = ma.asarray(value).astype(np.float)
        else:
            vtype = 'scalar'
            val = ma.array([value]).astype(np.float)

        self.autoscale_None(val)
        vmin, vmax = self.vmin, self.vmax

        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin==vmax:
            return 0.0 * val
        else:
            if clip:
                mask = ma.getmask(val)
                val = ma.array(np.clip(val.filled(vmax), vmin, vmax),
                                mask=mask)
            result = (val-vmin) * (1.0/(vmax-vmin))
            #result = (ma.arcsinh(val)-np.arcsinh(vmin))/(np.arcsinh(vmax)-np.arcsinh(vmin))
            result = result**(1./self.nthroot)
        if vtype == 'scalar':
            result = result[0]
        return result

    def autoscale_None(self, A):
        ' autoscale only None-valued vmin or vmax'
        if self.vmin is None:
            self.vmin = ma.min(A)
        if self.vmax is None:
            self.vmax = ma.max(A)



        #return np.arcsinh(array/midpoint) / np.arcsinh(1./midpoint)
