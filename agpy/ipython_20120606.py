
from pylab import *;import numpy,scipy,matplotlib,pyfits;
get_ipython().magic(u'run PCA_tools.py')
get_ipython().magic(u'paste')
print "PyMC linear tests"
MC1 = pymc_linear_fit(x,y,intercept=False,print_results=True,return_MC=True)
MC2 = pymc_linear_fit(x,y,xerr,yerr,intercept=False,print_results=True,return_MC=True)
get_ipython().magic(u'history ')
