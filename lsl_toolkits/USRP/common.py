"""
Module that contains common values for the USRP N210.  The values 
are:
  * f_S - Sampleng rate in samples per second
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info < (3,):
    range = xrange
    
__version__ = '0.1'
__all__ = ['fS',]

fS = 100.0e6	# Hz
