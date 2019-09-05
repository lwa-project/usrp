# -*- coding: utf-8 -*-

"""
Module that contains common values for the USRP N210.  The values 
are:
  * f_S - Sampleng rate in samples per second
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
__version__ = '0.1'
__revision__ = '$Rev$'
__all__ = ['fS', '__version__', '__revision__', '__all__']

fS = 100.0e6	# Hz
