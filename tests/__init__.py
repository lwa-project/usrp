# -*- coding: utf-8 -*-

"""
Modules defining package tests.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
__revision__  = "$Rev$"
__version__   = "0.2"
__author__    = "Jayce Dowell"

from . import test_usrp
from . import test_scripts
