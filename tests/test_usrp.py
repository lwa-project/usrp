# -*- coding: utf-8 -*-

"""
Unit test for lsl_toolkit.USRP module.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import unittest

from lsl_toolkits import USRP as usrp
from lsl_toolkits.USRP.common import fS


__revision__ = "$Rev$"
__version__  = "0.1"
__author__    = "Jayce Dowell"


usrpFile = os.path.join(os.path.dirname(__file__), 'data', 'usrp-test.dat')


class usrp_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the lsl_toolkits.USRP
    module."""
    
    def test_usrp_read(self):
        """Test reading in a frame from a USRP file."""
        
        fh = open(usrpFile, 'rb')
        
        # First frame makes it in with the correct number of elements
        frame1 = usrp.read_frame(fh)
        self.assertEqual(frame1.payload.data.shape[0], 8192)
        
        # Next frames make it in with the correct number of elements
        frame2 = usrp.read_frame(fh)
        self.assertEqual(frame2.payload.data.shape[0], 8192)
        
        # Next frames make it in with the correct number of elements
        frame3 = usrp.read_frame(fh)
        self.assertEqual(frame3.payload.data.shape[0], 8192)
        
        fh.close()
        
    def test_usrp_header(self):
        """Test the USRP metadata in a USRP file."""
        
        fh = open(usrpFile, 'rb')
        frame = usrp.read_frame(fh)
        
        # Tuning
        self.assertEqual(frame.central_freq, 50e6)
        
        # Sample Rate
        self.assertEqual(frame.sample_rate, 195312.5)
        
        # Filter Code
        self.assertEqual(frame.filter_code, 0)
        
        fh.close()
        
    def test_usrp_timetags(self):
        """Test the time tags in a USRP file."""
        
        fh = open(usrpFile, 'rb')
        frame = usrp.read_frame(fh)
        tt = 1*frame.data.timetag
        ttSkip = int(fS / frame.sample_rate * frame.payload.data.size)
        
        for i in xrange(2, 6):
            frame = usrp.read_frame(fh)
            
            self.assertEqual(frame.data.timetag, tt+ttSkip)
            tt = 1*frame.data.timetag
            
        fh.close()
        
    def test_usrp_errors(self):
        """Test reading in all frames from a truncated USRP file."""
        
        fh = open(usrpFile, 'rb')
        
        # Frames 1 through 5
        for i in range(1,6):
            frame = usrp.read_frame(fh)
            
        # Last frame should be an error (errors.numpyError)
        self.assertRaises(Exception, usrp.read_frame, fh)
        
        fh.close()


class usrp_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the lsl_toolkits.USRP units 
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(usrp_tests)) 


if __name__ == '__main__':
    unittest.main()
