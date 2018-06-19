# -*- coding: utf-8 -*-

"""Unit test for lsl_toolkit.S60 module"""

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
        frame1 = usrp.readFrame(fh)
        self.assertEqual(frame1.data.iq.shape[0], 8192)
        
        # Next frames make it in with the correct number of elements
        frame2 = usrp.readFrame(fh)
        self.assertEqual(frame2.data.iq.shape[0], 8192)
        
        # Next frames make it in with the correct number of elements
        frame3 = usrp.readFrame(fh)
        self.assertEqual(frame3.data.iq.shape[0], 8192)
        
        fh.close()
        
    def test_usrp_header(self):
        """Test the USRP metadata in a USRP file."""
        
        fh = open(usrpFile, 'rb')
        frame = usrp.readFrame(fh)
        
        # Tuning
        self.assertEqual(frame.getCentralFreq(), 50e6)
        
        # Sample Rate
        self.assertEqual(frame.getSampleRate(), 195312.5)
        
        # Filter Code
        self.assertEqual(frame.getFilterCode(), 0)
        
        fh.close()
        
    def test_usrp_timetags(self):
        """Test the time tags in a USRP file."""
        
        fh = open(usrpFile, 'rb')
        frame = usrp.readFrame(fh)
        tt = 1*frame.data.timeTag
        ttSkip = int(fS / frame.getSampleRate() * frame.data.iq.size)
        
        for i in xrange(2, 6):
            frame = usrp.readFrame(fh)
            
            self.assertEqual(frame.data.timeTag, tt+ttSkip)
            tt = 1*frame.data.timeTag
            
        fh.close()
        
    def test_usrp_errors(self):
        """Test reading in all frames from a truncated USRP file."""
        
        fh = open(usrpFile, 'rb')
        
        # Frames 1 through 5
        for i in range(1,6):
            frame = usrp.readFrame(fh)
            
        # Last frame should be an error (errors.numpyError)
        self.assertRaises(Exception, usrp.readFrame, fh)
        
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
