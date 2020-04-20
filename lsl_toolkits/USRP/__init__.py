# Python2 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info < (3,):
    range = xrange
    
__version__ = '0.2'


from .reader import FrameHeader, FramePayload, Frame, read_frame, get_sample_rate, get_frame_size, get_beam_count, get_frames_per_obs, FILTER_CODES

