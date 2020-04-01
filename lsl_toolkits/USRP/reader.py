# -*- coding: utf-8 -*-

"""
Python module to read in USRP data.  This module defines the following 
classes for storing the USRP data found in a file:

Frame
  object that contains all data associated with a particular DRX frame.  
  The primary constituents of each frame are:
    * FrameHeader - the USRP frame header object and
    * FramePayload   - the USRP frame data object.
Combined, these two objects contain all of the information found in the 
original USRP data block.

The functions defined in this module fall into two class:
  1. convert a frame in a file to a Frame object and
  2. describe the format of the data in the file.

For reading in data, use the read_frame function.  It takes a python file-
handle as an input and returns a fully-filled Frame object.

For describing the format of data in the file, two function are provided:

get_beam_count
  read in the first few frames of an open file handle and return how many 
  beams are present in the file.

get_frames_per_obs
  read in the first several frames to see how many frames (tunings/polarizations)
  are associated with each beam.
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import copy
import numpy
import struct

from .common import fS
from lsl.reader.base import *

__version__ = '0.3'
__all__ = ['FrameHeader', 'FramePayload', 'Frame', 'read_frame', 
           'get_sample_rate', 'get_frame_size', 'get_beam_count', 'get_frames_per_obs',
           'FILTER_CODES']


_type2name = {0: 'b', 
              1: 'h', 
              2: 'i', 
              3: 'l', 
              4: 'q', 
              5: 'f', 
              6: 'd'}


# List of filter codes and their corresponding sample rates in Hz
FILTER_CODES = {}
for i in xrange(9):
    FILTER_CODES[i] = fS / 2**(9-i)


class FrameHeader(FrameHeaderBase):
    """
    Class that stores the information found in the header of a USRP 
    frame.
    """
    
    _header_attrs = ['size', 'type', 'complex', 'sample_rate']
    
    def __init__(self, size=None, type=None, complex=False, sample_rate=0):
        self.size = size
        self.type = type
        self.complex = complex
        self.sample_rate = sample_rate
        FrameHeaderBase.__init__(self)
        
    @property
    def id(self):
        """
        Return the ID for a USRP stream.
        
        .. note:: 
            This isn't stored in the frame headers by default a three element 
            tuple of (0, 1, 0) is returned to be compatible with DRX.
        """
        
        return (0,1,0)
        
    @property
    def filter_code(self):
        """
        Function to convert the sample rate in Hz to a filter code.
        """
        
        sampleCodes = {}
        for key,value in FILTER_CODES.iteritems():
            sampleCodes[value] = key
            
        return sampleCodes[self.sample_rate]


class FramePayload(FramePayloadBase):
    """
    Class that stores the information found in the data section of a USRP
    frame.
    """
    
    _payload_attrs = ['size', 'timetag', 'central_freq']
    
    def __init__(self, size=None, timetag=None, central_freq=None, iq=None):
        self.size = size
        self.central_freq = central_freq
        self.timetag = timetag
        FramePayloadBase.__init__(self, iq)
        
    @property
    def time(self):
        """
        Function to convert the time tag from samples since the UNIX epoch
        (UTC 1970-01-01 00:00:00) to seconds since the UNIX epoch as a two-
        element tuple.
        """
        
        seconds_i = self.timetag // int(fS)
        seconds_f = (self.timetag % int(fS)) / fS
        
        return seconds_i, seconds_f


class Frame(FrameBase):
    """
    Class that stores the information contained within a single DRX 
    frame.  It's properties are FrameHeader and FramePayload objects.
    """
    
    _header_class = FrameHeader
    _payload_class = FramePayload
    
    @property
    def size(self):
        """
        Total frame size.
        """
        
        return self.header.size + self.payload.size
        
    @property
    def id(self):
        """
        Convenience wrapper for the Frame.FrameHeader.id 
        property.
        """
        
        return self.header.id
        
    @property
    def sample_rate(self):
        """
        Convenience wrapper for the Frame.FrameHeader.sample_rate 
        property.
        """
        
        return self.header.sample_rate
        
    @property
    def filter_code(self):
        """
        Convenience wrapper for the Frame.FrameHeader.filter_code property.
        """
        
        return self.header.filter_code
        
    @property
    def time(self):
        """
        Convenience wrapper for the Frame.FramePayload.time property
        """
        
        return self.payload.time
        
    @property
    def central_freq(self):
        """
        Convenience wrapper for the Frame.FramePayload.central_freq property.
        """
        
        return self.payload.central_freq


def read_frame(filehandle, verbose=False):
    """
    Function to read in a single USRP frame (header+data) and store the 
    contents as a Frame object.
    
    .. note::
        Even real-valued data is stored in the FramePayload instance as a
        complex64 array.
    """
    
    # Header
    header = {}
    rawHeader = filehandle.read(149)
    for key,typ in zip((b'strt', b'rx_rate', b'rx_time', b'bytes', b'type', b'cplx', b'version', b'size'), ('Q', 'd', 'Qbd', 'Q', 'i', '?', 'b', 'i')):
        start = rawHeader.find(key)
        stop = start + len(key) + 1
        tln = struct.calcsize(typ)
        
        ## The rx_time is store as a pair, deal with that fact
        if key == b'rx_time':
            stop += 5
            tln = 17
        
        ## Unpack
        out = struct.unpack('>%s' % typ, rawHeader[stop:stop+tln])
    
        ## Deal with the tuple.  The time is the only one that has more than 
        ## one elements, so save it that way
        if len(out) == 1:
            out = out[0]
            
        ## Deal the the 'type' key
        if key == b'type':
            out = _type2name[out]
            
        ## Store
        header[key] = out
        
    # Cleanup time
    header[b'rx_time'] = (numpy.uint64(header[b'rx_time'][0]), numpy.float128(header[b'rx_time'][2]))
        
    # Extended header (optional)
    if header[b'strt'] != 149:
        rawHeader = filehandle.read(header[b'strt']-149)
        
        for key,typ in zip((b'rx_freq',), ('d',)):
            start = rawHeader.find(key)
            stop = start + len(key) + 1
            tln = struct.calcsize(typ)
            
            ## Unpack
            out = struct.unpack('>%s' % typ, rawHeader[stop:stop+tln])
        
            ## Deal with the tuple.
            out = out[0]
                
            ## Store
            header[key] = out
    else:
        header[b'rx_freq'] = 0.0
        
    # Data
    dataRaw = filehandle.read(header[b'bytes'])
    if header[b'cplx']:
        dataRaw = struct.unpack('>%i%s' % (2*header[b'bytes']//header[b'size'], header[b'type']), dataRaw)
        
        data = numpy.zeros( header[b'bytes']//header[b'size'], dtype=numpy.complex64)
        data.real = dataRaw[0::2]
        data.imag = dataRaw[1::2]
    else:
        dataRaw = struct.unpack('>%i%s' % (header[b'bytes']//header[b'size'], header[b'type']), dataRaw)
        
        data = numpy.zeros( header[b'bytes']//header[b'size'], dtype=numpy.int32)
        data.real = dataRaw
        
    # Build the frame
    timetag = header[b'rx_time'][0]*numpy.uint64(fS) + numpy.uint64( header[b'rx_time'][1]*fS )
    fHeader = FrameHeader(size=header[b'strt'], type=header[b'type'], complex=header[b'cplx'], sample_rate=header[b'rx_rate'])
    fData = FramePayload(size=header[b'bytes'], timetag=timetag, central_freq=header[b'rx_freq'], iq=data)
    newFrame = Frame(header=fHeader, payload=fData)
    
    return newFrame


def get_sample_rate(filehandle, nframes=None, filter_code=False):
    """
    Find out what the sampling rate/filter code is from a single observations.  
    By default, the rate in Hz is returned.  However, the corresponding filter 
    code can be returned instead by setting the filter_code keyword to true.
    
    This function is included to make easier to write code for DRX analysis and 
    modify it for USRP data.
    """
    
    with FilePositionSaver(filehandle):
        # Read in one frame
        newFrame = read_frame(filehandle)
        
    if not filter_code:
        return newFrame.sample_rate
    else:
        return newFrame.filter_code


def get_frame_size(filehandle, nframes=None):
    """
    Find out what the frame size is in bytes from a single observation.
    """
    
    with FilePositionSaver(filehandle):
        # Read in one frame
        newFrame = read_frame(filehandle)
        
    return newFrame.size


def get_beam_count(filehandle):
    """
    Find out how many beams are present and return the number of beams found.
    
    This function is included to make easier to write code for DRX analysis 
    and modify it for USRP data.
    
    .. note::
        This function always returns 1.
    """
    
    return 1


def get_frames_per_obs(filehandle):
    """
    Find out how many frames are present per beam and return the number of 
    frames per observations as a four-element tuple, one for each beam.
    
    This function is included to make easier to write code for DRX analysis 
    and modify it for USRP data.
    
    ..note::
        This function always returns the four-element tuple of (1, 0, 0, 0).
    """
    
    return (1, 0, 0, 0)