# -*- coding: utf-8 -*-

"""
Python module to read in USRP data.  This module defines the following 
classes for storing the DRX data found in a file:

Frame
  object that contains all data associated with a particular DRX frame.  
  The primary constituents of each frame are:
    * FrameHeader - the USRP frame header object and
    * FrameData   - the USRP frame data object.
  Combined, these two objects contain all of the information found in the 
  original USRP data block.

The functions defined in this module fall into two class:
 1. convert a frame in a file to a Frame object and
 2. describe the format of the data in the file.

For reading in data, use the readFrame function.  It takes a python file-
handle as an input and returns a fully-filled Frame object.
"""

import copy
import numpy
import struct

from common import fS

__version__ = '0.1'
__revision__ = '$Rev$'
__all__ = ['FrameHeader', 'FrameData', 'Frame', 'readFrame', 'getSampleRate', 'getFrameSize', '__version__', '__revision__', '__all__']


_type2name = {0: 'b', 
	     1: 's', 
	     2: 'i', 
	     3: 'l', 
	     4: 'q', 
	     5: 'f', 
	     6: 'd'}


class FrameHeader(object):
	"""
	Class that stores the information found in the header of a USRP 
	frame.
	"""
	
	def __init__(self, size=None, type=None, complex=False, sampleRate=0):
		self.size = size
		self.type = type
		self.complex = complex
		self.sampleRate = sampleRate
		
	def parseID(self):
		"""
		Placeholder for the ID of a USRP stream.  This isn't stored in the frame 
		headers by default so a single value of '0' is returned.
		"""
		
		pol = 0

		return pol
	
	def getSampleRate(self):
		"""
		Return the sample rate of the data in samples/second.
		"""
		
		return self.sampleRate


class FrameData(object):
	"""
	Class that stores the information found in the data section of a USRP
	frame.
	"""

	def __init__(self, size=None, timeTag=None, centralFreq=None, iq=None):
		self.size = size
		self.centralFreq = centralFreq
		self.timeTag = timeTag
		self.iq = iq
		
	def getCentralFreq(self):
		"""
		Function to set the central frequency of the USRP data in Hz.
		"""

		return self.centralFreq


class Frame(object):
	"""
	Class that stores the information contained within a single DRX 
	frame.  It's properties are FrameHeader and FrameData objects.
	"""

	def __init__(self, header=None, data=None):
		if header is None:
			self.header = FrameHeader()
		else:
			self.header = header
			
		if data is None:
			self.data = FrameData()
		else:
			self.data = data
			
		self.valid = True
		
	def parseID(self):
		"""
		Convenience wrapper for the Frame.FrameHeader.parseID 
		function.
		"""
		
		return self.header.parseID()
		
	def getSampleRate(self):
		"""
		Convenience wrapper for the Frame.FrameHeader.getSampleRate 
		function.
		"""
		
		return self.header.getSampleRate()
		
	def getTime(self):
		"""
		Function to convert the time tag from samples since the UNIX epoch
		(UTC 1970-01-01 00:00:00) to seconds since the UNIX epoch.
		"""
		
		seconds = self.data.timeTag / fS
		
		return seconds
		
	def getCentralFreq(self):
		"""
		Convenience wrapper for the Frame.FrameData.getCentralFreq function.
		"""
		
		return self.data.getCentralFreq()
		
	def __add__(self, y):
		"""
		Add the data sections of two frames together or add a number 
		to every element in the data section.
		"""
	
		newFrame = copy.deepcopy(self)
		newFrame += y	
		return newFrame
			
	def __iadd__(self, y):
		"""
		In-place add the data sections of two frames together or add 
		a number to every element in the data section.
		"""
		
		try:
			self.data.iq += y.data.iq
		except AttributeError:
			self.data.iq += y
		return self
		
	def __mul__(self, y):
		"""
		Multiple the data sections of two frames together or multiply 
		a number to every element in the data section.
		"""
		
		newFrame = copy.deepcopy(self)
		newFrame *= y
		return newFrame
			
	def __imul__(self, y):
		"""
		In-place multiple the data sections of two frames together or 
		multiply a number to every element in the data section.
		"""
		
		try:
			self.data.iq *= y.data.iq
		except AttributeError:
			self.data.iq *= y
		return self
			
	def __eq__(self, y):
		"""
		Check if the time tags of two frames are equal or if the time
		tag is equal to a particular value.
		"""
		
		tX = self.data.timeTag
		try:
			tY = y.data.timeTag
		except AttributeError:
			tY = y
		
		if tX == tY:
			return True
		else:
			return False
			
	def __ne__(self, y):
		"""
		Check if the time tags of two frames are not equal or if the time
		tag is not equal to a particular value.
		"""
		
		tX = self.data.timeTag
		try:
			tY = y.data.timeTag
		except AttributeError:
			tY = y
		
		if tX != tY:
			return True
		else:
			return False
			
	def __gt__(self, y):
		"""
		Check if the time tag of the first frame is greater than that of a
		second frame or if the time tag is greater than a particular value.
		"""
		
		tX = self.data.timeTag
		try:
			tY = y.data.timeTag
		except AttributeError:
			tY = y
		
		if tX > tY:
			return True
		else:
			return False
			
	def __ge__(self, y):
		"""
		Check if the time tag of the first frame is greater than or equal to 
		that of a second frame or if the time tag is greater than a particular 
		value.
		"""
		
		tX = self.data.timeTag
		try:
			tY = y.data.timeTag
		except AttributeError:
			tY = y
		
		if tX >= tY:
			return True
		else:
			return False
			
	def __lt__(self, y):
		"""
		Check if the time tag of the first frame is less than that of a
		second frame or if the time tag is greater than a particular value.
		"""
		
		tX = self.data.timeTag
		try:
			tY = y.data.timeTag
		except AttributeError:
			tY = y
		
		if tX < tY:
			return True
		else:
			return False
			
	def __le__(self, y):
		"""
		Check if the time tag of the first frame is less than or equal to 
		that of a second frame or if the time tag is greater than a particular 
		value.
		"""
		
		tX = self.data.timeTag
		try:
			tY = y.data.timeTag
		except AttributeError:
			tY = y
		
		if tX <= tY:
			return True
		else:
			return False
			
	def __cmp__(self, y):
		"""
		Compare two frames based on the time tags.  This is helpful for 
		sorting things.
		"""
		
		tX = self.data.timeTag
		tY = y.data.timeTag
		if tY > tX:
			return -1
		elif tX > tY:
			return 1
		else:
			return 0


def readFrame(filehandle, Verbose=False):
	"""
	Function to read in a single USRP frame (header+data) and store the 
	contents as a Frame object.
	"""
	
	# Header
	header = {}
	rawHeader = filehandle.read(149)
	for key,typ in zip(('strt', 'rx_rate', 'rx_time', 'bytes', 'type', 'cplx', 'version', 'size'), ('Q', 'd', 'Qbd', 'Q', 'i', '?', 'b', 'i')):
		start = rawHeader.find(key)
		stop = start + len(key) + 1
		tln = struct.calcsize(typ)
		
		## The rx_time is store as a pair, deal with that fact
		if key == 'rx_time':
			stop += 5
			tln = 17
		
		## Unpack
		out = struct.unpack('>%s' % typ, rawHeader[stop:stop+tln])
	
		## Deal with the tuple.  The time is the only one that has two 
		## elements, so save them that way
		if len(out) == 1:
			out = out[0]
			
		## Deal the the 'type' key
		if key == 'type':
			out = _type2name[out]
			
		## Store
		header[key] = out
		
	# Extended header (optional)
	if header['strt'] != 149:
		rawHeader = filehandle.read(header['strt']-149)
		
		for key,typ in zip(('rx_freq',), ('d',)):
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
		header['rx_freq'] = 0.0
		
	# Data
	dataRaw = filehandle.read(header['bytes'])
	if header['cplx']:
		dataRaw = struct.unpack('>%ih' % (2*header['bytes']/header['size'],), dataRaw)
		
		data = numpy.zeros( header['bytes']/header['size'], dtype=numpy.complex64)
		data.real = dataRaw[0::2]
		data.imag = dataRaw[1::2]
	else:
		dataRaw = struct.unpack('>%ih' % (header['bytes']/header['size'],), dataRaw)
		
		data = numpy.zeros( header['bytes']/header['size'], dtype=numpy.int32)
		data[:] = dataRaw
		
	# Build the frame
	timeTag = header['rx_time'][0] * int(fS) + int(header['rx_time'][2]*fS)
	fHeader = FrameHeader(size=header['strt'], type=header['type'], complex=header['cplx'], sampleRate=header['rx_rate'])
	fData = FrameData(size=header['bytes'], timeTag=timeTag, centralFreq=header['rx_freq'], iq=data)
	newFrame = Frame(header=fHeader, data=fData)
	
	return newFrame


def getSampleRate(filehandle, nFrames=None):
	"""
	Find out what the sampling rate is from a single observations.  The rate 
	in Hz is returned.
	
	This function is included to make easier to write code for DRX analysis and 
	modify it for USRP data.
	"""
	
	# Save the current position in the file so we can return to that point
	fhStart = filehandle.tell()

	# Read in one frame
	newFrame = readFrame(filehandle)
	
	# Return to the place in the file where we started
	filehandle.seek(fhStart)
	
	return newFrame.getSampleRate()


def getFrameSize(filehandle, nFrames=None):
	"""
	Find out what the frame size is in bytes from a single observation.
	"""
	# Save the current position in the file so we can return to that point
	fhStart = filehandle.tell()

	# Read in one frame
	newFrame = readFrame(filehandle)
	
	# Return to the place in the file where we started
	filehandle.seek(fhStart)
	
	return newFrame.header.size + newFrame.data.size
	