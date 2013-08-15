# -*- coding: utf-8 -*-

"""
Python module to read in USRP data.  This module defines the following 
classes for storing the USRP data found in a file:

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

For describing the format of data in the file, two function are provided:

getBeamCount
  read in the first few frames of an open file handle and return how many 
  beams are present in the file.

getFramesPerObs
  read in the first several frames to see how many frames (tunings/polarizations)
  are associated with each beam.
"""

import copy
import numpy
import struct

from common import fS

__version__ = '0.2'
__revision__ = '$Rev$'
__all__ = ['FrameHeader', 'FrameData', 'Frame', 'readFrame', 'getSampleRate', 'getFrameSize', 'getBeamCount', 'getFramesPerObs', 'filterCodes', '__version__', '__revision__', '__all__']


_type2name = {0: 'b', 
		    1: 'h', 
		    2: 'i', 
		    3: 'l', 
		    4: 'q', 
		    5: 'f', 
		    6: 'd'}


# List of filter codes and their corresponding sample rates in Hz
filterCodes = {}
for i in xrange(9):
	filterCodes[i] = fS / 2**(9-i)


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
		Return the ID for a USRP stream.
		
		.. note:: 
			This isn't stored in the frame headers by default a three element 
			tuple of (0, 1, 0) is returned to be compatible with DRX.
		"""
		
		return (0,1,0)
		
	def getSampleRate(self):
		"""
		Return the sample rate of the data in samples/second.
		"""
		
		return self.sampleRate
		
	def getFilterCode(self):
		"""
		Function to convert the sample rate in Hz to a filter code.
		"""
		
		sampleCodes = {}
		for key,value in filterCodes.iteritems():
			sampleCodes[value] = key
			
		return sampleCodes[self.getSampleRate()]


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
		
		centralFreq = 1.0*self.centralFreq
		if centralFreq < 0:
			centralFreq += fS
			
		return centralFreq


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
		
	def getFilterCode(self):
		"""
		Convenience wrapper for the Frame.FrameHeader.getFilterCode function.
		"""
		
		return self.header.getFilterCode()
		
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
	
	.. note::
		Even real-valued data is stored in the FrameData instance as a
		complex64 array.
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
	
		## Deal with the tuple.  The time is the only one that has more than 
		## one elements, so save it that way
		if len(out) == 1:
			out = out[0]
			
		## Deal the the 'type' key
		if key == 'type':
			out = _type2name[out]
			
		## Store
		header[key] = out
		
	# Cleanup time
	header['rx_time'] = (header['rx_time'][0], header['rx_time'][2])
		
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
		dataRaw = struct.unpack('>%i%s' % (2*header['bytes']/header['size'], header['type']), dataRaw)
		
		data = numpy.zeros( header['bytes']/header['size'], dtype=numpy.complex64)
		data.real = dataRaw[0::2]
		data.imag = dataRaw[1::2]
	else:
		dataRaw = struct.unpack('>%i%s' % (header['bytes']/header['size'], header['type']), dataRaw)
		
		data = numpy.zeros( header['bytes']/header['size'], dtype=numpy.int32)
		data.real = dataRaw
		
	# Build the frame
	timeTag = int(header['rx_time'][0])*int(fS) + int(header['rx_time'][1]*int(fS))
	print header['rx_time'], timeTag, timeTag/fS
	fHeader = FrameHeader(size=header['strt'], type=header['type'], complex=header['cplx'], sampleRate=header['rx_rate'])
	fData = FrameData(size=header['bytes'], timeTag=timeTag, centralFreq=header['rx_freq'], iq=data)
	newFrame = Frame(header=fHeader, data=fData)
	
	return newFrame


def getSampleRate(filehandle, nFrames=None, FilterCode=False):
	"""
	Find out what the sampling rate/filter code is from a single observations.  
	By default, the rate in Hz is returned.  However, the corresponding filter 
	code can be returned instead by setting the FilterCode keyword to true.
	
	This function is included to make easier to write code for DRX analysis and 
	modify it for USRP data.
	"""
	
	# Save the current position in the file so we can return to that point
	fhStart = filehandle.tell()

	# Read in one frame
	newFrame = readFrame(filehandle)
	
	# Return to the place in the file where we started
	filehandle.seek(fhStart)
	
	if not FilterCode:
		return newFrame.getSampleRate()
	else:
		return newFrame.getFilterCode()


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


def getBeamCount(filehandle):
	"""
	Find out how many beams are present and return the number of beams found.
	
	This function is included to make easier to write code for DRX analysis 
	and modify it for USRP data.
	
	.. note::
		This function always returns 1.
	"""
	
	return 1


def getFramesPerObs(filehandle):
	"""
	Find out how many frames are present per beam and return the number of 
	frames per observations as a four-element tuple, one for each beam.
	
	This function is included to make easier to write code for DRX analysis 
	and modify it for USRP data.
	
	..note::
		This function always returns the four-element tuple of (1, 0, 0, 0).
	"""
	
	return (1, 0, 0, 0)