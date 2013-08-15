#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run through a DRX file and determine if it is bad or not.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import ephem
import numpy
import getopt

from lsl import astro
from lsl_toolkits import USRP as usrp


def usage(exitCode=None):
	print """usrpFileCheck.py - Run through a USRP file and determine if it is bad or not.

Usage: usrpFileCheck.py [OPTIONS] filename

Options:
-h, --help         Display this help information
-l, --length       Length of time in seconds to analyze (default 1 s)
-s, --skip         Skip period in seconds between chunks (default 900 s)
-t, --trim-level   Trim level for power analysis with clipping (default 32768^2)
"""

	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return None


def parseOptions(args):
	config = {}
	config['length'] = 1.0
	config['skip'] = 900.0
	config['trim'] = 32768**2
	
	try:
		opts, args = getopt.getopt(args, "hl:s:t:", ["help", "length=", "skip=", "trim-level="])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)

	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-l', '--length'):
			config['length'] = float(value)
		elif opt in ('-s', '--skip'):
			config['skip'] = float(value)
		elif opt in ('-t', '--trim-level'):
			config['trim'] = int(value)
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Return
	return config


def main(args):
	config = parseOptions(args)
	filename = config['args'][0]
	
	fh = open(filename, "rb")
	usrp.FrameSize = usrp.getFrameSize(fh)
	nFramesFile = os.path.getsize(filename) / usrp.FrameSize
	junkFrame = usrp.readFrame(fh)
	srate = junkFrame.getSampleRate()
	fh.seek(-usrp.FrameSize, 1)
	
	beam, tune, pol = junkFrame.parseID()
	tunepols = max(usrp.getFramesPerObs(fh))
	
	# Date & Central Frequnecy
	beginDate = ephem.Date(astro.unix_to_utcjd(junkFrame.getTime()) - astro.DJD_OFFSET)
	centralFreq1 = 0.0
	centralFreq2 = 0.0
	for i in xrange(tunepols):
		junkFrame = usrp.readFrame(fh)
		b,t,p = junkFrame.parseID()
		if p == 0 and t == 1:
			centralFreq1 = junkFrame.getCentralFreq()
		elif p == 0 and t == 2:
			centralFreq2 = junkFrame.getCentralFreq()
		else:
			pass
	fh.seek(-tunepols*usrp.FrameSize, 1)
	
	# Report on the file
	print "Filename: %s" % filename
	print "Date of First Frame: %s" % str(beginDate)
	print "Beam: %i" % beam
	print "Tune/Pols: %i" % tunepols
	print "Sample Rate: %i Hz" % srate
	print "Tuning Frequency: %.3f Hz (1); %.3f Hz (2)" % (centralFreq1, centralFreq2)
	print " "
	
	# Convert chunk length to total frame count
	chunkLength = int(config['length'] * srate / junkFrame.data.iq.size * tunepols)
	chunkLength = int(1.0 * chunkLength / tunepols) * tunepols
	
	# Convert chunk skip to total frame count
	chunkSkip = int(config['skip'] * srate / junkFrame.data.iq.size * tunepols)
	chunkSkip = int(1.0 * chunkSkip / tunepols) * tunepols
	
	# Output arrays
	clipFraction = []
	meanPower = []
	
	# Go!
	i = 1
	done = False
	print "   |        Clipping         |          Power          |"
	print "   |                         |                         |"
	print "---+-------------------------+-------------------------+"
	
	while True:
		count = {0:0, 1:0, 2:0, 3:0}
		data = numpy.empty((4,chunkLength*junkFrame.data.iq.size/tunepols), dtype=numpy.csingle)
		for j in xrange(chunkLength):
			# Read in the next frame and anticipate any problems that could occur
			try:
				cFrame = usrp.readFrame(fh, Verbose=False)
			except:
				done = True
				break
				
			beam,tune,pol = cFrame.parseID()
			aStand = 2*(tune-1) + pol
			
			try:
				data[aStand, count[aStand]*cFrame.data.iq.size:(count[aStand]+1)*cFrame.data.iq.size] = cFrame.data.iq
				
				# Update the counters so that we can average properly later on
				count[aStand] += 1
			except ValueError:
				pass
				
		if done:
			break
			
		else:
			data = numpy.abs(data)**2
			data = data.astype(numpy.int32)
			
			clipFraction.append( numpy.zeros(4) )
			meanPower.append( data.mean(axis=1) )
			for j in xrange(4):
				bad = numpy.nonzero(data[j,:] > config['trim'])[0]
				clipFraction[-1][j] = 1.0*len(bad) / data.shape[1]
			
			clip = clipFraction[-1]
			power = meanPower[-1]
			print "%2i | %23.2f | %23.2f |" % (i, clip[0]*100.0, power[0])
		
			i += 1
			fh.seek(usrp.FrameSize*chunkSkip, 1)
			
	clipFraction = numpy.array(clipFraction)
	meanPower = numpy.array(meanPower)
	
	clip = clipFraction.mean(axis=0)
	power = meanPower.mean(axis=0)
	
	print "---+-------------------------+-------------------------+"
	print "%2s | %23.2f | %23.2f |" % ('M', clip[0]*100.0, power[0])


if __name__ == "__main__":
	main(sys.argv[1:])
	
