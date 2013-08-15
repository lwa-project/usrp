#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check the time times in a USRP file for flow.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import ephem
import gc

from lsl import astro
from lsl_toolkits import USRP as usrp
from lsl_toolkits.USRP.common import fS

def main(args):
	if args[0] == '-s':
		skip = float(args[1])
		fh = open(args[2], "rb")
	else:
		skip = 0
		fh = open(args[0], "rb")

	# Get the first frame and find out what the firt time tag is, which the
	# first frame number is, and what the sample rate it.  From the sample 
	# rate, estimate how the time tag should advance between frames.
	usrp.FrameSize = usrp.getFrameSize(fh)
	junkFrame = usrp.readFrame(fh)
	sampleRate = junkFrame.getSampleRate()
	tagSkip = int(fS / sampleRate * junkFrame.data.iq.shape[0])
	fh.seek(-usrp.FrameSize, 1)
	
	# Store the information about the first frame and convert the timetag to 
	# an ephem.Date object.
	prevTime = junkFrame.data.timeTag
	prevDate = ephem.Date(astro.unix_to_utcjd(junkFrame.getTime()) - astro.DJD_OFFSET)
	
	# Skip ahead
	fh.seek(int(skip*sampleRate/junkFrame.data.iq.size)*usrp.FrameSize)
	
	# Report on the file
	print "Filename: %s" % os.path.basename(args[0])
	print "Date of first frame: %i -> %s" % (prevTime, str(prevDate))
	print "Sample rate: %i Hz" % sampleRate
	print "Time tag skip per frame: %i" % tagSkip
	if skip != 0:
		print "Skipping ahead %i frames (%.6f seconds)" % (int(skip*sampleRate/junkFrame.data.iq.size), int(skip*sampleRate/junkFrame.data.iq.size)*junkFrame.data.iq.size/sampleRate)
		
	k = 0
	#k = 1
	prevTime = [0, 0, 0, 0]
	prevDate = ['', '', '', '']
	prevNumb = [0, 0, 0, 0]
	for i in xrange(1):
		currFrame = usrp.readFrame(fh)
		beam, tune, pol = currFrame.parseID()
		rID = 2*(tune-1) + pol
		
		prevTime[rID] = currFrame.data.timeTag
		prevDate[rID] = ephem.Date(astro.unix_to_utcjd(currFrame.getTime()) - astro.DJD_OFFSET)
		prevNumb[rID] = 1 + k / 1
		#prevNumb[rID] = k
		
		k += 1
		
	while True:
		try:
			currFrame = usrp.readFrame(fh)
		except:
			break
			
		beam, tune, pol = currFrame.parseID()
		rID = 2*(tune-1) + pol
		currTime = currFrame.data.timeTag
		currDate = ephem.Date(astro.unix_to_utcjd(currFrame.getTime()) - astro.DJD_OFFSET)
		currNumb = 1 + k / 1
		#currNumb = k
		
		if tune == 1 and pol == 0 and currNumb % 50000 == 0:
			print "Beam %i, tune %i, pol %i: frame %8i -> %i (%s)" % (beam, tune, pol, currNumb, currTime, currDate)
			
		if currTime < prevTime[rID]:
			print "ERROR: t.t. %i @ frame %i < t.t. %i @ frame %i" % (currTime, currNumb, prevTime[rID], prevNumb[rID])
			print "       -> difference: %i (%.5f seconds); %s" % (currTime-prevTime[rID], float(currTime-prevTime[rID])/fS, str(currDate))
		elif currTime > (prevTime[rID] + tagSkip):
			print "ERROR: t.t. %i @ frame %i > t.t. %i @ frame %i + skip" % (currTime, currNumb, prevTime[rID], prevNumb[rID])
			print "       -> difference: %i (%.5f seconds); %s" % (currTime-prevTime[rID], float(currTime-prevTime[rID])/fS, str(currDate))
		elif currTime < (prevTime[rID] + tagSkip):
			print "ERROR: t.t %i @ frame %i < t.t. %i @ frame %i + skip" % (currTime, currNumb, prevTime[rID], prevNumb[rID])
			print "       -> difference: %i (%.5f seconds; %s" % (currTime-prevTime[rID], float(currTime-prevTime[rID])/fS, str(currDate))
			print "       -> beam %i tune %i pol %i" % (beam, tune, pol)
		else:
			pass
			
		prevTime[rID] = currTime
		prevDate[rID] = currDate
		prevNumb[rID] = currNumb
		k += 1
		
		del currFrame
		
	fh.close()


if __name__ == "__main__":
	main(sys.argv[1:])
