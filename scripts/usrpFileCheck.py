#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run through a USRP file and determine if it is bad or not.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

# Python3 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import sys
import ephem
import numpy
import argparse

from lsl import astro
from lsl_toolkits import USRP as usrp
from lsl.misc import parser as aph


def main(args):
    fh = open(args.filename, "rb")
    usrp.FrameSize = usrp.getFrameSize(fh)
    nFramesFile = os.path.getsize(args.filename) // usrp.FrameSize
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
    print("Filename: %s" % args.filename)
    print("Date of First Frame: %s" % str(beginDate))
    print("Beam: %i" % beam)
    print("Tune/Pols: %i" % tunepols)
    print("Sample Rate: %i Hz" % srate)
    print("Tuning Frequency: %.3f Hz (1); %.3f Hz (2)" % (centralFreq1, centralFreq2))
    print(" ")
    
    # Convert chunk length to total frame count
    chunkLength = int(args.length * srate / junkFrame.data.iq.size * tunepols)
    chunkLength = int(1.0 * chunkLength / tunepols) * tunepols
    
    # Convert chunk skip to total frame count
    chunkSkip = int(args.skip * srate / junkFrame.data.iq.size * tunepols)
    chunkSkip = int(1.0 * chunkSkip / tunepols) * tunepols
    
    # Output arrays
    clipFraction = []
    meanPower = []
    
    # Go!
    i = 1
    done = False
    print("   |        Clipping         |          Power          |")
    print("   |                         |                         |")
    print("---+-------------------------+-------------------------+")
    
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
                bad = numpy.nonzero(data[j,:] > args.trim_level)[0]
                clipFraction[-1][j] = 1.0*len(bad) / data.shape[1]
            
            clip = clipFraction[-1]
            power = meanPower[-1]
            print("%2i | %23.2f | %23.2f |" % (i, clip[0]*100.0, power[0]))
        
            i += 1
            fh.seek(usrp.FrameSize*chunkSkip, 1)
            
    clipFraction = numpy.array(clipFraction)
    meanPower = numpy.array(meanPower)
    
    clip = clipFraction.mean(axis=0)
    power = meanPower.mean(axis=0)
    
    print("---+-------------------------+-------------------------+")
    print("%2s | %23.2f | %23.2f |" % ('M', clip[0]*100.0, power[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='run through a USRP file and determine if it is bad or not.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, 
                        help='filename to check')
    parser.add_argument('-l', '--length', type=aph.positive_float, default=1.0, 
                        help='length of time in seconds to analyze')
    parser.add_argument('-s', '--skip', type=aph.positive_float, default=900.0, 
                        help='skip period in seconds between chunks')
    parser.add_argument('-t', '--trim-level', type=aph.positive_float, default=32768**2, 
                        help='trim level for power analysis with clipping')
    args = parser.parse_args()
    main(args)
    