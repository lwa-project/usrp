#!/usr/bin/env python

"""
Run through a USRP file and determine if it is bad or not.
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info < (3,):
    range = xrange
    
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
    usrp.FRAME_SIZE = usrp.get_frame_size(fh)
    nFramesFile = os.path.getsize(args.filename) // usrp.FRAME_SIZE
    junkFrame = usrp.read_frame(fh)
    srate = junkFrame.sample_rate
    fh.seek(-usrp.FRAME_SIZE, 1)
    
    beam, tune, pol = junkFrame.id
    tunepols = max(usrp.get_frames_per_obs(fh))
    
    # Date & Central Frequnecy
    beginDate = junkFrame.time.datetime
    central_freq1 = 0.0
    central_freq2 = 0.0
    for i in range(tunepols):
        junkFrame = usrp.read_frame(fh)
        b,t,p = junkFrame.id
        if p == 0 and t == 1:
            central_freq1 = junkFrame.central_freq
        elif p == 0 and t == 2:
            central_freq2 = junkFrame.central_freq
        else:
            pass
    fh.seek(-tunepols*usrp.FRAME_SIZE, 1)
    
    # Report on the file
    print("Filename: %s" % args.filename)
    print("Date of First Frame: %s" % str(beginDate))
    print("Beam: %i" % beam)
    print("Tune/Pols: %i" % tunepols)
    print("Sample Rate: %i Hz" % srate)
    print("Tuning Frequency: %.3f Hz (1); %.3f Hz (2)" % (central_freq1, central_freq2))
    print(" ")
    
    # Convert chunk length to total frame count
    chunkLength = int(args.length * srate / junkFrame.payload.data.size * tunepols)
    chunkLength = int(1.0 * chunkLength / tunepols) * tunepols
    
    # Convert chunk skip to total frame count
    chunkSkip = int(args.skip * srate / junkFrame.payload.data.size * tunepols)
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
        data = numpy.empty((4,chunkLength*junkFrame.payload.data.size/tunepols), dtype=numpy.csingle)
        for j in range(chunkLength):
            # Read in the next frame and anticipate any problems that could occur
            try:
                cFrame = usrp.read_frame(fh, Verbose=False)
            except:
                done = True
                break
                
            beam,tune,pol = cFrame.id
            aStand = 2*(tune-1) + pol
            
            try:
                data[aStand, count[aStand]*cFrame.payload.data.size:(count[aStand]+1)*cFrame.payload.data.size] = cFrame.payload.data
                
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
            for j in range(4):
                bad = numpy.nonzero(data[j,:] > args.trim_level)[0]
                clipFraction[-1][j] = 1.0*len(bad) / data.shape[1]
            
            clip = clipFraction[-1]
            power = meanPower[-1]
            print("%2i | %23.2f | %23.2f |" % (i, clip[0]*100.0, power[0]))
        
            i += 1
            fh.seek(usrp.FRAME_SIZE*chunkSkip, 1)
            
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
    
