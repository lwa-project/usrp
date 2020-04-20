#!/usr/bin/env python

"""
Check the time times in a USRP file for flow.
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info < (3,):
    range = xrange
    
import os
import sys
import ephem
import gc
import argparse

from lsl import astro
from lsl_toolkits import USRP as usrp
from lsl_toolkits.USRP.common import fS
from lsl.misc import parser as aph

def main(args):
    skip = args.skip
    fh = open(args.filename, "rb")
    
    # Get the first frame and find out what the firt time tag is, which the
    # first frame number is, and what the sample rate it.  From the sample 
    # rate, estimate how the time tag should advance between frames.
    usrp.FRAME_SIZE = usrp.get_frame_size(fh)
    junkFrame = usrp.read_frame(fh)
    sample_rate = junkFrame.sample_rate
    tagSkip = int(fS / sample_rate * junkFrame.payload.data.shape[0])
    fh.seek(-usrp.FRAME_SIZE, 1)
    
    # Store the information about the first frame and convert the timetag to 
    # an ephem.Date object.
    prevTime = junkFrame.payload.timetag
    prevDate = currFrame.time.datetime
    
    # Skip ahead
    fh.seek(int(skip*sample_rate/junkFrame.payload.data.size)*usrp.FRAME_SIZE)
    
    # Report on the file
    print("Filename: %s" % os.path.basename(args[0]))
    print("Date of first frame: %i -> %s" % (prevTime, str(prevDate)))
    print("Sample rate: %i Hz" % sample_rate)
    print("Time tag skip per frame: %i" % tagSkip)
    if skip != 0:
        print("Skipping ahead %i frames (%.6f seconds)" % (int(skip*sample_rate/junkFrame.payload.data.size), int(skip*sample_rate/junkFrame.payload.data.size)*junkFrame.payload.data.size/sample_rate))
        
    k = 0
    #k = 1
    prevTime = [0, 0, 0, 0]
    prevDate = ['', '', '', '']
    prevNumb = [0, 0, 0, 0]
    for i in range(1):
        currFrame = usrp.read_frame(fh)
        beam, tune, pol = currFrame.id
        rID = 2*(tune-1) + pol
        
        prevTime[rID] = currFrame.payload.timetag
        prevDate[rID] = currFrame.time.datetime
        prevNumb[rID] = 1 + k // 1
        #prevNumb[rID] = k
        
        k += 1
        
    while True:
        try:
            currFrame = usrp.read_frame(fh)
        except:
            break
            
        beam, tune, pol = currFrame.id
        rID = 2*(tune-1) + pol
        currTime = currFrame.payload.timetag
        currDate = currFrame.time.datetime
        currNumb = 1 + k // 1
        #currNumb = k
        
        if tune == 1 and pol == 0 and currNumb % 50000 == 0:
            print("Beam %i, tune %i, pol %i: frame %8i -> %i (%s)" % (beam, tune, pol, currNumb, currTime, currDate))
            
        if currTime < prevTime[rID]:
            print("ERROR: t.t. %i @ frame %i < t.t. %i @ frame %i" % (currTime, currNumb, prevTime[rID], prevNumb[rID]))
            print("       -> difference: %i (%.5f seconds); %s" % (currTime-prevTime[rID], float(currTime-prevTime[rID])/fS, str(currDate)))
        elif currTime > (prevTime[rID] + tagSkip):
            print("ERROR: t.t. %i @ frame %i > t.t. %i @ frame %i + skip" % (currTime, currNumb, prevTime[rID], prevNumb[rID]))
            print("       -> difference: %i (%.5f seconds); %s" % (currTime-prevTime[rID], float(currTime-prevTime[rID])/fS, str(currDate)))
        elif currTime < (prevTime[rID] + tagSkip):
            print("ERROR: t.t %i @ frame %i < t.t. %i @ frame %i + skip" % (currTime, currNumb, prevTime[rID], prevNumb[rID]))
            print("       -> difference: %i (%.5f seconds; %s" % (currTime-prevTime[rID], float(currTime-prevTime[rID])/fS, str(currDate)))
            print("       -> beam %i tune %i pol %i" % (beam, tune, pol))
        else:
            pass
            
        prevTime[rID] = currTime
        prevDate[rID] = currDate
        prevNumb[rID] = currNumb
        k += 1
        
        del currFrame
        
    fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='read in a USRP file and check the flow of time', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, 
                        help='filename to check')
    parser.add_argument('-s', '--skip', type=aph.positive_float, default=0.0, 
                        help='skip period in seconds between chunks')
    args = parser.parse_args()
    main(args)
    
