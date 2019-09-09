#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a USRP file, plot the time averaged spectra for each beam output over some 
period.

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
import math
import numpy
import ephem
import argparse

from lsl_toolkits import USRP as usrp
import lsl.statistics.robust as robust
import lsl.correlator.fx as fxc
from lsl.reader import errors
from lsl.astro import unix_to_utcjd, DJD_OFFSET
from lsl.misc import parser as aph

import matplotlib.pyplot as plt


def bestFreqUnits(freq):
    """Given a numpy array of frequencies in Hz, return a new array with the
    frequencies in the best units possible (kHz, MHz, etc.)."""

    # Figure out how large the data are
    scale = int(math.log10(freq.max()))
    if scale >= 9:
        divis = 1e9
        units = 'GHz'
    elif scale >= 6:
        divis = 1e6
        units = 'MHz'
    elif scale >= 3:
        divis = 1e3
        units = 'kHz'
    else:
        divis = 1
        units = 'Hz'

    # Convert the frequency
    newFreq = freq / divis

    # Return units and freq
    return (newFreq, units)


def main(args):
    # Length of the FFT and the window to use
    LFFT = args.fft_length
    if args.bartlett:
        window = numpy.bartlett
    elif args.blackman:
        window = numpy.blackman
    elif args.hanning:
        window = numpy.hanning
    else:
        window = fxc.noWindow
    args.window = window
    
    fh = open(args.filename, "rb")
    usrp.FrameSize = usrp.getFrameSize(fh)
    nFramesFile = os.path.getsize(args.filename) // usrp.FrameSize
    
    junkFrame = usrp.readFrame(fh)
    srate = junkFrame.getSampleRate()
    t0 = junkFrame.getTime()
    fh.seek(-usrp.FrameSize, 1)
    
    beams = usrp.getBeamCount(fh)
    tunepols = usrp.getFramesPerObs(fh)
    tunepol = tunepols[0] + tunepols[1] + tunepols[2] + tunepols[3]
    beampols = tunepol
    
    # Offset in frames for beampols beam/tuning/pol. sets
    offset = int(args.skip * srate / junkFrame.data.iq.size * beampols)
    offset = int(1.0 * offset / beampols) * beampols
    fh.seek(offset*usrp.FrameSize, 1)
    
    # Iterate on the offsets until we reach the right point in the file.  This
    # is needed to deal with files that start with only one tuning and/or a 
    # different sample rate.  
    while True:
        ## Figure out where in the file we are and what the current tuning/sample 
        ## rate is
        junkFrame = usrp.readFrame(fh)
        srate = junkFrame.getSampleRate()
        t1 = junkFrame.getTime()
        tunepols = usrp.getFramesPerObs(fh)
        tunepol = tunepols[0] + tunepols[1] + tunepols[2] + tunepols[3]
        beampols = tunepol
        fh.seek(-usrp.FrameSize, 1)
        
        ## See how far off the current frame is from the target
        tDiff = t1 - (t0 + args.skip)
        
        ## Half that to come up with a new seek parameter
        tCorr = -tDiff / 2.0
        cOffset = int(tCorr * srate / junkFrame.data.iq.size * beampols)
        cOffset = int(1.0 * cOffset / beampols) * beampols
        offset += cOffset
        
        ## If the offset is zero, we are done.  Otherwise, apply the offset
        ## and check the location in the file again/
        if cOffset is 0:
            break
        fh.seek(cOffset*usrp.FrameSize, 1)
        
    # Update the offset actually used
    args.skip = t1 - t0
    offset = int(round(args.skip * srate / junkFrame.data.iq.size * beampols))
    offset = int(1.0 * offset / beampols) * beampols
    
    # Make sure that the file chunk size contains is an integer multiple
    # of the FFT length so that no data gets dropped.  This needs to
    # take into account the number of beampols in the data, the FFT length,
    # and the number of samples per frame.
    maxFrames = int(1.0*28000/beampols*junkFrame.data.iq.size/float(LFFT))*LFFT/junkFrame.data.iq.size*beampols
    
    # Number of frames to integrate over
    nFramesAvg = int(args.average * srate / junkFrame.data.iq.size * beampols)
    nFramesAvg = int(1.0 * nFramesAvg / beampols*junkFrame.data.iq.size/float(LFFT))*LFFT/junkFrame.data.iq.size*beampols
    args.average = 1.0 * nFramesAvg / beampols * junkFrame.data.iq.size / srate
    maxFrames = nFramesAvg
    
    # Number of remaining chunks (and the correction to the number of
    # frames to read in).
    nChunks = int(round(args.duration / args.average))
    if nChunks == 0:
        nChunks = 1
    nFrames = nFramesAvg*nChunks
    
    # Date & Central Frequnecy
    beginDate = ephem.Date(unix_to_utcjd(junkFrame.getTime()) - DJD_OFFSET)
    centralFreq1 = 0.0
    centralFreq2 = 0.0
    for i in xrange(4):
        junkFrame = usrp.readFrame(fh)
        b,t,p = junkFrame.parseID()
        if p == 0 and t == 1:
            centralFreq1 = junkFrame.getCentralFreq()
        elif p == 0 and t == 2:
            centralFreq2 = junkFrame.getCentralFreq()
        else:
            pass
    fh.seek(-4*usrp.FrameSize, 1)
    
    centralFreq1 = centralFreq1
    centralFreq2 = centralFreq2
    
    # File summary
    print("Filename: %s" % args.filename)
    print("Date of First Frame: %s" % str(beginDate))
    print("Beams: %i" % beams)
    print("Tune/Pols: %i %i %i %i" % tunepols)
    print("Sample Rate: %i Hz" % srate)
    print("Tuning Frequency: %.3f Hz (1); %.3f Hz (2)" % (centralFreq1, centralFreq2))
    print("Frames: %i (%.3f s)" % (nFramesFile, 1.0 * nFramesFile / beampols * junkFrame.data.iq.size / srate))
    print("---")
    print("Offset: %.3f s (%i frames)" % (args.skip, offset))
    print("Integration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (args.average, nFramesAvg, nFramesAvg / beampols))
    print("Duration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (args.average*nChunks, nFrames, nFrames / beampols))
    print("Chunks: %i" % nChunks)
    
    # Sanity check
    if nFrames > (nFramesFile - offset):
        raise RuntimeError("Requested integration time+offset is greater than file length")
        
    # Estimate clip level (if needed)
    if args.estimate_clip_level:
        filePos = fh.tell()
        
        # Read in the first 100 frames for each tuning/polarization
        count = {0:0, 1:0, 2:0, 3:0}
        data = numpy.zeros((4, junkFrame.data.iq.size*100), dtype=numpy.csingle)
        for i in xrange(4*100):
            try:
                cFrame = usrp.readFrame(fh, Verbose=False)
            except errors.eofError:
                break
            except errors.syncError:
                continue
            
            beam,tune,pol = cFrame.parseID()
            aStand = 2*(tune-1) + pol
            
            data[aStand, count[aStand]*junkFrame.data.iq.size:(count[aStand]+1)*junkFrame.data.iq.size] = cFrame.data.iq
            count[aStand] +=  1
        
        # Go back to where we started
        fh.seek(filePos)
        
        # Correct the DC bias
        for j in xrange(data.shape[0]):
            data[j,:] -= data[j,:].mean()
            
        # Compute the robust mean and standard deviation for I and Q for each
        # tuning/polarization
        meanI = []
        meanQ = []
        stdsI = []
        stdsQ = []
        for i in xrange(4):
            meanI.append( robust.mean(data[i,:].real) )
            meanQ.append( robust.mean(data[i,:].imag) )
            
            stdsI.append( robust.std(data[i,:].real) )
            stdsQ.append( robust.std(data[i,:].imag) )
            
        # Report
        print("Statistics:")
        for i in xrange(4):
            print(" Mean %i: %.3f + %.3f j" % (i+1, meanI[i], meanQ[i]))
            print(" Std  %i: %.3f + %.3f j" % (i+1, stdsI[i], stdsQ[i]))
            
        # Come up with the clip levels based on 4 sigma
        clip1 = (meanI[0] + meanI[1] + meanQ[0] + meanQ[1]) / 4.0
        clip2 = (meanI[2] + meanI[3] + meanQ[2] + meanQ[3]) / 4.0
        
        clip1 += 5*(stdsI[0] + stdsI[1] + stdsQ[0] + stdsQ[1]) / 4.0
        clip2 += 5*(stdsI[2] + stdsI[3] + stdsQ[2] + stdsQ[3]) / 4.0
        
        clip1 = int(round(clip1))
        clip2 = int(round(clip2))
        
        # Report again
        print("Clip Levels:")
        print(" Tuning 1: %i" % clip1)
        print(" Tuning 2: %i" % clip2)
        
    else:
        clip1 = args.clip_level
        clip2 = args.clip_level
        
    # Master loop over all of the file chunks
    masterWeight = numpy.zeros((nChunks, 4, LFFT-1 if float(fxc.__version__) < 0.8 else LFFT))
    masterSpectra = numpy.zeros((nChunks, 4, LFFT-1 if float(fxc.__version__) < 0.8 else LFFT))
    masterTimes = numpy.zeros(nChunks)
    for i in xrange(nChunks):
        # Find out how many frames remain in the file.  If this number is larger
        # than the maximum of frames we can work with at a time (maxFrames),
        # only deal with that chunk
        framesRemaining = nFrames - i*maxFrames
        if framesRemaining > maxFrames:
            framesWork = maxFrames
        else:
            framesWork = framesRemaining
        print("Working on chunk %i, %i frames remaining" % (i, framesRemaining))
        
        count = {0:0, 1:0, 2:0, 3:0}
        data = numpy.zeros((4,framesWork*junkFrame.data.iq.size//beampols), dtype=numpy.csingle)
        # If there are fewer frames than we need to fill an FFT, skip this chunk
        if data.shape[1] < LFFT:
            break
            
        # Inner loop that actually reads the frames into the data array
        print("Working on %.1f ms of data" % ((framesWork*junkFrame.data.iq.size/beampols/srate)*1000.0))
        
        for j in xrange(framesWork):
            # Read in the next frame and anticipate any problems that could occur
            try:
                cFrame = usrp.readFrame(fh, Verbose=False)
            except errors.eofError:
                break
            except errors.syncError:
                continue
                
            beam,tune,pol = cFrame.parseID()
            aStand = 2*(tune-1) + pol
            if j is 0:
                cTime = cFrame.getTime()
                
            data[aStand, count[aStand]*cFrame.data.iq.size:(count[aStand]+1)*cFrame.data.iq.size] = cFrame.data.iq
            count[aStand] +=  1
            
        # Correct the DC bias
        for j in xrange(data.shape[0]):
            data[j,:] -= data[j,:].mean()
            
        # Calculate the spectra for this block of data and then weight the results by 
        # the total number of frames read.  This is needed to keep the averages correct.
        freq, tempSpec1 = fxc.SpecMaster(data[:2,:], LFFT=LFFT, window=args.window, verbose=True, SampleRate=srate, ClipLevel=clip1)
        
        freq, tempSpec2 = fxc.SpecMaster(data[2:,:], LFFT=LFFT, window=args.window, verbose=True, SampleRate=srate, ClipLevel=clip2)
        
        # Save the results to the various master arrays
        masterTimes[i] = cTime
        
        masterSpectra[i,0,:] = tempSpec1[0,:]
        masterSpectra[i,1,:] = tempSpec1[1,:]
        masterSpectra[i,2,:] = tempSpec2[0,:]
        masterSpectra[i,3,:] = tempSpec2[1,:]
        
        masterWeight[i,0,:] = int(count[0] * cFrame.data.iq.size / LFFT)
        masterWeight[i,1,:] = int(count[1] * cFrame.data.iq.size / LFFT)
        masterWeight[i,2,:] = int(count[2] * cFrame.data.iq.size / LFFT)
        masterWeight[i,3,:] = int(count[3] * cFrame.data.iq.size / LFFT)
        
        # We don't really need the data array anymore, so delete it
        del(data)
        
    # Now that we have read through all of the chunks, perform the final averaging by
    # dividing by all of the chunks
    outname = os.path.split(args.filename)[1]
    outname = os.path.splitext(outname)[0]
    outname = '%s-waterfall.npz' % outname
    numpy.savez(outname, freq=freq, freq1=freq+centralFreq1, freq2=freq+centralFreq2, times=masterTimes, spec=masterSpectra, tInt=(maxFrames*cFrame.data.iq.size/beampols/srate), srate=srate,  standMapper=[4*(beam-1) + i for i in xrange(masterSpectra.shape[1])])
    spec = numpy.squeeze( (masterWeight*masterSpectra).sum(axis=0) / masterWeight.sum(axis=0) )
    
    # The plots:  This is setup for the current configuration of 20 beampols
    fig = plt.figure()
    figsX = int(round(math.sqrt(1)))
    figsY = 1 // figsX
    
    # Put the frequencies in the best units possible
    freq, units = bestFreqUnits(freq)
    
    for i in xrange(1):
        ax = fig.add_subplot(figsX,figsY,i+1)
        currSpectra = numpy.squeeze( numpy.log10(masterSpectra[:,i,:])*10.0 )
        currSpectra = numpy.where( numpy.isfinite(currSpectra), currSpectra, -10)
        
        ax.imshow(currSpectra, interpolation='nearest', extent=(freq.min(), freq.max(), args.skip+0, args.skip+args.average*nChunks), origin='lower')
        print(currSpectra.min(), currSpectra.max())
        
        ax.axis('auto')
        ax.set_title('Beam %i, Tune. %i, Pol. %i' % (beam, i//2+1, i%2))
        ax.set_xlabel('Frequency Offset [%s]' % units)
        ax.set_ylabel('Time [s]')
        ax.set_xlim([freq.min(), freq.max()])
        
    print("RBW: %.4f %s" % ((freq[1]-freq[0]), units))
    if True:
        plt.show()
        
    # Save spectra image if requested
    if args.output is not None:
        fig.savefig(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='read in USRP files and create a collection of time-averaged spectra stored as an HDF5 file called <args.filename>-waterfall.hdf5',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('args.filename', type=str, 
                        help='args.filename to process')
    wgroup = parser.add_mutually_exclusive_group(required=False)
    wgroup.add_argument('-t', '--bartlett', action='store_true', 
                        help='apply a Bartlett window to the data')
    wgroup.add_argument('-b', '--blackman', action='store_true', 
                        help='apply a Blackman window to the data')
    wgroup.add_argument('-n', '--hanning', action='store_true', 
                        help='apply a Hanning window to the data')
    parser.add_argument('-s', '--skip', type=aph.positive_or_zero_float, default=0.0, 
                        help='skip the specified number of seconds at the beginning of the file')
    parser.add_argument('-a', '--average', type=aph.positive_float, default=1.0, 
                        help='number of seconds of data to average for spectra')
    parser.add_argument('-d', '--duration', type=aph.positive_or_zero_float, default=0.0, 
                        help='number of seconds to calculate the waterfall for; 0 for everything') 
    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
                        help='run %(prog)s in silent mode')
    parser.add_argument('-l', '--fft-length', type=aph.positive_int, default=4096, 
                        help='set FFT length')
    parser.add_argument('-c', '--clip-level', type=aph.positive_or_zero_int, default=0,  
                        help='FFT blanking clipping level in counts; 0 disables')
    parser.add_argument('-e', '--estimate-clip-level', action='store_true', 
                        help='use robust statistics to estimate an appropriate clip level; overrides -c/--clip-level')
    parser.add_argument('-o', '--output', type=str,
                        help='output file name for the waterfall image')
    args = parser.parse_args()
    main(args)
    