#!/usr/bin/env python

"""
Given a USRP file, plot the time averaged spectra.
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import

import os
import sys
import math
import numpy
import ephem
import argparse

import lsl_toolkits.USRP as usrp
import lsl.correlator.fx as fxc
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
    # Length of the FFT
    LFFT = args.fft_length

    fh = open(args.filename, "rb")
    usrp.FRAME_SIZE = usrp.get_frame_size(fh)
    nFramesFile = os.path.getsize(args.filename) // usrp.FRAME_SIZE
    
    junkFrame = usrp.read_frame(fh)
    srate = junkFrame.sample_rate
    t0i, t0f = junkFrame.time
    fh.seek(-usrp.FRAME_SIZE, 1)
    
    beams = usrp.get_beam_count(fh)
    tunepols = usrp.get_frames_per_obs(fh)
    tunepol = tunepols[0] + tunepols[1] + tunepols[2] + tunepols[3]
    beampols = tunepol
    
    # Offset in frames for beampols beam/tuning/pol. sets
    offset = int(args.skip * srate / junkFrame.payload.data.size * beampols)
    offset = int(1.0 * offset / beampols) * beampols
    fh.seek(offset*usrp.FRAME_SIZE)
    
    # Iterate on the offsets until we reach the right point in the file.  This
    # is needed to deal with files that start with only one tuning and/or a 
    # different sample rate.  
    while True:
        ## Figure out where in the file we are and what the current tuning/sample 
        ## rate is
        junkFrame = usrp.read_frame(fh)
        srate = junkFrame.sample_rate
        t1i, t1f = junkFrame.time
        tunepols = usrp.get_frames_per_obs(fh)
        tunepol = tunepols[0] + tunepols[1] + tunepols[2] + tunepols[3]
        beampols = tunepol
        fh.seek(-usrp.FRAME_SIZE, 1)
        
        ## See how far off the current frame is from the target
        tDiff = t1i - (t0i + args.skip) + t1f - t0f
        
        ## Half that to come up with a new seek parameter
        tCorr = -tDiff / 2.0
        cOffset = int(tCorr * srate / junkFrame.payload.data.size * beampols)
        cOffset = int(1.0 * cOffset / beampols) * beampols
        offset += cOffset
        
        ## If the offset is zero, we are done.  Otherwise, apply the offset
        ## and check the location in the file again/
        if cOffset is 0:
            break
        fh.seek(cOffset*usrp.FRAME_SIZE, 1)
        
    # Update the offset actually used
    args.skip = t1i - t0i + t1f - t0f
    
    # Make sure that the file chunk size contains is an integer multiple
    # of the FFT length so that no data gets dropped.  This needs to
    # take into account the number of beampols in the data, the FFT length,
    # and the number of samples per frame.
    maxFrames = int(1.0*19144/beampols*junkFrame.payload.data.size/float(LFFT))*LFFT/junkFrame.payload.data.size*beampols
    
    # Number of frames to integrate over
    nFrames = int(args.average * srate / junkFrame.payload.data.size * beampols)
    nFrames = int(1.0 * nFrames / beampols*junkFrame.payload.data.size/float(LFFT))*LFFT/junkFrame.payload.data.size*beampols
    args.average = 1.0 * nFrames / beampols * junkFrame.payload.data.size / srate
    
    # Number of remaining chunks
    nChunks = int(math.ceil(1.0*(nFrames)/maxFrames))
    
    # Date & Central Frequnecy
    beginDate = junkFrame.time.datetime
    central_freq1 = 0.0
    junkFrame = usrp.read_frame(fh)
    central_freq1 = junkFrame.central_freq
    fh.seek(-usrp.FRAME_SIZE, 1)
    
    # File summary
    print("Filename: %s" % args.filename)
    print("Date of First Frame: %s" % str(beginDate))
    print("Beams: %i" % beams)
    print("Tune/Pols: %i %i %i %i" % tunepols)
    print("Sample Rate: %i Hz" % srate)
    print("Tuning Frequency: %.3f Hz" % central_freq1)
    print("Frames: %i (%.3f s)" % (nFramesFile, 1.0 * nFramesFile / beampols * junkFrame.payload.data.size / srate))
    print("---")
    print("Offset: %.3f s (%i frames)" % (args.skip, offset))
    print("Integration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (args.average, nFrames, nFrames / beampols))
    print("Chunks: %i" % nChunks)
    
    # Sanity check
    if offset > nFramesFile:
        raise RuntimeError("Requested offset is greater than file length")
    if nFrames > (nFramesFile - offset):
        raise RuntimeError("Requested integration time+offset is greater than file length")
        
    # Setup the window function to use
    if args.bartlett:
        window = numpy.bartlett
    elif args.blackman:
        window = numpy.blackman
    elif args.hanning:
        window = numpy.hanning
    else:
        window = fxc.null_window
        
    # Master loop over all of the file chunks
    standMapper = []
    masterWeight = numpy.zeros((nChunks, beampols, LFFT-1 if float(fxc.__version__) < 0.8 else LFFT))
    masterSpectra = numpy.zeros((nChunks, beampols, LFFT-1 if float(fxc.__version__) < 0.8 else LFFT))
    for i in range(nChunks):
        # Find out how many frames remain in the file.  If this number is larger
        # than the maximum of frames we can work with at a time (maxFrames),
        # only deal with that chunk
        framesRemaining = nFrames - i*maxFrames
        if framesRemaining > maxFrames:
            framesWork = maxFrames
        else:
            framesWork = framesRemaining
        print("Working on chunk %i, %i frames remaining" % (i, framesRemaining))
        
        count = {}
        data = numpy.zeros((beampols,framesWork*junkFrame.payload.data.size/beampols), dtype=numpy.csingle)
        # If there are fewer frames than we need to fill an FFT, skip this chunk
        if data.shape[1] < LFFT:
            break
            
        # Inner loop that actually reads the frames into the data array
        print("Working on %.1f ms of data" % ((framesWork*junkFrame.payload.data.size//beampols//srate)*1000.0))
        
        for j in range(framesWork):
            # Read in the next frame and anticipate any problems that could occur
            try:
                cFrame = usrp.read_frame(fh, Verbose=False)
            except:
                break
                
            beam, tune, pol = cFrame.id
            aStand = 2*(tune - 1) + pol
            if aStand not in standMapper:
                standMapper.append(aStand)
                oStand = 1*aStand
                aStand = standMapper.index(aStand)
                print("Mapping beam %i, tune. %1i, pol. %1i (%2i) to array index %3i" % (beam, tune, pol, oStand, aStand))
            else:
                aStand = standMapper.index(aStand)
                
            if aStand not in count.keys():
                count[aStand] = 0
                
            data[aStand, count[aStand]*junkFrame.payload.data.size:(count[aStand]+1)*junkFrame.payload.data.size] = cFrame.payload.data
            count[aStand] += 1
            
        # Correct the DC bias
        for j in range(data.shape[0]):
            data[j,:] -= data[j,:].mean()
            
        # Calculate the spectra for this block of data and then weight the results by 
        # the total number of frames read.  This is needed to keep the averages correct.
        freq, tempSpec = fxc.SpecMaster(data, LFFT=LFFT, window=window, verbose=(not args.quiet), sample_rate=srate, clip_level=0)
        for stand in count.keys():
            masterSpectra[i,stand,:] = tempSpec[stand,:]
            masterWeight[i,stand,:] = int(count[stand] * junkFrame.payload.data.size / LFFT)
            
        # We don't really need the data array anymore, so delete it
        del(data)
        
    # Now that we have read through all of the chunks, perform the final averaging by
    # dividing by all of the chunks
    spec = numpy.squeeze( (masterWeight*masterSpectra).sum(axis=0) / masterWeight.sum(axis=0) )
    spec.shape = (1,spec.size)
    
    # Frequencies
    freq1 = freq + central_freq1
    
    # The plots:  This is setup for the current configuration of 20 beampols
    fig = plt.figure()
    figsX = int(round(math.sqrt(beampols)))
    figsY = beampols // figsX
    # Put the frequencies in the best units possible
    freq1, units1 = bestFreqUnits(freq1)
    
    sortedMapper = sorted(standMapper)
    for k, aStand in enumerate(sortedMapper):
        i = standMapper.index(aStand)
        if standMapper[i]%4//2+1 == 1:
            freq = freq1
            units = units1
        else:
            freq = freq1
            units = units1
            
        ax = fig.add_subplot(figsX,figsY,k+1)
        print(spec.shape)
        currSpectra = numpy.squeeze( numpy.log10(spec[i,:])*10.0 )
        ax.plot(freq, currSpectra, label='%i (avg)' % (i+1))
        
        # If there is more than one chunk, plot the difference between the global 
        # average and each chunk
        if nChunks > 1 and not args.disable_chunks:
            for j in range(nChunks):
                # Some files are padded by zeros at the end and, thus, carry no 
                # weight in the average spectra.  Skip over those.
                if masterWeight[j,i,:].sum() == 0:
                    continue
                    
                # Calculate the difference between the spectra and plot
                subspectra = numpy.squeeze( numpy.log10(masterSpectra[j,i,:])*10.0 )
                diff = subspectra - currSpectra
                ax.plot(freq, diff, label='%i' % j)
                
        ax.set_title('Beam %i, Tune. %i, Pol. %i' % (standMapper[i]//4+1, standMapper[i]%4//2+1, standMapper[i]%2))
        ax.set_xlabel('Frequency [%s]' % units)
        ax.set_ylabel('P.S.D. [dB/RBW]')
        ax.set_xlim([freq.min(), freq.max()])
        ax.legend(loc=0)
        
        print("For beam %i, tune. %i, pol. %i maximum in PSD at %.3f %s" % (standMapper[i]//4+1, standMapper[i]%4//2+1, standMapper[i]%2, freq[numpy.where( spec[i,:] == spec[i,:].max() )][0], units))
        
    print("RBW: %.4f %s" % ((freq[1]-freq[0]), units))
    plt.subplots_adjust(hspace=0.35, wspace=0.30)
    plt.show()
    
    # Save spectra image if requested
    if args.output is not None:
        fig.savefig(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='read in a USRP file and create a collection of time-averaged spectra', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('filename', type=str, 
                        help='filename to process')
    wgroup = parser.add_mutually_exclusive_group(required=False)
    wgroup.add_argument('-t', '--bartlett', action='store_true', 
                        help='apply a Bartlett window to the data')
    wgroup.add_argument('-b', '--blackman', action='store_true', 
                        help='apply a Blackman window to the data')
    wgroup.add_argument('-n', '--hanning', action='store_true', 
                        help='apply a Hanning window to the data')
    parser.add_argument('-s', '--skip', type=aph.positive_or_zero_float, default=0.0, 
                        help='skip the specified number of seconds at the beginning of the file')
    parser.add_argument('-a', '--average', type=aph.positive_float, default=10.0, 
                        help='number of seconds of data to average for spectra')
    parser.add_argument('-q', '--quiet', dest='verbose', action='store_false',
                        help='run %(prog)s in silent mode')
    parser.add_argument('-l', '--fft-length', type=aph.positive_int, default=4096, 
                        help='set FFT length')
    parser.add_argument('-d', '--disable-chunks', action='store_true', 
                        help='disable plotting chunks in addition to the global average')
    parser.add_argument('-o', '--output', type=str, 
                        help='output file name for spectra image')
    args = parser.parse_args()
    main(args)
    
