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
import h5py
import math
import numpy
import ephem
import getopt
from datetime import datetime

import lsl_toolkits.USRP as usrp
import lsl.statistics.robust as robust
import lsl.correlator.fx as fxc
from lsl.reader import errors
from lsl.astro import unix_to_utcjd, DJD_OFFSET
from lsl.common import progress, stations
from lsl.common import mcs, metabundle

import matplotlib.pyplot as plt


def usage(exitCode=None):
    print("""usrpHDFWaterfall.py -Read in USRP files and create a collection of 
time-averaged spectra.  These spectra are saved to a HDF5 file called <filename>-waterfall.hdf5.

Usage: usrpHDFWaterfall.py [OPTIONS] file

Options:
-h, --help                  Display this help information
-t, --bartlett              Apply a Bartlett window to the data
-b, --blackman              Apply a Blackman window to the data
-n, --hanning               Apply a Hanning window to the data
-s, --skip                  Skip the specified number of seconds at the beginning
                            of the file (default = 0)
-a, --average               Number of seconds of data to average for spectra 
                            (default = 1)
-d, --duration              Number of seconds to calculate the waterfall for 
                            (default = 0; run the entire file)
-q, --quiet                 Run drxSpectra in silent mode and do not show the plots
-l, --fft-length            Set FFT length (default = 4096)
-c, --clip-level            FFT blanking clipping level in counts (default = 0, 
                            0 disables)
-e, --estimate-clip         Use robust statistics to estimate an appropriate clip 
                            level (overrides the `-c` option)
-w, --without-sats          Do not generate saturation counts
""")

    if exitCode is not None:
        sys.exit(exitCode)
    else:
        return True


def parseOptions(args):
    config = {}
    # Command line flags - default values
    config['offset'] = 0.0
    config['average'] = 1.0
    config['LFFT'] = 4096
    config['freq1'] = 0
    config['freq2'] = 0
    config['maxFrames'] = 28000
    config['window'] = fxc.noWindow
    config['duration'] = 0.0
    config['verbose'] = True
    config['clip'] = 0
    config['estimate'] = False
    config['countSats'] = True
    config['args'] = []

    # Read in and process the command line flags
    try:
        opts, args = getopt.getopt(args, "hqtbnl:s:a:d:c:ew", ["help", "quiet", "bartlett", "blackman", "hanning", "fft-length=", "skip=", "average=", "duration=", "freq1=", "freq2=", "clip-level=", "estimate-clip", "without-sats"])
    except getopt.GetoptError, err:
        # Print help information and exit:
        print(str(err)) # will print something like "option -a not recognized"
        usage(exitCode=2)
    
    # Work through opts
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage(exitCode=0)
        elif opt in ('-q', '--quiet'):
            config['verbose'] = False
        elif opt in ('-t', '--bartlett'):
            config['window'] = numpy.bartlett
        elif opt in ('-b', '--blackman'):
            config['window'] = numpy.blackman
        elif opt in ('-n', '--hanning'):
            config['window'] = numpy.hanning
        elif opt in ('-l', '--fft-length'):
            config['LFFT'] = int(value)
        elif opt in ('-s', '--skip'):
            config['offset'] = float(value)
        elif opt in ('-a', '--average'):
            config['average'] = float(value)
        elif opt in ('-d', '--duration'):
            config['duration'] = float(value)
        elif opt in ('-c', '--clip-level'):
            config['clip'] = int(value)
        elif opt in ('-e', '--estimate-clip'):
            config['estimate'] = True
        elif opt in ('-w', '--without-sats'):
            config['countSats'] = False
        else:
            assert False
    
    # Add in arguments
    config['args'] = args

    # Return configuration
    return config


def createNewFile(filename):
    """
    Create a new HDF5 and return the handle for it.  This sets up all of 
    the required attributes and groups and fills them with dummy values.
    
    Returns an open h5py.File instance.
    """
    
    # Create the file
    f = h5py.File(filename, 'w')
    
    # Observer and Project Info.
    f.attrs['ObserverID'] = 0
    f.attrs['ObserverName'] = ''
    f.attrs['ProjectID'] = ''
    f.attrs['SessionsID'] = 0
    
    # File creation time
    f.attrs['FileCreation'] = datetime.utcnow().strftime("UTC %Y/%m/%d %H:%M:%S")
    f.attrs['FileGenerator'] = ''
    
    # Input file info.
    f.attrs['InputData'] = ''
    f.attrs['InputMetadata'] = ''
    
    return f


def fillMinimum(f, obsID, beam, srate, srateUnits='samples/s'):
    """
    Minimum metadata filling for a particular observation.
    """
    
    # Get the group or create it if it doesn't exist
    obs = f.get('/Observation%i' % obsID, None)
    if obs is None:
        obs = f.create_group('/Observation%i' % obsID)
        
    # Target info.
    obs.attrs['TargetName'] = ''
    obs.attrs['RA'] = -99.0
    obs.attrs['RA_Units'] = 'hours'
    obs.attrs['Dec'] = -99.0
    obs.attrs['Dec_Units'] = 'degrees'
    obs.attrs['Epoch'] = 2000.0
    obs.attrs['TrackingMode'] = 'Unknown'
    
    # Observation info
    obs.attrs['Beam'] = beam
    obs.attrs['DRX_Gain'] = -1.0
    obs.attrs['sampleRate'] = srate
    obs.attrs['sampleRate_Units'] = srateUnits
    obs.attrs['tInt'] = -1.0
    obs.attrs['tInt_Units'] = 's'
    obs.attrs['LFFT'] = -1
    obs.attrs['nChan'] = -1
    obs.attrs['RBW'] = -1.0
    obs.attrs['RBW_Units'] = 'Hz'
    
    return True


def getObservationSet(f, observation):
    """
    Return a reference to the specified observation.
    """
    
    # Get the observation
    obs = f.get('/Observation%i' % observation, None)
    if obs is None:
        raise RuntimeError('No such observation: %i' % observation)

    return obs


def createDataSets(f, observation, tuning, frequency, chunks, dataProducts=['XX',]):
    """
    Fill in a tuning group with the right set of dummy data sets and 
    attributes.
    """

    # Get the observation
    obs = f.get('/Observation%i' % observation, None)
    if obs is None:
        obs = f.create_group('/Observation%i' % observation)

        # Target info.
        obs.attrs['TargetName'] = ''
        obs.attrs['RA'] = -99.0
        obs.attrs['RA_Units'] = 'hours'
        obs.attrs['Dec'] = -99.0
        obs.attrs['Dec_Units'] = 'degrees'
        obs.attrs['Epoch'] = 2000.0
        obs.attrs['TrackingMode'] = 'Unknown'
    
        # Observation info
        obs.attrs['Beam'] = -1.0
        obs.attrs['DRX_Gain'] = -1.0
        obs.attrs['sampleRate'] = -1.0
        obs.attrs['sampleRate_Units'] = 'samples/s'
        obs.attrs['tInt'] = -1.0
        obs.attrs['tInt_Units'] = 's'
        obs.attrs['LFFT'] = -1
        obs.attrs['nChan'] = -1
        obs.attrs['RBW'] = -1.0
        obs.attrs['RBW_Units'] = 'Hz'
    
    # Get the group or create it if it doesn't exist
    grp = obs.get('Tuning%i' % tuning, None)
    if grp is None:
        grp = obs.create_group('Tuning%i' % tuning)
        
    grp['freq'] = frequency
    grp['freq'].attrs['Units'] = 'Hz'
    for p in dataProducts:
        d = grp.create_dataset(p, (chunks, frequency.size), 'f4')
        d.attrs['axis0'] = 'time'
        d.attrs['axis1'] = 'frequency'
    d = grp.create_dataset('Saturation', (chunks, 2), 'i8')
    d.attrs['axis0'] = 'time'
    d.attrs['axis1'] = 'polarization'
        
    return True


def getDataSet(f, observation, tuning, dataProduct):
    """
    Return a reference to the specified data set.
    """
    
    # Get the observation
    obs = f.get('/Observation%i' % observation, None)
    if obs is None:
        raise RuntimeError('No such observation: %i' % observation)
        
    # Get the groups
    grp = obs.get('Tuning%i' % tuning, None)
    if grp is None:
        raise RuntimeError("Unknown tuning: %i" % tuning)
        
    # Get the data set
    try:
        d = grp[dataProduct]
    except:
        raise RuntimeError("Unknown data product for Observation %i, Tuning %i: %s" % (observation, tuning, dataProduct))
        
    return d


def estimateClipLevel(fh, beampols):
    """
    Read in a set of 100 frames and come up with the 4-sigma clip levels 
    for each tuning.  These clip levels are returned as a two-element 
    tuple.
    """
    
    filePos = fh.tell()
    junkFrame = usrp.readFrame(fh, Verbose=False)
    fh.seek(filePos)
    
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
    
    return clip1, clip2


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


def processDataBatchLinear(fh, dataProducts, tStart, duration, sampleRate, config, dataSets, obsID=1, clip1=0, clip2=0):
    """
    Process a chunk of data in a raw DRX file into linear polarization 
    products and add the contents to an HDF5 file.
    """
    
    # Length of the FFT
    LFFT = config['LFFT']
    
    # Find the start of the observation
    junkFrame = usrp.readFrame(fh)
    srate = junkFrame.getSampleRate()
    t0 = junkFrame.getTime()
    fh.seek(-usrp.FrameSize, 1)
    
    print('Looking for #%i at %s with sample rate %.1f Hz...' % (obsID, tStart, sampleRate))
    while datetime.utcfromtimestamp(t0) < tStart or srate != sampleRate:
        junkFrame = usrp.readFrame(fh)
        srate = junkFrame.getSampleRate()
        t0 = junkFrame.getTime()
    print('... Found #%i at %s with sample rate %.1f Hz' % (obsID, datetime.utcfromtimestamp(t0), srate))
    tDiff = datetime.utcfromtimestamp(t0) - tStart
    try:
        duration = duration - tDiff.total_seconds()
    except:
        duration = duration - (tDiff.seconds + tDiff.microseconds/1e6)
        
    beam,tune,pol = junkFrame.parseID()
    beams = usrp.getBeamCount(fh)
    tunepols = usrp.getFramesPerObs(fh)
    tunepol = tunepols[0] + tunepols[1] + tunepols[2] + tunepols[3]
    beampols = tunepol
    
    # Make sure that the file chunk size contains is an integer multiple
    # of the FFT length so that no data gets dropped.  This needs to
    # take into account the number of beampols in the data, the FFT length,
    # and the number of samples per frame.
    maxFrames = int(1.0*config['maxFrames']/beampols*junkFrame.data.iq.size/float(LFFT))*LFFT/junkFrame.data.iq.size*beampols
    
    # Number of frames to integrate over
    nFramesAvg = int(config['average'] * srate / junkFrame.data.iq.size * beampols)
    nFramesAvg = int(1.0 * nFramesAvg / beampols*junkFrame.data.iq.size/float(LFFT))*LFFT/junkFrame.data.iq.size*beampols
    config['average'] = 1.0 * nFramesAvg / beampols * junkFrame.data.iq.size / srate
    maxFrames = nFramesAvg
    
    # Number of remaining chunks (and the correction to the number of
    # frames to read in).
    nChunks = int(round(duration / config['average']))
    if nChunks == 0:
        nChunks = 1
    nFrames = nFramesAvg*nChunks
    
    # Date & Central Frequency
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
    freq = numpy.fft.fftshift(numpy.fft.fftfreq(LFFT, d=1/srate))
    if float(fxc.__version__) < 0.8:
        freq = freq[1:]
        
    dataSets['obs%i-freq1' % obsID][:] = freq + centralFreq1
    dataSets['obs%i-freq2' % obsID][:] = freq + centralFreq2
    
    obs = dataSets['obs%i' % obsID]
    obs.attrs['tInt'] = config['average']
    obs.attrs['tInt_Unit'] = 's'
    obs.attrs['LFFT'] = LFFT
    obs.attrs['nChan'] = LFFT-1 if float(fxc.__version__) < 0.8 else LFFT
    obs.attrs['RBW'] = freq[1]-freq[0]
    obs.attrs['RBW_Units'] = 'Hz'
    
    done = False
    for i in xrange(nChunks):
        # Find out how many frames remain in the file.  If this number is larger
        # than the maximum of frames we can work with at a time (maxFrames),
        # only deal with that chunk
        framesRemaining = nFrames - i*maxFrames
        if framesRemaining > maxFrames:
            framesWork = maxFrames
        else:
            framesWork = framesRemaining
        print("Working on chunk %i, %i frames remaining" % (i+1, framesRemaining))
        
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
            print(pol)
            if j is 0:
                cTime = cFrame.getTime()
                
            data[aStand, count[aStand]*cFrame.data.iq.size:(count[aStand]+1)*cFrame.data.iq.size] = cFrame.data.iq
            count[aStand] +=  1
            
        # Correct the DC bias
        for j in xrange(data.shape[0]):
            data[j,:] -= data[j,:].mean()
            
        # Save out some easy stuff
        dataSets['obs%i-time' % obsID][i] = cTime
        
        if config['countSats']:
            sats = ((data.real**2 + data.imag**2) >= 32768**2).sum(axis=1)
            dataSets['obs%i-Saturation1' % obsID][i,:] = sats[0:2]
            dataSets['obs%i-Saturation2' % obsID][i,:] = sats[2:4]
        else:
            dataSets['obs%i-Saturation1' % obsID][i,:] = -1
            dataSets['obs%i-Saturation2' % obsID][i,:] = -1
            
        # Calculate the spectra for this block of data and then weight the results by 
        # the total number of frames read.  This is needed to keep the averages correct.
        if clip1 == clip2:
            freq, tempSpec1 = fxc.SpecMaster(data, LFFT=LFFT, window=config['window'], verbose=config['verbose'], SampleRate=srate, ClipLevel=clip1)
            
            l = 0
            for t in (1,2):
                for p in dataProducts:
                    dataSets['obs%i-%s%i' % (obsID, p, t)][i,:] = tempSpec1[l,:]
                    l += 1
                    
        else:
            freq, tempSpec1 = fxc.SpecMaster(data[:2,:], LFFT=LFFT, window=config['window'], verbose=config['verbose'], SampleRate=srate, ClipLevel=clip1)
            freq, tempSpec2 = fxc.SpecMaster(data[2:,:], LFFT=LFFT, window=config['window'], verbose=config['verbose'], SampleRate=srate, ClipLevel=clip2)
            
            for l,p in enumerate(dataProducts):
                dataSets['obs%i-%s%i' % (obsID, p, 1)][i,:] = tempSpec1[l,:]
                dataSets['obs%i-%s%i' % (obsID, p, 2)][i,:] = tempSpec2[l,:]
                
        # We don't really need the data array anymore, so delete it
        del(data)
        
        # Are we done yet?
        if done:
            break
            
    return True


def main(args):
    # Parse command line options
    config = parseOptions(args)

    # Length of the FFT
    LFFT = config['LFFT']
    
    filename = config['args'][0]
    fh = open(filename, "rb")
    usrp.FrameSize = usrp.getFrameSize(fh)
    nFramesFile = os.path.getsize(filename) // usrp.FrameSize
    
    junkFrame = usrp.readFrame(fh)
    srate = junkFrame.getSampleRate()
    t0 = junkFrame.getTime()
    fh.seek(-usrp.FrameSize, 1)
    
    beam,tune,pol = junkFrame.parseID()
    beams = usrp.getBeamCount(fh)
    tunepols = usrp.getFramesPerObs(fh)
    tunepol = tunepols[0] + tunepols[1] + tunepols[2] + tunepols[3]
    beampols = tunepol

    # Offset in frames for beampols beam/tuning/pol. sets
    offset = int(config['offset'] * srate / junkFrame.data.iq.size * beampols)
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
        tDiff = t1 - (t0 + config['offset'])
        
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
    config['offset'] = t1 - t0
    offset = int(round(config['offset'] * srate / junkFrame.data.iq.size * beampols))
    offset = int(1.0 * offset / beampols) * beampols

    # Make sure that the file chunk size contains is an integer multiple
    # of the FFT length so that no data gets dropped.  This needs to
    # take into account the number of beampols in the data, the FFT length,
    # and the number of samples per frame.
    maxFrames = int(1.0*config['maxFrames']/beampols*junkFrame.data.iq.size/float(LFFT))*LFFT/junkFrame.data.iq.size*beampols

    # Number of frames to integrate over
    nFramesAvg = int(config['average'] * srate / junkFrame.data.iq.size * beampols)
    nFramesAvg = int(1.0 * nFramesAvg / beampols*junkFrame.data.iq.size/float(LFFT))*LFFT/junkFrame.data.iq.size*beampols
    config['average'] = 1.0 * nFramesAvg / beampols * junkFrame.data.iq.size / srate
    maxFrames = nFramesAvg

    # Number of remaining chunks (and the correction to the number of
    # frames to read in).
    if config['duration'] == 0:
        config['duration'] = 1.0 * nFramesFile / beampols * junkFrame.data.iq.size / srate
    nChunks = int(round(config['duration'] / config['average']))
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
            try:
                centralFreq1 = junkFrame.getCentralFreq()
            except AttributeError:
                from lsl.common.dp import fS
                centralFreq1 = fS * ((junkFrame.data.flags>>32) & (2**32-1)) / 2**32
        elif p == 0 and t == 2:
            try:
                centralFreq2 = junkFrame.getCentralFreq()
            except AttributeError:
                from lsl.common.dp import fS
                centralFreq2 = fS * ((junkFrame.data.flags>>32) & (2**32-1)) / 2**32
        else:
            pass
    fh.seek(-4*usrp.FrameSize, 1)
    
    config['freq1'] = centralFreq1
    config['freq2'] = centralFreq2
    
    # File summary
    print("Filename: %s" % filename)
    print("Date of First Frame: %s" % str(beginDate))
    print("Beams: %i" % beams)
    print("Tune/Pols: %i %i %i %i" % tunepols)
    print("Sample Rate: %i Hz" % srate)
    print("Tuning Frequency: %.3f Hz (1); %.3f Hz (2)" % (centralFreq1, centralFreq2))
    print("Frames: %i (%.3f s)" % (nFramesFile, 1.0 * nFramesFile / beampols * junkFrame.data.iq.size / srate))
    print("---")
    print("Offset: %.3f s (%i frames)" % (config['offset'], offset))
    print("Integration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (config['average'], nFramesAvg, nFramesAvg / beampols))
    print("Duration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (config['average']*nChunks, nFrames, nFrames / beampols))
    print("Chunks: %i" % nChunks)
    
    # Estimate clip level (if needed)
    if config['estimate']:
        clip1, clip2 = estimateClipLevel(fh, beampols)
    else:
        clip1 = config['clip']
        clip2 = config['clip']
        
    # Setup the output file
    outname = os.path.split(filename)[1]
    outname = os.path.splitext(outname)[0]
    outname = '%s-waterfall.hdf5' % outname
    f = createNewFile(outname)
    
    # Look at the metadata and come up with a list of observations.  If 
    # there are no metadata, create a single "observation" that covers the
    # whole file.
    obsList = {}
    obsList[1] = (datetime.utcfromtimestamp(t1), datetime(2222,12,31,23,59,59), config['duration'], srate)
    fillMinimum(f, 1, beam, srate)
    
    if filename.find('pol0') != -1:
        dataProducts = ['XX',]
    elif filename.find('pol1') != -1:
        dataProducts = ['YY',]
    else:
        dataProducts = ['XX',]
        
    for o in sorted(obsList.keys()):
        for t in (1,2):
            createDataSets(f, o, t, numpy.arange(LFFT-1 if float(fxc.__version__) < 0.8 else LFFT, dtype=numpy.float32), int(round(obsList[o][2]/config['average'])), dataProducts)
            
    f.attrs['FileGenerator'] = 'usrpHDFWaterfall.py'
    f.attrs['InputData'] = os.path.basename(filename)
    
    # Create the various HDF group holders
    ds = {}
    for o in sorted(obsList.keys()):
        obs = getObservationSet(f, o)
        
        ds['obs%i' % o] = obs
        ds['obs%i-time' % o] = obs.create_dataset('time', (int(round(obsList[o][2]/config['average'])),), 'f8')
        
        for t in (1,2):
            ds['obs%i-freq%i' % (o, t)] = getDataSet(f, o, t, 'freq')
            for p in dataProducts:
                ds["obs%i-%s%i" % (o, p, t)] = getDataSet(f, o, t, p)
            ds['obs%i-Saturation%i' % (o, t)] = getDataSet(f, o, t, 'Saturation')
            
    # Load in the correct analysis function
    processDataBatch = processDataBatchLinear
    
    # Go!
    for o in sorted(obsList.keys()):
        try:
            processDataBatch(fh, dataProducts,  obsList[o][0], obsList[o][2], obsList[o][3], config, ds, obsID=o, clip1=clip1, clip2=clip2)
        except RuntimeError, e:
            print("Observation #%i: %s, abandoning this observation" % (o, str(e)))

    # Save the output to a HDF5 file
    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
