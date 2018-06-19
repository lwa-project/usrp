#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to combine two polarizations contained in two NPZ files 
created by usrpWaterfall.py into a single NPZ file.

$Rev$
$LastChangedBy$
$LastChangedDate$
"""

import os
import sys
import numpy


def main(args):
    # Sort and read in the data
    filename1 = args[0]
    filename2 = args[1]
    
    dd1 = numpy.load(filename1)
    dd2 = numpy.load(filename2)
    
    # Verify that the files are from the data data collection/processing batch
    for attr in ['tInt', 'srate', 'freq1', 'times']:
        try:
            try:
                ## For simple things, a direct comparison
                assert(dd1[attr] == dd2[attr])
            except ValueError:
                ## For arrays, check the shape and all values
                assert(dd1[attr].shape == dd2[attr].shape)
                assert((numpy.abs(dd1[attr]-dd2[attr])**2).sum() == 0)
                
        except AssertionError:
            raise RuntimeError("Attribute '%s' does not match between the two files." % attr)
            
    # Output
    srate = dd1['srate']
    tInt = dd1['tInt']
    freq = dd1['freq']
    freq1 = dd1['freq1']
    freq2 = dd1['freq2']
    times = dd1['times']
    standMapper = dd1['standMapper']
    
    spec = numpy.zeros_like(dd1['spec'])
    if filename1.find('_pol0'):
        spec[:,0,:] = dd1['spec'][:,0,:]
        spec[:,1,:] = dd2['spec'][:,0,:]
    else:
        spec[:,0,:] = dd2['spec'][:,0,:]
        spec[:,1,:] = dd1['spec'][:,0,:]
        
    filename = filename1.replace('_pol0', '_comb').replace('_pol1', '_comb')
    filename = os.path.basename(filename)
    numpy.savez(filename, srate=srate, tInt=tInt, freq=freq, freq1=freq1, freq2=freq2, times=times, spec=spec, standMapper=standMapper)
    
    # Close
    dd1.close()
    dd2.close()


if __name__ == "__main__":
    main(sys.argv[1:])

