#!/usr/bin/env python

"""
Simple script to combine two polarizations contained in two NPZ files 
created by usrpWaterfall.py into a single NPZ file.
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import
import sys
if sys.version_info < (3,):
    range = xrange
    
import os
import sys
import numpy
import argparse


def main(args):
    # Read in the data
    dd1 = numpy.load(args.pol0)
    dd2 = numpy.load(args.pol1)
    
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
    if args.pol0.find('_pol0'):
        spec[:,0,:] = dd1['spec'][:,0,:]
        spec[:,1,:] = dd2['spec'][:,0,:]
    else:
        spec[:,0,:] = dd2['spec'][:,0,:]
        spec[:,1,:] = dd1['spec'][:,0,:]
        
    filename = args.pol0.replace('_pol0', '_comb').replace('_pol1', '_comb')
    filename = os.path.basename(filename)
    numpy.savez(filename, srate=srate, tInt=tInt, freq=freq, freq1=freq1, freq2=freq2, times=times, spec=spec, standMapper=standMapper)
    
    # Close
    dd1.close()
    dd2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='combine two USRP polarization .npz files into a single .npz file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('pol0', type=str,
                        help='first polarization')
    parser.add_argument('pol1', type=str,
                        help='second polarization')
    args = parser.parse_args()
    main(args)
    
