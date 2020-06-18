#!/usr/bin/env python

"""
Script to combine two polarizations contained in two HDF5 files 
created by usrpHDFWaterfall.py into a single HDF5 file.
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import

import os
import sys
import h5py
import argparse


def _fillHDF(input1, input2, output):
    """
    Function to recursively copy the structure of a HDF5 file created by 
    usrpHDFWaterfall.py.
    """
    
    # Copy the attributes
    for key in input1.attrs:
        output.attrs[key] = input1.attrs[key]
        
    # Loop over the entities in the first input file for copying
    for entity in list(input1):
        ## Get both entities - This is needed for combinging things like the saturation
        ## counts
        entity1 = input1.get(entity, None)
        entity2 = input2.get(entity, None)
        
        ## Is it a group?
        if type(entity1).__name__ == 'Group':
            ### If so, add it and fill it in.
            if entity not in list(output):
                entityC = output.create_group(entity)
            _fillHDF(entity1, entity2, entityC)
            continue
            
        ## Is it a dataset?
        if type(entity1).__name__ == 'Dataset':
            ### If so, add it and fill it in.
            if entity == 'Saturation':
                entityC = output.create_dataset(entity, entity1.shape, entity1.dtype.descr[0][1])
                entityC[:,0] = entity1[:,0]
                entityC[:,1] = entity2[:,0]
                
            else:
                entityC = output.create_dataset(entity, entity1.shape, entity1.dtype.descr[0][1])
                entityC[:] = entity1[:]
            for key in entity1.attrs:
                entityC.attrs[key] = entity1.attrs[key]
                
    # Loop over the entities in the second input file for copying
    for entity in list(input2):
        ## Get both the entity
        entity2 = input2.get(entity, None)
        
        ## Is it a dataset?
        if type(entity2).__name__ == 'Dataset':
            ### If so, add it and fill it in.
            if entity not in list(output):
                entityC = output.create_dataset(entity, entity2.shape, entity2.dtype.descr[0][1])
                entityC[:] = entity2[:]
                for key in entity2.attrs:
                    entityC.attrs[key] = entity2.attrs[key]
                    
    return True


def main(args):
    # Read in the data
    h1 = h5py.File(args.pol0)
    h2 = h5py.File(args.pol1)
    
    # Verify that the files are from the data data collection/processing batch
    i = 1
    while True:
        obs1 = h1.get('/Observation%i' % i, None)
        obs2 = h2.get('/Observation%i' % i, None)
        
        # Are we still aligned?
        if obs1 is None and obs2 is None:
            break
        elif obs1 is None and obs2 is not None:
            raise RuntimeError("Files contain a different number of observations")
        elif obs1 is not None and obs2 is None:
            raise RuntimeError("Files contain a different number of observations")
            
        # Do the attributes match?
        for key in ['Beam', 'sample_rate', 'tInt']:
            try:
                assert(obs1.attrs[key] == obs2.attrs[key])
            except AssertionError:
                raise RuntimeError("Attribute '%s' does not match between the two files, obs #%i." % (key, i))
                
        # Do the times match?
        for key in ['time',]:
            q1 = obs1[key]
            q2 = obs2[key]
            try:
                assert(q1.shape == q2.shape)
                assert(q1.dtype == q2.dtype)
                assert(((q1[:] - q2[:])**2).sum() == 0)
            except AssertionError:
                raise RuntimeError("Metadata '%s' does not match between the two files, obs #%i." % (key, i))
                
        # Do the frequencies match?
        for j in [1, 2]:
            tuning1 = obs1.get('Tuning%i' % j, None)
            tuning2 = obs2.get('Tuning%i' % j, None)
            
            if tuning1 is None and tuning2 is None:
                break
            elif tuning1 is None and tuning2 is not None:
                raise RuntimeError("Files contain a different number of tunings for obs #%i", i)
            elif tuning1 is not None and tuning2 is None:
                raise RuntimeError("Files contain a different number of tunings for obs #%i", i)
                
            for key in ['freq',]:
                q1 = tuning1[key]
                q2 = tuning2[key]
                try:
                    assert(q1.shape == q2.shape)
                    assert(q1.dtype == q2.dtype)
                    assert(((q1[:] - q2[:])**2).sum() == 0)
                except AssertionError:
                    raise RuntimeError("Metadata '%s' does not match between the two files, obs #%i, tuning %i" % (key, i, j))
                    
        i += 1
        
    # Output
    filename = args.pol0.replace('_pol0', '_comb').replace('_pol1', '_comb')
    filename = os.path.basename(filename)
    hC = h5py.File(filename, mode='w')
    _fillHDF(h1, h2, hC)
    
    # Update various values
    hC.attrs['FileGenerator'] = 'usrpHDFCombinePols.py'
    hC.attrs['InputData'] = '%s,%s' % (h1.attrs['InputData'], h2.attrs['InputData'])
    
    # Close
    h1.close()
    h2.close()
    hC.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='combine two USRP polarization HDF5 files into a single HDF5 file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('pol0', type=str,
                        help='first polarization')
    parser.add_argument('pol1', type=str,
                        help='second polarization')
    args = parser.parse_args()
    main(args)
    
