LSL Toolkit for GNURadio/USRP Data
==================================

[![Build and Test](https://github.com/lwa-project/usrp/actions/workflows/main.yml/badge.svg)](https://github.com/lwa-project/usrp/actions/workflows/main.yml)  [![codecov](https://codecov.io/gh/lwa-project/usrp/branch/master/graph/badge.svg?token=2ABAOPIS0G)](https://codecov.io/gh/lwa-project/usrp)


DESCRIPTION
-----------
This package provides a reader for USRP data generated through GNURadio.

REQUIREMENTS
------------
  * python >= 2.7
  * numpy >= 1.2
  * h5py

INSTALLING
----------
The USRP package is installed as a regular Python package using distutils.  
Unzip and untar the source distribution. Setup the python interpreter you 
wish to use for running the package applications and switch to the root of 
the source distribution tree.

Install USRP by running:
    
    pip install [--root=<prefix>|--user] .

If the '--root' option is not provided, then the installation 
tree root directory is the same as for the python interpreter used 
to run `setup.py`.  For instance, if the python interpreter is in
'/usr/local/bin/python', then <prefix> will be set to '/usr/local'.
Otherwise, the explicit <prefix> value is taken from the command line
option.  The package will install files in the following locations:
  * <prefix>/bin
  * <prefix>/lib/python2.6/site-packages
  * <prefix>/share/doc
  * <prefix>/share/install
If an alternate <prefix> value is provided, you should set the PATH
environment to include directory '<prefix>/bin' and the PYTHONPATH
environment to include directory '<prefix>/bin' and the PYTHONPATH
environment to include directory '<prefix>/lib/python2.6/site-packages'.

If the '--user' option is provided, then then installation tree root 
directory will be in the current user's home directory.

UNIT TESTS
----------
Unit tests for the package may be found in the 'USRP/tests' sub-directory in
the package source distribution tree.  To run the complete suite of package unit 
tests:

    cd tests
    python -m unittest discover

DOCUMENTATION
-------------
See the module's internal documentation.

RELEASE NOTES
-------------
See the CHANGELOG for a detailed list of changes and notes.
