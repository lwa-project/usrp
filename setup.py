# -*- coding: utf-8 -*-

import ez_setup
ez_setup.use_setuptools()

import glob

from setuptools import setup, Extension, Distribution, find_packages

try:
    import numpy
except ImportError:
    pass

setup(
    name                 = "USRP",
    version              = "0.2.3",
    description          = "Python reader for GNURadio/USRP data",
    url                  = "https://fornax.phys.unm.edu/lwa/trac/", 
    author               = "Jayce Dowell",
    author_email         = "jdowell@unm.edu",
    license              = 'GPL',
    classifiers          = ['Development Status :: 5 - Production/Stable',
                            'Intended Audience :: Developers',
                            'Intended Audience :: Science/Research',
                            'License :: OSI Approved :: GNU General Public License (GPL)',
                            'Topic :: Scientific/Engineering :: Astronomy',
                            'Programming Language :: Python :: 2',
                            'Programming Language :: Python :: 2.6',
                            'Programming Language :: Python :: 2.7',
                            'Operating System :: MacOS :: MacOS X',
                            'Operating System :: POSIX :: Linux'],
    packages             = find_packages(exclude="tests"), 
    namespace_packages   = ['lsl_toolkits',],
    scripts              = glob.glob('scripts/*.py'), 
    include_package_data = True,  
    zip_safe             = False,  
    test_suite           = "tests.test_usrp.usrp_tests"
)
