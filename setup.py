# -*- coding: utf-8 -*-

import unittest
try:
	import numpy
except ImportError:
	pass
from distutils.core  import setup, Command

packages          = ['lsl_toolkits','lsl_toolkits.USRP']
package_dirs       = {'':''}

class TestSuite(Command):
	user_options = []
	def initialize_options(self):
		pass
	def finalize_options(self):
		pass
	def run(self):
		from tests import test_usrp
		t = unittest.TestSuite()
		r = unittest.TextTestRunner(verbosity=2)
		loader = unittest.TestLoader()
		t.addTests(loader.loadTestsFromModule(test_usrp))
		r.run(t)

cmdClasses = {'test': TestSuite}
setup(
  cmdclass          = cmdClasses,
  name              = "USRP",
  version           = "0.1.0",
  description       = "Python reader for GNURadio/USRP data",
  url               = "http://fornax.phys.unm.edu/lwa/trac/", 
  author            = "Jayce Dowell",
  author_email      = "jdowell@unm.edu",
  license           = 'GPL',
  classifiers       = ['Development Status :: 4 - Beta',
				   'Intended Audience :: Science/Research',
				   'License :: OSI Approved :: GNU General Public License (GPL)',
				   'Topic :: Scientific/Engineering :: Astronomy'],
  packages          = packages,
  package_dir       = package_dirs,
  scripts           = ['scripts/usrpCheckTimetags.py', 'scripts/usrpFileCheck.py', 'scripts/usrpSpectra.py', 'scripts/usrpTimeseries.py', 'scripts/usrpWaterfall.py']
)
