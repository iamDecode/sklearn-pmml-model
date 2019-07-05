from __future__ import division, print_function, absolute_import

# Choose build type.
build_type="optimized" # "debug"

# Long description for package homepage on PyPI
with open("README.md", "r") as fh:
  long_description = fh.read()

# Directories (relative to the top-level directory where setup.py resides) in which to look for data files.
datadirs = ("tests", "sklearn_pmml_model", "sklearn_pmml_model/tree")

# File extensions to be considered as data files. (Literal, no wildcards.)
dataexts = (".py",  ".pyx", ".pxd",  ".c",)

# Standard documentation to detect (and package if it exists).
standard_docs     = ["README", "LICENSE", "TODO", "CHANGELOG", "AUTHORS"]  # just the basename without file extension
standard_doc_exts = [".md", ".rst", ".txt", ""]  # commonly .md for GitHub projects, but other projects may use .rst or .txt (or even blank).


#########################################################
# Init
#########################################################

# check for Python 2.7 or later
# http://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
import sys
if sys.version_info < (2,7):
  sys.exit('Sorry, Python < 2.7 is not supported')

import os

from setuptools import setup, find_packages
from setuptools.extension import Extension

try:
  from Cython.Build import cythonize
except ImportError:
  sys.exit("Cython not found. Cython is needed to build the extension modules.")


#########################################################
# Definitions
#########################################################

# Define our base set of compiler and linker flags.
#
# This is geared toward x86_64, see
#    https://gcc.gnu.org/onlinedocs/gcc-4.6.4/gcc/i386-and-x86_002d64-Options.html
#
# Customize these as needed.
#
# Note that -O3 may sometimes cause mysterious problems, so we limit ourselves to -O2.

# Modules involving numerical computations
extra_compile_args_math_optimized    = ['-march=native', '-O2', '-msse', '-msse2', '-mfma', '-mfpmath=sse']
extra_compile_args_math_debug        = ['-march=native', '-O0', '-g']
extra_link_args_math_optimized       = []
extra_link_args_math_debug           = []

# Modules that do not involve numerical computations
extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug     = ['-O0', '-g']
extra_link_args_nonmath_optimized    = []
extra_link_args_nonmath_debug        = []

# Additional flags to compile/link with OpenMP
openmp_compile_args = ['-fopenmp']
openmp_link_args    = ['-fopenmp']


#########################################################
# Helpers
#########################################################

# Make absolute cimports work.
#
# See
#     https://github.com/cython/cython/wiki/PackageHierarchy
#
# For example: my_include_dirs = [np.get_include()]
import numpy as np
my_include_dirs = [".", np.get_include()]


# Choose the base set of compiler and linker flags.
if build_type == 'optimized':
  my_extra_compile_args_math    = extra_compile_args_math_optimized
  my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
  my_extra_link_args_math       = extra_link_args_math_optimized
  my_extra_link_args_nonmath    = extra_link_args_nonmath_optimized
  my_debug = False
  print( "build configuration selected: optimized" )
elif build_type == 'debug':
  my_extra_compile_args_math    = extra_compile_args_math_debug
  my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
  my_extra_link_args_math       = extra_link_args_math_debug
  my_extra_link_args_nonmath    = extra_link_args_nonmath_debug
  my_debug = True
  print( "build configuration selected: debug" )
else:
  raise ValueError("Unknown build configuration '%s'; valid: 'optimized', 'debug'" % (build_type))


def declare_cython_extension(extName, use_math=False, use_openmp=False, include_dirs=None):
  """Declare a Cython extension module for setuptools.
  Parameters:
    extName : str
        Absolute module name, e.g. use `mylibrary.mypackage.mymodule`
        for the Cython source file `mylibrary/mypackage/mymodule.pyx`.
    use_math : bool
        If True, set math flags and link with ``libm``.
    use_openmp : bool
        If True, compile and link with OpenMP.
  Return value:
    Extension object
        that can be passed to ``setuptools.setup``.
  """
  extPath = extName.replace(".", os.path.sep)+".pyx"

  if use_math and os.name != 'nt': # Windows crashes when using m library
    compile_args = list(my_extra_compile_args_math) # copy
    link_args    = list(my_extra_link_args_math)
    libraries    = ["m"]  # link libm; this is a list of library names without the "lib" prefix
  else:
    compile_args = list(my_extra_compile_args_nonmath)
    link_args    = list(my_extra_link_args_nonmath)
    libraries    = None  # value if no libraries, see setuptools.extension._Extension

  # OpenMP
  if use_openmp:
    compile_args.insert( 0, openmp_compile_args )
    link_args.insert( 0, openmp_link_args )

  # See
  #    http://docs.cython.org/src/tutorial/external.html
  #
  # on linking libraries to your Cython extensions.
  return Extension(
    extName,
    [extPath],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    include_dirs=include_dirs,
    libraries=libraries
  )


# Gather user-defined data files
# http://stackoverflow.com/questions/13628979/setuptools-how-to-make-package-contain-extra-data-folder-and-all-folders-inside
datafiles = []
getext = lambda filename: os.path.splitext(filename)[1]
for datadir in datadirs:
  datafiles.extend( [(root, [os.path.join(root, f) for f in files if getext(f) in dataexts])
                     for root, dirs, files in os.walk(datadir)] )


# Add standard documentation (README et al.), if any, to data files
detected_docs = []
for docname in standard_docs:
  for ext in standard_doc_exts:
    filename = "".join( (docname, ext) )  # relative to the directory in which setup.py resides
    if os.path.isfile(filename):
      detected_docs.append(filename)
datafiles.append( ('.', detected_docs) )


#########################################################
# Set up modules
#########################################################

ext_module_tree      = declare_cython_extension("sklearn_pmml_model.tree._tree", use_math=True, use_openmp=False, include_dirs=my_include_dirs)
ext_module_quad_tree = declare_cython_extension("sklearn_pmml_model.tree.quad_tree", use_math=True, use_openmp=False, include_dirs=my_include_dirs)
ext_module_criterion = declare_cython_extension("sklearn_pmml_model.tree._criterion", use_math=True, use_openmp=False, include_dirs=my_include_dirs)
ext_module_splitter  = declare_cython_extension("sklearn_pmml_model.tree._splitter", use_math=True, use_openmp=False, include_dirs=my_include_dirs)
ext_module_utils     = declare_cython_extension("sklearn_pmml_model.tree._utils", use_math=True, use_openmp=False, include_dirs=my_include_dirs)

cython_ext_modules = [ext_module_tree, ext_module_quad_tree, ext_module_criterion, ext_module_splitter, ext_module_utils]

# Call cythonize() explicitly, as recommended in the Cython documentation. See
#     http://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiling-with-distutils
#
# This will favor Cython's own handling of '.pyx' sources over that provided by setuptools.
#
# Note that my_ext_modules is just a list of Extension objects. We could add any C sources (not coming from Cython modules) here if needed.
# cythonize() just performs the Cython-level processing, and returns a list of Extension objects.
my_ext_modules = cythonize(cython_ext_modules, include_path=my_include_dirs, gdb_debug=my_debug)


#########################################################
# Call setup()
#########################################################

setup(
  name="sklearn-pmml-model",
  version="0.0.8",
  author="Dennis Collaris",
  author_email="d.collaris@me.com",
  description = "A library to parse PMML models into Scikit-learn estimators.",
  long_description = long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/iamDecode/sklearn-pmml-model",
  license = "BSD-2-Clause",
  classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering"
  ],

  setup_requires = ["cython", "numpy", "pytest-runner"],
  install_requires = [
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
    'cached-property'
  ],
  tests_require = [
    'pytest',
  ],
  provides = ["sklearn_pmml_model"],
  ext_modules = my_ext_modules,
  packages=find_packages(),

  # Install also Cython headers so that other Cython modules can cimport ours
  #
  # Fileglobs relative to each package, **does not** automatically recurse into subpackages.
  # FIXME: force sdist, but sdist only, to keep the .pyx files (this puts them also in the bdist)
  package_data={'sklearn_pmml_model': ['*.pxd', '*.pyx'],
                'sklearn_pmml_model.tree': ['*.pxd', '*.pyx']},

  # Disable zip_safe, because:
  #   - Cython won't find .pxd files inside installed .egg, hard to compile libs depending on this one
  #   - dynamic loader may need to have the library unzipped to a temporary directory anyway (at import time)
  zip_safe = False,

  # Custom data files not inside a Python package
  data_files = datafiles
)
