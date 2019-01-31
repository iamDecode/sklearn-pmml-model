from setuptools import setup, find_packages, Extension
import numpy as np
from Cython.Build import cythonize, build_ext

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name="sklearn-pmml-model",
  version="0.0.2",
  author="Dennis Collaris",
  author_email="d.collaris@me.com",
  description="A library to parse PMML models into Scikit-learn estimators.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/iamDecode/sklearn-pmml-model",
  packages=find_packages(),
  classifiers=(
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering"
  ),
  install_requires = [
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
    'cached-property'
  ],
  setup_requires = [
    'pytest-runner',
  ],
  tests_require = [
    'pytest',
  ],
  ext_modules = cythonize([
    Extension("sklearn_pmml_model.tree._tree", ["sklearn_pmml_model/tree/_tree.pyx"], include_dirs=[np.get_include()]),
    Extension("sklearn_pmml_model.tree.quad_tree", ["sklearn_pmml_model/tree/quad_tree.pyx"], include_dirs=[np.get_include()]),
    Extension("sklearn_pmml_model.tree._criterion", ["sklearn_pmml_model/tree/_criterion.pyx"], include_dirs=[np.get_include()]),
    Extension("sklearn_pmml_model.tree._splitter", ["sklearn_pmml_model/tree/_splitter.pyx"], include_dirs=[np.get_include()]),
    Extension("sklearn_pmml_model.tree._utils", ["sklearn_pmml_model/tree/_utils.pyx"], include_dirs=[np.get_include()]),
  ]),
  build_ext = build_ext
)