import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="sklearn-pmml-model",
  version="0.0.1",
  author="Dennis Collaris",
  author_email="d.collaris@me.com",
  description="A library to parse PMML models into Scikit-learn estimators.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/iamDecode/sklearn-pmml-model",
  packages=setuptools.find_packages(),
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
  ]
)