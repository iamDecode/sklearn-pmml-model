Installing sklearn-pmml-model
=============================

The easiest way to install :code:`sklearn-pmml-model` is with :ref:`install-pip`. Alternatively, you can install it :ref:`from source<install-from-source>`.

.. _install-pip:

pip
--------

Pre-built binary packages (wheels) are provided for Linux, MacOS, and Windows through PyPI.
To install using :code:`pip`, simply run::

  $ pip install sklearn-pmml-model

More details on using `pip` can be found `here <https://packaging.python.org/tutorials/installing-packages/#use-pip-for-installing>`_.

.. _install-from-source:

From source
-----------

If you want to build :code:`sklearn-pmml-model` from source, you
will need a C/C++ compiler to compile extensions.

**Linux**

On Linux, you need to install :code:`gcc`, which in most cases is available
via your distribution's packaging system.
Please follow your distribution's instructions on how to install packages.

**MacOS**

On MacOS, you need to install :code:`clang`, which is available from
the *Command Line Tools* package. Open a terminal and execute::

  $ xcode-select --install

Alternatively, you can download it from the
`Apple Developers page <https://developer.apple.com/downloads/index.action>`_.
Log in with your Apple ID, then search and download the
*Command Line Tools for Xcode* package.

**Windows**

On Windows, the compiler you need depends on the Python version
you are using. See `this guide <https://wiki.python.org/moin/WindowsCompilers>`_
to determine which Microsoft Visual C++ compiler to use with a specific Python version.

**Installing**

Grab a local copy of the source::

  $ git clone http://github.com/iamDecode/sklearn-pmml-model
  $ cd sklearn-pmml-model

create a virtual environment and activate it::

  $ python3 -m venv venv
  $ source venv/bin/activate

and install the dependencies::

  $ pip install -r requirements.txt

The final step is to build the Cython extensions (this part requires the C/C++ compiler)::

  $ python setup.py build_ext --inplace


.. _dependencies:

Dependencies
------------

The current minimum dependencies to run :code:`sklearn-pmml-model` are:

- numpy 1.16 or later
- pandas
- scikit-learn
- scipy
- cached-property