.. _installation:

************
Installation
************

To install the latest version of *bayesloop*, download the `zipped version <https://github.com/christophmark/bayesloop/zipball/master>`__ or clone the repository and install *bayesloop* using ``python setup.py install``.

Another option is to install *bayesloop* from the master branch using pip (requires git):

::

    pip install git+https://github.com/christophmark/bayesloop

Dependencies
------------

*bayesloop* is tested on both Python 2.7 and Python 3.5. It depends on NumPy, SciPy, SymPy and matplotlib.

Optional dependencies
---------------------

*bayesloop* uses `dill <https://pypi.python.org/pypi/dill>`__, an extension to Python's `pickle <https://docs.python.org/2/library/pickle.html>`__ module, to save/load on-going data studies to/from file. It can be installed via pip:

::

    pip install dill

*bayesloop* further supports multiprocessing for computationally expensive analyses, based on the `pathos <https://github.com/uqfoundation/pathos>`__ module. The latest version can be obtained directly from GitHub using pip (requires git):

::

    pip install git+https://github.com/uqfoundation/pathos

.. note::
    
    Windows users need to install a C compiler *before* installing pathos. One possible solution for 64bit systems is to install `Microsoft Visual C++ 2008 SP1 Redistributable Package (x64) <http://www.microsoft.com/en-us/download/confirmation.aspx?id=2092>`__ and `Microsoft Visual C++ Compiler for Python 2.7 <http://www.microsoft.com/en-us/download/details.aspx?id=44266>`__.
