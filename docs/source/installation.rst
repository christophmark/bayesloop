.. _installation:

************
Installation
************

The easiest way to install the latest release version of *bayesloop* is via ``pip``:

::

    pip install bayesloop

Alternatively, a zipped version can be downloaded `here <https://github.com/christophmark/bayesloop/releases>`__. The module is installed by calling ``python setup.py install``.

Development version
-------------------

The latest development version of *bayesloop* can be installed from the master branch using pip (requires git):

::

    pip install git+https://github.com/christophmark/bayesloop

Alternatively, use this `zipped version <https://github.com/christophmark/bayesloop/zipball/master>`__ or clone the repository.

Dependencies
------------

*bayesloop* is tested on both Python 2.7 and Python 3.5. It depends on NumPy, SciPy, SymPy, matplotlib and tqdm. All except the latter are already included in the `Anaconda distribution <https://www.continuum.io/downloads>`__ of Python. Windows users may also take advantage of pre-compiled binaries for all dependencies, which can be found at `Christoph Gohlke's page <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`__.

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
