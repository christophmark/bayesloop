.. _installation:

************
Installation
************

The easiest way to install the latest release version of *bayesloop* is via ``pip``:

::

    pip install bayesloop

Alternatively, a zipped version can be downloaded `here <https://github.com/christophmark/bayesloop/releases>`__. The module is installed by calling ``python -m pip install .`` from the project root.

Development version
-------------------

The latest development version of *bayesloop* can be installed from the ``v2`` branch using pip (requires git):

::

    pip install git+https://github.com/christophmark/bayesloop@v2

Alternatively, clone the repository and install it in editable mode:

::

    python -m pip install -e ".[test]"

Dependencies
------------

*bayesloop* v2 supports Python 3.10 and newer. It depends on NumPy, SciPy, SymPy, matplotlib, tqdm and cloudpickle.

Optional dependencies
---------------------

*bayesloop* supports multiprocessing for computationally expensive analyses, based on the `joblib <https://joblib.readthedocs.io/>`__ module:

::

    python -m pip install "bayesloop[parallel]"
