.. _api:

*************
API Reference
*************

Study types
-----------

.. currentmodule:: bayesloop.core
.. autosummary::

    Study
    HyperStudy
    ChangepointStudy
    OnlineStudy

.. note::

    These Study classes are imported directly into the module namespace for convenient access.
    
    .. code-block:: python

        import bayesloop as bl
        S = bl.Study()

.. automodule:: bayesloop.core
    :members:

Observation models
------------------

.. currentmodule:: bayesloop.observation_models
.. autosummary::

   SymPy
   SciPy
   NumPy
   Bernoulli
   Poisson
   Gaussian
   GaussianMean
   WhiteNoise
   AR1
   ScaledAR1

.. note::

    You can use the short-form `om` to access all observation models:
    
    .. code-block:: python

        import bayesloop as bl
        L = bl.om.SymPy(...)
   
.. automodule:: bayesloop.observation_models
   :members:
   
Transition models
-----------------

.. currentmodule:: bayesloop.transition_models
.. autosummary::

    Static
    Deterministic
    GaussianRandomWalk
    AlphaStableRandomWalk
    ChangePoint
    RegimeSwitch
    Independent
    NotEqual
    CombinedTransitionModel
    SerialTransitionModel

.. note::

    You can use the short-form `tm` to access all transition models:
    
    .. code-block:: python

        import bayesloop as bl
        T = bl.tm.ChangePoint(...)

.. automodule:: bayesloop.transition_models
    :members:

File I/O
--------

.. automodule:: bayesloop.file_io
    :members:

.. note::

    Both file I/O functions are imported directly into the module namespace for convenient access.
    
    .. code-block:: python

        import bayesloop as bl
        S = bl.Study()
        ...
        bl.save('test.bl', S)
        ...
        S = bl.load('test.bl')
