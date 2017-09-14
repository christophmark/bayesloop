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

.. currentmodule:: bayesloop.observationModels
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
   
.. automodule:: bayesloop.observationModels
   :members:
   
Transition models
-----------------

.. currentmodule:: bayesloop.transitionModels
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

.. automodule:: bayesloop.transitionModels
    :members:

File I/O
--------

.. automodule:: bayesloop.fileIO
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

Probability Parser
------------------

.. autoclass:: Parser