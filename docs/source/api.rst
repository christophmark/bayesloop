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

.. automodule:: bayesloop.core
    :members:

Observation models
------------------

.. currentmodule:: bayesloop.observationModels
.. autosummary::

   Custom
   Bernoulli
   Poisson
   Gaussian
   ZeroMeanGaussian
   AR1
   ScaledAR1
   LinearRegression
   
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

.. automodule:: bayesloop.transitionModels
    :members:
