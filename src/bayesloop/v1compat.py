#!/usr/bin/env python
"""
Opt-in backwards-compatibility layer for the bayesloop 1.x API.

bayesloop 2.0 renamed the public API from camelCase to snake_case and removed a number of shorthand aliases. Importing
this module once restores the 1.x names on top of the 2.x code base, so that existing 1.x scripts keep running:

    import bayesloop as bl
    import bayesloop.v1compat  # activates the 1.x API

Every use of a 1.x name emits a ``DeprecationWarning`` that points to the snake_case replacement. The layer covers:

* camelCase method, attribute and property names on all study and model classes (``loadData``,
  ``getParameterDistribution``, ``logEvidence``, ...)
* the 1.x shorthand aliases (``setOM``, ``setTM``, ``getPD``, ...)
* camelCase keyword arguments (``forwardOnly``, ``nJobs``, ...)
* the 1.x module names ``bayesloop.observationModels``, ``bayesloop.transitionModels`` and ``bayesloop.fileIO``
* custom observation/transition models that implement 1.x hooks (``computeForwardPrior``,
  ``estimateParameterValues``, ...)
* loading study files saved with bayesloop 1.x (instance attributes are migrated to snake_case on load)

Not covered: the probability ``Parser`` and ``Study.eval()`` were removed in bayesloop 2.0, and the default parameter
names of some transition models changed (``'tChange'`` -> ``'t_change'``, ``'tBreak'`` -> ``'t_break'``,
``'log10pMin'`` -> ``'log10p_min'``); scripts that rely on those default names need to set the names explicitly.
"""

from __future__ import division, print_function
import functools
import re
import sys
import warnings

from . import core, file_io, helper, jeffreys, observation_models, preprocessing, transition_models

_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')


def _snake(name):
    """loadData -> load_data"""
    return _CAMEL_RE.sub('_', name).lower()


def _camel(name):
    """load_data -> loadData"""
    first, _, rest = name.partition('_')
    return first + ''.join(part.title() for part in rest.split('_'))


# 1.x shorthand aliases that cannot be derived by case conversion
_ALIASES = {
    'setOM': 'set_observation_model',
    'setTM': 'set_transition_model',
    'load': 'load_data',
    'add': 'add_transition_model',
    'addTM': 'add_transition_model',
    'getPD': 'get_parameter_distribution',
    'getPDs': 'get_parameter_distributions',
    'getHPD': 'get_hyper_parameter_distribution',
    'getHPDs': 'get_hyper_parameter_distributions',
    'getJHPD': 'get_joint_hyper_parameter_distribution',
    'getDD': 'get_duration_distribution',
    'getCPD': 'get_current_parameter_distribution',
    'getCHPD': 'get_current_hyper_parameter_distribution',
    'getCTMD': 'get_current_transition_model_distribution',
    'getCTMP': 'get_current_transition_model_probability',
    'getTMPs': 'get_transition_model_probabilities',
}

# 1.x keyword arguments that were renamed in 2.0
_LEGACY_KWARGS = {
    'forwardOnly': 'forward_only',
    'evidenceOnly': 'evidence_only',
    'nJobs': 'n_jobs',
    'customHyperGrid': 'custom_hyper_grid',
    'parameterList': 'parameter_list',
    'storeHistory': 'store_history',
    'dataPoint': 'data_point',
    'transitionModel': 'transition_model',
    'transitionModelPrior': 'transition_model_prior',
    'fixedParameters': 'fixed_parameters',
    'determineJeffreysPrior': 'determine_jeffreys_prior',
}

# 1.x features that were removed in 2.0 and cannot be emulated by this layer
_REMOVED = {
    'eval': "'eval' was removed in bayesloop 2.0 together with the probability Parser and is not available through "
            "bayesloop.v1compat. Evaluate the parameter distributions returned by the get_*_distribution methods "
            "directly instead.",
}


def _warn(old, new, stacklevel=3):
    warnings.warn("bayesloop 1.x name '{}' is deprecated, use '{}' instead.".format(old, new),
                  DeprecationWarning, stacklevel=stacklevel)


def _translate_kwargs(kwargs):
    for old, new in _LEGACY_KWARGS.items():
        if old in kwargs:
            if new in kwargs:
                raise TypeError("got values for both '{}' and its 1.x spelling '{}'".format(new, old))
            _warn(old, new, stacklevel=4)
            kwargs[new] = kwargs.pop(old)


def _legacy_kwargs_proxy(func):
    """Wrap a callable so that it accepts the renamed 1.x keyword arguments."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _translate_kwargs(kwargs)
        return func(*args, **kwargs)
    wrapper._v1compat = True
    return wrapper


def _compat_getattr(self, name):
    # only called when normal attribute lookup fails
    if name.startswith('_'):
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
    if name in _REMOVED:
        raise AttributeError(_REMOVED[name])

    candidates = []
    if name in _ALIASES:
        candidates.append(_ALIASES[name])
    snake = _snake(name)
    if snake != name:
        candidates.append(snake)
    camel = _camel(name)
    if camel != name:
        candidates.append(camel)

    for candidate in candidates:
        if candidate in self.__dict__ or hasattr(type(self), candidate):
            if candidate == camel:
                # 2.x code asked for a snake_case name that a 1.x-style custom model implements in camelCase
                warnings.warn("'{}' uses the bayesloop 1.x name '{}'; rename it to '{}'."
                              .format(type(self).__name__, candidate, name), DeprecationWarning, stacklevel=2)
                return getattr(self, candidate)
            _warn(name, candidate)
            attr = getattr(self, candidate)
            if callable(attr):
                return _legacy_kwargs_proxy(attr)
            return attr

    raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))


def _compat_setattr(self, name, value):
    if not name.startswith('_'):
        if not name.islower():
            # 1.x code sets a camelCase attribute that exists in snake_case
            candidate = _snake(name)
            if candidate != name and (candidate in self.__dict__ or hasattr(type(self), candidate)):
                _warn(name, candidate)
                object.__setattr__(self, candidate, value)
                return
        elif '_' in name and name not in self.__dict__:
            # 2.x code sets a snake_case attribute on a 1.x-style custom model: write through to the existing
            # camelCase attribute so that the model's own methods keep seeing the updated value
            camel = _camel(name)
            if camel != name and camel in self.__dict__:
                object.__setattr__(self, camel, value)
                return
    object.__setattr__(self, name, value)


class Parser:
    """
    The probability Parser was removed in bayesloop 2.0 and is not available through bayesloop.v1compat.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('The probability Parser was removed in bayesloop 2.0 and is not available through '
                                  'bayesloop.v1compat. Evaluate the parameter distributions returned by the '
                                  'get_*_distribution methods directly instead.')


def _migrate_v1_attributes(obj, _seen=None):
    """
    Rename camelCase instance attributes of a study saved with bayesloop 1.x (including its observation and
    transition models) to their snake_case spelling, so that 2.x code operates on the same attributes it writes.
    """
    if _seen is None:
        _seen = set()

    if isinstance(obj, (list, tuple)):
        for item in obj:
            _migrate_v1_attributes(item, _seen)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            _migrate_v1_attributes(item, _seen)
        return
    if not isinstance(obj, (core.Study, observation_models.ObservationModel, transition_models.TransitionModel)):
        return
    if id(obj) in _seen:
        return
    _seen.add(id(obj))

    for old in [key for key in obj.__dict__ if not key.startswith('_') and not key.islower()]:
        new = _snake(old)
        if new != old and new not in obj.__dict__:
            obj.__dict__[new] = obj.__dict__.pop(old)

    for value in list(obj.__dict__.values()):
        _migrate_v1_attributes(value, _seen)


# methods whose names are unchanged in 2.0, but whose keyword arguments were renamed
_KWARG_METHODS = [
    (core.Study, 'fit'),
    (core.Study, 'optimize'),
    (core.HyperStudy, 'fit'),
    (core.ChangepointStudy, 'fit'),
    (core.OnlineStudy, '__init__'),
    (core.OnlineStudy, 'step'),
    (core.OnlineStudy, 'add_transition_model'),
    (core.OnlineStudy, 'set_transition_model_prior'),
    (observation_models.SciPy, '__init__'),
    (observation_models.SymPy, '__init__'),
]

# module-level functions that were renamed in 2.0, as (module, 1.x name, 2.0 name)
_FUNCTION_ALIASES = [
    (jeffreys, 'getJeffreysPrior', 'get_jeffreys_prior'),
    (jeffreys, 'computeJeffreysPriorAR1', 'compute_jeffreys_prior_ar1'),
    (helper, 'assignNestedItem', 'assign_nested_item'),
    (helper, 'recursiveIndex', 'recursive_index'),
    (helper, 'createColormap', 'create_colormap'),
    (helper, 'freeSymbols', 'free_symbols'),
    (preprocessing, 'movingWindow', 'moving_window'),
]


def _deprecated_function(func, old, new):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _warn(old, new, stacklevel=2)
        return func(*args, **kwargs)
    wrapper._v1compat = True
    return wrapper


def _load_with_migration(func):
    @functools.wraps(func)
    def wrapper(filename):
        study = func(filename)
        _migrate_v1_attributes(study)
        return study
    wrapper._v1compat = True
    return wrapper


def _activate():
    package = sys.modules[__package__]

    # restore 1.x attribute and method names on studies and models
    for cls in (core.Study, observation_models.ObservationModel, transition_models.TransitionModel):
        cls.__getattr__ = _compat_getattr
        cls.__setattr__ = _compat_setattr

    # accept 1.x keyword arguments on methods whose names survived the renaming
    for cls, name in _KWARG_METHODS:
        method = cls.__dict__[name]
        if not getattr(method, '_v1compat', False):
            setattr(cls, name, _legacy_kwargs_proxy(method))

    # restore 1.x module names (also makes study files saved with 1.x unpicklable again)
    sys.modules['bayesloop.observationModels'] = observation_models
    sys.modules['bayesloop.transitionModels'] = transition_models
    sys.modules['bayesloop.fileIO'] = file_io
    package.observationModels = observation_models
    package.transitionModels = transition_models
    package.fileIO = file_io

    # restore renamed module-level functions
    for module, old, new in _FUNCTION_ALIASES:
        if not hasattr(module, old):
            setattr(module, old, _deprecated_function(getattr(module, new), old, new))
    package.getJeffreysPrior = jeffreys.getJeffreysPrior
    package.computeJeffreysPriorAR1 = jeffreys.computeJeffreysPriorAR1
    package.Parser = Parser

    # migrate attribute names when loading study files saved with 1.x
    if not getattr(file_io.load, '_v1compat', False):
        file_io.load = _load_with_migration(file_io.load)
    package.load = file_io.load


_activate()
