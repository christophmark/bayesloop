#!/usr/bin/env python
"""
Observation models refer to likelihood functions, describing the probability (density) of a measurement at a certain
time step, given the time-varying parameter values and past measurements. Observation models thus represent the low-
level model in a bayesloop study, as compared to transition models that represent the high-level models and specify
how the time-varying parameter change over time.
"""

from __future__ import division, print_function
import numpy as np
import sympy.abc as abc
from sympy import lambdify
from sympy.stats import density
from .jeffreys import getJeffreysPrior
from scipy.special import factorial
from scipy.special import iv
from .exceptions import ConfigurationError, PostProcessingError
from .helper import cint, oint, freeSymbols
from inspect import getargspec
import warnings


class ObservationModel:
    """
    Observation model class that handles missing data points and multi-dimensional data. All observation
    models included in bayesloop inherit from this class.
    """

    def __str__(self):
        return self.name

    def processedPdf(self, grid, dataSegment):
        """
        This method is called by the fit-method of the Study class (and the step method of the OnlineStudy class) and
        processes multidimensional data and missing data and passes it to the pdf-method of the child class.

        Args:
            grid(list): Discrete parameter grid
            dataSegment(ndarray): Data segment from formatted data

        Returns:
            ndarray: Discretized pdf (with same shape as grid)
        """
        # if self.multipyLikelihoods == True, multi-dimensional data is processed one dimension at a time;
        # likelihoods are then multiplied
        if len(dataSegment.shape) == 2 and self.multiplyLikelihoods:
            return np.prod(np.array([self.processedPdf(grid, d) for d in dataSegment.T]), axis=0)

        # check for missing data
        if np.isnan(dataSegment).any():
            return np.ones_like(grid[0])  # grid of ones does not alter the current prior distribution

        return self.pdf(grid, dataSegment)


class NumPy(ObservationModel):
    """
    Model based on NumPy functions. This observation model class allows the user to create new observation models by
    expressing the likelihood function as a Python function that takes a data point (or vector) and arrays of parameter
    values as input, and outputs the probability density of those parameter values. Note that the Python function must
    be able to broadcast the arrays of parameter values, so that the output array has the same shape as the input
    arrays.

    Args:
        function: Likelihood function that takes a data point as the first argument and one NumPy array per model
            parameter (see example below).
        args: succession of names and corresponding parameter values (using bayesloop.cint() or
            bayesloop.oint()) Example: 'mu', bl.cint(-1, 1, 100), 'sigma', bl.oint(0, 3, 100)
        prior: custom prior distribution that may be passed as a NumPy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)

    Example:
    ::
        # Assume that we have a data set of Gaussian random variates. We know the standard deviation for each random
        # variate, but not the mean value. The data has the form [[variate_1, std_1], [variate_2, std_2], ...]. We can
        # design an observation model to infer the mean value of the data taking into account the known standard
        # deviation as follows:

        import bayesloop as bl
        S = bl.Study()

        data = np.array([[0.12, 0.2], [-0.23, 0.2], [-0.03, 0.1], [0.12, 0.1]])
        S.loadData(data)

        def likelihood(data, mu):
            # read in one data point of the form [variate_n, std_n]
            x, std = data

            # define Gaussian likelihood function (pdf) with known standard deviation
            pdf = np.exp((x - mu)**2./(2*std**2.))/np.sqrt(2*np.pi*std**2.)

            return pdf

        L = bl.om.NumPy(likelihood, 'mu', bl.cint(-3, 3, 1000))
        S.setOM(L)
    """
    def __init__(self, function, *args, **kwargs):
        # check if first argument is valid
        if not hasattr(function, '__call__'):
            raise ConfigurationError('Expected a function as the first argument of NumPy observation model')

        self.function = function
        self.name = function.__name__
        self.segmentLength = 1  # all required data for one time step is bundled
        self.multiplyLikelihoods = False  # no more than one observation per time step

        # get specified parameter names/values
        self.parameterNames = args[::2]
        self.parameterValues = args[1::2]

        # check if number of specified parameters matches number of arguments of function (-1 for data)
        nArgs = len(getargspec(self.function).args)
        if not len(self.parameterNames) == nArgs-1:
            raise ConfigurationError('Supplied function has {} parameters, observation model has {}'
                                     .format(nArgs-1, len(self.parameterNames)))

        # check if first argument of supplied function is called 'data'
        if not getargspec(self.function).args[0] == 'data':
            raise ConfigurationError('First argument of supplied function must be called "data"')

        # check for unknown keyword-arguments
        for key in kwargs.keys():
            if key not in ['prior']:
                raise TypeError("__init__() got an unexpected keyword argument '{}'".format(key))

        # get allowed keyword-arguments
        self.prior = kwargs.get('prior', None)

    def pdf(self, grid, dataSegment):
        """
        Probability density function of custom models

        Args:
            grid(list): Parameter grid for discrete parameter values
            dataSegment(ndarray): Data segment from formatted data

        Returns:
            ndarray: Discretized pdf (with same shape as grid)
        """
        return self.function(dataSegment[0], *grid)


class SciPy(ObservationModel):
    """
    Model based on scipy.stats distribution. This observation model class allows to create new observation models
    on-the-fly from scipy.stats probability distributions.

    Args:
        rv: SciPy random distribution
        args: succession of names and corresponding parameter values (using bayesloop.cint() or
            bayesloop.oint()) Example: 'mu', bl.cint(-1, 1, 100), 'sigma', bl.oint(0, 3, 100)
        fixedParameters(dict): Dictionary defining the names and values of fixed parameters
        prior: custom prior distribution that may be passed as a Numpy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)

    Note that scipy.stats does not use the canonical way of naming the parameters of the probability distributions, but
    instead includes the parameter 'loc' (for discrete & continuous distributions) and 'scale' (for continuous only).

    See http://docs.scipy.org/doc/scipy/reference/stats.html for further information on the available distributions and
    the parameter notation.

    Example:
    ::
        import bayesloop as bl
        import scipy.stats
        L = bl.om.SciPy(scipy.stats.poisson, 'mu', bl.oint(0, 6, 1000), fixedParameters={'loc': 0})

    This will result in a model for poisson-distributed observations with a rate parameter 'mu' between 0 and 6. The
    distribution is not shifted (loc = 0).

    Note that while the parameters 'loc' and 'scale' have default values in scipy.stats and do not necessarily need
    to be set, they have to be added to the fixedParameters dictionary in bayesloop to be treated as a constant.
    Using SciPy.stats distributions, bayesloop uses a flat prior by default.
    """
    def __init__(self, rv, *args, **kwargs):
        # def __init__(self, rv, valueDict={}, prior=None, fixedParameters={}):
        # check if first argument is valid
        try:
            self.module = rv.__module__.split('.')
            assert self.module[0] == 'scipy'
            assert self.module[1] == 'stats'
        except:
            raise ConfigurationError('SciPy observation model must contain SciPy probability distribution')

        self.rv = rv
        self.name = rv.name  # scipy.stats name is used

        # get specified parameter names/values
        if len(args) == 1 and isinstance(args[0], dict):
            warnings.warn(
                "Using a dictionary to define parameter names/values is deprecated and will be removed in a "
                "future version. Pass parameter names and values as successive arguments.",
                DeprecationWarning)
            valueDict = args[0]
            self.parameterNames = valueDict.keys()
            self.parameterValues = valueDict.values()
        else:
            self.parameterNames = args[::2]
            self.parameterValues = args[1::2]

        # check for unknown keyword-arguments
        for key in kwargs.keys():
            if key not in ['prior', 'fixedParameters']:
                raise TypeError("__init__() got an unexpected keyword argument '{}'".format(key))

        # get allowed keyword-arguments
        self.prior = kwargs.get('prior', None)
        self.fixedParameterDict = kwargs.get('fixedParameters', {})

        self.segmentLength = 1  # currently only independent observations are supported by Custom class
        self.multiplyLikelihoods = True

        # check whether random variable is a continuous variable
        if "pdf" in dir(self.rv):
            self.isContinuous = True
        else:
            self.isContinuous = False

        # list of all possible parameters is stored in 'shapes'
        if rv.shapes is None:  # for some distributions, shapes is set to None (e.g. normal distribution)
            shapes = []
        else:
            shapes = rv.shapes.split(', ')

        shapes.append('loc')
        if self.isContinuous:
            shapes.append('scale')  # scale parameter is only available for continuous variables

        # list of free parameters
        rvNames = [param for param in shapes if not (param in self.fixedParameterDict)]

        # if no parameters are provided, take all free parameters and assign "None" as values
        if len(self.parameterNames) == 0:
            self.parameterNames = rvNames
            self.parameterValues = [None] * len(rvNames)

        # check if free parameters from SymPy RV match supplied parameter names
        diff = set(self.parameterNames).difference(set(rvNames))
        if len(diff) > 0:
            raise ConfigurationError(
                'The following parameter names from the observation model do not match the parameter names '
                'of the SciPy distribution: {} (options: {})'.format(list(diff), rvNames))

    def pdf(self, grid, dataSegment):
        """
        Probability density function of custom scipy.stats models

        Args:
            grid(list): Parameter grid for discrete rate values
            dataSegment(ndarray): Data segment from formatted data

        Returns:
            ndarray: Discretized pdf (with same shape as grid)
        """
        # create dictionary from list
        freeParameterDict = {key: value for key, value in zip(self.parameterNames, grid)}

        # merge free/fixed parameter dictionaries
        parameterDict = freeParameterDict.copy()
        parameterDict.update(self.fixedParameterDict)

        # scipy.stats differentiates between 'pdf' and 'pmf' for continuous and discrete variables, respectively
        if self.isContinuous:
            return self.rv.pdf(dataSegment[0], **parameterDict)
        else:
            return self.rv.pmf(dataSegment[0], **parameterDict)


class SymPy(ObservationModel):
    """
    Model based on sympy.stats random variable. This observation model class allows to create new observation models
    on-the-fly from sympy.stats random variables.

    Args:
        rv: SymPy random symbol
        args: succession of names and corresponding parameter values (using bayesloop.cint() or
            bayesloop.oint()) Example: 'mu', bl.cint(-1, 1, 100), 'sigma', bl.oint(0, 3, 100)
        determineJeffreysPrior(bool): If set to true, Jeffreys prior is analytically derived
        prior: custom prior distribution that may be passed as a Numpy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)

    Observation models can be defined symbolically using the SymPy module in a convenient way. In contrast to the
    SciPy probability distributions, fixed parameters are directly set and do not have to be passed as a dictionary.

    See http://docs.sympy.org/dev/modules/stats.html for further information on the available distributions and the
    parameter notation.

    Example:
    ::
        import bayesloop as bl
        from sympy import Symbol
        from sympy.stats import Normal

        mu = 4
        sigma = Symbol('sigma', positive=True)
        rv = Normal('normal', mu, sigma)

        L = bl.om.SymPy(rv, {'sigma': bl.oint(0, 3, 1000)})

    This will result in a model for normally distributed observations with a fixed 'mu' (mean) of 4, leaving 'sigma'
    (the standard deviation) as the only free parameter to be inferred. Using SymPy random variables to create an
    observation model, bayesloop tries to determine the corresponding Jeffreys prior. This behavior can be turned
    off by setting the keyword-argument 'determineJeffreysPrior=False'.
    """
    def __init__(self, rv, *args, **kwargs):
        # check if first argument is valid
        try:
            self.module = rv.__module__.split('.')
            assert self.module[0] == 'sympy'
            assert self.module[1] == 'stats'
        except:
            raise ConfigurationError('SymPy observation model must contain SymPy random variable.')

        self.rv = rv
        self.name = str(rv)  # user-defined name for random variable is used

        # get specified parameter names/values
        if len(args) == 1 and isinstance(args[0], dict):
            warnings.warn("Using a dictionary to define parameter names/values is deprecated and will be removed in a "
                          "future version. Pass parameter names and values as successive arguments.",
                          DeprecationWarning)
            valueDict = args[0]
            self.parameterNames = valueDict.keys()
            self.parameterValues = valueDict.values()
        else:
            self.parameterNames = args[::2]
            self.parameterValues = args[1::2]

        rvParams = freeSymbols(self.rv)
        rvNames = [str(p) for p in rvParams]

        # if no parameters are provided, take the ones from the random variables and assign "None" as values
        if len(self.parameterNames) == 0:
            self.parameterNames = rvNames
            self.parameterValues = [None]*len(rvNames)

        # check if free parameters from SymPy RV match supplied parameter names
        diff = set(self.parameterNames).difference(set(rvNames))
        if len(diff) > 0:
            raise ConfigurationError('The following parameter names from the observation model do not match the names '
                                     'of SymPy random variables: {}'.format(list(diff)))

        # order of rvParams seems to be random, so we need to adjust it to self.parameterNames
        rvParamsSorted = [rvParams[rvNames.index(name)] for name in self.parameterNames]

        # check for unknown keyword-arguments
        for key in kwargs.keys():
            if key not in ['prior', 'determineJeffreysPrior']:
                raise TypeError("__init__() got an unexpected keyword argument '{}'".format(key))

        # get allowed keyword-arguments
        self.prior = kwargs.get('prior', None)
        determineJeffreysPrior = kwargs.get('determineJeffreysPrior', True)

        self.segmentLength = 1  # currently only independent observations are supported by Custom class
        self.multiplyLikelihoods = True

        # determine Jeffreys prior
        if self.prior is None:
            if determineJeffreysPrior:
                try:
                    print('    + Trying to determine Jeffreys prior. This might take a moment...')
                    symPrior, self.prior = getJeffreysPrior(self.rv)
                    print('    + Successfully determined Jeffreys prior: {}. Will use corresponding lambda function.'
                          .format(symPrior))
                except:
                    print('    ! WARNING: Failed to determine Jeffreys prior. Will use flat prior instead.')
                    self.prior = None
            else:
                self.prior = None

        # provide lambda function for probability density
        x = abc.x
        symDensity = density(rv)(x)
        self.density = lambdify([x]+rvParamsSorted, symDensity,
                                modules=['numpy', {'factorial': factorial, 'besseli': iv}])

    def pdf(self, grid, dataSegment):
        """
        Probability density function of custom sympy.stats models

        Args:
            grid(list): Parameter grid for discrete rate values
            dataSegment(ndarray): Data segment from formatted data

        Returns:
            ndarray: Discretized pdf (with same shape as grid)
        """
        return self.density(dataSegment[0], *grid)


class Bernoulli(ObservationModel):
    """
    Bernoulli trial. This distribution models a random variable that takes the value 1 with a probability of p, and
    a value of 0 with the probability of (1-p). Subsequent data points are considered independent. The model has one
    parameter, p, which describes the probability of "success", i.e. to take the value 1.

    Args:
        name(str): custom name for model parameter p
        value(list, tuple, ndarray): Regularly spaced parameter values for the model parameter p
        prior: custom prior distribution that may be passed as a Numpy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)
    """

    def __init__(self, name='p', value=None, prior='Jeffreys'):
        self.name = 'Bernoulli'
        self.segmentLength = 1  # number of measurements in one data segment
        self.parameterNames = [name]
        self.parameterValues = [value]
        self.multiplyLikelihoods = True

        if isinstance(prior, str) and prior == 'Jeffreys':
            self.prior = self.jeffreys  # default: Jeffreys prior
        else:
            self.prior = prior

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Bernoulli model

        Args:
            grid(list): Parameter grid for discrete values of the parameter p
            dataSegment(ndarray): Data segment from formatted data (in this case a single number of events)

        Returns:
            ndarray: Discretized Bernoulli pdf (with same shape as grid)
        """
        temp = grid[0][:]  # make copy of parameter grid
        temp[temp > 1.] = 0.  # p < 1
        temp[temp < 0.] = 0.  # p > 0

        if dataSegment[0]:
            pass  # pdf = p
        else:
            temp = 1. - temp  # pdf = 1 - p

        return temp

    def estimateParameterValues(self, name, rawData):
        """
        Returns appropriate boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            name(str): name of a parameter of the observation model
            rawData(ndarray): observed data points that may be used to determine appropriate parameter boundaries

        Returns:
            list: Regularly spaced parameter values for the specified parameter.
        """

        if name == self.parameterNames[0]:
            # The parameter of the Bernoulli model is naturally constrained to the [0, 1] interval
            return cint(0, 1, 1000)
        else:
            raise ConfigurationError('Bernoulli model does not contain a parameter "{}".'.format(name))

    def jeffreys(self, x):
        """
        Jeffreys prior for Bernoulli model.
        """
        return 1./np.sqrt(x*(1.-x))


class Poisson(ObservationModel):
    """
    Poisson observation model. Subsequent data points are considered independent and distributed according to the
    Poisson distribution. Input data consists of integer values, typically the number of events in a fixed time
    interval. The model has one parameter, often denoted by lambda, which describes the rate of the modeled events.

    Args:
        name(str): custom name for rate parameter lambda
        value(list, tuple, ndarray): Regularly spaced parameter values for the model parameter lambda
        prior: custom prior distribution that may be passed as a Numpy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)
    """
    def __init__(self, name='lambda', value=None, prior='Jeffreys'):
        self.name = 'Poisson'
        self.segmentLength = 1  # number of measurements in one data segment
        self.parameterNames = [name]
        self.parameterValues = [value]
        self.multiplyLikelihoods = True

        if isinstance(prior, str) and prior == 'Jeffreys':
            self.prior = self.jeffreys  # default: Jeffreys prior
        else:
            self.prior = prior

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Poisson model

        Args:
            grid(list): Parameter grid for discrete rate (lambda) values
            dataSegment(ndarray): Data segment from formatted data (in this case a single number of events)

        Returns:
            ndarray: Discretized Poisson pdf (with same shape as grid)
        """
        return (grid[0] ** dataSegment[0]) * (np.exp(-grid[0])) / (np.math.factorial(dataSegment[0]))

    def estimateParameterValues(self, name, rawData):
        """
        Returns appropriate boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            name(str): name of a parameter of the observation model
            rawData(ndarray): observed data points that may be used to determine appropriate parameter boundaries

        Returns:
            list: parameter boundaries.
        """
        if name == self.parameterNames[0]:
            # lower is boundary is zero by definition, upper boundary is chosen as 1.25*(largest observation)
            return oint(0, 1.25*np.nanmax(np.ravel(rawData)), 1000)
        else:
            raise ConfigurationError('Poisson model does not contain a parameter "{}".'.format(name))

    def jeffreys(self, x):
        """
        Jeffreys prior for Poisson model.
        """
        return np.sqrt(1. / x)


class Gaussian(ObservationModel):
    """
    Gaussian observations. All observations are independently drawn from a Gaussian distribution. The model has two
    parameters, mean and standard deviation.

    Args:
        name1(str): custom name for mean
        value1(list, tuple, ndarray): Regularly spaced parameter values for the model parameter mean
        name2(str): custom name for standard deviation
        value2(list, tuple, ndarray): Regularly spaced parameter values for the model parameter standard deviation
        prior: custom prior distribution that may be passed as a Numpy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)
    """

    def __init__(self, name1='mean', value1=None, name2='std', value2=None, prior='Jeffreys'):
        self.name = 'Gaussian observations'
        self.segmentLength = 1  # number of measurements in one data segment
        self.parameterNames = [name1, name2]
        self.parameterValues = [value1, value2]
        self.multiplyLikelihoods = True

        if isinstance(prior, str) and prior == 'Jeffreys':
            self.prior = self.jeffreys  # default: Jeffreys prior
        else:
            self.prior = prior

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Gaussian model.

        Args:
            grid(list): Parameter grid for discrete values of mean and standard deviation
            dataSegment(ndarray): Data segment from formatted data (containing a single measurement)

        Returns:
            ndarray: Discretized Normal pdf (with same shape as grid).
        """
        return np.exp(
            -((dataSegment[0] - grid[0]) ** 2.) / (2. * grid[1] ** 2.) - .5 * np.log(2. * np.pi * grid[1] ** 2.))

    def estimateParameterValues(self, name, rawData):
        """
        Returns appropriate boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            name(str): name of a parameter of the observation model
            rawData(ndarray): observed data points that may be used to determine appropriate parameter boundaries

        Returns:
            list: parameter boundaries.
        """
        mean = np.nanmean(np.ravel(rawData))
        std = np.nanstd(np.ravel(rawData))

        if name == self.parameterNames[0]:
            return cint(mean-2*std, mean+2*std, 200)
        elif name == self.parameterNames[1]:
            return oint(0, 2 * std, 200)
        else:
            raise ConfigurationError('Gaussian model does not contain a parameter "{}".'.format(name))

    def jeffreys(self, mu, sigma):
        """
        Jeffreys prior for Gaussian model.
        """
        return 1./sigma**2.


class Laplace(ObservationModel):
    """
    Laplace model. All observations are independently drawn from a Laplace (double-sided exponential) distribution. The
    model has two parameters, mean and scale.

    Args:
        name1(str): custom name for mean
        value1(list, tuple, ndarray): Regularly spaced parameter values for the model parameter mean
        name2(str): custom name for the scale parameter
        value2(list, tuple, ndarray): Regularly spaced parameter values for the scale parameter
        prior: custom prior distribution that may be passed as a Numpy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)
    """

    def __init__(self, name1='mean', value1=None, name2='scale', value2=None, prior='Jeffreys'):
        self.name = 'Laplace observations'
        self.segmentLength = 1  # number of measurements in one data segment
        self.parameterNames = [name1, name2]
        self.parameterValues = [value1, value2]
        self.multiplyLikelihoods = True

        if isinstance(prior, str) and prior == 'Jeffreys':
            self.prior = self.jeffreys  # default: Jeffreys prior
        else:
            self.prior = prior

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Laplace model.

        Args:
            grid(list): Parameter grid for discrete values of mean and scale
            dataSegment(ndarray): Data segment from formatted data (containing a single measurement)

        Returns:
            ndarray: Discretized Normal pdf (with same shape as grid).
        """
        return np.exp(-np.abs(dataSegment[0] - grid[0])/grid[1])/(2.*grid[1])

    def estimateParameterValues(self, name, rawData):
        """
        Returns appropriate boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            name(str): name of a parameter of the observation model
            rawData(ndarray): observed data points that may be used to determine appropriate parameter boundaries

        Returns:
            list: parameter boundaries.
        """
        mean = np.nanmean(np.ravel(rawData))
        std = np.nanstd(np.ravel(rawData))

        if name == self.parameterNames[0]:
            return cint(mean-2*std, mean+2*std, 200)
        elif name == self.parameterNames[1]:
            return oint(0, np.sqrt(2) * std, 200)
        else:
            raise ConfigurationError('Gaussian model does not contain a parameter "{}".'.format(name))

    def jeffreys(self, mu, scale):
        """
        Jeffreys prior for the Laplace model.
        """
        return 1./scale**2.


class GaussianMean(ObservationModel):
    """
    Observations with given error interval. This observation model represents a Gaussian distribution with given
    standard deviation, only the mean of the distribution is a free parameter. It can be used if the data at hand
    contains for example mean values and corresponding error intervals. The data is supplied as an array of tuples,
    where each tuple contains the observed mean value and the corresponding standard deviation for an individual time
    step:

    ::

        [["mean (t=1)", "std (t=1)"], ["mean (t=2)", "std (t=2)"], ...]

    Args:
        name(str): custom name for the mean parameter
        value(list, tuple, ndarray): Regularly spaced parameter values for the mean parameter
        prior: custom prior distribution that may be passed as a Numpy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)
    """

    def __init__(self, name='mean', value=None, prior=None):
        self.name = 'Gaussian mean model'
        self.segmentLength = 1
        self.parameterNames = [name]
        self.parameterValues = [value]
        self.multiplyLikelihoods = False
        self.prior = prior  # default: flat prior

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Gaussian mean model.

        Args:
            grid(list): Parameter grid for discrete values of the mean
            dataSegment(ndarray): Data segment from formatted data (containing a tuple of observed mean value and the
                given standard deviation)

        Returns:
            ndarray: Discretized Normal pdf (with same shape as grid).
        """
        return np.exp(-((dataSegment[0, 0] - grid[0]) ** 2.) / (2. * dataSegment[0, 1] ** 2.) -
                      .5 * np.log(2. * np.pi * dataSegment[0, 1] ** 2.))

    def estimateParameterValues(self, name, rawData):
        """
        Returns appropriate boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            name(str): name of a parameter of the observation model
            rawData(ndarray): observed data points that may be used to determine appropriate parameter boundaries

        Returns:
            list: parameter boundaries.
        """
        observations = np.array([d[0] for d in rawData])
        min = np.nanmin(observations)
        max = np.nanmax(observations)
        delta = max - min

        if name == self.parameterNames[0]:
            return oint(min-delta, max+delta, 1000)
        else:
            raise ConfigurationError('Gaussian mean model does not contain a parameter "{}".'.format(name))


class WhiteNoise(ObservationModel):
    """
    White noise process. All observations are independently drawn from a Gaussian distribution with zero mean and
    a finite standard deviation, the noise amplitude. This process is basically an auto-regressive process with zero
    correlation.

    Args:
        name(str): custom name for standard deviation
        value(list, tuple, ndarray): Regularly spaced parameter values for the model parameter standard deviation
        prior: custom prior distribution that may be passed as a Numpy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)
    """

    def __init__(self, name='std', value=None, prior='Jeffreys'):
        self.name = 'White noise process (Zero-mean Gaussian)'
        self.segmentLength = 1  # number of measurements in one data segment
        self.parameterNames = [name]
        self.parameterValues = [value]
        self.multiplyLikelihoods = True

        if isinstance(prior, str) and prior == 'Jeffreys':
            self.prior = self.jeffreys  # default: Jeffreys prior
        else:
            self.prior = prior

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the white noise process.

        Args:
            grid(list): Parameter grid for discrete values of noise amplitude
            dataSegment(ndarray): Data segment from formatted data (containing a single measurement)

        Returns:
            ndarray: Discretized pdf (with same shape as grid).
        """
        return np.exp(-(dataSegment[0] ** 2.) / (2. * grid[0] ** 2.) - .5 * np.log(2. * np.pi * grid[0] ** 2.))

    def estimateParameterValues(self, name, rawData):
        """
        Returns appropriate boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            name(str): name of a parameter of the observation model
            rawData(ndarray): observed data points that may be used to determine appropriate parameter boundaries

        Returns:
            list: parameter boundaries.
        """
        std = np.nanstd(np.ravel(rawData))

        if name == self.parameterNames[0]:
            return oint(0, 2 * std, 1000)
        else:
            raise ConfigurationError('White noise model does not contain a parameter "{}".'.format(name))

    def jeffreys(self, sigma):
        """
        Jeffreys prior for Gaussian model.
        """
        return 1. / sigma


class AR1(ObservationModel):
    """
    Auto-regressive process of first order. This model describes a simple stochastic time series model with an
    exponential autocorrelation-function. It can be recursively defined as: d_t = r_t * d_(t-1) + s_t * e_t, with d_t
    being the data point at time t, r_t the correlation coefficient of subsequent data points and s_t being the noise
    amplitude of the process. e_t is distributed according to a standard normal distribution.

    Args:
        name1(str): custom name for correlation coefficient
        value1(list, tuple, ndarray): Regularly spaced parameter values for the correlation coefficient
        name2(str): custom name for noise amplitude
        value2(list, tuple, ndarray): Regularly spaced parameter values for the noise amplitude
        prior: custom prior distribution that may be passed as a Numpy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)
    """

    def __init__(self, name1='correlation coefficient', value1=None, name2='noise amplitude', value2=None, prior=None):
        self.name = 'Autoregressive process of first order (AR1)'
        self.segmentLength = 2  # number of measurements in one data segment
        self.parameterNames = [name1, name2]
        self.parameterValues = [value1, value2]
        self.prior = prior  # default: flat prior
        self.multiplyLikelihoods = True

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Auto-regressive process of first order

        Args:
            grid(list): Parameter grid for discrete values of the correlation coefficient and noise amplitude
            dataSegment(ndarray): Data segment from formatted data (in this case a pair of measurements)

        Returns:
            ndarray: Discretized pdf (for data point d_t, given d_(t-1) and parameters).
        """
        return np.exp(-((dataSegment[1] - grid[0] * dataSegment[0]) ** 2.) / (2. * grid[1] ** 2.) - .5 * np.log(
            2. * np.pi * grid[1] ** 2.))

    def estimateParameterValues(self, name, rawData):
        """
        Returns estimated boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            name(str): name of a parameter of the observation model
            rawData(ndarray): observed data points that may be used to determine appropriate parameter boundaries

        Returns:
            list: parameter boundaries.
        """
        std = np.nanstd(np.ravel(rawData))

        if name == self.parameterNames[0]:
            return oint(-1, 1, 200)
        elif name == self.parameterNames[1]:
            return oint(0, 2 * std, 200)
        else:
            raise ConfigurationError('AR1 model does not contain a parameter "{}".'.format(name))


class ScaledAR1(ObservationModel):
    """
    Scaled auto-regressive process of first order. Recusively defined as
        d_t = r_t * d_(t-1) + s_t*sqrt(1 - (r_t)^2) * e_t,
    with r_t the correlation coefficient of subsequent data points and s_t being the standard deviation of the
    observations d_t. For the standard AR1 process, s_t defines the noise amplitude of the process. For uncorrelated
    data, the two observation models are equal.

    Args:
        name1(str): custom name for correlation coefficient
        value1(list, tuple, ndarray): Regularly spaced parameter values for the correlation coefficient
        name2(str): custom name for standard deviation
        value2(list, tuple, ndarray): Regularly spaced parameter values for the standard deviation
        prior: custom prior distribution that may be passed as a Numpy array that has tha same shape as the parameter
            grid, as a(lambda) function or as a (list of) SymPy random variable(s)
    """

    def __init__(self, name1='correlation coefficient', value1=None, name2='standard deviation', value2=None,
                 prior=None):
        self.name = 'Scaled autoregressive process of first order (AR1)'
        self.segmentLength = 2  # number of measurements in one data segment
        self.parameterNames = [name1, name2]
        self.parameterValues = [value1, value2]
        self.prior = prior  # default: flat prior
        self.multiplyLikelihoods = True

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Auto-regressive process of first order

        Args:
            grid(list): Parameter grid for discerete values of the correlation coefficient and standard deviation
            dataSegment(ndarray): Data segment from formatted data (in this case a pair of measurements)

        Returns:
            ndarray: Discretized pdf (for data point d_t, given d_(t-1) and parameters).
        """
        r = grid[0]
        s = grid[1]
        sScaled = s*np.sqrt(1 - r**2.)
        return np.exp(-((dataSegment[1] - r * dataSegment[0]) ** 2.) / (2. * sScaled ** 2.) - .5 * np.log(
            2. * np.pi * sScaled ** 2.))

    def estimateParameterValues(self, name, rawData):
        """
        Returns estimated boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            name(str): name of a parameter of the observation model
            rawData(ndarray): observed data points that may be used to determine appropriate parameter boundaries

        Returns:
            list: parameter boundaries.
        """
        std = np.nanstd(np.ravel(rawData))

        if name == self.parameterNames[0]:
            return oint(-1, 1, 200)
        elif name == self.parameterNames[1]:
            return oint(0, 2 * std, 200)
        else:
            raise ConfigurationError('AR1 model does not contain a parameter "{}".'.format(name))
