#!/usr/bin/env python
"""
This file introduces the observation models that can be used by the Study class for data analysis. An observation model
here refers to a likelihood function, stating the probability of a measurement at a certain time step, given the
parameter values.
"""

from __future__ import division, print_function
import numpy as np
import sympy.abc as abc
from sympy import lambdify
from sympy.stats import density
from .jeffreys import getJeffreysPrior
from scipy.misc import factorial
from .exceptions import *


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
            grid: Discrete parameter grid
            dataSegment: Data segment from formatted data

        Returns:
            Discretized pdf as numpy array (with same shape as grid)
        """
        # if self.multipyLikelihoods == True, multi-dimensional data is processed one dimension at a time;
        # likelihoods are then multiplied
        if len(dataSegment.shape) == 2 and self.multiplyLikelihoods:
            return np.prod(np.array([self.processedPdf(grid, d) for d in dataSegment.T]), axis=0)

        # check for missing data
        if np.isnan(dataSegment).any():
            return np.ones_like(grid[0])  # grid of ones does not alter the current prior distribution

        return self.pdf(grid, dataSegment)


class Custom(ObservationModel):
    """
    This observation model class allows to create new observation models on-the-fly from scipy.stats probability
    distributions or from sympy.stats random variables.

    SciPy.stats
    -----------
    Note that scipy.stats does not use the canonical way of naming the parameters of the probability distributions, but
    instead includes the parameter 'loc' (for discrete & continuous distributions) and 'scale' (for continuous only).

    See http://docs.scipy.org/doc/scipy/reference/stats.html for further information on the available distributions and
    the parameter notation.

    Example::
        M = bl.observationModels.Custom(scipy.stats.norm, fixedParameters={'loc': 4})

    This will result in a model for normally distributed observations with a fixed 'loc' (mean) of 4, leaving the
    'scale' (standard deviation) as the only free parameter to be inferred.

    Note that while the parameters 'loc' and 'scale' have default values in scipy.stats and do not necessarily need to
    be set, they have to be added to the fixedParameters dictionary in bayesloop to be treated as a constant. Using
    SciPy.stats distributions, bayesloop uses a flat prior by default.

    SymPy.stats
    -----------
    Observation models can be defined symbolically using the SymPy module in a convenient way. In contrast to the
    SciPy probability distributions, fixed parameters are directly set and not passed as a dictionary.

    See http://docs.sympy.org/dev/modules/stats.html for further information on the available distributions and the
    parameter notation.

    Example:
        from sympy import Symbol
        from sympy.stats import Normal

        mu = 4
        sigma = Symbol('sigma', positive=True)
        rv = Normal('normal', mu, sigma)

        M = bl.observationModels.Custom(rv)

    This will result in a model for normally distributed observations with a fixed 'mu' (mean) of 4, leaving 'sigma'
    (the standard deviation) as the only free parameter to be inferred. Using SymPy random variables to create an
    observation model, bayesloop tries to determine the corresponding Jeffreys prior. This behavior can be turned off
    by setting the keyword-argument 'determineJeffreysPrior=False'.
    """

    def __init__(self, rv, fixedParameters={}, determineJeffreysPrior=True):
        self.rv = rv
        self.fixedParameterDict = fixedParameters

        # check if first argument is valid
        try:
            self.module = rv.__module__.split('.')
            assert self.module[0] == 'scipy' or self.module[0] == 'sympy'
            assert self.module[1] == 'stats'
        except:
            raise ConfigurationError('Custom observation models have to be based on probability distributions from '
                                     'SciPy or random variables from SymPy.')

        # SciPy probability distribution
        if self.module[0] == 'scipy':
            print('+ Creating custom observation model based on probability distribution from SciPy.')

            # auto-Jeffreys is only available for SymPy RVs
            if determineJeffreysPrior:
                print('  ! WARNING: A flat prior is used.')
                print('    Automatic determination of Jeffreys priors is only available for SymPy RVs.')

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
            self.freeParameters = [param for param in shapes if not (param in self.fixedParameterDict)]

            # set class attributes similar to other observation models
            self.name = rv.name  # scipy.stats name is used
            self.segmentLength = 1  # currently only independent observations are supported by Custom class
            self.parameterNames = self.freeParameters
            self.defaultGridSize = [1000]*len(self.parameterNames)
            self.prior = None
            self.multiplyLikelihoods = True

        # SymPy random variable
        elif self.module[0] == 'sympy':
            print('+ Creating custom observation model based on random variable from SymPy.')

            if fixedParameters:
                raise ConfigurationError('The keyword argument "fixedParameters" can only be used for SciPy '
                                         'probability distributions')

            # extract free variables from SymPy random variable
            parameters = list(self.rv._sorted_args[0].distribution.free_symbols)

            self.name = str(rv)  # user-defined name for random variable is used
            self.segmentLength = 1  # currently only independent observations are supported by Custom class
            self.parameterNames = [str(p) for p in parameters]
            self.defaultGridSize = [1000]*len(self.parameterNames)
            self.multiplyLikelihoods = True

            # determine Jeffreys prior
            if determineJeffreysPrior:
                try:
                    print('    + Trying to determine Jeffreys prior. This might take a moment...')
                    symPrior, self.prior = getJeffreysPrior(self.rv)
                    print('    + Successfully determined Jeffreys prior: {}'.format(symPrior))
                except:
                    print('    ! WARNING: Failed to determine Jeffreys prior. Will use flat prior instead.')
                    self.prior = None
            else:
                self.prior = None

            # provide lambda function for probability density
            x = abc.x
            symDensity = density(rv)(x)
            self.density = lambdify([x]+parameters, symDensity, modules=['numpy', {'factorial': factorial}])

    def pdf(self, grid, dataSegment):
        """
        Probability density function of custom scipy.stats or sympy.stats models

        Args:
            grid: Parameter grid for discrete rate values
            dataSegment: Data segment from formatted data

        Returns:
            Discretized pdf as numpy array (with same shape as grid)
        """
        # SciPy probability distribution
        if self.module[0] == 'scipy':
            # create dictionary from list
            freeParameterDict = {key: value for key, value in zip(self.freeParameters, grid)}

            # merge free/fixed parameter dictionaries
            parameterDict = freeParameterDict.copy()
            parameterDict.update(self.fixedParameterDict)

            # scipy.stats differentiates between 'pdf' and 'pmf' for continuous and discrete variables, respectively
            if self.isContinuous:
                return self.rv.pdf(dataSegment[0], **parameterDict)
            else:
                return self.rv.pmf(dataSegment[0], **parameterDict)

        # SymPy random variable
        elif self.module[0] == 'sympy':
            return self.density(dataSegment[0], *grid)


class Bernoulli(ObservationModel):
    """
    Bernoulli trial. This distribution models a random variable that takes the value 1 with a probability of p, and
    a value of 0 with the probability of (1-p). Subsequent data points are considered independent. The model has one
    parameter, p, which describes the probability of "success", i.e. to take the value 1.
    """

    def __init__(self):
        self.name = 'Bernoulli'
        self.segmentLength = 1  # number of measurements in one data segment
        self.parameterNames = ['p']
        self.defaultGridSize = [1000]
        self.prior = lambda p: 1./np.sqrt(p*(1.-p))  # Jeffreys prior
        self.multiplyLikelihoods = True

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Bernoulli model

        Args:
            grid: Parameter grid for discrete values of the parameter p
            dataSegment: Data segment from formatted data (in this case a single number of events)

        Returns:
            Discretized Bernoulli pdf as numpy array (with same shape as grid)
        """
        temp = grid[0][:]  # make copy of parameter grid
        temp[temp > 1.] = 0.  # p < 1
        temp[temp < 0.] = 0.  # p > 0

        if dataSegment[0]:
            pass  # pdf = p
        else:
            temp = 1. - temp  # pdf = 1 - p

        return temp

    def estimateBoundaries(self, rawData):
        """
        Returns appropriate boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            rawData: observed data points that may be used to determine appropriate parameter boundaries

        Returns:
            List of parameter boundaries.
        """

        # The parameter of the Bernoulli model is naturally constrained to the [0, 1] interval
        return [[0, 1]]


class Poisson(ObservationModel):
    """
    Poisson observation model. Subsequent data points are considered independent and distributed according to the
    Poisson distribution. Input data consists of integer values, typically the number of events in a fixed time
    interval. The model has one parameter, often denoted by lambda, which describes the rate of the modeled events.
    """

    def __init__(self):
        self.name = 'Poisson'
        self.segmentLength = 1  # number of measurements in one data segment
        self.parameterNames = ['lambda']
        self.defaultGridSize = [1000]
        self.prior = lambda x: np.sqrt(1./x)  # Jeffreys prior
        self.multiplyLikelihoods = True

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Poisson model

        Args:
            grid: Parameter grid for discrete rate (lambda) values
            dataSegment: Data segment from formatted data (in this case a single number of events)

        Returns:
            Discretized Poisson pdf as numpy array (with same shape as grid)
        """
        return (grid[0] ** dataSegment[0]) * (np.exp(-grid[0])) / (np.math.factorial(dataSegment[0]))

    def estimateBoundaries(self, rawData):
        """
        Returns appropriate boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            rawData: observed data points that may be used to determine appropiate parameter boundaries

        Returns:
            List of parameter boundaries.
        """

        # lower is boundary is zero by definition, upper boundary is chosen as 1.25*(largest observation)
        return [[0, 1.25*np.nanmax(np.ravel(rawData))]]


class Gaussian(ObservationModel):
    """
    Gaussian observations. All observations are independently drawn from a Gaussian distribution. The model has two
    parameters, mean and standard deviation.
    """

    def __init__(self):
        self.name = 'Gaussian observations'
        self.segmentLength = 1  # number of measurements in one data segment
        self.parameterNames = ['mean', 'standard deviation']
        self.defaultGridSize = [200, 200]
        self.prior = lambda mu, sigma: 1./sigma**3.  # Jeffreys prior
        self.multiplyLikelihoods = True

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Gaussian model.

        Args:
            grid: Parameter grid for discrete values of mean and standard deviation
            dataSegment: Data segment from formatted data (containing a single measurement)

        Returns:
            Discretized Normal pdf as numpy array (with same shape as grid).
        """
        return np.exp(
            -((dataSegment[0] - grid[0]) ** 2.) / (2. * grid[1] ** 2.) - .5 * np.log(2. * np.pi * grid[1] ** 2.))

    def estimateBoundaries(self, rawData):
        """
        Returns appropriate boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            rawData: observed data points that may be used to determine appropiate parameter boundaries

        Returns:
            List of parameter boundaries.
        """
        mean = np.mean(np.ravel(rawData))
        std = np.std(np.ravel(rawData))

        mean_boundaries = [mean - 2*std, mean + 2*std]
        std_boundaries = [0, 2*std]

        return [mean_boundaries, std_boundaries]


class ZeroMeanGaussian(ObservationModel):
    """
    White noise process. All observations are independently drawn from a Gaussian distribution with zero mean and
    a finite standard deviation, the noise amplitude. This process is basically an autoregressive process with zero
    correlation.
    """

    def __init__(self):
        self.name = 'White noise process (zero mean Gaussian)'
        self.segmentLength = 1  # number of measurements in one data segment
        self.parameterNames = ['standard deviation']
        self.defaultGridSize = [1000]
        self.prior = lambda sigma: 1./sigma  # Jeffreys prior
        self.multiplyLikelihoods = True

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the white noise process.

        Args:
            grid: Parameter grid for discrete values of noise amplitude
            dataSegment: Data segment from formatted data (containing a single measurement)

        Returns:
            Discretized Normal pdf (with zero mean) as numpy array (with same shape as grid).
        """
        return np.exp(-(dataSegment[0] ** 2.) / (2. * grid[0] ** 2.) - .5 * np.log(2. * np.pi * grid[0] ** 2.))

    def estimateBoundaries(self, rawData):
        """
        Returns appropriate boundaries based on the imported data. Is called in case fit method is called and no
        boundaries are defined.

        Args:
            rawData: observed data points that may be used to determine appropiate parameter boundaries

        Returns:
            List of parameter boundaries.
        """
        std = np.std(np.ravel(rawData))
        return [[0, 2*std]]


class AR1(ObservationModel):
    """
    Auto-regressive process of first order. This model describes a simple stochastic time series model with an
    exponential autocorrelation-function. It can be recursively defined as: d_t = r_t * d_(t-1) + s_t * e_t, with d_t
    being the data point at time t, r_t the correlation coefficient of subsequent data points and s_t being the noise
    amplitude of the process. e_t is distributed according to a standard normal distribution.
    """

    def __init__(self):
        self.name = 'Autoregressive process of first order (AR1)'
        self.segmentLength = 2  # number of measurements in one data segment
        self.parameterNames = ['correlation coefficient', 'noise amplitude']
        self.defaultGridSize = [200, 200]
        self.multiplyLikelihoods = True

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Auto-regressive process of first order

        Args:
            grid: Parameter grid for discerete values of the correlation coefficient and noise amplitude
            dataSegment: Data segment from formatted data (in this case a pair of measurements)

        Returns:
            Discretized pdf (for data point d_t, given d_(t-1) and parameters) as numpy array (with same shape as grid).
        """
        return np.exp(-((dataSegment[1] - grid[0] * dataSegment[0]) ** 2.) / (2. * grid[1] ** 2.) - .5 * np.log(
            2. * np.pi * grid[1] ** 2.))


class ScaledAR1(ObservationModel):
    """
    Scaled auto-regressive process of first order. Recusively defined as
        d_t = r_t * d_(t-1) + s_t*sqrt(1 - (r_t)^2) * e_t,
    with r_t the correlation coefficient of subsequent data points and s_t being the standard deviation of the
    observations d_t. For the standard AR1 process, s_t defines the noise amplitude of the process. For uncorrelated
    data, the two observation models are equal.
    """

    def __init__(self):
        self.name = 'Scaled autoregressive process of first order (AR1)'
        self.segmentLength = 2  # number of measurements in one data segment
        self.parameterNames = ['correlation coefficient', 'standard deviation']
        self.defaultGridSize = [200, 200]
        self.multiplyLikelihoods = True

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Auto-regressive process of first order

        Args:
            grid: Parameter grid for discerete values of the correlation coefficient and standard deviation
            dataSegment: Data segment from formatted data (in this case a pair of measurements)

        Returns:
            Discretized pdf (for data point d_t, given d_(t-1) and parameters) as numpy array (with same shape as grid).
        """
        r = grid[0]
        s = grid[1]
        sScaled = s*np.sqrt(1 - r**2.)
        return np.exp(-((dataSegment[1] - r * dataSegment[0]) ** 2.) / (2. * sScaled ** 2.) - .5 * np.log(
            2. * np.pi * sScaled ** 2.))


class LinearRegression(ObservationModel):
    """
    Linear regression model. It consists of three parameters: slope, offset and standard deviation. It assumes that the
    observed data follows the relation: data_y = slope * data_x + offset + error. Here, each data point consists of a
    list of two values, data_x and data_y. The error term is assumed to be normally distributed and the corresponding
    standard deviation can be inferred from data or can be set to a fixed value (e.g. fixedError=1). If the keyword-
    argument offset is set to False, this parameter is not included in the fitting process.

    Args:
        offset: If true, the constant term of the linear regression model is inferred from data.
        fixedError: If the error of data_y is known, it can be set using this argument.
    """

    def __init__(self, offset=True, fixedError=False):
        self.offset = offset
        self.fixedError = fixedError
        self.segmentLength = 1
        self.multiplyLikelihoods = False  # multivariate input is needed to compute pdf

        if self.offset and not self.fixedError:
            self.name = 'Linear regression model (including offset)'
            self.parameterNames = ['slope', 'offset', 'standard deviation']
            self.defaultGridSize = [40, 40, 40]
        elif self.offset and self.fixedError:
            self.name = 'Linear regression model (including offset; fixed error = {})'.format(self.fixedError)
            self.parameterNames = ['slope', 'offset']
            self.defaultGridSize = [200, 200]
        elif not self.offset and not self.fixedError:
            self.name = 'Linear regression model'
            self.parameterNames = ['slope', 'standard deviation']
            self.defaultGridSize = [200, 200]
        elif not self.offset and self.fixedError:
            self.name = 'Linear regression model (fixed error = {})'.format(self.fixedError)
            self.parameterNames = ['slope']
            self.defaultGridSize = [1000]

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Auto-regressive process of first order

        Args:
            grid: Parameter grid for discrete parameter values
            dataSegment: Data segment from formatted data (in this case a pair consisting of a x and y-value)

        Returns:
            Discretized pdf.

        """
        slope = grid[0]
        offset = grid[1] if self.offset else 0.
        if not self.fixedError:
            if self.offset:
                sigma = grid[2]
            else:
                sigma = grid[1]
        else:
            sigma = self.fixedError

        return np.exp(-((dataSegment[0, 1] - slope * dataSegment[0, 0] - offset) ** 2.) /
                      (2. * sigma ** 2.) - .5 * np.log(2. * np.pi * sigma ** 2.))
