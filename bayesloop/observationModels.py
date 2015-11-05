#!/usr/bin/env python
"""
This file introduces the observation models that can be used by the Study class for data analysis. An observation model
here refers to a likelihood function, stating the probability of a measurement at a certain time step, given the
parameter values.
"""

import numpy as np


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

        Parameters:
            grid - Discrete parameter grid
            dataSegment - Data segment from formatted data

        Returns:
            Discretized pdf as numpy array (with same shape as grid)
        """
        # check for multi-dimensional data
        if len(dataSegment.shape) == 2:
            # multi-dimensional data is processed one dimension at a time; likelihoods are then multiplied
            return np.prod(np.array([self.processedPdf(grid, d) for d in dataSegment.T]), axis=0)

        # check for missing data
        if np.isnan(dataSegment).any():
            if self.uninformativePdf is not None:
                return self.uninformativePdf  # arbitrary likelihood
            else:
                return np.ones_like(grid[0]) / np.sum(np.ones_like(grid[0]))  # uniform likelihood

        return self.pdf(grid, dataSegment)


class Custom(ObservationModel):
    """
    This observation model class allows to create new observation models on-the-fly from scipy.stats random variables.
    Note that scipy.stats does not use the canonical way of naming the parameters of the probability distributions, but
    instead includes the parameter 'loc' (for discrete & continuous distributions) and 'scale' (for continuous only).

    See http://docs.scipy.org/doc/scipy/reference/stats.html for further information on the available distributions and
    the parameter notation.

    Example usage:
        M = bl.observationModels.Custom(scipy.stats.norm, fixedParameters={'loc': 4})

    This will result in a model for normally distributed observations with a fixed 'loc' (mean) of 4, leaving the
    'scale' (standard deviation) as the only free parameter to be inferred.

    Note that while the parameters 'loc' and 'scale' have default values in scipy.stats and do not necessarily need to
    be set, they have to be added to the fixedParameters dictionary in bayesloop to be treated as a constant.
    """

    def __init__(self, rv, fixedParameters={}):
        self.rv = rv
        self.fixedParameterDict = fixedParameters

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
        print '+ Created custom observation model with {0} free parameter(s).'.format(len(self.freeParameters))

        # set class attributes similar to other observation models
        self.name = rv.name  # scipy.stats name is used
        self.segmentLength = 1  # currently only independent observations are supported by Custom class
        self.parameterNames = self.freeParameters
        self.defaultGridSize = [1000]*len(self.parameterNames)
        self.defaultBoundaries = [[0, 1]]*len(self.parameterNames)
        self.defaultPrior = None
        self.uninformativePdf = None

    def pdf(self, grid, dataSegment):
        """
        Probability density function of custom scipy.stats models

        Parameters:
            grid - Parameter grid for discrete rate values
            dataSegment - Data segment from formatted data

        Returns:
            Discretized pdf as numpy array (with same shape as grid)
        """
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
        self.defaultBoundaries = [[0, 1]]
        self.defaultPrior = lambda x: 1./np.sqrt(x*(1.-x))  # Jeffreys prior
        self.uninformativePdf = None

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Bernoulli model

        Parameters:
            grid - Parameter grid for discrete values of the parameter p
            dataSegment - Data segment from formatted data (in this case a single number of events)

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
        self.defaultBoundaries = [[0, 1]]
        self.defaultPrior = lambda x: np.sqrt(1./x)  # Jeffreys prior
        self.uninformativePdf = None

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Poisson model

        Parameters:
            grid - Parameter grid for discrete rate (lambda) values
            dataSegment - Data segment from formatted data (in this case a single number of events)

        Returns:
            Discretized Poisson pdf as numpy array (with same shape as grid)
        """
        return (grid[0] ** dataSegment[0]) * (np.exp(-grid[0])) / (np.math.factorial(dataSegment[0]))


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
        self.defaultBoundaries = [[-1, 1], [0, 1]]
        self.defaultPrior = None
        self.uninformativePdf = None

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Gaussian model.

        Parameters:
            grid - Parameter grid for discrete values of mean and standard deviation
            dataSegment - Data segment from formatted data (containing a single measurement)

        Returns:
            Discretized Normal pdf as numpy array (with same shape as grid).
        """
        return np.exp(
            -((dataSegment[0] - grid[0]) ** 2.) / (2. * grid[1] ** 2.) - .5 * np.log(2. * np.pi * grid[1] ** 2.))


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
        self.defaultBoundaries = [[0, 1]]
        self.defaultPrior = None
        self.uninformativePdf = None

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the white noise process.

        Parameters:
            grid - Parameter grid for discrete values of noise amplitude
            dataSegment - Data segment from formatted data (containing a single measurement)

        Returns:
            Discretized Normal pdf (with zero mean) as numpy array (with same shape as grid).
        """
        return np.exp(-(dataSegment[0] ** 2.) / (2. * grid[0] ** 2.) - .5 * np.log(2. * np.pi * grid[0] ** 2.))


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
        self.defaultBoundaries = [[-1, 1], [0, 1]]
        self.defaultPrior = None
        self.uninformativePdf = None

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Auto-regressive process of first order

        Parameters:
            grid - Parameter grid for discerete values of the correlation coefficient and noise amplitude
            dataSegment - Data segment from formatted data (in this case a pair of measurements)

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
        self.defaultBoundaries = [[-1, 1], [0, 1]]
        self.defaultPrior = None
        self.uninformativePdf = None

    def pdf(self, grid, dataSegment):
        """
        Probability density function of the Auto-regressive process of first order

        Parameters:
            grid - Parameter grid for discerete values of the correlation coefficient and standard deviation
            dataSegment - Data segment from formatted data (in this case a pair of measurements)

        Returns:
            Discretized pdf (for data point d_t, given d_(t-1) and parameters) as numpy array (with same shape as grid).
        """
        r = grid[0]
        s = grid[1]
        sScaled = s*np.sqrt(1 - r**2.)
        return np.exp(-((dataSegment[1] - r * dataSegment[0]) ** 2.) / (2. * sScaled ** 2.) - .5 * np.log(
            2. * np.pi * sScaled ** 2.))
