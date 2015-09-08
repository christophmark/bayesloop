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
        # check for multi-dimensional data
        if len(dataSegment.shape) == 2:
            # multi-dimensional data is processed one dimension at a time; likelihoods are then multiplied
            return np.prod(np.array([self.pdf(grid, d) for d in dataSegment.T]), axis=0)

        # check for missing data
        if np.isnan(dataSegment).any():
            if self.uninformativePdf is not None:
                return self.uninformativePdf  # arbitrary likelihood
            else:
                return np.ones_like(grid[0]) / np.sum(np.ones_like(grid[0]))  # uniform likelihood

        return self.pdf(grid, dataSegment)


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
