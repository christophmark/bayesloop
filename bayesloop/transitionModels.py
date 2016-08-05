#!/usr/bin/env python
"""
This file introduces the transition models that can be used by the Study class for data analysis. A transition model
here refers to a stochastic or deterministic model that describes how the parameter values of a given time series
model change from one time step to another. The transition model can thus be compared to the state transition matrix
of Hidden Markov models. However, instead of explicitly stating transition probabilities for all possible states, a
transformation is defined that alters the distribution of the model parameters in one time step according to the
transition model. This altered distribution is subsequently used as a prior distribution in the next time step.
"""

from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.ndimage.interpolation import shift
from collections import OrderedDict
from inspect import getargspec
from .exceptions import *


class Static:
    """
    Static transition model. This trivial model assumes no change of parameter values over time.
    """
    def __init__(self):
        self.study = None
        self.latticeConstant = None
        self.hyperParameters = {}
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Static/constant parameter values'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior: Parameter distribution (numpy array shaped according to grid size) from current time step
            t: integer time step

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """
        return posterior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class GaussianRandomWalk:
    """
    Gaussian random walk model. This model assumes that parameter changes are Gaussian-distributed. The standard
    deviation can be set individually for each model parameter.

    Args:
        sigma: Float or list of floats defining the standard deviation of the Gaussian random walk for each parameter
    """
    def __init__(self, sigma=None, param=None):
        self.study = None
        self.latticeConstant = None
        self.hyperParameters = OrderedDict([('sigma', sigma)])
        self.selectedParameter = param
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Gaussian random walk'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior: Parameter distribution (numpy array shaped according to grid size) from current time step
            t: integer time step

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """

        normedSigma = []
        if type(self.hyperParameters['sigma']) is not list:
            for c in self.latticeConstant:
                normedSigma.append(self.hyperParameters['sigma'] / c)
        else:
            for i, c in enumerate(self.latticeConstant):
                normedSigma.append(self.hyperParameters['sigma'][i] / c)

        # check if only one axis is to be transformed
        if self.selectedParameter is not None:
            axisToTransform = self.study.observationModel.parameterNames.index(self.selectedParameter)
            selectedSigma = normedSigma[axisToTransform]
            if selectedSigma < 0.0:  # gaussian_filter1d cannot handle negative st.dev.
                selectedSigma = 0.0
            newPrior = gaussian_filter1d(posterior, selectedSigma, axis=axisToTransform)
        else:
            newPrior = gaussian_filter(posterior, normedSigma)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class AlphaStableRandomWalk:
    """
    Alpha stable random walk model. This model assumes that parameter changes are distributed according to the symmetric
    alpha-stable distribution. For each parameter, two hyper-parameters can be set: the width of the distribution (c)
    and the shape (alpha).

    Args:
        c: Float or list of floats defining the width of the distribution (c >= 0).
        alpha: Float or list of floats defining the shape of the distribution (0 < alpha <= 2).
    """
    def __init__(self, c=None, alpha=None, param=None):
        self.study = None
        self.latticeConstant = None
        self.hyperParameters = OrderedDict([('c', c), ('alpha', alpha)])
        self.selectedParameter = param
        self.kernel = None
        self.kernelParameters = None
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Alpha-stable random walk'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior: Parameter distribution (numpy array shaped according to grid size) from current time step
            t: integer time step

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """

        # if hyper-parameter values have changed, a new convolution kernel needs to be created
        if not self.kernelParameters == self.hyperParameters:
            normedC = []
            if type(self.hyperParameters['c']) is not list:
                for lc in self.latticeConstant:
                    normedC.append(self.hyperParameters['c'] / lc)
            else:
                for i, lc in enumerate(self.latticeConstant):
                    normedC.append(self.hyperParameters['c'][i] / lc)

            if type(self.hyperParameters['alpha']) is not list:
                alpha = [self.hyperParameters['alpha']] * len(normedC)
            else:
                alpha = self.hyperParameters['alpha']

            # check if only one axis is to be transformed
            if self.selectedParameter is not None:
                axisToTransform = self.study.observationModel.parameterNames.index(self.selectedParameter)
                selectedC = normedC[axisToTransform]
                normedC = [0.]*len(normedC)
                normedC[axisToTransform] = selectedC

            self.kernel = self.createKernel(normedC[0], alpha[0], 0)
            for i, (a, c) in enumerate(zip(alpha, normedC)[1:]):
                self.kernel *= self.createKernel(c, a, i+1)

            self.kernel = self.kernel.T
            self.kernelParameters = self.hyperParameters

        newPrior = self.convolve(posterior)
        newPrior /= np.sum(newPrior)
        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)

    def createKernel(self, c, alpha, axis):
        """
        Create alpha-stable distribution on a grid as a kernel for concolution.

        Args:
            c: Scale parameter.
            alpha: Tail parameter (alpha = 1: Cauchy, alpha = 2: Gauss)
            axis: Axis along which the distribution is defined, for 2D-Kernels

        Returns:
            Numpy array containing kernel.
        """
        gs = self.study.gridSize
        if len(gs) == 2:
            if axis == 1:
                l1 = gs[1]
                l2 = gs[0]
            elif axis == 0:
                l1 = gs[0]
                l2 = gs[1]
            else:
                raise ConfigurationError('Transformation axis must either be 0 or 1.')
        elif len(gs) == 1:
            l1 = gs[0]
            l2 = 0
            axis = 0
        else:
            raise ConfigurationError('Parameter grid must either be 1- or 2-dimensional.')

        kernel_fft = np.exp(-np.abs(c*np.linspace(0, np.pi, int(3*l1/2+1)))**alpha)
        kernel = np.fft.irfft(kernel_fft)
        kernel = np.roll(kernel, int(3*l1/2-1))

        if len(gs) == 2:
            kernel = np.array([kernel]*(3*l2))

        if axis == 1:
            return kernel.T
        elif axis == 0:
            return kernel

    def convolve(self, distribution):
        """
        Convolves distribution with alpha-stabel kernel.

        Args:
            distribution: Discrete probability distribution to convolve.

        Returns:
            Numpy array containing convolution.
        """
        gs = np.array(self.study.gridSize)
        padded_distribution = np.zeros(3*np.array(gs))
        if len(gs) == 2:
            padded_distribution[gs[0]:2*gs[0], gs[1]:2*gs[1]] = distribution
        elif len(gs) == 1:
            padded_distribution[gs[0]:2*gs[0]] = distribution

        padded_convolution = fftconvolve(padded_distribution, self.kernel, mode='same')
        if len(gs) == 2:
            convolution = padded_convolution[gs[0]:2*gs[0], gs[1]:2*gs[1]]
        elif len(gs) == 1:
            convolution = padded_convolution[gs[0]:gs[0]]

        return convolution


class ChangePoint:
    """
    Change-point model. Parameter values are allowed to change only at a single point in time, right after a specified
    time step (Hyper-parameter tChange). Note that a uniform parameter distribution is used at this time step to
    achieve this "reset" of parameter values.

    Args:
        tChange: Integer value of the time step of the change point
    """
    def __init__(self, tChange=None):
        self.study = None
        self.latticeConstant = None
        self.hyperParameters = OrderedDict([('tChange', tChange)])
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Change-point model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior: Parameter distribution (numpy array shaped according to grid size) from current time step
            t: integer time step

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """
        if t == self.hyperParameters['tChange']:
            # check if custom prior is used by observation model
            if self.study.observationModel.prior is not None:
                prior = self.study.observationModel.prior(*self.study.grid)
            else:
                prior = np.ones(self.study.gridSize)  # flat prior

            # normalize prior (necessary in case an improper prior is used)
            prior /= np.sum(prior)
            return prior
        else:
            return posterior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class RegimeSwitch:
    """
    Regime-switching model. In case the number of change-points in a given data set is unknown, the regime-switching
    model may help to identify potential abrupt changes in parameter values. At each time step, all parameter values
    within the set boundaries are assigned a minimal probability density of being realized in the next time step,
    effectively allowing abrupt parameter changes at every time step.

    Args:
        log10pMin: Minimal probability density (log10 value) that is assigned to every parameter value
    """
    def __init__(self, log10pMin=None):
        self.study = None
        self.latticeConstant = None
        self.hyperParameters = OrderedDict([('log10pMin', log10pMin)])
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Regime-switching model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Parameters:
            posterior: Parameter distribution (numpy array shaped according to grid size) from current time step
            t: integer time step

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """
        newPrior = posterior.copy()
        limit = (10**self.hyperParameters['log10pMin'])*np.prod(self.latticeConstant)  # convert prob. density to prob.
        newPrior[newPrior < limit] = limit

        # transformation above violates proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class Deterministic:
    """
    Generic deterministic model. Given a function with time as the first argument and further  keyword-arguments as
    hyper-parameters, plus the name of a parameter of the observation model that is supposed to follow this function
    over time, this transition model shifts the parameter distribution accordingly. Note that these models are entirely
    deterministic, as the hyper-parameter values are entered by the user. However, the hyper-parameter distributions can
    be inferred using a Hyper-study or can be optimized using the 'optimize' method of the Study class.

    Args:
        function: A function that takes the time as its first argument and further takes keyword-arguments that
            correspond to the hyper-parameters of the transition model which the function defines.
        param: The observation model parameter that is manipulated according to the function defined above.

    Example:
        def quadratic(t, a=0, b=0):
            return a*(t**2) + b*t

        S = bl.Study()
        ...
        S.setObservationModel(bl.om.Gaussian())
        S.setTransitionModel(bl.tm.Deterministic(quadratic, param='standard deviation'))
    """
    def __init__(self, function=None, param=None):
        self.study = None
        self.latticeConstant = None
        self.function = function
        self.selectedParameter = param
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

        # create ordered dictionary of hyper-parameters from keyword-arguments of function
        self.hyperParameters = OrderedDict()
        argspec = getargspec(self.function)

        # only keyword arguments are allowed
        if not len(argspec.args) == len(argspec.defaults)+1:
            raise ConfigurationError('Function to define deterministic transition model can only contain one '
                                     'non-keyword argument (time; first argument) and keyword-arguments '
                                     '(hyper-parameters) with default values.')

        # define hyper-parameters of transition model
        self.hyperParameters = OrderedDict()
        for arg, default in zip(argspec.args[1:], argspec.defaults):
            self.hyperParameters[arg] = default

    def __str__(self):
        return 'Deterministic model ({})'.format(self.function.__name__)

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior: Parameter distribution (numpy array shaped according to grid size) from current time step
            t: time stamp (integer time index by default)

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """

        # compute normed shift of grid values
        normedHyperParameters = self.hyperParameters.copy()
        for key, value in self.hyperParameters.items():
            if type(value) is not list:
                normedHyperParameters[key] = [value / c for c in self.latticeConstant]
            else:
                normedHyperParameters[key] = [v / c for v, c in zip(value, self.latticeConstant)]

        d = []
        for i in range(len(self.latticeConstant)):
            params = {key: value[i] for (key, value) in normedHyperParameters.items()}
            ftp1 = self.function(t + 1 - self.tOffset, **params)
            ft = self.function(t-self.tOffset, **params)
            d.append(ftp1-ft)

        # check if only one axis is to be transformed
        if self.selectedParameter is not None:
            axisToTransform = self.study.observationModel.parameterNames.index(self.selectedParameter)
            selectedD = d[axisToTransform]

            # reinitiate coefficient list (setting only the selected axis to a non-zero value)
            d = [0] * len(d)
            d[axisToTransform] = selectedD

        # shift interpolated version of distribution
        newPrior = shift(posterior, d, order=3, mode='nearest')

        # transformation above may violate proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        # compute normed shift of grid values
        normedHyperParameters = self.hyperParameters.copy()
        for key, value in self.hyperParameters.items():
            if type(value) is not list:
                normedHyperParameters[key] = [value / c for c in self.latticeConstant]
            else:
                normedHyperParameters[key] = [v / c for v, c in zip(value, self.latticeConstant)]

        d = []
        for i in range(len(self.latticeConstant)):
            params = {key: value[i] for (key, value) in normedHyperParameters.items()}
            ftm1 = self.function(t - 1 - self.tOffset, **params)
            ft = self.function(t - self.tOffset, **params)
            d.append(ftm1 - ft)

        # check if only one axis is to be transformed
        if self.selectedParameter is not None:
            axisToTransform = self.study.observationModel.parameterNames.index(self.selectedParameter)
            selectedD = d[axisToTransform]

            # reinitiate coefficient list (setting only the selected axis to a non-zero value)
            d = [0] * len(d)
            d[axisToTransform] = selectedD

        # shift interpolated version of distribution
        newPrior = shift(posterior, d, order=3, mode='nearest')

        # transformation above may violate proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior


class CombinedTransitionModel:
    """
    Combined transition model. This class allows to combine different transition models to be able to explore more
    complex parameter dynamics. All sub-models are passed to this class as arguments on initialization. Note that a
    different order of the sub-models can result in different parameter dynamics.

    Args:
        *args: Sequence of transition models
    """
    def __init__(self, *args):
        self.study = None
        self.latticeConstant = None
        self.models = args
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Combined transition model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior: Parameter distribution (numpy array shaped according to grid size) from current time step
            t: integer time step

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """
        newPrior = posterior.copy()

        for m in self.models:
            m.latticeConstant = self.latticeConstant  # latticeConstant needs to be propagated to sub-models
            m.study = self.study  # study needs to be propagated to sub-models
            m.tOffset = self.tOffset
            newPrior = m.computeForwardPrior(newPrior, t)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        newPrior = posterior.copy()

        for m in self.models:
            m.latticeConstant = self.latticeConstant
            m.study = self.study
            m.tOffset = self.tOffset
            newPrior = m.computeBackwardPrior(newPrior, t)

        return newPrior


class SerialTransitionModel:
    """
    Serial transition model. To model fundamental changes in parameter dynamics, different transition models can be
    serially coupled. Depending on the time step, a corresponding sub-model is chosen to compute the new prior
    distribution from the posterior distribution.

    Args:
        *args: Sequence of transition models and integer time steps for structural breaks
            (for n models, n-1 time steps have to be provided)

    Example:
        K = bl.transitionModels.SerialTransitionModel(bl.transitionModels.Static(),
                                                      50,
                                                      bl.transitionModels.RegimeSwitch(log10pMin=-7),
                                                      100,
                                                      bl.transitionModels.GaussianRandomWalk(sigma=0.2))

        In this example, parameters are assumed to be constant until time step 50, followed by a regime-switching-
        process until time step 100. Finally, we assume Gaussian parameter fluctuations until the last time step. Note
        that models and time steps do not necessarily have to be passed in an alternating way.
    """
    def __init__(self, *args):
        self.study = None
        self.latticeConstant = None

        # determine time steps of structural breaks and store them as hyper-parameter 'tBreak'
        self.hyperParameters = OrderedDict([('tBreak', [t for t in args if isinstance(t, int)])])

        # determine sub-models corresponding to break times
        self.models = [m for m in args if not isinstance(m, int)]

        # check: break times have to be passed in monotonically increasing order
        if not all(x < y for x, y in zip(self.hyperParameters['tBreak'], self.hyperParameters['tBreak'][1:])):
            raise ConfigurationError('Time steps for structural breaks have to be passed in monotonically increasing '
                                     'order.')

        # check: n models require n-1 break times
        if not (len(self.models)-1 == len(self.hyperParameters['tBreak'])):
            raise ConfigurationError('Wrong number of structural breaks/models. For n models, n-1 structural breaks '
                                     'are required.')

    def __str__(self):
        return 'Serial transition model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Args:
            posterior: Parameter distribution (numpy array shaped according to grid size) from current time step
            t: integer time step

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """
        # the index of the model to choose at time t is given by the number of break times <= t
        modelIndex = np.sum(np.array(self.hyperParameters['tBreak']) <= t)

        self.models[modelIndex].latticeConstant = self.latticeConstant  # latticeConstant needs to be propagated
        self.models[modelIndex].study = self.study  # study needs to be propagated
        self.models[modelIndex].tOffset = self.hyperParameters['tBreak'][modelIndex-1] if modelIndex > 0 else 0
        newPrior = self.models[modelIndex].computeForwardPrior(posterior, t)
        return newPrior

    def computeBackwardPrior(self, posterior, t):
        # the index of the model to choose at time t is given by the number of break times <= t
        modelIndex = np.sum(np.array(self.hyperParameters['tBreak']) <= t-1)

        self.models[modelIndex].latticeConstant = self.latticeConstant  # latticeConstant needs to be propagated
        self.models[modelIndex].study = self.study  # study needs to be propagated
        self.models[modelIndex].tOffset = self.hyperParameters['tBreak'][modelIndex-1] if modelIndex > 0 else 0
        newPrior = self.models[modelIndex].computeBackwardPrior(posterior, t)
        return newPrior
