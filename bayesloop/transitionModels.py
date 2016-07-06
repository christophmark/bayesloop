#!/usr/bin/env python
"""
This file introduces the transition models that can be used by the Study class for data analysis. A transition model
here refers to a stochastic or deterministic model that describes how the parameter values of a given time series
model change from one time step to another. The transition model can thus be compared to the state transition matrix
of Hidden Markov models. However, instead of explicitly stating transition probabilities for all possible states, a
transformation is defined that alters the distribution of the model parameters in one time step according to the
transition model. This altered distribution is subsequently used as a prior distribution in the next time step.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.ndimage.interpolation import shift
from collections import OrderedDict


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

        Parameters:
            posterior - Parameter distribution (numpy array shaped according to grid size) from current time step
            t - integer time step

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

    Parameters (on initialization):
        sigma - Float or list of floats defining the standard deviation of the Gaussian random walk for each parameter
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

        Parameters:
            posterior - Parameter distribution (numpy array shaped according to grid size) from current time step
            t - integer time step

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

    Parameters (on initialization):
        c - Float or list of floats defining the width of the distribution (c >= 0).
        alpha - Float or list of floats defining the shape of the distribution (0 < alpha <= 2).
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

        Parameters:
            posterior - Parameter distribution (numpy array shaped according to grid size) from current time step
            t - integer time step

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
        gs = self.study.gridSize
        if len(gs) == 2:
            if axis == 1:
                l1 = gs[1]
                l2 = gs[0]
            elif axis == 0:
                l1 = gs[0]
                l2 = gs[1]
            else:
                print '! Transformation axis must either be 0 or 1.'
                return
        elif len(gs) == 1:
            l1 = gs[0]
            l2 = 0
            axis = 0
        else:
            print '! Parameter grid must either be 1- or 2-dimensional.'
            return

        kernel_fft = np.exp(-np.abs(c*np.linspace(0, np.pi, 3*l1/2+1))**alpha)
        kernel = np.fft.irfft(kernel_fft)
        kernel = np.roll(kernel, 3*l1/2-1)

        if len(gs) == 2:
            kernel = np.array([kernel]*(3*l2))

        if axis == 1:
            return kernel.T
        elif axis == 0:
            return kernel

    def convolve(self, distribution):
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

    Parameters (on initialization):
        tChange - Integer value of the time step of the change point
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

        Parameters:
            posterior - Parameter distribution (numpy array shaped according to grid size) from current time step
            t - integer time step

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
    within the set boundaries are assigned a minimal probability of being realized in the next time step, effectively
    allowing abrupt parameter changes at every time step.

    Parameters (on initialization):
        log10pMin - Minimal probability (on a log10 scale) that is assigned to every parameter value
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
            posterior - Parameter distribution (numpy array shaped according to grid size) from current time step
            t - integer time step

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """
        newPrior = posterior.copy()
        newPrior[newPrior < 10**self.hyperParameters['log10pMin']] = 10**self.hyperParameters['log10pMin']

        # transformation above violates proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class Linear:
    """
    Linear deterministic model. This model assumes a constant change of parameter values, resulting in a linear
    parameter evolution. Note that this model is entirely deterministic, as the slope for all parameters has to be
    entered by the user. However, the slope can be optimized by maximizing the model evidence.

    Parameters (on initialization):
        slope - Float or list of floats defining the change in parameter value for each time step for all parameters
    """
    def __init__(self, slope=None, param=None):
        self.study = None
        self.latticeConstant = None
        self.hyperParameters = OrderedDict([('slope', slope)])
        self.selectedParameter = param
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Linear deterministic model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Parameters:
            posterior - Parameter distribution (numpy array shaped according to grid size) from current time step
            t - integer time step

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """
        # slope has to be adjusted based on grid size
        normedSlope = []
        if type(self.hyperParameters['slope']) is not list:
            for c in self.latticeConstant:
                normedSlope.append(self.hyperParameters['slope'] / c)
        else:
            for i, c in enumerate(self.latticeConstant):
                normedSlope.append(self.hyperParameters['slope'][i] / c)

        # check if only one axis is to be transformed
        if self.selectedParameter is not None:
            axisToTransform = self.study.observationModel.parameterNames.index(self.selectedParameter)
            selectedSlope = normedSlope[axisToTransform]

            # reinitiate slope list (setting only the selected axis to a non-zero value)
            normedSlope = [0]*len(normedSlope)
            normedSlope[axisToTransform] = selectedSlope

        # shift interpolated version of distribution according to slope
        newPrior = shift(posterior, normedSlope, order=3, mode='nearest')

        # transformation above may violate proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        # slope has to be adjusted based on grid size
        normedSlope = []
        if type(self.hyperParameters['slope']) is not list:
            for c in self.latticeConstant:
                normedSlope.append(self.hyperParameters['slope'] / c)
        else:
            for i, c in enumerate(self.latticeConstant):
                normedSlope.append(self.hyperParameters['slope'][i] / c)

        # shift interpolated version of distribution according to negative (!) slope
        newPrior = shift(posterior, -np.array(normedSlope), order=3, mode='nearest')

        # transformation above may violate proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior


class Quadratic:
    """
    Quadratic deterministic model. This model assumes a quadratic change of parameter values. Note that this model is
    entirely deterministic, as the coefficient for all parameters has to be entered by the user. However, the
    coefficient can be optimized by maximizing the model evidence.

    Parameters (on initialization):
        coefficient - multiplicative coefficient for quadratic model: coefficient*x^2
    """
    def __init__(self, coefficient=None, param=None):
        self.study = None
        self.latticeConstant = None
        self.hyperParameters = OrderedDict([('coefficient', coefficient)])
        self.selectedParameter = param
        self.tOffset = 0  # is set to the time of the last Breakpoint by SerialTransition model

    def __str__(self):
        return 'Quadratic deterministic model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Parameters:
            posterior - Parameter distribution (numpy array shaped according to grid size) from current time step
            t - integer time step

        Returns:
            Prior parameter distribution for subsequent time step (numpy array shaped according to grid size)
        """
        # slope has to be adjusted based on grid size
        normedCoeff = []
        if type(self.hyperParameters['coefficient']) is not list:
            for c in self.latticeConstant:
                normedCoeff.append(self.hyperParameters['coefficient'] / c)
        else:
            for i, c in enumerate(self.latticeConstant):
                normedCoeff.append(self.hyperParameters['coefficient'][i] / c)

        # check if only one axis is to be transformed
        if self.selectedParameter is not None:
            axisToTransform = self.study.observationModel.parameterNames.index(self.selectedParameter)
            selectedCoeff = normedCoeff[axisToTransform]

            # reinitiate coefficient list (setting only the selected axis to a non-zero value)
            normedCoeff = [0]*len(normedCoeff)
            normedCoeff[axisToTransform] = selectedCoeff

        # shift interpolated version of distribution according to coefficient
        newPrior = shift(posterior, np.array(normedCoeff)*(t-self.tOffset)**2., order=3, mode='nearest')

        # transformation above may violate proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        # coefficient has to be adjusted based on grid size
        normedCoeff = []
        if type(self.hyperParameters['coefficient']) is not list:
            for c in self.latticeConstant:
                normedCoeff.append(self.hyperParameters['coefficient'] / c)
        else:
            for i, c in enumerate(self.latticeConstant):
                normedCoeff.append(self.hyperParameters['coefficient'][i] / c)

        # shift interpolated version of distribution according to negative (!) coefficient
        newPrior = shift(posterior, -np.array(normedCoeff)*(t-self.tOffset)**2., order=3, mode='nearest')

        # transformation above may violate proper normalization; re-normalization needed
        newPrior /= np.sum(newPrior)

        return newPrior


class CombinedTransitionModel:
    """
    Combined transition model. This class allows to combine different transition models to be able to explore more
    complex parameter dynamics. All sub-models are passed to this class as arguments on initialization. Note that a
    different order of the sub-models can result in different parameter dynamics.

    Parameters (on initialization):
        args - Sequence of transition models
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

        Parameters:
            posterior - Parameter distribution (numpy array shaped according to grid size) from current time step
            t - integer time step

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

    Parameters (on initialization):
        args - Sequence of transition models and integer time steps for structural breaks
               (for n models, n-1 time steps have to be provided)

    Usage example:
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
            print '! Time steps for structural breaks ave to be passed in monotonically increasing order.'

        # check: n models require n-1 break times
        if not (len(self.models)-1 == len(self.hyperParameters['tBreak'])):
            print '! Wrong number of structural breaks/models. For n models, n-1 structural breaks are required.'

    def __str__(self):
        return 'Serial transition model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior (moving forwards in time).

        Parameters:
            posterior - Parameter distribution (numpy array shaped according to grid size) from current time step
            t - integer time step

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
