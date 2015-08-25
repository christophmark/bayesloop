#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file introduces the transition models that can be used by the Study class for data analysis. A transition model
here referes to a stochastic or deterministic model that describes how the parameter values of a given time series
model change from one time step to another. The transition model can thus be compared to the state transition matrix
of Hidden Markov models. However, instead of explicitely stating transition probabilities for all possible states, a
transformation is defined that alters the distribution of the model parameters in one time step according to the
transition model. This altered distribution is subsequently used as a prior distribution in the next time step.
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter


class Static:
    """
    Static transition model. This trivial model assumes no change of parameter values over time.
    """
    def __init__(self):
        self.latticeConstant = None
        self.hyperParameters = {}

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
        return posterior.copy()

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class GaussianRandomWalk:
    """
    Gaussian random walk model. This model assumes that parameter changes are Gaussian-distributed. The standard
    deviation can be set individually for each model parameter.
    """
    def __init__(self, sigma=None):
        self.latticeConstant = None
        self.hyperParameters = {'sigma': sigma}

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
        newPrior = posterior.copy()

        normedSigma = []
        if type(self.hyperParameters['sigma']) is not list:
            for c in self.latticeConstant:
                normedSigma.append(self.hyperParameters['sigma'] / c)
        else:
            for i, c in enumerate(self.latticeConstant):
                normedSigma.append(self.hyperParameters['sigma'][i] / c)


        newPrior = gaussian_filter(newPrior, normedSigma)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class ChangePoint:
    """
    Change-point model. Parameter values are allowed to change only at a single point in time, right after a specified
    time step (Hyper-parameter tChange). Note that a uniform parameter distribution is used at this time step to
    achieve this "reset" of parameter values.
    """
    def __init__(self, tChange=None):
        self.latticeConstant = None
        self.hyperParameters = {'tChange': tChange}

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
            return np.ones_like(posterior) / np.sum(np.ones_like(posterior))  # return flat distribution
        else:
            return posterior.copy()

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class RegimeSwitch:
    """
    Regime-switching model. In case the number of change-points in a given data set are unknown, the regime-switching
    model may help to identify potential abrupt changes in parameter values. At each time step, all parameter values
    within the set boundaries are assigned a minimal probability of being realized in the next time step, effectively
    allowing abrupt parameter changes at every time step.
    """
    def __init__(self, pMin=None):
        self.latticeConstant = None
        self.hyperParameters = {'pMin': pMin}

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
        newPrior[newPrior < self.hyperParameters['pMin']] = self.hyperParameters['pMin']

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class CombinedTransitionModel:
    """
    Combined transition model. This class allows to combine different transition models to be able to explore more
    complex parameter dynamics. All sub-models are passed to this class as arguments on initialization. Note that a
    different order of the sub-models can result in different parameter dynamics.
    """
    def __init__(self, *args):
        self.latticeConstant = None
        self.models = args

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
            newPrior = m.computeForwardPrior(newPrior, t)

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        newPrior = posterior.copy()

        for m in self.models:
            m.latticeConstant = self.latticeConstant
            newPrior = m.computeBackwardPrior(newPrior, t)

        return newPrior
