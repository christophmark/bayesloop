# -*- coding: utf-8 -*-
"""
transitionModel.py introduces models for parameter variations.
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class Static:
    def __init__(self):
        self.latticeConstant = None
        self.hyperParameters = {}

    def __str__(self):
        return 'Static/constant parameter values'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior.

        :param posterior: grid-like posterior distribution
        :param t: integer time step
        :return: grid-like prior distribution
        """
        return posterior.copy()

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class GaussianRandomWalk:
    def __init__(self, sigma=None):
        self.latticeConstant = None
        self.hyperParameters = {'sigma': sigma}

    def __str__(self):
        return 'Gaussian random walk'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior.

        :param posterior: grid-like posterior distribution
        :param t: integer time step
        :return: grid-like prior distribution
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
    def __init__(self, tChange=None):
        self.latticeConstant = None
        self.hyperParameters = {'tChange': tChange}

    def __str__(self):
        return 'Change-point model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior.

        :param posterior: grid-like posterior distribution
        :param t: integer time step
        :return: grid-like prior distribution
        """
        if t == self.hyperParameters['tChange']:
            return np.ones_like(posterior) / np.sum(np.ones_like(posterior))  # return flat distribution
        else:
            return posterior.copy()

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class RegimeSwitch:
    def __init__(self, pMin=None):
        self.latticeConstant = None
        self.hyperParameters = {'pMin': pMin}

    def __str__(self):
        return 'Regime-switching model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior.

        :param posterior: grid-like posterior distribution
        :param t: integer time step
        :return: grid-like prior distribution
        """
        newPrior = posterior.copy()
        newPrior[newPrior < self.hyperParameters['pMin']] = self.hyperParameters['pMin']

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t - 1)


class CombinedTransitionModel:
    def __init__(self, *args):
        self.latticeConstant = None
        self.models = args

    def __str__(self):
        return 'Combined transition model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior.

        :param posterior: grid-like posterior distribution
        :param t: integer time step
        :return: grid-like prior distribution
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
