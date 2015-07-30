# -*- coding: utf-8 -*-
"""
transitionModel.py introduces models for parameter variations.
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

class Static:
    def __init__(self):
        self.latticeConstant = None

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
        return self.computeForwardPrior(posterior, t)

class GaussianRandomWalk:
    def __init__(self, sigma=None):
        self.latticeConstant = None
        self.sigma = sigma

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
        newPrior = gaussian_filter1d(newPrior, self.sigma/self.latticeConstant[0])

        return newPrior

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t)

class ChangePoint:
    def __init__(self, tChange=None):
        self.latticeConstant = None
        self.tChange = tChange

    def __str__(self):
        return 'Change-point model'

    def computeForwardPrior(self, posterior, t):
        """
        Compute new prior from old posterior.

        :param posterior: grid-like posterior distribution
        :param t: integer time step
        :return: grid-like prior distribution
        """
        if t == self.tChange:
            return np.ones_like(posterior)/np.sum(np.ones_like(posterior))  # return flat distribution
        else:
            return posterior.copy()

    def computeBackwardPrior(self, posterior, t):
        return self.computeForwardPrior(posterior, t-1)
