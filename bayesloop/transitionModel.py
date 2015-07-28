# -*- coding: utf-8 -*-
"""
transitionModel.py introduces models for parameter variations.
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

class Static:
    def __init__(self, sigma=None):
        self.latticeConstant = None
        self.name = 'Static/constant parameter values'

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
        self.name = 'Gaussian random walk'

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
