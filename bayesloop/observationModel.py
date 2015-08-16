# -*- coding: utf-8 -*-
"""
observationModel.py introduces likelihood functions.
"""
import numpy as np

class Poisson:
    def __init__(self):
        self.segmentLength = 1  # number of measurements in one data segment
        self.defaultGridSize = [1000]
        self.defaultBoundaries = [[0, 1]]
        self.uninformativePdf = None

    def __str__(self):
        return 'Poisson'

    def pdf(self, grid, x):
        """
        Probability density function of the observational model

        :param grid: parameter grid
        :param x: data segment from formatted data
        :return: poisson pdf (shape like grid)
        """

        # check for missing data
        if np.isnan(x[0]):
            if self.uninformativePdf is not None:
                return self.uninformativePdf  # arbitrary likelihood
            else:
                return np.ones_like(grid[0])/np.sum(np.ones_like(grid[0]))  # uniform likelihood

        return (grid[0]**x[0])*(np.exp(-grid[0]))/(np.math.factorial(x[0]))

class AR1:
    def __init__(self):
        self.segmentLength = 2  # number of measurements in one data segment
        self.defaultGridSize = [200, 200]
        self.defaultBoundaries = [[-1, 1], [0, 1]]
        self.uninformativePdf = None

    def __str__(self):
        return 'Autoregressive process of first order (AR1)'

    def pdf(self, grid, x):
        """
        Probability density function of the observational model

        :param grid: parameter grid
        :param x: data segment from formatted data
        :return: pdf for the autoregressive process of first order (shape like grid)
        """

        # check for missing data
        if np.isnan(x[0]) or np.isnan(x[1]):
            if self.uninformativePdf is not None:
                return self.uninformativePdf  # arbitrary likelihood
            else:
                return np.ones_like(grid[0])/np.sum(np.ones_like(grid[0]))  # uniform likelihood

        return np.exp(-((x[1] - grid[0]*x[0])**2.)/(2.*grid[1]**2.) - np.log(2.*np.pi*grid[1]**2.))
