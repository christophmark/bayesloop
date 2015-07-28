# -*- coding: utf-8 -*-
"""
observationModel.py introduces likelihood functions.
"""
import numpy as np

class Poisson:
    def __init__(self):
        self.name = 'Poisson'

        self.segmentLength = 1 # number of measurements in one data segment
        self.defaultGridSize = [1000]
        self.defaultBoundaries = [[0, 1]]

    def pdf(self, grid, x):
        """
        Probability density function of the observational model

        :param grid: parameter grid
        :param x: data segment from formatted data
        :return: poisson pdf (shape like grid)
        """
        return (grid[0]**x[0])*(np.exp(-grid[0]))/(np.math.factorial(x[0]))
