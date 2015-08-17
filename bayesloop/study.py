# -*- coding: utf-8 -*-
"""
study.py introduces the main class of bayesloop.
"""

import numpy as np
from .preprocessing import *


class Study(object):
    def __init__(self):
        self.observationModel = None
        self.transitionModel = None

        self.gridSize = []
        self.boundaries = []
        self.marginalGrid = []
        self.grid = []
        self.latticeConstant = []

        self.rawData = []
        self.formattedData = []

        self.posteriorSequence = []
        self.posteriorMeanValues = []
        self.logEvidence = None
        self.localEvidence = []

        print '+ Created new study.'

    def loadExampleData(self):
        """
        Load UK coal mining disaster data.

        :return:
        """
        self.rawData = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                                 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                                 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                                 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                                 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                                 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                                 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

        print '    + Successfully imported example data.'

    def loadData(self, array):
        """
        Load Numpy array as data.

        :param array: numpy array containing data
        :return:
        """
        self.rawData = array
        print '    + Successfully imported array.'

    def createGrid(self):
        """
        Based on given boundaries and grid size, create parameter grid.

        :return:
        """

        self.marginalGrid = [np.linspace(b[0], b[1], g+2)[1:-1] for b, g in zip(self.boundaries, self.gridSize)]
        self.grid = [m for m in np.meshgrid(*self.marginalGrid, indexing='ij')]
        self.latticeConstant = [g[1]-g[0] for g in self.marginalGrid]

        if self.transitionModel != None:
            self.transitionModel.latticeConstant = self.latticeConstant

    def setBoundaries(self, newBoundaries):
        """
        Set lower and upper parameter boundaries.

        :param newBoundaries: list of lists with lower and upper parameter boundaries
        :return:
        """
        self.boundaries = newBoundaries
        self.createGrid()

    def setGridSize(self, newGridSize):
        """
        Set grid size for discretization of parameter distributions.

        :param newGridSize: list containing sizes for each grid dimension
        :return:
        """
        self.gridSize = newGridSize
        self.createGrid()

    def setObservationModel(self, M, silent=False):
        """
        Set observation model (likelihood function) for analysis.

        :param M: observation model class (see observationModel.py)
        :return:
        """
        self.observationModel = M

        self.gridSize = M.defaultGridSize
        self.boundaries = M.defaultBoundaries

        if not silent:
            print '    + Observation model:', M

    def setTransitionModel(self, K, silent=False):
        """
        Set transition model (for parameter variations).

        :param K: transition model class (see transitionModel.py)
        :return:
        """
        self.transitionModel = K
        self.transitionModel.latticeConstant = self.latticeConstant

        if not silent:
            print '    + Transition model:', K

    def fit(self, forwardOnly=False, evidenceOnly=False, silent=False):
        """
        Computes the posterior sequence and evidence of a data set + models
        """
        if not silent:
            print '+ Started new fit:'

        self.formattedData = movingWindow(self.rawData, self.observationModel.segmentLength)
        if not silent:
            print '    + Formatted data.'

        # initialize array for posterior distributions
        self.posteriorSequence = np.empty([len(self.formattedData)]+self.gridSize)

        # initialize array for computed evidence (marginal likelihood)
        self.logEvidence = 0
        self.localEvidence = np.empty(len(self.formattedData))

        # forward pass
        alpha = np.ones(self.gridSize)/np.prod(np.array(self.gridSize))  # initial flat prior
        for i in np.arange(0, len(self.formattedData)):

            # compute likelihood
            likelihood = self.observationModel.pdf(self.grid, self.formattedData[i])

            # update alpha based on likelihood
            alpha *= likelihood

            # normalization constant of alpha is used to compute evidence
            norm = np.sum(alpha)
            self.logEvidence += np.log(norm)
            self.localEvidence[i] = norm  # in case we return before backward pass (forwardOnly = True)

            # normalize alpha (for numerical stability)
            alpha /= norm

            # alphas are stored as preliminary posterior distributions
            self.posteriorSequence[i] = alpha

            # compute alpha for next iteration
            alpha = self.transitionModel.computeForwardPrior(alpha, i)

        if not silent:
            print '    + Finished forward pass.'
            print '    + Log10-evidence: {:.5f}'.format(self.logEvidence / np.log(10))

        if not forwardOnly:
            # backward pass
            beta = np.ones(self.gridSize)/np.prod(np.array(self.gridSize))  # initial flat prior
            for i in np.arange(0, len(self.formattedData))[::-1]:
                # posterior ~ alpha*beta
                self.posteriorSequence[i] *= beta  # alpha*beta

                # compute local evidence (before normalizing posterior wrt the parameters)
                self.localEvidence[i] = np.sum(self.posteriorSequence[i])
                self.posteriorSequence[i] /= self.localEvidence[i]

                # re-compute likelihood
                likelihood = self.observationModel.pdf(self.grid, self.formattedData[i])

                # compute beta for next iteration
                beta = self.transitionModel.computeBackwardPrior(beta*likelihood, i)

                # normalize beta (for numerical stability)
                beta /= np.sum(beta)

            if not silent:
                print '    + Finished backward pass.'

        # posterior mean values do not need to be computed for evidence
        if evidenceOnly:
            self.posteriorMeanValues = []
        else:
            self.posteriorMeanValues = np.empty([len(self.grid), len(self.posteriorSequence)])

            for i in range(len(self.grid)):
                self.posteriorMeanValues[i] = np.array([np.sum(p*self.grid[i]) for p in self.posteriorSequence])

            if not silent:
                print '    + Computed mean parameter values.'
