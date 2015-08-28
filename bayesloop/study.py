#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file introduces the main class used for data analysis.
"""

import numpy as np
from scipy.optimize import minimize
from .preprocessing import *
from .transitionModel import CombinedTransitionModel


class Study(object):
    """
    This class implements a forward-backward-algorithm for analyzing time series data using hierarchical models. For
    efficient computation, all parameter distribution are dicretized on a parameter grid.
    """
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
        Loads UK coal mining disaster data.

        Parameters:
            None

        Returns:
            None
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
        Loads Numpy array as data.

        Parameters:
            array - Numpy array containing time series data

        Returns:
            None
        """
        self.rawData = array
        print '    + Successfully imported array.'

    def createGrid(self):
        """
        Creates parameter grid, based on given boundaries and grid size. Note that the given parameter boundaries are
        not included as points on the grid (to avoid e.g. division by zero in case zero is chosen as boundary value).

        Parameters:
            None

        Returns:
            None
        """

        self.marginalGrid = [np.linspace(b[0], b[1], g+2)[1:-1] for b, g in zip(self.boundaries, self.gridSize)]
        self.grid = [m for m in np.meshgrid(*self.marginalGrid, indexing='ij')]
        self.latticeConstant = [g[1]-g[0] for g in self.marginalGrid]

        if self.transitionModel != None:
            self.transitionModel.latticeConstant = self.latticeConstant

    def setBoundaries(self, newBoundaries):
        """
        Sets lower and upper parameter boundaries (and updates grid accordingly).

        Parameters:
            newBoundaries: A list of lists which each contain a lower & upper parameter boundary
                           Example: [[-1, 1],[0, 2]]

        Returns:
            None
        """
        self.boundaries = newBoundaries
        self.createGrid()

    def setGridSize(self, newGridSize):
        """
        Sets grid size for discretization of parameter distributions (and updates grid accordingly).

        Parameters:
            newGridSize - List of integers describing the size of the parameter grid for each dimension

        Returns:
            None
        """
        self.gridSize = newGridSize
        self.createGrid()

    def setObservationModel(self, M, silent=False):
        """
        Sets observation model (likelihood function) for analysis.

        Parameters:
            M - Observation model class (see observationModel.py)

        Returns:
            None
        """
        self.observationModel = M

        self.gridSize = M.defaultGridSize
        self.boundaries = M.defaultBoundaries

        if not silent:
            print '    + Observation model:', M

    def setTransitionModel(self, K, silent=False):
        """
        Set transition model which describes the parameter dynamics.

        Parameters:
            K - Transition model class (see transitionModel.py)

        Returns:
            None
        """
        self.transitionModel = K
        self.transitionModel.latticeConstant = self.latticeConstant

        if not silent:
            print '    + Transition model:', K

    def fit(self, forwardOnly=False, evidenceOnly=False, silent=False):
        """
        Computes the sequence of posterior distributions and evidence for each time step. Evidence is also computed for
        the complete data set.

        Parameters:
            forwardOnly - If set to True, the fitting process is terminated after the forward pass. The resulting
                posterior distributions are so-called "filtering distributions" which - at each time step -
                only incorporate the information of past data points. This option thus emulates an online
                analysis.

            evidenceOnly - If set to True, only forward pass is run and evidence is calculated. In contrast to the
                forwardOnly option, no posterior mean values are computed.

            silent - If set to True, no output is generated by the fitting method.

        Returns:
            None
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

        if not (forwardOnly or evidenceOnly):
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

    def optimize(self):
        if self.transitionModel is None:
            print '! ERROR: No transition model chosen.'
            return

        print '+ Starting optimization...'

        # create parameter list to set start values for optimization
        x0 = self.unpackHyperParameters()

        # perform optimization (maximization of log-evidence)
        result = minimize(self.optimizationStep, x0, method='COBYLA')

        print '+ Finished optimization.'

        # set optimal hyperparameters in transition model
        self.setHyperParameters(result.x)

        # run analysis with optimal parameter values
        self.fit()

    def optimizationStep(self, x):
        # set new hyperparameters in transition model
        self.setHyperParameters(x)

        # compute log-evidence
        self.fit(evidenceOnly=True, silent=True)

        print '    + Log10-evidence: {:.5f}'.format(self.logEvidence / np.log(10)), '- Parameter values:', x

        # return negative log-evidence (is minimized to maximize evidence)
        return -self.logEvidence

    def unpackHyperParameters(self):
        if isinstance(self.transitionModel, CombinedTransitionModel):
            models = self.transitionModel.models
        else:
            # only one model in a non-combined transition model
            models = [self.transitionModel]

        x0 = []  # initialize parameter list
        for model in models:
            for i, key in enumerate(model.hyperParameters.keys()):
                # if parameter itself is a list, we need to unpack
                if type(model.hyperParameters[key]) is list:
                    length = len(model.hyperParameters[key])
                    for i in range(length):
                        x0.append(model.hyperParameters[key][i])
                # if parameter is single float, no unpacking is needed
                else:
                    x0.append(model.hyperParameters[key])

        return x0

    def setHyperParameters(self, x):
        if isinstance(self.transitionModel, CombinedTransitionModel):
            models = self.transitionModel.models
        else:
            # only one model in a non-combined transition model
            models = [self.transitionModel]

        paramList = list(x[:])  # make copy of previous parameter list
        for model in models:
            for i, key in enumerate(model.hyperParameters.keys()):
                # if parameter itself is a list, we need to unpack
                if type(model.hyperParameters[key]) is list:
                    length = len(model.hyperParameters[key])
                    model.hyperParameters[key] = []
                    for i in range(length):
                        model.hyperParameters[key].append(paramList[0])
                        paramList.pop(0)
                # if parameter is single float, no unpacking is needed
                else:
                    model.hyperParameters[key] = paramList[0]
                    paramList.pop(0)
