#!/usr/bin/env python
"""
This file introduces an extension to the basic Study-class which allows to add data step by step and keep an updated
parameter distribution, allowing to analyze data streams online.
"""

from __future__ import division, print_function
import numpy as np
from .study import Study
import matplotlib.pyplot as plt
from .helper import create_colormap


class OnlineStudy(Study):
    """
    This class builds on the Study-class and features a step-method to include new data points in the study as they
    arrive from a data stream. This online-analysis is performed in an forward-only way, resulting in filtering-
    distributions only. If one is interested in smoothing-distributions, use the conventional fit-method.

    Note: Only the complete rawData is preserved; all results are only current results and have to be stored by external
        functions to be used later.
    """
    def __init__(self, storeHistory=False):
        super(OnlineStudy, self).__init__()

        self.transitionModels = None
        self.tmCount = None
        self.tmCounts = None
        self.alpha = None
        self.beta = None
        self.normi = None
        self.hyperPrior = None
        self.parameterPosterior = None
        self.parameterPosteriorSequence = None
        self.hyperPosterior = None
        self.marginalizedPosterior = None

        self.storeHistory = storeHistory

        self.posteriorMeanValues = []
        self.posteriorSequence = []
        self.hyperPosteriorSequence = []
        self.parameterPosteriorSequence = []
        print('  --> Loops model.')

    def addTransitionModel(self, transitionModel):
        """
        TODO

        Args:
            transitionModel: TODO
        """

        if self.transitionModels is None:
            self.transitionModels = [transitionModel]
        else:
            self.transitionModels.append(transitionModel)

        # set first hypothesis as transition model (so consistency check of bayesloop passes)
        self.setTransitionModel(self.transitionModels[0][0](self.transitionModels[0][1][0]), silent=True)

        # count individual transition models
        self.tmCounts = []
        for tm in self.transitionModels:
            self.tmCounts.append(len(tm[1]))
        self.tmCount = np.sum(self.tmCounts)

    def step(self, dataPoint):
        """
        Update the current parameter distribution by adding a new data point to the data set.

        Args:
            dataPoint: Float, int, or 1D-array of those (for multidimensional data).
        """
        if not isinstance(dataPoint, list):
            dataPoint = [dataPoint]

        if len(self.rawData) == 0:
            self.rawData = np.array(dataPoint)
            self.rawTimestamps = np.array([0])
        else:
            self.rawData = np.append(self.rawData, np.array(dataPoint), axis=0)
            self.rawTimestamps = np.append(self.rawTimestamps, self.rawTimestamps[-1]+1)

        # only proceed if at least one data segment can be created
        if len(self.rawData) < self.observationModel.segmentLength:
            print('    ! Not enough data points to start analysis. Will wait for more data.')
            return

        # initialize hyper-prior as flat
        if self.hyperPrior is None:
            self.hyperPrior = np.ones(self.tmCount) / self.tmCount
            print('    + Initialized flat hyper-prior.')

        # initialize alpha as flat distribution
        if self.alpha is None:
            self.alpha = np.ones(self.gridSize)/np.prod(np.array(self.gridSize))
            print('    + Initialized flat alpha.')

        # initialize normi as an array of ones
        if self.normi is None:
            self.normi = np.ones(self.tmCount)
            print('    + Initialized normalization factors.')

        # initialize parameter posterior
        if self.parameterPosterior is None:
            self.parameterPosterior = np.zeros([self.tmCount] + self.gridSize)

        # initialize hyper-posterior
        if self.hyperPosterior is None:
            self.hyperPosterior = np.zeros(self.tmCount)

        # initialize marginalized posterior
        self.marginalizedPosterior = np.zeros(self.gridSize)

        # select data segment
        dataSegment = self.rawData[-self.observationModel.segmentLength:]

        # compute current likelihood only once
        likelihood = self.observationModel.processedPdf(self.grid, dataSegment)

        # loop over all hypotheses
        idx = 0
        for tm in self.transitionModels:
            model = tm[0]
            params = tm[1]
            for p in params:
                self.setTransitionModel(model(p), silent=True)

                # compute alpha_i
                alphai = self.transitionModel.computeForwardPrior(self.alpha, len(self.formattedData)-1)*likelihood
                ni = np.sum(alphai)

                # hyper-post. values are not normalized at this point: hyper-like. * hyper-prior
                self.hyperPosterior[idx] = (ni/self.normi[idx])*self.hyperPrior[idx]

                # store parameter posterior
                self.parameterPosterior[idx] = alphai/ni

                # add weighted parameter posterior to marginalized posterior
                self.marginalizedPosterior += self.parameterPosterior[idx]*self.hyperPosterior[idx]

                # update normalization constant
                self.normi[idx] = ni

                idx += 1

        # normalize hyper-posterior
        self.hyperPosterior /= np.sum(self.hyperPosterior)

        # normalize marginalized posterior
        self.marginalizedPosterior /= np.sum(self.marginalizedPosterior)

        # compute new alpha
        self.alpha = self.marginalizedPosterior  # *np.sum(self.normi)

        # store results for future plotting
        if self.storeHistory:
            self.posteriorMeanValues.append(np.array([np.sum(self.marginalizedPosterior*g) for g in self.grid]))
            self.posteriorSequence.append(self.marginalizedPosterior.copy())
            self.hyperPosteriorSequence.append(self.hyperPosterior.copy())

    def plotParameterEvolution(self, param=0, xLower=None, xUpper=None, color='b', gamma=0.5, **kwargs):
        """
        Plots a series of marginal posterior distributions corresponding to a single model parameter, together with the
        posterior mean values.

        Args:
            param: parameter name or index of parameter to display; default: 0 (first model parameter)

            color: color from which a light colormap is created

            gamma: exponent for gamma correction of the displayed marginal distribution; default: 0.5

            **kwargs: all further keyword-arguments are passed to the plot of the posterior mean values
        """
        posteriorSequence = np.array(self.posteriorSequence)
        posteriorMeanValues = np.array(self.posteriorMeanValues).T

        if self.posteriorSequence == []:
            print('! Cannot plot posterior sequence as it has not yet been computed. Run complete fit.')
            return

        if isinstance(param, int):
            paramIndex = param
        elif isinstance(param, str):
            paramIndex = -1
            for i, name in enumerate(self.observationModel.parameterNames):
                if name == param:
                    paramIndex = i

            # check if match was found
            if paramIndex == -1:
                print('! Wrong parameter name. Available options: {0}'.format(self.observationModel.parameterNames))
                return
        else:
            print('! Wrong parameter format. Specify parameter via name or index.')
            return

        axesToMarginalize = list(range(1, len(self.observationModel.parameterNames) + 1))  # axis 0 is time
        axesToMarginalize.remove(paramIndex + 1)

        marginalPosteriorSequence = np.squeeze(np.apply_over_axes(np.sum, posteriorSequence, axesToMarginalize))

        if not xLower:
            xLower = 0
        if not xUpper:
            if xLower:
                print('! If lower x limit is provided, upper limit has to be provided, too.')
                print('  Setting lower limit to zero.')
            xLower = 0
            xUpper = len(marginalPosteriorSequence)

        plt.imshow((marginalPosteriorSequence.T) ** gamma,
                   origin=0,
                   cmap=create_colormap(color),
                   extent=[xLower, xUpper - 1] + self.boundaries[paramIndex],
                   aspect='auto')

        # set default color of plot to black
        if (not 'c' in kwargs) and (not 'color' in kwargs):
            kwargs['c'] = 'k'

        # set default linewidth to 1.5
        if (not 'lw' in kwargs) and (not 'linewidth' in kwargs):
            kwargs['lw'] = 1.5

        plt.plot(np.arange(xLower, xUpper), posteriorMeanValues[paramIndex], **kwargs)

        plt.ylim(self.boundaries[paramIndex])
        plt.ylabel(self.observationModel.parameterNames[paramIndex])
        plt.xlabel('time step')