#!/usr/bin/env python
"""
This file introduces an extension to the basic Study-class which allows to add data step by step and keep an updated
parameter distribution, enabling the analysis of on-going data streams.
"""

from __future__ import division, print_function
import numpy as np
from .study import Study
from .exceptions import *
from collections import OrderedDict, Iterable


class OnlineStudy(Study):
    """
    This class builds on the Study-class and features a step-method to include new data points in the study as they
    arrive from a data stream. This online-analysis is performed in an forward-only way, resulting in filtering-
    distributions only. In contrast to a normal study, however, one can add multiple transition models to account for
    different types of parameter dynamics (similar to a Hyper study). The Online study then computes the probability
    distribution over all transition models for each new data point, enabling real-time model selection.

    Args:
        storeHistory: If true, posterior distributions and their mean values, as well as hyper-posterior distributions
            are stored for all time steps.
    """
    def __init__(self, storeHistory=False):
        super(OnlineStudy, self).__init__()

        self.transitionModels = None
        self.tmCount = None
        self.tmCounts = None
        self.hyperParameters = None
        self.selectedHyperParameters = None
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
        print('  --> Online study')

    def addTransitionModel(self, transitionModel):
        """
        Adds a transition model to the list of transition models that are fitted in each time step. Note that a list of
        hyper-parameter values can be supplied.

        Args:
            transitionModel: instance of a transition model class.

        Example:
            Here, 'S' denotes the OnlineStudy instance. In the first example, we assume a Poisson observation model and
            add a Gaussian random walk with varying standard deviation to the rate parameter 'lambda':

                S.setObservationModel(bl.om.Poisson())
                S.addTransitionModel(bl.tm.GaussianRandomWalk(sigma=[0, 0.1, 0.2, 0.3], param='lambda'))
        """
        if str(transitionModel) == 'Serial transition model':
            raise NotImplementedError('Serial transition models are not supported by OnlineStudy.')
        if str(transitionModel) == 'Combined transition model':
            raise NotImplementedError('Combined transition models are not supported by OnlineStudy.')

        if self.transitionModels is None:
            self.transitionModels = []
            self.hyperParameters = []
            self.selectedHyperParameters = []

        self.transitionModels.append(transitionModel)
        self.hyperParameters.append(transitionModel.hyperParameters.copy())

        # if a single hyper-parameter value is provided, pack it in a list
        for key, values in self.hyperParameters[-1].items():
            if not isinstance(values, Iterable):
                self.hyperParameters[-1][key] = [values]

        try:
            self.selectedHyperParameters.append(transitionModel.selectedHyperParameter)
        except:  # transition model does not support the selection of a specific hyper-parameter
            self.selectedHyperParameters.append(None)

        # set first hypothesis as transition model (so consistency check of bayesloop passes)
        self.setTransitionModel(self.transitionModels[0], silent=True)

        # count individual transition models
        self.tmCounts = []
        for i, tm in enumerate(self.transitionModels):
            self.tmCounts.append(len(self.hyperParameters[i].values()[0]))
        self.tmCount = np.sum(self.tmCounts)

        print('    + Added transition model: {}'.format(transitionModel))

    def step(self, dataPoint):
        """
        Update the current parameter distribution by adding a new data point to the data set.

        Args:
            dataPoint: Float, int, or 1D-array of those (for multidimensional data).
        """
        # at least one transition model has to be set or added
        if (self.tmCount is None) and (self.transitionModel is None):
            raise ConfigurationError('No transition model set or added.')

        # if one only sets a transition model, but does not use addTransitionModel, we add it here
        if (self.tmCount is None) and (self.transitionModel is not None):
            self.addTransitionModel(self.transitionModel)

        if not isinstance(dataPoint, list):
            dataPoint = [dataPoint]

        if len(self.rawData) == 0:
            # to check the model consistency the first time that 'step' is called
            self.rawData = np.array(dataPoint)
            Study.checkConsistency(self)

            self.rawTimestamps = np.array([0])
            self.formattedTimestamps = []
        else:
            self.rawData = np.append(self.rawData, np.array(dataPoint), axis=0)
            self.rawTimestamps = np.append(self.rawTimestamps, self.rawTimestamps[-1]+1)

        # only proceed if at least one data segment can be created
        if len(self.rawData) < self.observationModel.segmentLength:
            print('    + Not enough data points to start analysis. Will wait for more data.')
            return

        self.formattedTimestamps.append(self.rawTimestamps[-1])

        # initialize hyper-prior as flat
        if self.hyperPrior is None:
            self.hyperPrior = np.ones(self.tmCount) / self.tmCount
            print('    + Initialized flat hyper-prior.')

        # initialize alpha with prior distribution
        if self.alpha is None:
            if self.observationModel.prior is not None:
                if isinstance(self.observationModel.prior, np.ndarray):
                    self.alpha = self.observationModel.prior
                else:  # prior is set as a function
                    self.alpha = self.observationModel.prior(*self.grid)
            else:
                self.alpha = np.ones(self.gridSize)  # flat prior

            # normalize prior (necessary in case an improper prior is used)
            self.alpha /= np.sum(self.alpha)
            print('    + Initialized prior.')

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
        for tm, hp, sp in zip(self.transitionModels, self.hyperParameters, self.selectedHyperParameters):

            for j in range(len(hp.values()[0])):  # loop over all hyper-parameter values to fit
                # create dictionary with keyword-arguments to pass to transition model
                odict = OrderedDict()
                for param, value in hp.items():
                    odict[param] = value[j]
                tm.hyperParameters = odict

                try:
                    tm.selectedHyperParameter = sp
                except:  # transition model does not support the selection of a specific hyper-parameter
                    pass

                # set current transition model and hyper-parameter values
                self.setTransitionModel(tm, silent=True)

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
        self.alpha = self.marginalizedPosterior

        # store results for future plotting
        if self.storeHistory:
            self.posteriorMeanValues.append(np.array([np.sum(self.marginalizedPosterior*g) for g in self.grid]))
            self.posteriorSequence.append(self.marginalizedPosterior.copy())
            self.hyperPosteriorSequence.append(self.hyperPosterior.copy())

    def plotParameterEvolution(self, param=0, color='b', gamma=0.5, **kwargs):
        """
        Plots a series of marginal posterior distributions corresponding to a single model parameter, together with the
        posterior mean values.

        Args:
            param: parameter name or index of parameter to display; default: 0 (first model parameter)
            color: color from which a light colormap is created
            gamma: exponent for gamma correction of the displayed marginal distribution; default: 0.5
            kwargs: all further keyword-arguments are passed to the plot of the posterior mean values
        """
        # plotting function of Study class can only handle arrays, not lists
        self.formattedTimestamps = np.array(self.formattedTimestamps)
        self.posteriorMeanValues = np.array(self.posteriorMeanValues).T
        self.posteriorSequence = np.array(self.posteriorSequence)

        Study.plotParameterEvolution(self, param=param, color='b', gamma=0.5, **kwargs)

        # re-transform arrays to lists, so online study may continue to append values
        self.formattedTimestamps = list(self.formattedTimestamps)
        self.posteriorMeanValues = list(self.posteriorMeanValues)
        self.posteriorSequence = list(self.posteriorSequence)
