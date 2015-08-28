#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file introduces an extension to the basic Study-class which allows to add data step by step and keep an updated
parameter distribution, allowing to analyze data streams online.
"""

import numpy as np
from .study import Study


class OnlineStudy(Study):
    """
    This class builds on the Study-class and features a step-method to include new data points in the study as they
    arrive from a data stream. This online-analysis is performed in an forward-only way, resulting in filtering-
    distributions only. If one is interested in smoothing-distributions, use the conventional fit-method.

    Note: Only the complete rawData is preserved; all results are only current results and have to be stored by external
        functions to be used later.
    """
    def __init__(self):
        super(OnlineStudy, self).__init__()

        self.posteriorDistribution = None
        self.posteriorMeanValue = None
        print '  --> Online study'

    def step(self, dataPoint):
        self.rawData = np.append(self.rawData, dataPoint)

        # initialize posterior distribution as flat prior
        if self.posteriorDistribution is None:
            self.posteriorDistribution = np.ones(self.gridSize)/np.prod(np.array(self.gridSize))
            print '    + Initialized flat prior distribution.'

        # only proceed if at least one data segment can be created
        if len(self.rawData) < self.observationModel.segmentLength:
            print '    ! Not enough data points to start analysis. Will wait for more data.'
            return

        dataSegment = self.rawData[-self.observationModel.segmentLength:]

        # update posterior distribution with new data
        # posterior-prior transformation is done here to ensure that after the call of update, a true posterior
        # distribution is stored in self.posteriorDistribution; time parameter is given by length of formatted data
        # minus one
        self.posteriorDistribution = self.transitionModel.computeForwardPrior(self.posteriorDistribution,
                                                                              len(self.formattedData)-1)

        likelihood = self.observationModel.pdf(self.grid, dataSegment)  # compute likelihood
        self.posteriorDistribution *= likelihood  # update alpha based on likelihood

        norm = np.sum(self.posteriorDistribution)  # normalization constant of posterior is used to compute evidence
        self.logEvidence += np.log(norm)
        self.localEvidence = norm

        self.posteriorDistribution /= norm  # normalize posterior distribution

        # compute posterior values
        self.posteriorMeanValue = np.array([np.sum(self.posteriorDistribution*g) for g in self.grid])

        print '    + Updated posterior distribution. Data points: ' + str(len(self.rawData)) + \
              '; Local evidence: {:.4f}'.format(self.localEvidence)

    def reset(self):
        self.posteriorDistribution = None
        self.posteriorMeanValue = None
        self.logEvidence = 0
        print '    ! Resetted posterior distribution, mean value and log-evidence.'

    def update(self):
        print '    + Updating posterior distribution and mean value using all available data...'
        self.logEvidence = 0
        self.fit(forwardOnly=True, silent=True)
        self.posteriorDistribution = self.posteriorSequence[-1]
        self.posteriorMeanValue = self.posteriorMeanValues[-1]
        self.localEvidence = self.localEvidence[-1]
        self.formattedData = np.array([])
        print '    + Update done.'
