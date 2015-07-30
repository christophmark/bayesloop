# -*- coding: utf-8 -*-
"""
changepointStudy.py builds on Study-class to perform a series of runs with varying change point.
"""

import numpy as np
from .study import Study
from .preprocessing import *
from .transitionModel import ChangePoint


class changepointStudy(Study):
    def __init__(self):
        super(changepointStudy, self).__init__()

        self.changepointPrior = None
        self.changepointDistribution = None
        self.averagePosteriorSequence = None

        print '  --> Change-point study'

    def fit(self, silent=False):
        if not silent:
            print '+ Started new fit.'

        # prepare arrays for change-point distribution and average posterior sequence
        self.formattedData = movingWindow(self.rawData, self.observationModel.segmentLength)
        if self.changepointPrior is None:
            self.changepointPrior = np.ones(len(self.formattedData))/len(self.formattedData)
        self.averagePosteriorSequence = np.zeros([len(self.formattedData)]+self.gridSize)
        logEvidenceList = []

        for tChange in range(len(self.formattedData)):
            # configure transistion model
            K = ChangePoint(tChange=tChange)
            self.setTransitionModel(K, silent=True)

            # call fit method from parent class
            Study.fit(self, silent=True)

            logEvidenceList.append(self.logEvidence)
            self.averagePosteriorSequence += self.posteriorSequence*np.exp(self.logEvidence)*self.changepointPrior[tChange]

            if not silent:
                print '    + t = {} -- log10-evidence = {:.5f}'.format(tChange, self.logEvidence / np.log(10))

        # compute average posterior distribution
        normalization = self.averagePosteriorSequence.sum(axis=1)
        self.averagePosteriorSequence /= normalization[:,None]

        # set self.posteriorSequence to average posterior sequence for plotting reasons
        self.posteriorSequence = self.averagePosteriorSequence

        if not silent:
            print '    + Computed average posterior sequence'

        # compute log-evidence of averaged model
        self.logEvidence = np.log(np.sum(np.exp(np.array(logEvidenceList))*self.changepointPrior))

        if not silent:
            print '    + Log10-evidence of average model: {:.5f}'.format(self.logEvidence / np.log(10))

        # compute change-point distribution
        self.changepointDistribution = np.exp(np.array(logEvidenceList))*self.changepointPrior
        self.changepointDistribution /= np.sum(self.changepointDistribution)

        if not silent:
            print '    + Computed change-point distribution'

        # compute posterior mean values
        self.posteriorMeanValues = np.empty([len(self.grid), len(self.posteriorSequence)])
        for i in range(len(self.grid)):
            self.posteriorMeanValues[i] = np.array([np.sum(p*self.grid[i]) for p in self.posteriorSequence])

        if not silent:
            print '    + Computed mean parameter values.'

        # local evidence is not supported at the moment
        self.localEvidence = None
