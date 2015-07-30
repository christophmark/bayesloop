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

        print '  --> change-point study'

    def fit(self):
        # prepare arrays for change-point distribution and average posterior sequence
        self.formattedData = movingWindow(self.rawData, self.observationModel.segmentLength)
        if self.changepointPrior is None:
            self.changepointPrior = np.ones(len(self.formattedData))/len(self.formattedData)
        self.averagePosteriorSequence = np.zeros([len(self.formattedData)]+self.gridSize)
        logEvidenceList = []

        for tChange in range(len(self.formattedData)):
            # configure transistion model
            K = ChangePoint(tChange=tChange)
            self.setTransitionModel(K)

            # call fit method from parent class
            Study.fit(self)

            logEvidenceList.append(self.logEvidence)
            self.averagePosteriorSequence += self.posteriorSequence*np.exp(self.logEvidence)*self.changepointPrior[tChange]

        # compute average posterior distribution
        normalization = self.averagePosteriorSequence.sum(axis=1)
        self.averagePosteriorSequence /= normalization[:,None]

        # compute log-evidence of averaged model
        self.logEvidence = np.log(np.sum(np.exp(np.array(logEvidenceList))*self.changepointPrior))

        # compute change-point distribution
        self.changepointDistribution = np.exp(np.array(logEvidenceList))*self.changepointPrior
        self.changepointDistribution /= np.sum(self.changepointDistribution)

        # compute posterior mean values
        self.posteriorMeanValues = np.empty([len(self.grid), len(self.posteriorSequence)])
        for i in range(len(self.grid)):
            self.posteriorMeanValues[i] = np.array([np.sum(p*self.grid[i]) for p in self.posteriorSequence])

        # local evidence is not supported at the moment
        self.localEvidence = None
