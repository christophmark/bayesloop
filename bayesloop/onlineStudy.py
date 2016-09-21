#!/usr/bin/env python
"""
This file introduces an extension to the basic Study-class which allows to add data step by step and keep an updated
parameter distribution, enabling the analysis of on-going data streams.
"""

from __future__ import division, print_function
import numpy as np
from .study import *
from .helper import *
from .exceptions import *
from collections import OrderedDict, Iterable
from inspect import getargspec
from copy import deepcopy


class OnlineStudy(Study):
    """
    This class builds on the Study-class and features a step-method to include new data points in the study as they
    arrive from a data stream. This online-analysis is performed in an forward-only way, resulting in filtering-
    distributions only. In contrast to a normal study, however, one can add multiple transition models to account for
    different types of parameter dynamics (similar to a Hyper study). The Online study then computes the probability
    distribution over all transition models for each new data point, enabling real-time model selection.

    Args:
        storeHistory(bool): If true, posterior distributions and their mean values, as well as hyper-posterior
            distributions are stored for all time steps.
    """
    def __init__(self, storeHistory=False):
        super(OnlineStudy, self).__init__()

        self.transitionModels = None
        self.tmCount = None
        self.tmCounts = None
        self.hyperParameterValues = None
        self.hyperParameterNames = None
        self.hyperGridConstants = None

        self.alpha = None
        self.beta = None
        self.normi = None
        self.hyperPrior = None
        self.hyperPriorValues = None
        self.transitionModelPrior = None

        self.parameterPosterior = None
        self.transitionModelPosterior = None
        self.marginalizedPosterior = None

        self.hyperParameterDistribution = None
        self.transitionModelDistribution = None
        self.localHyperEvidence = None

        self.storeHistory = storeHistory
        self.posteriorMeanValues = []
        self.posteriorSequence = []
        self.hyperParameterSequence = []
        self.transitionModelSequence = []
        print('  --> Online study')

        self.debug = []

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
        if self.transitionModels is None:
            self.transitionModels = []
            self.hyperParameterValues = []
            self.hyperParameterNames = []
            self.hyperGridConstants = []

        # extract hyper-parameter values and names
        self.setTransitionModel(transitionModel, silent=True)
        flatHyperParameters = self.unpackAllHyperParameters()
        flatHyperParameterNames = self.unpackAllHyperParameters(values=False)
        if len(flatHyperParameterNames) > 0:
            reshapedHyperParameters = np.array(np.meshgrid(*flatHyperParameters)).T.reshape(-1, len(flatHyperParameters))
        else:
            reshapedHyperParameters = []

        # find lattice constants for equally spaced hyper-parameter values
        self.hyperGridConstants.append([])
        for values in flatHyperParameters:
            if isinstance(values, Iterable) and len(values) > 1:
                d = values[1:] - values[:-1]
                dd = d[1:] - d[:-1]
                if np.all(dd < 10 ** -10):  # for equally spaced values, set difference as grid-constant
                    self.hyperGridConstants[-1].append(np.abs(d[0]))
            else:  # for irregularly spaced values (e.g. categorical) or single value, set grid-constant to 1
                self.hyperGridConstants[-1].append(1)
        self.hyperGridConstants[-1] = np.array(self.hyperGridConstants[-1])

        self.transitionModels.append(transitionModel)
        self.hyperParameterValues.append(reshapedHyperParameters)
        self.hyperParameterNames.append(flatHyperParameterNames)

        # count individual transition models
        self.tmCounts = []
        for hpv in self.hyperParameterValues:
            if len(hpv) > 0:
                self.tmCounts.append(len(hpv))
            else:
                self.tmCounts.append(1)
        self.tmCount = np.sum(self.tmCounts)

        if len(reshapedHyperParameters) > 0:
            print('    + Added transition model: {} ({} combination(s) of the following hyper-parameters: {})'
                  .format(transitionModel, len(reshapedHyperParameters), self.hyperParameterNames[-1]))
        else:
            print('    + Added transition model: {} (no hyper-parameters)'.format(transitionModel))

    def setHyperPrior(self, hyperPrior):
        """
        Sets prior probabilities for hyper-parameter values of each transition model added to the online study instance.

        Args:
            hyperPrior: List/array of functions, where each function takes exactly as many arguments as the number of
                hyper-parameters in the corresponding transition model. Note: If only a single transition model is used,
                the function does not need to be contained in a list/array.

        Example:
            S = bl.OnlineStudy()
            S.setObservationModel(bl.om.ScaledAR1())
            S.setGrid([[-1, 1, 200], [0, 1, 200]])

            S.addTransitionModel(bl.tm.CombinedTransitionModel(
                bl.tm.GaussianRandomWalk(np.linspace(0, 0.1, 5), param='correlation coefficient'),
                bl.tm.GaussianRandomWalk(np.linspace(0, 0.1, 5), param='standard deviation'))
                )

            def linear(t, slope=np.linspace(0, 0.1, 5)):
                return slope*t

            S.addTransitionModel(bl.tm.Deterministic(linear, param='correlation coefficient'))

            def hyperPriorGaussian(sigma_r, sigma_s):
                return 1./(sigma_r*sigma_s + 10**-20)

            def hyperPriorLinear(slope):
                return 1.

            S.setHyperPrior([hyperPriorGaussian, hyperPriorLinear])
        """
        # check if setTransitionModel() has been used instead of addTransitionModel
        if self.transitionModels is None and self.transitionModel is not None:
            self.addTransitionModel(self.transitionModel)

        # check if only one function is provided that is not contained in a list/array
        if hasattr(hyperPrior, '__call__') and len(self.transitionModels) == 1:
            hyperPrior = [hyperPrior]

        # here, 'hyperPrior' must be a list/array
        if isinstance(hyperPrior, Iterable):
            # check number of prior functions
            if not len(hyperPrior) == len(self.transitionModels):
                raise ConfigurationError('Expected list of {} functions as hyper-prior, got list with {} elements.'
                                         .format(len(self.transitionModels), len(hyperPrior)))
            # check if all priors are indeed functions or set to None
            if not np.all([np.any([hasattr(hp, '__call__'), hp is None]) for hp in hyperPrior]):
                raise ConfigurationError('All elements of the list of hyper-priors must be functions or "None".')

            # loop over all added transition models
            hyperPriorProbabilities = []
            for i, hpv in enumerate(self.hyperParameterValues):
                if len(hpv) == 0:
                    if hyperPrior[i] is None:
                        hyperPriorProbabilities.append(np.array([1.]))
                    else:
                        raise ConfigurationError('Expected hyper-prior to be "None" for Transition model "{}", as it '
                                                 'contains no hyper-parameters.'.format(str(self.transitionModels[i])))
                else:
                    if hyperPrior[i] is None:
                        hyperPriorProbabilities.append(np.ones(len(hpv))/len(hpv))
                    else:
                        argspec = getargspec(hyperPrior[i])
                        # check if number of hyper-parameters matches number of function arguments
                        if not len(argspec.args) == len(hpv[0]):
                            raise ConfigurationError('Expected {} arguments for prior function corresponding to the '
                                                     'transition model: {}; got function with {} arguments.'
                                                     .format(len(hpv[0]), str(self.transitionModels[i]), len(argspec.args)))

                        # evaluate prior probabilities (or probability densities)
                        values = np.array([hyperPrior[i](*x) for x in hpv])
                        values = values / (np.sum(values)*np.prod(self.hyperGridConstants[i]))
                        hyperPriorProbabilities.append(values)

            # if all is well, attributes are set
            self.hyperPriorValues = hyperPriorProbabilities
            self.hyperPrior = hyperPrior
            print('    + Set custom hyper-prior.')
        else:
            raise ConfigurationError('OnlineStudy expects a list of functions as a hyper-prior.')

    def setTransitionModelPrior(self, transitionModelPrior):
        """
        Sets prior probabilities for transition models added to the online study instance.

        Args:
            transitionModelPrior: List/Array of probabilities, one for each transition model. If the list does not sum
                to one, it will be re-normalised.
        """
        if not (isinstance(transitionModelPrior, Iterable) and len(transitionModelPrior) == len(self.transitionModels)):
            raise ConfigurationError('Length of transition model prior ({}) does not fit number of transition models '
                                     '({})'.format(len(transitionModelPrior), len(self.transitionModels)))

        self.transitionModelPrior = np.array(transitionModelPrior)

        if not np.sum(transitionModelPrior) == 1.:
            print('    + WARNING: Transition model prior does not sum up to one. Will re-normalize.')
            self.transitionModelPrior /= np.sum(self.transitionModelPrior)

    def adoptHyperParameterDistribution(self):
        """
        Will set the current hyper-parameter distribution as the new hyper-parameter prior, if a distribution has
        already been computed. Is usually called after the 'step' method.
        """
        if self.hyperParameterDistribution is not None:
            self.hyperPriorValues = deepcopy(self.hyperParameterDistribution)

    def adoptTransitionModelDistribution(self):
        """
        Will set the current transition model distribution as the new transition model prior, if a distribution has
        already been computed. Is usually called after the 'step' method.
        """
        if self.transitionModelDistribution is not None:
            self.transitionModelPrior = deepcopy(self.transitionModelDistribution)

    def step(self, dataPoint):
        """
        Update the current parameter distribution by adding a new data point to the data set.

        Args:
            dataPoint(float, int, ndarray): Float, int, or 1D-array of those (for multidimensional data).
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
            self.hyperPrior = 'flat hyper-prior'
            self.hyperPriorValues = [np.ones(tmc) / tmc for tmc in self.tmCounts]
            print('    + Initialized flat hyper-prior.')

        # initialize transition model prior as flat
        if self.transitionModelPrior is None:
            self.transitionModelPrior = np.ones(len(self.transitionModels))/len(self.transitionModels)
            print('    + Initialized flat transition mode prior.')

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
            self.normi = [np.ones(tmc) for tmc in self.tmCounts]
            print('    + Initialized normalization factors.')

        # initialize parameter posterior
        if self.parameterPosterior is None:
            self.parameterPosterior = [np.zeros([tmc] + self.gridSize) for tmc in self.tmCounts]

        # initialize hyper-posterior
        if self.hyperParameterDistribution is None:
            self.hyperParameterDistribution = [np.zeros(tmc) for tmc in self.tmCounts]

        # initialize transition model evidence
        if self.localHyperEvidence is None:
            self.localHyperEvidence = np.zeros(len(self.transitionModels))

        # initialize transition model distribution (normalized version of transition model evidence array)
        if self.transitionModelDistribution is None:
            self.transitionModelDistribution = np.zeros(len(self.transitionModels))

        # initialize transition model posterior
        self.transitionModelPosterior = np.zeros([len(self.transitionModels)] + self.gridSize)

        # initialize marginalized posterior
        self.marginalizedPosterior = np.zeros(self.gridSize)

        # select data segment
        dataSegment = self.rawData[-self.observationModel.segmentLength:]

        # compute current likelihood only once
        likelihood = self.observationModel.processedPdf(self.grid, dataSegment)

        # loop over all hypotheses/transition models
        for i, (tm, hpv) in enumerate(zip(self.transitionModels, self.hyperParameterValues)):
            self.setTransitionModel(tm, silent=True)  # set current transition model

            if len(hpv) == 0:
                hpv = [None]

            # loop over all hyper-parameter values to fit
            for j, x in enumerate(hpv):
                # set current hyper-parameter values
                if x is not None:
                    self.setAllHyperParameters(x)

                # compute alpha_i
                alphai = self.transitionModel.computeForwardPrior(self.alpha, len(self.formattedData)-1)*likelihood
                ni = np.sum(alphai)

                # hyper-post. values are not normalized at this point: hyper-like. * hyper-prior
                self.hyperParameterDistribution[i][j] = (ni/self.normi[i][j])*self.hyperPriorValues[i][j]

                # store parameter posterior
                self.parameterPosterior[i][j] = alphai/ni

                # update normalization constant
                self.normi[i][j] = ni

            # compute hyper-evidence to properly normalize hyper-parameter distribution
            self.localHyperEvidence[i] = np.sum(self.hyperParameterDistribution[i] *
                                                np.prod(self.hyperGridConstants[i]))

            # normalize hyper-parameter distribution of current transition model
            self.hyperParameterDistribution[i] /= self.localHyperEvidence[i]

            # compute parameter posterior, marginalized over current hyper-parameter values of current transition model
            hpd = deepcopy(self.hyperParameterDistribution[i])
            while len(self.parameterPosterior[i].shape) > len(hpd.shape):
                hpd = np.expand_dims(hpd, axis=-1)
            self.transitionModelPosterior[i] = np.sum(self.parameterPosterior[i] *
                                                      hpd *
                                                      np.prod(self.hyperGridConstants[i]), axis=0)

        # compute distribution of transition models; normalizing constant of this distribution represents the local
        # evidence of current data point, marginalizing over all transition models
        self.transitionModelDistribution = self.localHyperEvidence * self.transitionModelPrior
        self.localEvidence = np.sum(self.transitionModelDistribution)
        self.transitionModelDistribution /= self.localEvidence

        # normalize marginalized posterior
        tmd = deepcopy(self.transitionModelDistribution)
        while len(self.transitionModelPosterior.shape) > len(tmd.shape):
            tmd = np.expand_dims(tmd, axis=-1)
        self.marginalizedPosterior = np.sum(self.transitionModelPosterior * tmd, axis=0)

        # compute new alpha
        self.alpha = self.marginalizedPosterior

        # store results for future plotting
        if self.storeHistory:
            self.posteriorMeanValues.append(np.array([np.sum(self.marginalizedPosterior*g) for g in self.grid]))
            self.posteriorSequence.append(self.marginalizedPosterior.copy())
            self.hyperParameterSequence.append(deepcopy(self.hyperParameterDistribution))
            self.transitionModelSequence.append(deepcopy(self.transitionModelDistribution))

    def getMarginalParameterDistribution(self, t, param=0, plot=False, **kwargs):
        """
        Compute the marginal parameter distribution at a given time step.

        Args:
            t(int, float): Time step/stamp for which the parameter distribution is evaluated
            param(str, int): Parameter name or index of the parameter to display; default: 0 (first model parameter)
            plot(bool): If True, a plot of the distribution is created
            **kwargs: All further keyword-arguments are passed to the plot (see matplotlib documentation)

        Returns:
            ndarray, ndarray: The first array contains the parameter values, the second one the corresponding
                probability (density) values
        """
        # plotting function of Study class can only handle arrays, not lists
        self.formattedTimestamps = np.array(self.formattedTimestamps)
        self.posteriorSequence = np.array(self.posteriorSequence)

        Study.getMarginalParameterDistribution(self, t, param=param, plot=plot, **kwargs)

        # re-transform arrays to lists, so online study may continue to append values
        self.formattedTimestamps = list(self.formattedTimestamps)
        self.posteriorSequence = list(self.posteriorSequence)

    def getMarginalParameterDistributions(self, param=0, plot=False, **kwargs):
        """
        Computes the time series of marginal posterior distributions with respect to a given model parameter.

        Args:
            param(str, int): Parameter name or index of the parameter to display; default: 0 (first model parameter)
            plot(bool): If True, a plot of the series of distributions is created (density map)
            **kwargs: All further keyword-arguments are passed to the plot (see matplotlib documentation)

        Returns:
            ndarray, ndarray: The first array contains the parameter values, the second one the sequence of
                corresponding posterior distirbutions.
        """
        # plotting function of Study class can only handle arrays, not lists
        self.formattedTimestamps = np.array(self.formattedTimestamps)
        self.posteriorSequence = np.array(self.posteriorSequence)

        Study.getMarginalParameterDistributions(self, param=param, plot=plot, **kwargs)

        # re-transform arrays to lists, so online study may continue to append values
        self.formattedTimestamps = list(self.formattedTimestamps)
        self.posteriorSequence = list(self.posteriorSequence)

    def plotParameterEvolution(self, param=0, color='b', gamma=0.5, **kwargs):
        """
        Plots a series of marginal posterior distributions corresponding to a single model parameter, together with the
        posterior mean values.

        Args:
            param(str, int): parameter name or index of parameter to display; default: 0 (first model parameter)
            color: color from which a light colormap is created
            gamma(float): exponent for gamma correction of the displayed marginal distribution; default: 0.5
            kwargs: all further keyword-arguments are passed to the plot of the posterior mean values
        """
        # plotting function of Study class can only handle arrays, not lists
        self.formattedTimestamps = np.array(self.formattedTimestamps)
        self.posteriorMeanValues = np.array(self.posteriorMeanValues).T
        self.posteriorSequence = np.array(self.posteriorSequence)

        Study.plotParameterEvolution(self, param=param, color='b', gamma=0.5, **kwargs)

        # re-transform arrays to lists, so online study may continue to append values
        self.formattedTimestamps = list(self.formattedTimestamps)
        self.posteriorMeanValues = list(self.posteriorMeanValues.T)
        self.posteriorSequence = list(self.posteriorSequence)

    def getTransitionModelDistributions(self):
        """
        """
        return np.array(self.transitionModelSequence)

    def getHyperParameterIndex(self, transitionModel, param):
        """
        Helper function that returns the index at which a hyper-parameter is found in the flattened list of
        hyper-parameter names.

        Args:
            transitionModel: transition model instance in which to search
            param(str): Name of a hyper-parameter. If the name occurs multiple times, the index of the submodel can be
                supplied (starting at 1 for the first submodel).

        Returns:
            int: index of the hyper-parameter
        """
        # extract hyper-parameter index from given string
        l = param.split('_')
        name = []
        index = []
        for s in l:
            try:
                index.append(int(s))
            except:
                name.append(s)
        name = '_'.join(name)

        if len(index) == 0:
            # no index provided: choose first occurrence and determine axis of hyper-parameter on grid of
            # hyper-parameter values
            hpn = self.unpackAllHyperParameters(values=False)
            if name in hpn:
                paramIndex = hpn.index(name)
            else:
                raise PostProcessingError('Could not find any hyper-parameter with name: {}.'.format(name))
        else:
            # index provided: check if proposed hyper-parameter can indeed be found at that index
            # (and refine index if necessary)
            hpn = self.unpackHyperParameters(transitionModel)
            for i in index:
                try:
                    hpn = hpn[i - 1]
                except IndexError:
                    raise PostProcessingError('Could not find any hyper-parameter at index {}.'.format(index))

            if not hpn == name:
                if name in hpn:
                    index.append(hpn.index(name))
                else:
                    raise PostProcessingError('Could not find hyper-parameter {} at index {}.'.format(name, index))

            # find index of hyper-parameter in flattened list of hyper-parameter values
            paramIndex = 0
            # effectively loop through all hyper-parameters before the one specified by the user
            for i, idx in enumerate(index[:-1]):
                for j in range(idx - 1):
                    hpn = self.unpackHyperParameters(transitionModel)
                    temp = index[:i] + [j]
                    for t in temp:
                        hpn = hpn[t]
                    paramIndex += len(list(flatten(hpn)))
            paramIndex += index[-1]

        return paramIndex

    def getHyperParameterMeanValue(self, t, transitionModelIndex=0):
        """
        Computes the mean value of the joint hyper-parameter distribution for a given transition model at a given time
        step.

        Args:
            t(int): Time step at which to compute distribution
            transitionModelIndex(int): Index of the transition model that contains the hyper-parameter; default: 0
                (first transition model)

        Returns:
            ndarray: Array containing the mean values of all hyper-parameters of the given transition model
        """
        try:
            hyperParameterDistribution = self.hyperParameterSequence[t]
        except IndexError:
            raise PostProcessingError('No hyper-parameter distribution found for t={}. Choose 0 <= t <= {}.'
                                      .format(t, len(self.formattedTimestamps) - 1))

        try:
            hyperParameterDistribution = hyperParameterDistribution[transitionModelIndex][:, None]
            hyperParameterValues = self.hyperParameterValues[transitionModelIndex]
            hyperGridConstants = self.hyperGridConstants[transitionModelIndex]
        except IndexError:
            raise PostProcessingError('Transition model with index {} does not exist. Options: 0-{}.'
                                      .format(transitionModelIndex, len(self.transitionModels) - 1))

        return np.sum(hyperParameterValues*hyperParameterDistribution*np.prod(hyperGridConstants), axis=0)

    def getHyperParameterMeanValues(self, transitionModelIndex=0):
        """
        Computes the sequence of mean value of the joint hyper-parameter distribution for a given transition model for
        all time steps.

        Args:
            transitionModelIndex(int): Index of the transition model that contains the hyper-parameter; default: 0
                (first transition model)

        Returns:
            ndarray: Array containing the sequences of mean values for all hyper-parameters of the given transition
                model
        """
        try:
            hyperParameterSequence = np.array(self.hyperParameterSequence)[:, transitionModelIndex][:, :, None]
            hyperParameterValues = self.hyperParameterValues[transitionModelIndex]
            hyperGridConstants = self.hyperGridConstants[transitionModelIndex]
        except IndexError:
            raise PostProcessingError('Transition model with index {} does not exist. Options: 0-{}.'
                                      .format(transitionModelIndex, len(self.transitionModels) - 1))

        return np.sum(hyperParameterSequence * hyperParameterValues * np.prod(hyperGridConstants), axis=1).T

    def getHyperParameterDistribution(self, t, transitionModelIndex=0, param=0, plot=False, **kwargs):
        """
        Computes marginal hyper-parameter distribution of a single hyper-parameter at a specific time step in an
        OnlineStudy fit.

        Args:
            t(int): Time step at which to compute distribution
            transitionModelIndex(int): Index of the transition model that contains the hyper-parameter; default: 0
                (first transition model)
            param(str, int): Parameter name or index of hyper-parameter to display; default: 0
                (first model hyper-parameter)
            plot(bool): If True, a bar chart of the distribution is created
            **kwargs: All further keyword-arguments are passed to the bar-plot (see matplotlib documentation)

        Returns:
            ndarray, ndarray: The first array contains the hyper-parameter values, the second one the
                corresponding probability (density) values
        """
        try:
            hyperParameterDistribution = self.hyperParameterSequence[t]
        except IndexError:
            raise PostProcessingError('No hyper-parameter distribution found for t={}. Choose 0 <= t <= {}.'
                                      .format(t, len(self.formattedTimestamps)-1))

        try:
            hyperParameterDistribution = hyperParameterDistribution[transitionModelIndex]
        except IndexError:
            raise PostProcessingError('Transition model with index {} does not exist. Options: 0-{}.'
                                      .format(transitionModelIndex, len(self.transitionModels)-1))

        if isinstance(param, int):
            if param >= len(self.hyperParameterNames[transitionModelIndex]):
                raise PostProcessingError('Hyper-parameter with index {} does not exist. Options: 0-{}.'
                                          .format(param, len(self.hyperParameterNames[transitionModelIndex])-1))
            paramIndex = param
        if isinstance(param, str):
            paramIndex = self.getHyperParameterIndex(self.transitionModels[transitionModelIndex], param)

        # marginalize the hyper-posterior probabilities
        hpv = np.array(self.hyperParameterValues[transitionModelIndex])
        paramHpv = hpv[:, paramIndex]
        uniqueValues = np.sort(np.unique(paramHpv))

        marginalDistribution = []
        for value in uniqueValues:
            indices = np.where(paramHpv == value)
            probabilities = hyperParameterDistribution[indices]
            marginalDistribution.append(np.sum(probabilities))
        marginalDistribution = np.array(marginalDistribution)

        d = uniqueValues[1:] - uniqueValues[:-1]
        dd = d[1:] - d[:-1]
        if np.all(dd < 10 ** -10):  # for equally spaced values, set difference as grid-constant
            gridConstant = np.abs(d[0])
            marginalDistribution /= gridConstant  # re-normalize to get probability density
        else:  # for irregularly spaced values (e.g. categorical), set grid-constant to 1
            gridConstant = 1

        if plot:
            plt.bar(uniqueValues, marginalDistribution, align='center', width=gridConstant, **kwargs)

            plt.xlabel(self.hyperParameterNames[transitionModelIndex][paramIndex])

            # in case an integer step size for hyper-parameter values is chosen, probability is displayed
            # (probability density otherwise)
            if gridConstant == 1.:
                plt.ylabel('probability')
            else:
                plt.ylabel('probability density')

        return uniqueValues, marginalDistribution

    def getHyperParameterDistributions(self, transitionModelIndex=0, param=0):
        """
        Computes marginal hyper-parameter distributions of a single hyper-parameter for all time steps in an OnlineStudy
        fit.

        Args:
            transitionModelIndex(int): Index of the transition model that contains the hyper-parameter; default: 0
                (first transition model)
            param(str, int): Parameter name or index of hyper-parameter to display; default: 0
                (first model hyper-parameter)

        Returns:
            ndarray, ndarray: The first array contains the hyper-parameter values, the second one the
                corresponding probability (density) values (first axis is time).
        """
        try:
            hyperParameterSequence = np.array(self.hyperParameterSequence)[:, transitionModelIndex]
        except IndexError:
            raise PostProcessingError('Transition model with index {} does not exist. Options: 0-{}.'
                                      .format(transitionModelIndex, len(self.transitionModels) - 1))

        if isinstance(param, int):
            if param >= len(self.hyperParameterNames[transitionModelIndex]):
                raise PostProcessingError('Hyper-parameter with index {} does not exist. Options: 0-{}.'
                                          .format(param, len(self.hyperParameterNames[transitionModelIndex]) - 1))
            paramIndex = param
        elif isinstance(param, str):
            paramIndex = self.getHyperParameterIndex(self.transitionModels[transitionModelIndex], param)
        else:
            raise PostProcessingError("Keyword argument 'param' must be integer or string")

        # marginalize the hyper-posterior probabilities
        hpv = np.array(self.hyperParameterValues[transitionModelIndex])
        paramHpv = hpv[:, paramIndex]
        uniqueValues = np.sort(np.unique(paramHpv))

        marginalDistribution = []
        for value in uniqueValues:
            marginalDistribution.append([])
            indices = np.where(paramHpv == value)
            for hp in hyperParameterSequence:
                probabilities = hp[indices]
                marginalDistribution[-1].append(np.sum(probabilities))
        marginalDistribution = np.array(marginalDistribution).T
        marginalDistribution /= np.sum(marginalDistribution, axis=1)[:, None]

        return uniqueValues, marginalDistribution

    def plotHyperParameterEvolution(self, transitionModelIndex=0, param=0, color='b', gamma=0.5, **kwargs):
        """
        Plot method to display a series of marginal posterior distributions corresponding to a single model parameter.
        This method includes the removal of plotting artefacts, gamma correction as well as an overlay of the posterior
        mean values.

        Args:
            transitionModelIndex(int): Index of the transition model that contains the hyper-parameter; default: 0
                (first transition model)
            param(str, int): parameter name or index of parameter to display; default: 0 (first model parameter)
            color: color from which a light colormap is created
            gamma(float): exponent for gamma correction of the displayed marginal distribution; default: 0.5
            kwargs: all further keyword-arguments are passed to the plot of the posterior mean values
        """
        # get sequence of hyper-parameter distributions
        uniqueValues, marginalDistribution = self.getHyperParameterDistributions(transitionModelIndex, param)

        # re-compute hyper-parameter index
        if isinstance(param, int):
            if param >= len(self.hyperParameterNames[transitionModelIndex]):
                raise PostProcessingError('Hyper-parameter with index {} does not exist. Options: 0-{}.'
                                          .format(param, len(self.hyperParameterNames[transitionModelIndex]) - 1))
            paramIndex = param
        elif isinstance(param, str):
            paramIndex = self.getHyperParameterIndex(self.transitionModels[transitionModelIndex], param)
        else:
            raise PostProcessingError("Keyword argument 'param' must be integer or string")

        # compute hyper-posterior mean values
        meanValues = self.getHyperParameterMeanValues(transitionModelIndex)[paramIndex]

        # clean up very small probability values, as they may create image artefacts
        pmax = np.amax(marginalDistribution)
        marginalDistribution[marginalDistribution < pmax * (10 ** -20)] = 0

        plt.imshow(marginalDistribution.T ** gamma,
                   origin=0,
                   cmap=createColormap(color),
                   extent=[self.formattedTimestamps[0], self.formattedTimestamps[-1]] + [uniqueValues[0], uniqueValues[-1]],
                   aspect='auto')

        # set default color of plot to black
        if ('c' not in kwargs) and ('color' not in kwargs):
            kwargs['c'] = 'k'

        # set default linewidth to 1.5
        if ('lw' not in kwargs) and ('linewidth' not in kwargs):
            kwargs['lw'] = 1.5

        plt.plot(self.formattedTimestamps, meanValues, **kwargs)

        plt.ylim(uniqueValues[0], uniqueValues[-1])
        plt.ylabel(self.hyperParameterNames[transitionModelIndex][paramIndex])
        plt.xlabel('time step')
