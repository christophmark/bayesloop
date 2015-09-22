#!/usr/bin/env python
"""
This file introduces the main class used for data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from .preprocessing import *
from .transitionModels import CombinedTransitionModel
from .transitionModels import SerialTransitionModel


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

        self.rawData = np.array([])
        self.formattedData = np.array([])

        self.posteriorSequence = []
        self.posteriorMeanValues = []
        self.logEvidence = 0
        self.localEvidence = []

        self.parametersToOptimize = []

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

        print '+ Successfully imported example data.'

    def loadData(self, array):
        """
        Loads Numpy array as data.

        Parameters:
            array - Numpy array containing time series data

        Returns:
            None
        """
        self.rawData = array
        print '+ Successfully imported array.'

    def createGrid(self):
        """
        Creates parameter grid, based on given boundaries and grid size. Note that the given parameter boundaries are
        not included as points on the grid (to avoid e.g. division by zero in case zero is chosen as boundary value).

        Parameters:
            None

        Returns:
            None
        """

        if not self.gridSize:
            print '! Grid-size not set.'
            print '  Setting default grid-size:', self.observationModel.defaultGridSize
            self.gridSize = self.observationModel.defaultGridSize

        self.marginalGrid = [np.linspace(b[0], b[1], g+2)[1:-1] for b, g in zip(self.boundaries, self.gridSize)]
        self.grid = [m for m in np.meshgrid(*self.marginalGrid, indexing='ij')]
        self.latticeConstant = [g[1]-g[0] for g in self.marginalGrid]

        if self.transitionModel != None:
            self.transitionModel.latticeConstant = self.latticeConstant

    def setBoundaries(self, newBoundaries):
        """
        Sets lower and upper parameter boundaries (and updates grid accordingly).

        Parameters:
            newBoundaries - A list of lists which each contain a lower & upper parameter boundary
                            Example: [[-1, 1], [0, 2]]

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

        if not silent:
            print '+ Observation model:', M

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
            print '+ Transition model:', K

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
        if not self.checkConsistency():
            return

        if not silent:
            print '+ Started new fit:'

        self.formattedData = movingWindow(self.rawData, self.observationModel.segmentLength)
        if not silent:
            print '    + Formatted data.'

        # initialize array for posterior distributions
        if not evidenceOnly:
            self.posteriorSequence = np.empty([len(self.formattedData)]+self.gridSize)

        # initialize array for computed evidence (marginal likelihood)
        self.logEvidence = 0
        self.localEvidence = np.empty(len(self.formattedData))

        # forward pass
        alpha = np.ones(self.gridSize)/np.prod(np.array(self.gridSize))  # initial flat prior
        for i in np.arange(0, len(self.formattedData)):

            # compute likelihood
            likelihood = self.observationModel.processedPdf(self.grid, self.formattedData[i])

            # update alpha based on likelihood
            alpha *= likelihood

            # normalization constant of alpha is used to compute evidence
            norm = np.sum(alpha)
            self.logEvidence += np.log(norm)
            self.localEvidence[i] = norm  # in case we return before backward pass (forwardOnly = True)

            # normalize alpha (for numerical stability)
            alpha /= norm

            # alphas are stored as preliminary posterior distributions
            if not evidenceOnly:
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
                likelihood = self.observationModel.processedPdf(self.grid, self.formattedData[i])

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

    def optimize(self, parameterList=[]):
        """
        Uses the COBYLA minimization algorithm from SciPy to perform a maximization of the log-evidence with respect
        to all hyper-parameters (the parameters of the transition model) of a time seris model. The starting values
        are the values set by the user when defining the transition model.

        For the optimization, only the log-evidence is computed and no parameter distributions are stored. When a local
        maximum is found, the parameter distribution is computed based on the optimal values for the hyper-parameters.

        Parameters:
            None

        Returns:
            None
        """
        if not self.checkConsistency():
            return

        # set list of parameters to optimize
        if isinstance(parameterList, basestring):  # in case only a single parameter name is provided as a string
            self.parametersToOptimize = [parameterList]
        else:
            self.parametersToOptimize = parameterList

        print '+ Starting optimization...'
        if self.parametersToOptimize:
            print '  --> Parameter(s) to optmimize:', self.parametersToOptimize
        else:
            print '  --> All model parameters are optmimized.'

        # create parameter list to set start values for optimization
        x0 = self.unpackHyperParameters()

        # check if valid parameter names were entered
        if len(x0) == 0:
            print '! No parameters to optimize. Check parameter names.'
            # reset list of parameters to optimize, so that unpacking and setting hyper-parameters works as expected
            self.parametersToOptimize = []
            return

        # perform optimization (maximization of log-evidence)
        result = minimize(self.optimizationStep, x0, method='COBYLA')

        print '+ Finished optimization.'

        # set optimal hyperparameters in transition model
        self.setHyperParameters(result.x)

        # run analysis with optimal parameter values
        self.fit()

        # reset list of parameters to optimize, so that unpacking and setting hyper-parameters works as expected
        self.parametersToOptimize = []

    def optimizationStep(self, x):
        """
        Wrapper for the fit method to use it in conjunction with scipy.optimize.minimize.

        Parameters:
            x - unpacked list of current hyper-parameter values

        Returns:
            negative log-evidence that is subject to minimization
        """
        # set new hyperparameters in transition model
        self.setHyperParameters(x)

        # compute log-evidence
        self.fit(evidenceOnly=True, silent=True)

        print '    + Log10-evidence: {:.5f}'.format(self.logEvidence / np.log(10)), '- Parameter values:', x

        # return negative log-evidence (is minimized to maximize evidence)
        return -self.logEvidence

    def unpackHyperParameters(self):
        """
        The parameters of a transition model can be split between several submodels (using CombinedTransitionModel) and
        can be lists of values (multiple standard deviations in GaussianRandomWalk). This function unpacks the hyper-
        parameters, resulting in a single list of values that can be fed to the optimization step routine.

        Parameters:
            None

        Returns:
            list of current hyper-parameter values
        """
        if isinstance(self.transitionModel, (CombinedTransitionModel, SerialTransitionModel)):
            models = self.transitionModel.models
        else:
            # only one model in a non-combined transition model
            models = [self.transitionModel]

        x0 = []  # initialize parameter list
        for i, model in enumerate(models):
            for key in model.hyperParameters.keys():
                # check whether list of parameters to optimize is set and contains the current parameter
                if self.parametersToOptimize and not (key in self.parametersToOptimize):
                    ignoreParameter = True  # by default, these parameters are ignored

                    # check for special notation (e.g. 'sigma_2')
                    for p in self.parametersToOptimize:
                        stringList = p.split('_')
                        # check if notation & key is correct & integer is at the end
                        if len(stringList) > 1 and '_'.join(stringList[:-1]) == key and stringList[-1].isdigit():
                            # check whether current model is the right one
                            if int(stringList[-1]) - 1 == i:
                                ignoreParameter = False  # exception due to special notation

                    if ignoreParameter:
                        continue

                # if parameter itself is a list, we need to unpack
                if type(model.hyperParameters[key]) is list:
                    length = len(model.hyperParameters[key])
                    for j in range(length):
                        x0.append(model.hyperParameters[key][j])
                # if parameter is single float, no unpacking is needed
                else:
                    x0.append(model.hyperParameters[key])

        return x0

    def setHyperParameters(self, x):
        """
        The parameters of a transition model can be split between several submodels (using CombinedTransitionModel) and
        can be lists of values (multiple standard deviations in GaussianRandomWalk). This function takes a list of
        values and sets the corresponding variables in the transition model instance.

        Parameters:
            x - list of values (e.g. from unpackHyperparameters)

        Returns:
            None
        """
        if isinstance(self.transitionModel, (CombinedTransitionModel, SerialTransitionModel)):
            models = self.transitionModel.models
        else:
            # only one model in a non-combined transition model
            models = [self.transitionModel]

        paramList = list(x[:])  # make copy of previous parameter list
        for i, model in enumerate(models):
            for key in model.hyperParameters.keys():
                # check whether list of parameters to optimize is set and contains the current parameter
                if self.parametersToOptimize and not (key in self.parametersToOptimize):
                    ignoreParameter = True  # by default, these parameters are ignored

                    # check for special notation (e.g. 'sigma_2')
                    for p in self.parametersToOptimize:
                        stringList = p.split('_')
                        # check if notation & key is correct & integer is at the end
                        if len(stringList) > 1 and '_'.join(stringList[:-1]) == key and stringList[-1].isdigit():
                            # check whether current model is the right one
                            if int(stringList[-1]) - 1 == i:
                                ignoreParameter = False  # exception due to special notation

                    if ignoreParameter:
                        continue

                # if parameter itself is a list, we need to unpack
                if type(model.hyperParameters[key]) is list:
                    length = len(model.hyperParameters[key])
                    model.hyperParameters[key] = []
                    for j in range(length):
                        model.hyperParameters[key].append(paramList[0])
                        paramList.pop(0)
                # if parameter is single float, no unpacking is needed
                else:
                    model.hyperParameters[key] = paramList[0]
                    paramList.pop(0)

    def plotParameterEvolution(self, param=0, xLower=None, xUpper=None, color='b'):
        """
        Plots a series of marginal posterior distributions corresponding to a single model parameter, together with the
        posterior mean values.

        Parameters:
            param - parameter name or index of parameter to display; default: 0 (first model parameter)

            color - color from which a light colormap is created

        Returns:
            None
        """

        if isinstance(param, (int, long)):
            paramIndex = param
        elif isinstance(param, basestring):
            paramIndex = -1
            for i, name in enumerate(self.observationModel.parameterNames):
                if name == param:
                    paramIndex = i

            # check if match was found
            if paramIndex == -1:
                print 'ERROR: Wrong parameter name. Available options: {0}'.format(self.observationModel.parameterNames)
                return
        else:
            print 'ERROR: Wrong parameter format. Specify parameter via name or index.'
            return

        axesToMarginalize = range(1, len(self.observationModel.parameterNames) + 1)  # axis 0 is time
        axesToMarginalize.remove(paramIndex + 1)

        marginalPosteriorSequence = np.squeeze(np.apply_over_axes(np.sum, self.posteriorSequence, axesToMarginalize))

        if not xLower:
            xLower = 0
        if not xUpper:
            if xLower:
                print '! If lower x limit is provided, upper limit has to be provided, too.'
                print '  Setting lower limit to zero.'
            xLower = 0
            xUpper = len(marginalPosteriorSequence)

        plt.imshow(marginalPosteriorSequence.T,
                   origin=0,
                   cmap=sns.light_palette(color, as_cmap=True),
                   extent=[xLower, xUpper - 1] + self.boundaries[paramIndex],
                   aspect='auto')

        plt.plot(np.arange(xLower, xUpper), self.posteriorMeanValues[paramIndex], c='k', lw=1.5)

        plt.ylim(self.boundaries[paramIndex])
        plt.ylabel(self.observationModel.parameterNames[paramIndex])
        plt.xlabel('time step')

    def checkConsistency(self):
        """
        This method is called at the very beginning of analysis methods to ensure that all necessary elements of the
        model are set correctly.

        Parameters:
            None

        Returns:
            True if all is well; False if problem with user input is detected.
        """
        if not self.observationModel:
            print '! No observation model chosen.'
            return False
        if not self.transitionModel:
            print '! No transition model chosen.'
            return False
        if not self.boundaries:
            print '! No parameter boundaries are set.'
            print '  Setting default boundaries:', self.observationModel.defaultBoundaries
            print '  Restart analysis to use these boundary values. To change boundaries, call setBoundaries().'
            self.setBoundaries(self.observationModel.defaultBoundaries)
            return False
        if not len(self.observationModel.defaultGridSize) == len(self.gridSize):
            print '! Specified parameter grid expects {0} parameter(s), but observation model has {1} parameter(s).'\
                .format(len(self.gridSize), len(self.observationModel.defaultGridSize))
            print '  Default grid-size for the chosen observation model: {0}'\
                .format(self.observationModel.defaultGridSize)
            return False
        if not len(self.observationModel.defaultBoundaries) == len(self.boundaries):
            print '! Parameter boundaries specify {0} parameter(s), but observation model has {1} parameter(s).'\
                .format(len(self.boundaries), len(self.observationModel.defaultBoundaries))
            print '  Default boundaries for the chosen observation model: {0}'\
                .format(self.observationModel.defaultBoundaries)
            return False

        return True
