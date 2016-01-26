#!/usr/bin/env python
"""
This file introduces the main class used for data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.misc import factorial
import sympy.abc as abc
from sympy import Symbol
from sympy import lambdify
from sympy.stats import density
import sympy.stats
from .helper import *
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

        self.selectedHyperParameters = []

        print '+ Created new study.'

    def loadExampleData(self):
        """
        Loads UK coal mining disaster data.

        Parameters:
            None

        Returns:
            None
        """
        self.rawData = np.array([5, 4, 1, 0, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4,
                                 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0,
                                 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0,
                                 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 3, 3, 0,
                                 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])

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
            print '! Grid-size not set (needed to set boundaries).'
            print '  Setting default grid-size:', self.observationModel.defaultGridSize
            self.gridSize = self.observationModel.defaultGridSize

        self.marginalGrid = [np.linspace(b[0], b[1], g+2)[1:-1] for b, g in zip(self.boundaries, self.gridSize)]
        self.grid = [m for m in np.meshgrid(*self.marginalGrid, indexing='ij')]
        self.latticeConstant = [g[1]-g[0] for g in self.marginalGrid]

        if self.transitionModel != None:
            self.transitionModel.latticeConstant = self.latticeConstant

    def setBoundaries(self, newBoundaries, silent=False):
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
        if not silent:
            print '+ Boundaries: {}'.format(self.boundaries)

    def setGridSize(self, newGridSize, silent=False):
        """
        Sets grid size for discretization of parameter distributions (and updates grid accordingly).

        Parameters:
            newGridSize - List of integers describing the size of the parameter grid for each dimension

        Returns:
            None
        """
        self.gridSize = newGridSize
        self.createGrid()
        if not silent:
            print '+ Grid size: {}'.format(self.gridSize)

    def setGrid(self, newGrid, silent=False):
        """
        Sets parameter boundaries and corresponding grid size. Provides a more convenient way to specify the parameter
        grid than calling 'setBoundaries' and 'setGridSize' separately.

        Parameters:
            newGrid - List of lists, one for each parameter, containing the lower and upper parameter boundaries and
                an integer value describing the size of the grid in the corresponding dimension.
                Example: [[0., 1., 1000], [-1., 1., 100]]

        Returns:
            None
        """
        newBoundaries = []
        newGridSize = []
        for entry in newGrid:
            newBoundaries.append([entry[0], entry[1]])
            newGridSize.append(entry[2])

        self.boundaries = newBoundaries
        self.gridSize = newGridSize
        self.createGrid()

        if not silent:
            print '+ Boundaries: {}'.format(self.boundaries)
            print '+ Grid size: {}'.format(self.gridSize)

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
            print '+ Observation model: {}. Parameter(s): {}'.format(M, M.parameterNames)

    def setPrior(self, prior, silent=False):
        """
        Sets a prior distribution for the parameters of the observation model. The custom prior distribution may be
        passed as a (lambda) function or as a (list of) SymPy random variable(s).

        Parameters:
            prior - SymPy Random Symbol (for a one-parameter model), list of SymPy Random Symbols (for multi-parameter
                models), or a (lambda) function that takes as many arguments as there are parameters in the observation
                model.

            silent - If set to True, no output is generated by this method.

        Returns:
            None
        """
        # check whether observation model is defined
        if self.observationModel is None:
            print '! Observation model has to be defined before setting prior distribution.'
            return

        # check whether function is provided
        if hasattr(prior, '__call__'):
            self.observationModel.prior = prior
            if not silent:
                print '+ Set custom prior: {}'.format(prior.__name__)
            return

        # check whether single random variable is provided
        if type(prior) is sympy.stats.rv.RandomSymbol:
            prior = [prior]

        # check if list/tuple is provided
        if isinstance(prior, (list, tuple)) and not isinstance(prior, basestring):
            if len(prior) != len(self.observationModel.parameterNames):
                print '! Observation model contains {} parameters, but {} priors were provided.'.format(
                     len(self.observationModel.parameterNames),
                    len(prior)
                    )
                print '  Using flat prior.'
                return

            pdf = 1
            x = [abc.x]*len(prior)
            for i, rv in enumerate(prior):
                if type(rv) is not sympy.stats.rv.RandomSymbol:
                    print '! Only lambda functions or SymPy random variables can be used as a prior.'
                    return
                if len(list(rv._sorted_args[0].distribution.free_symbols)) > 0:
                    print '! Prior distribution must not contain free parameters.'
                    return

                # multiply total pdf with density for current parameter
                pdf = pdf*density(rv)(x[i])

            # set density as lambda function
            self.observationModel.prior = lambdify(x, pdf, modules=['numpy', {'factorial': factorial}])
            if not silent:
                print '+ Set custom prior: {}'.format(pdf)

    def setTransitionModel(self, K, silent=False):
        """
        Set transition model which describes the parameter dynamics.

        Parameters:
            K - Transition model class (see transitionModel.py)

        Returns:
            None
        """
        self.transitionModel = K
        self.transitionModel.study = self
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
                forwardOnly option, no posterior mean values are computed and no posterior distributions are stored.

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

        # set prior distribution for forward-pass
        if self.observationModel.prior is not None:
            alpha = self.observationModel.prior(*self.grid)
        else:
            alpha = np.ones(self.gridSize)  # flat prior

        # normalize prior (necessary in case an improper prior is used)
        alpha /= np.sum(alpha)

        # forward pass
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
            if norm > 0.:
                alpha /= norm
            else:
                # if all probability values are zero, normalization is not possible
                print '    ! Forward pass distribution contains only zeros, check parameter boundaries!'
                print '      Stopping inference process. Setting model evidence to zero.'
                self.logEvidence = -np.inf
                return

            # alphas are stored as preliminary posterior distributions
            if not evidenceOnly:
                self.posteriorSequence[i] = alpha

            # compute alpha for next iteration
            alpha = self.transitionModel.computeForwardPrior(alpha, i)

        if not silent:
            print '    + Finished forward pass.'
            print '    + Log10-evidence: {:.5f}'.format(self.logEvidence / np.log(10))

        if not (forwardOnly or evidenceOnly):
            # set prior distribution for backward-pass
            if self.observationModel.prior is not None:
                beta = self.observationModel.prior(*self.grid)
            else:
                beta = np.ones(self.gridSize)  # flat prior

            # normalize prior (necessary in case an improper prior is used)
            beta /= np.sum(beta)

            # backward pass
            for i in np.arange(0, len(self.formattedData))[::-1]:
                # posterior ~ alpha*beta
                self.posteriorSequence[i] *= beta  # alpha*beta

                # normalize posterior wrt the parameters
                norm = np.sum(self.posteriorSequence[i])
                if norm > 0.:
                    self.posteriorSequence[i] /= np.sum(self.posteriorSequence[i])
                else:
                    # if all posterior probabilities are zero, normalization is not possible
                    print '    ! Posterior distribution contains only zeros, check parameter boundaries!'
                    print '      Stopping inference process. Setting model evidence to zero.'
                    self.logEvidence = -np.inf
                    return

                # re-compute likelihood
                likelihood = self.observationModel.processedPdf(self.grid, self.formattedData[i])

                # compute local evidence
                try:
                    self.localEvidence[i] = 1./np.sum(self.posteriorSequence[i]/likelihood)
                except:  # in case division by zero happens
                    self.localEvidence[i] = np.nan

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

    def optimize(self, parameterList=[], **kwargs):
        """
        Uses the COBYLA minimization algorithm from SciPy to perform a maximization of the log-evidence with respect
        to all hyper-parameters (the parameters of the transition model) of a time seris model. The starting values
        are the values set by the user when defining the transition model.

        For the optimization, only the log-evidence is computed and no parameter distributions are stored. When a local
        maximum is found, the parameter distribution is computed based on the optimal values for the hyper-parameters.

        Parameters:
            parameterList - List of hyper-parameter names to optimize. For nested transition models with multiple,
                identical hyper-parameter names, the sub-model index can be provided. By default, all hyper-parameters
                are optimized. For further information, see:
                http://nbviewer.ipython.org/github/christophmark/bayesloop/blob/master/docs/bayesloop_tutorial.ipynb#section_3.2

            kwargs - All other keyword parameters are passed to the 'minimize' routine of scipy.optimize.

        Returns:
            None
        """
        # set list of parameters to optimize
        if isinstance(parameterList, basestring):  # in case only a single parameter name is provided as a string
            self.selectedHyperParameters = [parameterList]
        else:
            self.selectedHyperParameters = parameterList

        print '+ Starting optimization...'
        if not self.checkConsistency():
            return

        if self.selectedHyperParameters:
            print '  --> Parameter(s) to optimize:', self.selectedHyperParameters
        else:
            print '  --> All model parameters are optimized (except change/break-points).'
            # load all hyper-parameter names
            self.selectedHyperParameters = list(flatten(self.unpackHyperParameters(self.transitionModel)))
            # delete all occurrences of 'tChange' or 'tBreak'
            self.selectedHyperParameters = [x for x in self.selectedHyperParameters
                                            if (x != 'tChange') and (x != 'tBreak')]

        # create parameter list to set start values for optimization
        x0 = self.unpackSelectedHyperParameters()

        # check if valid parameter names were entered
        if len(x0) == 0:
            print '! No parameters to optimize. Check parameter names.'
            # reset list of parameters to optimize, so that unpacking and setting hyper-parameters works as expected
            self.selectedHyperParameters = []
            return

        # perform optimization (maximization of log-evidence)
        result = minimize(self.optimizationStep, x0, method='COBYLA', **kwargs)

        print '+ Finished optimization.'

        # set optimal hyperparameters in transition model
        self.setSelectedHyperParameters(result.x)

        # run analysis with optimal parameter values
        self.fit()

        # reset list of parameters to optimize, so that unpacking and setting hyper-parameters works as expected
        self.selectedHyperParameters = []

    def optimizationStep(self, x):
        """
        Wrapper for the fit method to use it in conjunction with scipy.optimize.minimize.

        Parameters:
            x - unpacked list of current hyper-parameter values

        Returns:
            negative log-evidence that is subject to minimization
        """
        # set new hyperparameters in transition model
        self.setSelectedHyperParameters(x)

        # compute log-evidence
        self.fit(evidenceOnly=True, silent=True)

        print '    + Log10-evidence: {:.5f}'.format(self.logEvidence / np.log(10)), '- Parameter values:', x

        # return negative log-evidence (is minimized to maximize evidence)
        return -self.logEvidence

    def unpackHyperParameters(self, transitionModel, values=False):
        """
        Returns list of all hyper-parameters (names or values), nested as the transition model.

        Parameters:
            transitionModel - An instance of a transition model
            values - By default, parameter names are returned; if set to True, parameter values are returned

        Returns:
            List of hyper-parameters (names or values)
        """
        paramList = []
        # recursion step for sub-models
        if hasattr(transitionModel, 'models'):
            for m in transitionModel.models:
                paramList.append(self.unpackHyperParameters(m, values=values))

        # extend hyper-parameter based on current (sub-)model
        if hasattr(transitionModel, 'hyperParameters'):
            if values:
                paramList.extend(transitionModel.hyperParameters.values())
            else:
                paramList.extend(transitionModel.hyperParameters.keys())

        return paramList

    def unpackAllHyperParameters(self):
        """
        Returns a flattened list of all hyper-parameter values of the current transition model.
        """
        return list(flatten(self.unpackHyperParameters(self.transitionModel, values=True)))

    def unpackSelectedHyperParameters(self):
        """
        The parameters of a transition model can be split between several sub-models (using CombinedTransitionModel or
        SerialTransitionModel) and can be lists of values (multiple standard deviations in GaussianRandomWalk). This
        function unpacks the hyper-parameters, resulting in a single list of values that can be fed to the optimization
        step routine. Note that only the hyper-parameters that are noted (by name) in the attribute
        selectedHyperParameters are regarded.

        Parameters:
            None

        Returns:
            list of current hyper-parameter values if successful, 0 otherwise
        """
        # if no hyper-parameters are selected, choose all
        if not self.selectedHyperParameters:
            return self.unpackAllHyperParameters()

        # if self.selectedHyperParameters is not empty
        nameTree = self.unpackHyperParameters(self.transitionModel)
        valueTree = self.unpackHyperParameters(self.transitionModel, values=True)
        output = []

        # loop over selected hyper-parameters
        for x in self.selectedHyperParameters:
            l = x.split('_')
            name = []
            index = []
            for s in l:
                try:
                    index.append(int(s))
                except:
                    name.append(s)
            name = '_'.join(name)

            # no index provided
            if len(index) == 0:
                iFound = recursiveIndex(nameTree, name)  # choose first hit
                if len(iFound) == 0:
                    print '! Could not find any hyper-parameter named {}.'.format(name)
                    return 0

                value = valueTree[:]
                for i in iFound:
                    value = value[i]

                # if parameter value is list, we need to unpack
                if isinstance(value, (list, tuple)):
                    for v in value:
                        output.append(v)
                else:
                    output.append(value)

                # remove occurrence from nameTree (if name is listed twice, use second occurrence...)
                assignNestedItem(nameTree, iFound, ' ')

            # index provided
            if len(index) > 0:
                temp = nameTree[:]
                value = valueTree[:]
                for i in index:
                    try:
                        temp = temp[i-1]
                        value = value[i-1]  # notation starts to count at 1
                    except:
                        print '! Could not find any hyper-parameter at index {}.'.format(index)
                        return 0

                # find parameter within sub-model
                try:
                    value = value[temp.index(name)]
                    # if parameter value is list, we need to unpack
                    if isinstance(value, (list, tuple)):
                        for v in value:
                            output.append(v)
                    else:
                        output.append(value)
                except:
                    print '! Could not find hyper-parameter {} at index {}.'.format(name, index)
                    return 0

        # return selected values of hyper-parameters
        return output

    def setAllHyperParameters(self, x):
        """
        Sets all current hyper-parameters, based on a flattened list of parameter values.

        Parameters:
            x - list of values (e.g. from unpackSelectedHyperParameters)

        Returns:
            None
        """
        paramList = list(x[:])  # make copy of parameter list

        nameTree = self.unpackHyperParameters(self.transitionModel)
        namesFlat = list(flatten(self.unpackHyperParameters(self.transitionModel)))

        for name in namesFlat:
            index = recursiveIndex(nameTree, name)

            # get correct sub-model
            model = self.transitionModel
            for i in index[:-1]:
                model = model.models[i]

            # if parameter value is list, we need to unpack
            if isinstance(model.hyperParameters[name], (list, tuple)):
                lst = []
                for i in range(len(model.hyperParameters[name])):
                    lst.append(paramList[0])
                    paramList.pop(0)
                model.hyperParameters[name] = lst
            else:
                model.hyperParameters[name] = paramList[0]
                paramList.pop(0)

            # remove occurrence from nameTree (if name is listed twice, use second occurrence...)
            assignNestedItem(nameTree, index, ' ')

    def setSelectedHyperParameters(self, x):
        """
        The parameters of a transition model can be split between several sub-models (using CombinedTransitionModel or
        SerialTransitionModel) and can be lists of values (multiple standard deviations in GaussianRandomWalk). This
        function takes a list of values and sets the corresponding variables in the transition model instance. Note that
        only the hyper-parameters that are noted (by name) in the attribute selectedHyperParameters are regarded.

        Parameters:
            x - list of values (e.g. from unpackSelectedHyperParameters)

        Returns:
            1, if successful, 0 otherwise
        """
        # if no hyper-parameters are selected, choose all
        if not self.selectedHyperParameters:
            self.setAllHyperParameters(x)
            return 1

        paramList = list(x[:])  # make copy of parameter list
        nameTree = self.unpackHyperParameters(self.transitionModel)

        # loop over selected hyper-parameters
        for x in self.selectedHyperParameters:
            l = x.split('_')
            name = []
            index = []
            for s in l:
                try:
                    index.append(int(s))
                except:
                    name.append(s)
            name = '_'.join(name)

            # no index provided
            if len(index) == 0:
                iFound = recursiveIndex(nameTree, name)  # choose first hit
                if len(iFound) == 0:
                    print '! Could not find any hyper-parameter named {}.'.format(name)
                    return 0

                # get correct sub-model
                model = self.transitionModel
                for i in iFound[:-1]:
                    model = model.models[i]

                # if parameter value is list, we need to unpack
                if isinstance(model.hyperParameters[name], (list, tuple)):
                    lst = []
                    for i in range(len(model.hyperParameters[name])):
                        lst.append(paramList[0])
                        paramList.pop(0)
                    model.hyperParameters[name] = lst
                else:
                    model.hyperParameters[name] = paramList[0]
                    paramList.pop(0)

                # remove occurrence from nameTree (if name is listed twice, use second occurrence...)
                assignNestedItem(nameTree, iFound, ' ')

            # index provided
            if len(index) > 0:
                model = self.transitionModel
                for i in index:
                    try:
                        model = model.models[i-1]  # user indices begin at 1
                    except:
                        print '! Could not find any hyper-parameter at index {}.'.format(index)
                        return 0

                # find parameter within sub-model
                try:
                    # if parameter value is list, we need to unpack
                    if isinstance(model.hyperParameters[name], (list, tuple)):
                        lst = []
                        for i in range(len(model.hyperParameters[name])):
                            lst.append(paramList[0])
                            paramList.pop(0)
                        model.hyperParameters[name] = lst
                    else:
                        model.hyperParameters[name] = paramList[0]
                        paramList.pop(0)
                except:
                    print '! Could not find hyper-parameter {} at index {}.'.format(name, index)
                    return 0
        return 1

    def plotParameterEvolution(self, param=0, xLower=None, xUpper=None, color='b', gamma=0.5, **kwargs):
        """
        Plots a series of marginal posterior distributions corresponding to a single model parameter, together with the
        posterior mean values.

        Parameters:
            param - parameter name or index of parameter to display; default: 0 (first model parameter)

            color - color from which a light colormap is created

            gamma - exponent for gamma correction of the displayed marginal distribution; default: 0.5

            kwargs - all further keyword-arguments are passed to the plot of the posterior mean values

        Returns:
            None
        """
        if self.posteriorSequence == []:
            print '! Cannot plot posterior sequence as it has not yet been computed. Run complete fit.'
            return

        if isinstance(param, (int, long)):
            paramIndex = param
        elif isinstance(param, basestring):
            paramIndex = -1
            for i, name in enumerate(self.observationModel.parameterNames):
                if name == param:
                    paramIndex = i

            # check if match was found
            if paramIndex == -1:
                print '! Wrong parameter name. Available options: {0}'.format(self.observationModel.parameterNames)
                return
        else:
            print '! Wrong parameter format. Specify parameter via name or index.'
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

        plt.imshow((marginalPosteriorSequence.T)**gamma,
                   origin=0,
                   cmap=sns.light_palette(color, as_cmap=True),
                   extent=[xLower, xUpper - 1] + self.boundaries[paramIndex],
                   aspect='auto')

        # set default color of plot to black
        if (not 'c' in kwargs) and (not 'color' in kwargs):
            kwargs['c'] = 'k'

        # set default linewidth to 1.5
        if (not 'lw' in kwargs) and (not 'linewidth' in kwargs):
            kwargs['lw'] = 1.5

        plt.plot(np.arange(xLower, xUpper), self.posteriorMeanValues[paramIndex], **kwargs)

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
            print '! No parameter boundaries are set. Trying to estimate appropriate boundaries.'
            try:
                estimatedBoundaries = self.observationModel.estimateBoundaries(self.rawData)
                print '  Setting estimated parameter boundaries:', estimatedBoundaries
                self.setBoundaries(estimatedBoundaries, silent=True)
            except:
                print '! Parameter boundaries could not be estimated. To set boundaries, call setBoundaries().'
                return False
        if not len(self.observationModel.defaultGridSize) == len(self.gridSize):
            print '! Specified parameter grid expects {0} parameter(s), but observation model has {1} parameter(s).'\
                .format(len(self.gridSize), len(self.observationModel.defaultGridSize))
            print '  Default grid-size for the chosen observation model: {0}'\
                .format(self.observationModel.defaultGridSize)
            return False
        if not len(self.observationModel.parameterNames) == len(self.boundaries):
            print '! Parameter boundaries specify {0} parameter(s), but observation model has {1} parameter(s).'\
                .format(len(self.boundaries), len(self.observationModel.parameterNames))
            return False

        # check if assigning values to selected hyper-parameters is successful
        if self.unpackSelectedHyperParameters() == 0:
            print '! Setting hyper-parameter values failed. Check hyper-parameter names.'
            return False

        # all is well
        return True
