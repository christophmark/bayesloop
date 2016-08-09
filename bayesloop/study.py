#!/usr/bin/env python
"""
This file introduces the main class used for data analysis.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
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
from .exceptions import *


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
        self.rawTimestamps = None
        self.formattedTimestamps = None

        self.posteriorSequence = []
        self.posteriorMeanValues = []
        self.logEvidence = 0
        self.localEvidence = []

        self.selectedHyperParameters = []

        print('+ Created new study.')

    def loadExampleData(self):
        """
        Loads UK coal mining disaster data.
        """
        self.rawData = np.array([5, 4, 1, 0, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4,
                                 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0,
                                 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0,
                                 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 3, 3, 0,
                                 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])

        self.rawTimestamps = np.arange(1852, 1962)

        print('+ Successfully imported example data.')

    def loadData(self, array, timestamps=None):
        """
        Loads Numpy array as data.

        Args:
            array: Numpy array containing time series data
            timestamps: Array of timestamps (same length as data array)
        """
        self.rawData = array
        if timestamps is not None:  # load custom timestamps
            if len(timestamps) == len(array):
                self.rawTimestamps = np.array(timestamps)
            else:
                print('! WARNING: Number of timestamps does not match number of data points. Omitting timestamps.')
        else:  # set default timestamps (integer range)
            self.rawTimestamps = np.arange(len(self.rawData))
        print('+ Successfully imported array.')

    def createGrid(self):
        """
        Creates parameter grid, based on given boundaries and grid size. Note that the given parameter boundaries are
        not included as points on the grid (to avoid e.g. division by zero in case zero is chosen as boundary value).
        """

        if not self.gridSize:
            print('! WARNING: Grid-size not set (needed to set boundaries).')
            print('  Setting default grid-size:', self.observationModel.defaultGridSize)
            self.gridSize = self.observationModel.defaultGridSize

        self.marginalGrid = [np.linspace(b[0], b[1], g+2)[1:-1] for b, g in zip(self.boundaries, self.gridSize)]
        self.grid = [m for m in np.meshgrid(*self.marginalGrid, indexing='ij')]
        self.latticeConstant = [np.abs(g[0]-g[1]) for g in self.marginalGrid]

        if self.transitionModel is not None:
            self.transitionModel.latticeConstant = self.latticeConstant

    def setBoundaries(self, newBoundaries):
        """
        Sets lower and upper parameter boundaries (and updates grid accordingly).

        Args:
            newBoundaries: A list of lists which each contain a lower & upper parameter boundary
                Example: [[-1, 1], [0, 2]]
        """
        self.boundaries = newBoundaries
        self.createGrid()
        print('+ Boundaries: {}'.format(self.boundaries))

    def setGridSize(self, newGridSize):
        """
        Sets grid size for discretization of parameter distributions (and updates grid accordingly).

        Args:
            newGridSize: List of integers describing the size of the parameter grid for each dimension
        """
        self.gridSize = newGridSize
        self.createGrid()
        print('+ Grid size: {}'.format(self.gridSize))

    def setGrid(self, newGrid):
        """
        Sets parameter boundaries and corresponding grid size. Provides a more convenient way to specify the parameter
        grid than calling 'setBoundaries' and 'setGridSize' separately.

        Args:
            newGrid: List of lists, one for each parameter, containing the lower and upper parameter boundaries and
                an integer value describing the size of the grid in the corresponding dimension.
                Example: [[0., 1., 1000], [-1., 1., 100]]
        """
        newBoundaries = []
        newGridSize = []
        for entry in newGrid:
            newBoundaries.append([entry[0], entry[1]])
            newGridSize.append(entry[2])

        self.boundaries = newBoundaries
        self.gridSize = newGridSize
        self.createGrid()

        print('+ Boundaries: {}'.format(self.boundaries))
        print('+ Grid size: {}'.format(self.gridSize))

    def setObservationModel(self, M, silent=False):
        """
        Sets observation model (likelihood function) for analysis.

        Args:
            M: Observation model class (see observationModel.py)
            silent: If set to True, no output is generated by this method.
        """
        self.observationModel = M
        print('+ Observation model: {}. Parameter(s): {}'.format(M, M.parameterNames))

    def setPrior(self, prior):
        """
        Sets a prior distribution for the parameters of the observation model. The custom prior distribution may be
        passed as a Numpy array that has tha same shape as the parameter grid, as a(lambda) function or as a (list of)
        SymPy random variable(s).

        Args:
            prior: SymPy Random Symbol (for a one-parameter model), list of SymPy Random Symbols (for multi-parameter
                models), a (lambda) function that takes as many arguments as there are parameters in the observation
                model, or a Numpy array with the same shape as the parameter grid.
        """
        # check whether observation model is defined
        if self.observationModel is None:
            raise ConfigurationError('Observation model has to be defined before setting prior distribution.')

        # check whether correctly shaped numpy array is provided
        if isinstance(prior, np.ndarray):
            if np.all(prior.shape == self.grid[0].shape):
                self.observationModel.prior = prior
                print('+ Set custom prior.')
                return
            else:
                raise ConfigurationError('Prior array does not match parameter grid size.')

        # check whether function is provided
        if hasattr(prior, '__call__'):
            self.observationModel.prior = prior
            print('+ Set custom prior: {}'.format(prior.__name__))
            return

        # check whether single random variable is provided
        if type(prior) is sympy.stats.rv.RandomSymbol:
            prior = [prior]

        # check if list/tuple is provided
        if isinstance(prior, (list, tuple)) and not isinstance(prior, str):
            if len(prior) != len(self.observationModel.parameterNames):
                raise ConfigurationError('Observation model contains {} parameters, but {} priors were provided.'
                                         .format(len(self.observationModel.parameterNames), len(prior)))

            pdf = 1
            x = [abc.x]*len(prior)
            for i, rv in enumerate(prior):
                if type(rv) is not sympy.stats.rv.RandomSymbol:
                    raise ConfigurationError('Only lambda functions or SymPy random variables can be used as a prior.')
                if len(list(rv._sorted_args[0].distribution.free_symbols)) > 0:
                    raise ConfigurationError('Prior distribution must not contain free parameters.')

                # multiply total pdf with density for current parameter
                pdf = pdf*density(rv)(x[i])

            # set density as lambda function
            self.observationModel.prior = lambdify(x, pdf, modules=['numpy', {'factorial': factorial}])
            print('+ Set custom prior: {}'.format(pdf))

    def setTransitionModel(self, K, silent=True):
        """
        Set transition model which describes the parameter dynamics.

        Args:
            K: Transition model class (see transitionModel.py)
            silent: If true, no output is printed by this method
        """
        self.transitionModel = K
        self.transitionModel.study = self
        self.transitionModel.latticeConstant = self.latticeConstant
        if not silent:
            print('+ Transition model:', K)

    def fit(self, forwardOnly=False, evidenceOnly=False, silent=False):
        """
        Computes the sequence of posterior distributions and evidence for each time step. Evidence is also computed for
        the complete data set.

        Args:
            forwardOnly: If set to True, the fitting process is terminated after the forward pass. The resulting
                posterior distributions are so-called "filtering distributions" which - at each time step -
                only incorporate the information of past data points. This option thus emulates an online
                analysis.
            evidenceOnly: If set to True, only forward pass is run and evidence is calculated. In contrast to the
                forwardOnly option, no posterior mean values are computed and no posterior distributions are stored.
            silent: If set to True, no output is generated by the fitting method.
        """
        self.checkConsistency()

        if not silent:
            print('+ Started new fit:')

        self.formattedData = movingWindow(self.rawData, self.observationModel.segmentLength)
        self.formattedTimestamps = self.rawTimestamps[self.observationModel.segmentLength-1:]
        if not silent:
            print('    + Formatted data.')

        # initialize array for posterior distributions
        if not evidenceOnly:
            self.posteriorSequence = np.empty([len(self.formattedData)]+self.gridSize)

        # initialize array for computed evidence (marginal likelihood)
        self.logEvidence = 0
        self.localEvidence = np.empty(len(self.formattedData))

        # set prior distribution for forward-pass
        if self.observationModel.prior is not None:
            if isinstance(self.observationModel.prior, np.ndarray):
                alpha = self.observationModel.prior
            else:  # prior is set as a function
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
            self.localEvidence[i] = norm*np.prod(self.latticeConstant)  # integration yields evidence, not only sum

            # normalize alpha (for numerical stability)
            if norm > 0.:
                alpha /= norm
            else:
                # if all probability values are zero, normalization is not possible
                print('    ! WARNING: Forward pass distribution contains only zeros, check parameter boundaries!')
                print('      Stopping inference process. Setting model evidence to zero.')
                self.logEvidence = -np.inf
                return

            # alphas are stored as preliminary posterior distributions
            if not evidenceOnly:
                self.posteriorSequence[i] = alpha

            # compute alpha for next iteration
            alpha = self.transitionModel.computeForwardPrior(alpha, self.formattedTimestamps[i])

        self.logEvidence += np.log(np.prod(self.latticeConstant))  # integration yields evidence, not only sum
        if not silent:
            print('    + Finished forward pass.')
            print('    + Log10-evidence: {:.5f}'.format(self.logEvidence / np.log(10)))

        if not (forwardOnly or evidenceOnly):
            # set prior distribution for backward-pass
            if self.observationModel.prior is not None:
                if isinstance(self.observationModel.prior, np.ndarray):
                    beta = self.observationModel.prior
                else:  # prior is set as a function
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
                    print('    ! WARNING: Posterior distribution contains only zeros, check parameter boundaries!')
                    print('      Stopping inference process. Setting model evidence to zero.')
                    self.logEvidence = -np.inf
                    return

                # re-compute likelihood
                likelihood = self.observationModel.processedPdf(self.grid, self.formattedData[i])

                # compute local evidence
                try:
                    self.localEvidence[i] = 1./(np.sum(self.posteriorSequence[i]/likelihood) *
                                                np.prod(self.latticeConstant))  # integration, not only sum
                except:  # in case division by zero happens
                    self.localEvidence[i] = np.nan

                # compute beta for next iteration
                beta = self.transitionModel.computeBackwardPrior(beta*likelihood, self.formattedTimestamps[i])

                # normalize beta (for numerical stability)
                beta /= np.sum(beta)

            if not silent:
                print('    + Finished backward pass.')

        # posterior mean values do not need to be computed for evidence
        if evidenceOnly:
            self.posteriorMeanValues = []
        else:
            self.posteriorMeanValues = np.empty([len(self.grid), len(self.posteriorSequence)])

            for i in range(len(self.grid)):
                self.posteriorMeanValues[i] = np.array([np.sum(p*self.grid[i]) for p in self.posteriorSequence])

            if not silent:
                print('    + Computed mean parameter values.')

    def optimize(self, parameterList=[], **kwargs):
        """
        Uses the COBYLA minimization algorithm from SciPy to perform a maximization of the log-evidence with respect
        to all hyper-parameters (the parameters of the transition model) of a time seris model. The starting values
        are the values set by the user when defining the transition model.

        For the optimization, only the log-evidence is computed and no parameter distributions are stored. When a local
        maximum is found, the parameter distribution is computed based on the optimal values for the hyper-parameters.

        Args:
            parameterList: List of hyper-parameter names to optimize. For nested transition models with multiple,
                identical hyper-parameter names, the sub-model index can be provided. By default, all hyper-parameters
                are optimized.
            **kwargs - All other keyword parameters are passed to the 'minimize' routine of scipy.optimize.
        """
        # set list of parameters to optimize
        if isinstance(parameterList, str):  # in case only a single parameter name is provided as a string
            self.selectedHyperParameters = [parameterList]
        else:
            self.selectedHyperParameters = parameterList

        print('+ Starting optimization...')
        self.checkConsistency()

        if self.selectedHyperParameters:
            print('  --> Parameter(s) to optimize:', self.selectedHyperParameters)
        else:
            print('  --> All model parameters are optimized (except change/break-points).')
            # load all hyper-parameter names
            self.selectedHyperParameters = list(flatten(self.unpackHyperParameters(self.transitionModel)))
            # delete all occurrences of 'tChange' or 'tBreak'
            self.selectedHyperParameters = [x for x in self.selectedHyperParameters
                                            if (x != 'tChange') and (x != 'tBreak')]

        # create parameter list to set start values for optimization
        x0 = self.unpackSelectedHyperParameters()

        # check if valid parameter names were entered
        if len(x0) == 0:
            # reset list of parameters to optimize, so that unpacking and setting hyper-parameters works as expected
            self.selectedHyperParameters = []
            raise ConfigurationError('No parameters to optimize. Check parameter names.')

        # perform optimization (maximization of log-evidence)
        result = minimize(self.optimizationStep, x0, method='COBYLA', **kwargs)

        print('+ Finished optimization.')

        # set optimal hyperparameters in transition model
        self.setSelectedHyperParameters(result.x)

        # run analysis with optimal parameter values
        self.fit()

        # reset list of parameters to optimize, so that unpacking and setting hyper-parameters works as expected
        self.selectedHyperParameters = []

    def optimizationStep(self, x):
        """
        Wrapper for the fit method to use it in conjunction with scipy.optimize.minimize.

        Args:
            x: unpacked list of current hyper-parameter values
        """
        # set new hyperparameters in transition model
        self.setSelectedHyperParameters(x)

        # compute log-evidence
        self.fit(evidenceOnly=True, silent=True)

        print('    + Log10-evidence: {:.5f}'.format(self.logEvidence / np.log(10)), '- Parameter values:', x)

        # return negative log-evidence (is minimized to maximize evidence)
        return -self.logEvidence

    def unpackHyperParameters(self, transitionModel, values=False):
        """
        Returns list of all hyper-parameters (names or values), nested as the transition model.

        Args:
            transitionModel: An instance of a transition model
            values: By default, parameter names are returned; if set to True, parameter values are returned

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
                    raise ConfigurationError('Could not find any hyper-parameter named {}.'.format(name))

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
                        raise ConfigurationError('Could not find any hyper-parameter at index {}.'.format(index))

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
                    raise ConfigurationError('Could not find hyper-parameter {} at index {}.'.format(name, index))

        # return selected values of hyper-parameters
        return output

    def setAllHyperParameters(self, x):
        """
        Sets all current hyper-parameters, based on a flattened list of parameter values.

        Args:
            x: list of values (e.g. from unpackSelectedHyperParameters)
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

        Args:
            x: list of values (e.g. from unpackSelectedHyperParameters)

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
                    raise ConfigurationError('Could not find any hyper-parameter named {}.'.format(name))

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
                        raise ConfigurationError('Could not find any hyper-parameter at index {}.'.format(index))

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
                    raise ConfigurationError('Could not find hyper-parameter {} at index {}.'.format(name, index))
        return 1

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
        if self.posteriorSequence == []:
            raise PostProcessingError('Cannot plot posterior sequence as it has not yet been computed. '
                                      'Run complete fit.')

        dt = self.formattedTimestamps[1:] - self.formattedTimestamps[:-1]
        if not np.all(dt == dt[0]):
            print('! WARNING: Time stamps are not equally spaced. This may result in false plotting of parameter '
                  'distributions.')

        if isinstance(param, int):
            paramIndex = param
        elif isinstance(param, str):
            paramIndex = -1
            for i, name in enumerate(self.observationModel.parameterNames):
                if name == param:
                    paramIndex = i

            # check if match was found
            if paramIndex == -1:
                raise PostProcessingError('Wrong parameter name. Available options: {0}'
                                          .format(self.observationModel.parameterNames))
        else:
            raise PostProcessingError('Wrong parameter format. Specify parameter via name or index.')

        axesToMarginalize = list(range(1, len(self.observationModel.parameterNames) + 1))  # axis 0 is time
        try:
            axesToMarginalize.remove(paramIndex + 1)
        except ValueError:
            raise PostProcessingError('Wrong parameter index to plot. Available indices: {}'
                                      .format(np.array(axesToMarginalize)-1))
        marginalPosteriorSequence = np.squeeze(np.apply_over_axes(np.sum, self.posteriorSequence, axesToMarginalize))

        # clean up very small probability values, as they may create image artefacts
        pmax = np.amax(marginalPosteriorSequence)
        marginalPosteriorSequence[marginalPosteriorSequence < pmax*(10**-20)] = 0

        plt.imshow(marginalPosteriorSequence.T**gamma,
                   origin=0,
                   cmap=createColormap(color),
                   extent=[self.formattedTimestamps[0], self.formattedTimestamps[-1]] + self.boundaries[paramIndex],
                   aspect='auto')

        # set default color of plot to black
        if ('c' not in kwargs) and ('color' not in kwargs):
            kwargs['c'] = 'k'

        # set default linewidth to 1.5
        if ('lw' not in kwargs) and ('linewidth' not in kwargs):
            kwargs['lw'] = 1.5

        plt.plot(self.formattedTimestamps, self.posteriorMeanValues[paramIndex], **kwargs)

        plt.ylim(self.boundaries[paramIndex])
        plt.ylabel(self.observationModel.parameterNames[paramIndex])
        plt.xlabel('time step')

    def checkConsistency(self):
        """
        This method is called at the very beginning of analysis methods to ensure that all necessary elements of the
        model are set correctly.

        Returns:
            True if all is well; False if problem with user input is detected.
        """
        if len(self.rawData) == 0:
            raise ConfigurationError('No data loaded.')
        if not self.observationModel:
            raise ConfigurationError('No observation model chosen.')
        if not self.transitionModel:
            raise ConfigurationError('No transition model chosen.')

        if not self.boundaries:
            print('! WARNING: No parameter boundaries are set. Trying to estimate appropriate boundaries.')
            try:
                estimatedBoundaries = self.observationModel.estimateBoundaries(self.rawData)
                assert np.all(~np.isnan(estimatedBoundaries))  # check if estimation yielded NaN value(s)
                assert np.all([b[0] != b[1] for b in estimatedBoundaries])  # check if lower boundary != upper boundary
                self.setBoundaries(estimatedBoundaries)
                print('  Using estimated boundaries.')
            except AttributeError:
                raise NotImplementedError("Observation model does not support the estimation of boundaries, use "
                                          "'setBoundaries().")
            except AssertionError:
                raise ConfigurationError("Estimation of boundaries failed. Check supplied number of data points and "
                                         "for invalid data or directly use 'setBoundaries()'.")

        if not len(self.observationModel.defaultGridSize) == len(self.gridSize):
            raise ConfigurationError('Specified parameter grid expects {0} parameter(s), but observation model has {1} '
                                     'parameter(s). Default grid-size for the chosen observation model: {2}'
                                     .format(len(self.gridSize),
                                             len(self.observationModel.parameterNames),
                                             self.observationModel.defaultGridSize))

        if not len(self.observationModel.parameterNames) == len(self.boundaries):
            raise ConfigurationError('Parameter boundaries specify {0} parameter(s), but observation model has {1} '
                                     'parameter(s).'
                                     .format(len(self.boundaries), len(self.observationModel.parameterNames)))
