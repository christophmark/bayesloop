#!/usr/bin/env python
"""
In bayesloop, each new data study is handled by an instance of a ``Study``-class. In this way, all data, the inference
results and the appropriate post-processing routines are stored in one object that can be accessed conveniently or
stored in a file. Apart from the basic ``Study`` class, there exist a number of specialized classes that extend the
basic fit method, for example to infer the full distribution of hyper-parameters or to apply model selection to on-line
data streams.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.misc import factorial
from scipy.misc import logsumexp
import sympy.abc as abc
from sympy import Symbol
from sympy import lambdify
from sympy.stats import density
import sympy.stats
from copy import copy, deepcopy
from collections import OrderedDict, Iterable
from inspect import getargspec
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
        if not silent:
            print('+ Observation model: {}. Parameter(s): {}'.format(M, M.parameterNames))

    def setPrior(self, prior, silent=False):
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
                if not silent:
                    print('+ Set custom prior.')
                return
            else:
                raise ConfigurationError('Prior array does not match parameter grid size.')

        # check whether function is provided
        if hasattr(prior, '__call__'):
            self.observationModel.prior = prior
            if not silent:
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
            if not silent:
                print('+ Set custom prior: {}'.format(pdf))

    def setTransitionModel(self, K, silent=False):
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

    def unpackAllHyperParameters(self, values=True):
        """
        Returns a flattened list of all hyper-parameter values of the current transition model.
        """
        return list(flatten(self.unpackHyperParameters(self.transitionModel, values=values)))

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

    def getParameterDistribution(self, t, param=0, plot=False, **kwargs):
        """
        Compute the marginal parameter distribution at a given time step.

        Args:
            t: Time step/stamp for which the parameter distribution is evaluated
            param: Parameter name or index of the parameter to display; default: 0 (first model parameter)
            plot: If True, a plot of the distribution is created
            **kwargs: All further keyword-arguments are passed to the plot (see matplotlib documentation)

        Returns:
            Two numpy arrays. The first array contains the parameter values, the second one the corresponding
            probability (density) values
        """
        if self.posteriorSequence == []:
            raise PostProcessingError('Cannot plot posterior sequence as it has not yet been computed. '
                                      'Run complete fit.')

        # check if supplied time stamp exists
        if t not in self.formattedTimestamps:
            raise PostProcessingError('Supplied time ({}) does not exist in data or is out of range.'.format(t))
        timeIndex = list(self.formattedTimestamps).index(t)  # to select corresponding posterior distribution

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

        axesToMarginalize = list(range(len(self.observationModel.parameterNames)))
        try:
            axesToMarginalize.remove(paramIndex)
        except ValueError:
            raise PostProcessingError('Wrong parameter index. Available indices: {}'.format(axesToMarginalize))

        x = self.marginalGrid[paramIndex]
        marginalDistribution = np.squeeze(np.apply_over_axes(np.sum, self.posteriorSequence[timeIndex],
                                                             axesToMarginalize))

        if plot:
            plt.fill_between(x, 0, marginalDistribution, **kwargs)

            plt.xlabel(self.observationModel.parameterNames[paramIndex])

            # in case an integer step size for hyper-parameter values is chosen, probability is displayed
            # (probability density otherwise)
            if self.latticeConstant[paramIndex] == 1.:
                plt.ylabel('probability')
            else:
                plt.ylabel('probability density')

        return x, marginalDistribution

    def getParameterDistributions(self, param=0, plot=False, **kwargs):
        """
        Computes the time series of marginal posterior distributions with respect to a given model parameter.

        Args:
            param: Parameter name or index of the parameter to display; default: 0 (first model parameter)
            plot: If True, a plot of the series of distributions is created (density map)
            **kwargs: All further keyword-arguments are passed to the plot (see matplotlib documentation)

        Returns:
            Two numpy arrays. The first array contains the parameter values, the second one the sequence of
            corresponding posterior distirbutions.
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
            raise PostProcessingError('Wrong parameter index. Available indices: {}'
                                      .format(np.array(axesToMarginalize) - 1))

        x = self.marginalGrid[paramIndex]
        marginalPosteriorSequence = np.squeeze(np.apply_over_axes(np.sum, self.posteriorSequence, axesToMarginalize))

        if plot:
            if 'c' in kwargs:
                cmap = createColormap(kwargs['c'])
            elif 'color' in kwargs:
                cmap = createColormap(kwargs['color'])
            else:
                cmap = createColormap('b')

            plt.imshow(marginalPosteriorSequence.T,
                       origin=0,
                       cmap=cmap,
                       extent=[self.formattedTimestamps[0], self.formattedTimestamps[-1]] + self.boundaries[paramIndex],
                       aspect='auto')

        return x, marginalPosteriorSequence

    def plotParameterEvolution(self, param=0, color='b', gamma=0.5, **kwargs):
        """
        Extended plot method to display a series of marginal posterior distributions corresponding to a single model
        parameter. In constrast to getMarginalParameterDistributions(), this method includes the removal of plotting
        artefacts, gamma correction as well as an overlay of the posterior mean values.

        Args:
            param(str, int): parameter name or index of parameter to display; default: 0 (first model parameter)
            color: color from which a light colormap is created
            gamma(float): exponent for gamma correction of the displayed marginal distribution; default: 0.5
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
            bool: True if all is well; False if problem with user input is detected.
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


class HyperStudy(Study):
    """
    This class serves as an extension to the basic Study class and allows to compute the distribution of hyper-
    parameters of a given transition model. For further information, see the documentation of the fit-method of this
    class.
    """
    def __init__(self):
        super(HyperStudy, self).__init__()

        self.hyperGrid = []
        self.hyperGridValues = []
        self.hyperGridConstant = []
        self.hyperPrior = None
        self.hyperPriorValues = []
        self.hyperParameterDistribution = None
        self.averagePosteriorSequence = None
        self.logEvidenceList = []
        self.localEvidenceList = []

        print('  --> Hyper-study')

    def setHyperGrid(self, hyperGrid):
        """
        Creates a regular grid of hyper-parameter values to be analyzed by the fit-method of the HyperStudy class.

        Args:
            hyperGrid(list): List of lists with each containing the name of a hyper-parameter together with a lower and
                upper boundary as well as a number of steps in between, or a list containing the name of a hyper-
                parameter together with a list of discrete values to fit.
                Example: hyperGrid = [['sigma', 0, 1, 20], ['log10pMin', [-7, -4, -1]]
        """
        # in case no hyper-grid is provided, return directly
        if not hyperGrid:
            raise ConfigurationError('No hyper-grid provided.')
        self.hyperGrid = hyperGrid

        # create array with hyper-grid values
        temp = []
        self.hyperGridConstant = []
        for x in self.hyperGrid:
            if len(x) == 2:  # format ['sigma', [0, 0.1, 0.2, 0.3]]
                temp.append(x[1])
                d = np.array(x[1])[1:] - np.array(x[1])[:-1]
                dd = d[1:] - d[:-1]
                if np.all(dd < 10 ** -10):  # for equally spaced values, set difference as grid-constant
                    self.hyperGridConstant.append(d[0])
                else:  # for irregularly spaced values (e.g. categorical), set grid-constant to 1
                    self.hyperGridConstant.append(1)
            elif len(x) == 4:  # format ['sigma', 0, 0.3, 4]
                name, lower, upper, steps = x
                temp.append(np.linspace(lower, upper, steps))
                self.hyperGridConstant.append(np.abs(upper-lower)/(float(steps)-1))
            else:
                self.hyperGrid = []
                self.hyperGridConstant = []
                raise ConfigurationError('Wrong hyper-grid format: {}. Use either '
                                         'hyperGrid=[["sigma", 0, 1, 3], ...] or [["sigma", [0, 0.5, 1]], ...]'
                                         .format(x))

        temp = np.meshgrid(*temp, indexing='ij')
        self.hyperGridValues = np.array([t.ravel() for t in temp]).T

        print('+ Set hyper-grid for the following hyper-parameters:')
        print('  {}'.format([x[0] for x in self.hyperGrid]))

    def setHyperPrior(self, hyperPrior):
        """
        Assigns prior probability values to all points on the hyper-parameter grid that is used in a hyper-study or
        change-point study.

        Args:
            hyperPrior(list): List of SymPy random variables, each of which represents the prior distribution of one
                hyper-parameter. The multiplicative probability (density) will be assigned to the individual raster
                points. Alternatively, a function can be provided that takes exactly as many arguments as there are
                hyper-parameters in the transition model. The resulting prior distribution is renormalized such that
                the sum over all points specified by the hyper-grid equals one.
        """
        # Actual hyper-parameter prior values have to be evaluated inside the fit-method, because hyper-grid values have
        # to be evaluated first. In the case of a custom hyper-grid, it cannot be ensured, that calling setHyperPrior is
        # possible before the fit-method is called.
        self.hyperPrior = hyperPrior
        print('+ Will use custom hyper-parameter prior.')

    def fit(self, forwardOnly=False, evidenceOnly=False, silent=False, nJobs=1, referenceLogEvidence=None):
        """
        This method over-rides the according method of the Study-class. It runs the algorithm for equally spaced hyper-
        parameter values as defined by the variable 'hyperGrid'. The posterior sequence represents the average
        model of all analyses. Posterior mean values are computed from this average model.

        Args:
            forwardOnly(bool): If set to True, the fitting process is terminated after the forward pass. The resulting
                posterior distributions are so-called "filtering distributions" which - at each time step -
                only incorporate the information of past data points. This option thus emulates an online
                analysis.
            evidenceOnly(bool): If set to True, only forward pass is run and evidence is calculated. In contrast to the
                forwardOnly option, no posterior mean values are computed and no posterior distributions are stored.
            silent(bool): If set to true, reduced output is created by this method.
            nJobs(int): Number of processes to employ. Multiprocessing is based on the 'pathos' module.
            referenceLogEvidence: Reference value to increase numerical stability when computing average posterior
                sequence. Ideally, this value represents the mean value of all log-evidence values. As an approximation,
                the default behavior sets it to the log-evidence of the first set of hyper-parameter values.
        """
        print('+ Started new fit.')

        self.formattedData = movingWindow(self.rawData, self.observationModel.segmentLength)

        self.checkConsistency()

        if not self.hyperGrid:
            print('! WARNING: No hyper-grid defined for hyper-parameter values. Using standard fit-method.')
            Study.fit(self, forwardOnly=forwardOnly, evidenceOnly=evidenceOnly, silent=silent)
            return

        # in case a custom hyper-grid is defined by the user, check if all attributes are set
        if self.hyperGridValues == [] or self.hyperGridConstant == []:
            raise ConfigurationError("To set a custom hyper-grid, the attributes 'hyperGrid', 'hyperGridValues' and "
                                     "'hyperGridConstant' have to be set manually.")

        # determine prior distribution
        # check whether function is provided
        if hasattr(self.hyperPrior, '__call__'):
            try:
                self.hyperPriorValues = [self.hyperPrior(*value) for value in self.hyperGridValues]
                self.hyperPriorValues /= np.sum(self.hyperPriorValues)  # renormalize hyper-parameter prior
                if not silent:
                    print('+ Set custom hyper-parameter prior: {}'.format(self.hyperPrior.__name__))
            except:
                self.hyperPriorValues = None
                raise ConfigurationError('Failed to set hyper-parameter prior. Check number of variables of passed '
                                         'function.')

        # check whether single random variable is provided
        elif type(self.hyperPrior) is sympy.stats.rv.RandomSymbol:
            self.hyperPrior = [self.hyperPrior]

        # check if list/tuple is provided
        elif isinstance(self.hyperPrior, Iterable):
            # check if given prior is correctly formatted to fit length of hyper-grid array.
            # we use 'len(self.hyperGridValues[0])' because self.hyperGrid is reformatted within changepointStudy, when
            # using break-points.
            if len(self.hyperPrior) != len(self.hyperGridValues[0]):
                self.hyperPriorValues = None
                raise ConfigurationError('{} hyper-parameters are specified in hyper-grid. Priors are provided for {}.'
                                         .format(len(self.hyperGridValues[0]), len(self.hyperPrior)))
            else:
                print('+ Setting custom hyper-parameter priors')
                self.hyperPriorValues = np.ones(len(self.hyperGridValues))
                for i, rv in enumerate(self.hyperPrior):  # loop over all specified priors
                    if len(list(rv._sorted_args[0].distribution.free_symbols)) > 0:
                        self.hyperPriorValues = None
                        raise ConfigurationError('Prior distribution must not contain free parameters.')

                    # get symbolic representation of probability density
                    x = abc.x
                    symDensity = density(rv)(x)
                    print('  {}'.format(symDensity))

                    # get density as lambda function
                    pdf = lambdify([x], symDensity, modules=['numpy', {'factorial': factorial}])

                    # update hyper-parameter prior
                    self.hyperPriorValues *= pdf(self.hyperGridValues[:, i])

                # renormalize hyper-parameter prior
                self.hyperPriorValues /= np.sum(self.hyperPriorValues)

        # if no hyper-prior could be assigned, or the one defined does not fit, assign flat prior
        if len(self.hyperPriorValues) != len(self.hyperGridValues):
            self.hyperPriorValues = np.ones(len(self.hyperGridValues))/len(self.hyperGridValues)

        if not evidenceOnly:
            self.averagePosteriorSequence = np.zeros([len(self.formattedData)]+self.gridSize)

        self.logEvidenceList = []
        self.localEvidenceList = []

        # we use the setSelectedHyperParameters-method from the Study class
        self.selectedHyperParameters = [x[0] for x in self.hyperGrid]

        print('    + {} analyses to run.'.format(len(self.hyperGridValues)))

        # check if multiprocessing is available
        if nJobs > 1:
            try:
                from pathos.multiprocessing import ProcessPool
            except ImportError:
                raise ImportError('No module named pathos.multiprocessing. This module represents an optional '
                                  'dependency of bayesloop and is therefore not installed alongside bayesloop.')

        # prepare parallel execution if necessary
        if nJobs > 1:
            # compute reference log-evidence value for numerical stability when computing average posterior sequence
            if referenceLogEvidence is None:
                self.setSelectedHyperParameters(self.hyperGridValues[0])
                Study.fit(self, forwardOnly=forwardOnly, evidenceOnly=evidenceOnly, silent=True)
                referenceLogEvidence = self.logEvidence

            print('    + Creating {} processes.'.format(nJobs))
            pool = ProcessPool(nodes=nJobs)

            # use parallelFit method to create copies of this HyperStudy instance with only partial hyper-grid values
            subStudies = pool.map(self.parallelFit,
                                  range(nJobs),
                                  [nJobs]*nJobs,
                                  [forwardOnly]*nJobs,
                                  [evidenceOnly]*nJobs,
                                  [silent]*nJobs,
                                  [referenceLogEvidence]*nJobs)

            # prevent memory pile-up in main process
            pool.close()
            pool.join()
            pool.terminate()
            pool.restart()

            # merge all sub-studies
            for S in subStudies:
                self.logEvidenceList += S.logEvidenceList
                self.localEvidenceList += S.localEvidenceList
                if not evidenceOnly:
                    self.averagePosteriorSequence += S.averagePosteriorSequence
        # single process fit
        else:
            for i, hyperParamValues in enumerate(self.hyperGridValues):
                self.setSelectedHyperParameters(hyperParamValues)

                # call fit method from parent class
                Study.fit(self, forwardOnly=forwardOnly, evidenceOnly=evidenceOnly, silent=True)

                self.logEvidenceList.append(self.logEvidence)
                self.localEvidenceList.append(self.localEvidence)

                # compute reference log-evidence value for numerical stability when computing average posterior sequence
                if i == 0 and referenceLogEvidence is None:
                    referenceLogEvidence = self.logEvidence

                if (not evidenceOnly) and np.isfinite(self.logEvidence):
                    # note: averagePosteriorSequence has no proper normalization
                    self.averagePosteriorSequence += self.posteriorSequence *\
                                                     np.exp(self.logEvidence - referenceLogEvidence) *\
                                                     self.hyperPriorValues[i]

                if not silent:
                    print('    + Analysis #{} of {} -- Hyper-parameter values {} -- log10-evidence = {:.5f}'
                          .format(i+1, len(self.hyperGridValues), hyperParamValues, self.logEvidence / np.log(10)))

        # reset list of parameters to optimize, so that unpacking and setting hyper-parameters works as expected
        self.selectedHyperParameters = []

        if not evidenceOnly:
            # compute average posterior distribution
            normalization = np.array([np.sum(posterior) for posterior in self.averagePosteriorSequence])
            for i in range(len(self.grid)):
                normalization = normalization[:, None]  # add axis; needs to match averagePosteriorSequence
            self.averagePosteriorSequence /= normalization

            # set self.posteriorSequence to average posterior sequence for plotting reasons
            self.posteriorSequence = self.averagePosteriorSequence

            if not silent:
                print('    + Computed average posterior sequence')

        # compute log-evidence of average model
        self.logEvidence = logsumexp(np.array(self.logEvidenceList) + np.log(self.hyperPriorValues))
        print('    + Log10-evidence of average model: {:.5f}'.format(self.logEvidence / np.log(10)))

        # compute hyper-parameter distribution
        logHyperParameterDistribution = self.logEvidenceList + np.log(self.hyperPriorValues)
        # ignore evidence values of -inf when computing mean value for scaling
        scaledLogHyperParameterDistribution = logHyperParameterDistribution - \
                                              np.mean(np.ma.masked_invalid(logHyperParameterDistribution))
        self.hyperParameterDistribution = np.exp(scaledLogHyperParameterDistribution)
        self.hyperParameterDistribution /= np.sum(self.hyperParameterDistribution)
        self.hyperParameterDistribution /= np.prod(self.hyperGridConstant)  # probability density

        if not silent:
            print('    + Computed hyper-parameter distribution')

        # compute local evidence of average model
        self.localEvidence = np.sum((np.array(self.localEvidenceList).T*self.hyperPriorValues).T, axis=0)

        if not silent:
            print('    + Computed local evidence of average model')

        # compute posterior mean values
        if not evidenceOnly:
            self.posteriorMeanValues = np.empty([len(self.grid), len(self.posteriorSequence)])
            for i in range(len(self.grid)):
                self.posteriorMeanValues[i] = np.array([np.sum(p*self.grid[i]) for p in self.posteriorSequence])

            if not silent:
                print('    + Computed mean parameter values.')

        # clear localEvidenceList (to keep file size small for stored studies)
        self.localEvidenceList = []

        print('+ Finished fit.')

    def parallelFit(self, idx, nJobs, forwardOnly, evidenceOnly, silent, referenceLogEvidence):
        """
        This method is called by the fit method of the HyperStudy class. It creates a copy of the current class
        instance and performs a fit based on a subset of the specified hyper-parameter grid. The method thus allows
        to distribute a HyperStudy fit among multiple processes for multiprocessing.

        Args:
            idx(int): Index from 0 to (nJobs-1), indicating which part of the hyper-grid values are to be analyzed.
            nJobs(int): Number of processes to employ. Multiprocessing is based on the 'pathos' module.
            forwardOnly(bool): If set to True, the fitting process is terminated after the forward pass. The resulting
                posterior distributions are so-called "filtering distributions" which - at each time step -
                only incorporate the information of past data points. This option thus emulates an online
                analysis.
            evidenceOnly(bool): If set to True, only forward pass is run and evidence is calculated. In contrast to the
                forwardOnly option, no posterior mean values are computed and no posterior distributions are stored.
            silent(bool): If set to True, no output is generated by the fitting method.
            referenceLogEvidence(float): Reference value to increase numerical stability when computing average
                posterior sequence. Ideally, this value represents the mean value of all log-evidence values.

        Returns:
            HyperStudy instance
        """
        S = copy(self)
        S.hyperGridValues = np.array_split(S.hyperGridValues, nJobs)[idx]
        S.hyperPriorValues = np.array_split(S.hyperPriorValues, nJobs)[idx]

        for i, hyperParamValues in enumerate(S.hyperGridValues):
            S.setSelectedHyperParameters(hyperParamValues)

            # call fit method from parent class
            Study.fit(S, forwardOnly=forwardOnly, evidenceOnly=evidenceOnly, silent=True)

            S.logEvidenceList.append(S.logEvidence)
            S.localEvidenceList.append(S.localEvidence)
            if (not evidenceOnly) and np.isfinite(S.logEvidence):
                S.averagePosteriorSequence += S.posteriorSequence *\
                                              np.exp(S.logEvidence - referenceLogEvidence) *\
                                              S.hyperPriorValues[i]

            if not silent:
                print('    + Process {} -- Analysis #{} of {}'.format(idx, i+1, len(S.hyperGridValues)))

        print('    + Process {} finished.'.format(idx))
        return S

    # optimization methods are inherited from Study class, but cannot be used in this case
    def optimize(self, *args, **kwargs):
        raise NotImplementedError('HyperStudy object has no optimizing method.')

    def optimizationStep(self, *args, **kwargs):
        raise NotImplementedError('HyperStudy object has no optimizing method.')

    def getHyperParameterDistribution(self, param=0, plot=False, **kwargs):
        """
        Computes marginal hyper-parameter distribution of a single hyper-parameter in a HyperStudy fit.

        Args:
            param(string, int): Parameter name or index of hyper-parameter to display; default: 0
                (first model hyper-parameter)
            plot(bool): If True, a bar chart of the distribution is created
            **kwargs: All further keyword-arguments are passed to the bar-plot (see matplotlib documentation)

        Returns:
            ndarray, ndarray: The first array contains the hyper-parameter values, the second one the
                corresponding probability (density) values
        """
        hyperParameterNames = [x[0] for x in self.hyperGrid]

        if isinstance(param, int):
            paramIndex = param
        elif isinstance(param, str):
            paramIndex = -1
            for i, name in enumerate(hyperParameterNames):
                if name == param:
                    paramIndex = i

            # check if match was found
            if paramIndex == -1:
                raise PostProcessingError('Wrong parameter name. Available options: {0}'.format(hyperParameterNames))
        else:
            raise PostProcessingError('Wrong parameter format. Specify parameter via name or index.')

        axesToMarginalize = list(range(len(hyperParameterNames)))
        axesToMarginalize.remove(paramIndex)

        # reshape hyper-parameter distribution for easy marginalizing
        hyperGridSteps = []
        for x in self.hyperGrid:
            if len(x) == 2:
                hyperGridSteps.append(len(x[1]))
            else:
                hyperGridSteps.append(x[3])

        distribution = self.hyperParameterDistribution.reshape(hyperGridSteps, order='C')
        marginalDistribution = np.squeeze(np.apply_over_axes(np.sum, distribution, axesToMarginalize))

        # marginal distribution is not created by sum, but by the integral
        integrationFactor = np.prod([self.hyperGridConstant[axis] for axis in axesToMarginalize])
        marginalDistribution *= integrationFactor

        if len(self.hyperGrid[paramIndex]) == 2:
            x = self.hyperGrid[paramIndex][1]
        else:
            x = np.linspace(*self.hyperGrid[paramIndex][1:])

        if plot:
            plt.bar(x, marginalDistribution, align='center', width=self.hyperGridConstant[paramIndex], **kwargs)

            plt.xlabel(hyperParameterNames[paramIndex])

            # in case an integer step size for hyper-parameter values is chosen, probability is displayed
            # (probability density otherwise)
            if self.hyperGridConstant[paramIndex] == 1.:
                plt.ylabel('probability')
            else:
                plt.ylabel('probability density')

        return x, marginalDistribution

    def getJointHyperParameterDistribution(self, params=[0, 1], plot=False, figure=None, subplot=111, **kwargs):
        """
        Computes the joint distribution of two hyper-parameters of a HyperStudy and optionally creates a 3D bar chart.
        Note that the 3D plot can only be included in an existing plot by passing a figure object and subplot
        specification.

        Args:
            params(list): List of two parameter names or indices of hyper-parameters to display; default: [0, 1]
                (first and second model parameter)
            plot(bool): If True, a 3D-bar chart of the distribution is created
            figure: In case the plot is supposed to be part of an existing figure, it can be passed to the method. By
                default, a new figure is created.
            subplot: Characterization of subplot alignment, as in matplotlib. Default: 111
            **kwargs: all further keyword-arguments are passed to the bar3d-plot (see matplotlib documentation)

        Returns:
            ndarray, ndarray, ndarray: The first and second array contains the hyper-parameter values, the
                third one the corresponding probability (density) values
        """
        hyperParameterNames = [x[0] for x in self.hyperGrid]

        # check if list with two elements is provided
        if not isinstance(params, Iterable):
            raise PostProcessingError('A list of exactly two hyper-parameters has to be provided.')
        elif not len(params) == 2:
            raise PostProcessingError('A list of exactly two hyper-parameters has to be provided.')

        # check for type of parameters (indices or names)
        if all(isinstance(p, int) for p in params):
            paramIndices = params
        elif all(isinstance(p, str) for p in params):
            paramIndices = []
            for i, name in enumerate(hyperParameterNames):
                for p in params:
                    if name == p:
                        paramIndices.append(i)

            # check if match was found
            if paramIndices == []:
                raise PostProcessingError('Wrong hyper-parameter name. Available options: {0}'
                                          .format(hyperParameterNames))
        else:
            raise PostProcessingError('Wrong parameter format. Specify parameter via name or index.')

        # check if one of the parameter names provided is wrong
        if not len(paramIndices) == 2:
            raise PostProcessingError('Probably one wrong hyper-parameter name. Available options: {0}'
                                      .format(hyperParameterNames))

        # check if parameter indices are in ascending order (so axes are labeled correctly)
        if not paramIndices[0] < paramIndices[1]:
            print('! WARNING: Switching hyper-parameter order for plotting.')
            paramIndices = paramIndices[::-1]

        axesToMarginalize = list(range(len(hyperParameterNames)))
        for p in paramIndices:
            axesToMarginalize.remove(p)

        # reshape hyper-parameter distribution for easy marginalizing
        hyperGridSteps = []
        for x in self.hyperGrid:
            if len(x) == 2:
                hyperGridSteps.append(len(x[1]))
            else:
                hyperGridSteps.append(x[3])

        distribution = self.hyperParameterDistribution.reshape(hyperGridSteps, order='C')
        marginalDistribution = np.squeeze(np.apply_over_axes(np.sum, distribution, axesToMarginalize))

        # marginal distribution is not created by sum, but by the integral
        integrationFactor = np.prod([self.hyperGridConstant[axis] for axis in axesToMarginalize])
        marginalDistribution *= integrationFactor

        if len(self.hyperGrid[paramIndices[0]]) == 2:
            x = self.hyperGrid[paramIndices[0]][1]
        else:
            x = np.linspace(*self.hyperGrid[paramIndices[0]][1:])

        if len(self.hyperGrid[paramIndices[1]]) == 2:
            y = self.hyperGrid[paramIndices[1]][1]
        else:
            y = np.linspace(*self.hyperGrid[paramIndices[1]][1:])

        x2 = np.tile(x, (len(y), 1)).T
        y2 = np.tile(y, (len(x), 1))

        z = marginalDistribution

        if plot:
            # allow to add plot to predefined figure
            if figure is None:
                fig = plt.figure()
            else:
                fig = figure
            ax = fig.add_subplot(subplot, projection='3d')

            ax.bar3d(x2.flatten() - self.hyperGridConstant[paramIndices[0]]/2.,
                     y2.flatten() - self.hyperGridConstant[paramIndices[1]]/2.,
                     z.flatten()*0.,
                     self.hyperGridConstant[paramIndices[0]],
                     self.hyperGridConstant[paramIndices[1]],
                     z.flatten(),
                     zsort='max',
                     **kwargs
                     )

            ax.set_xlabel(hyperParameterNames[paramIndices[0]])
            ax.set_ylabel(hyperParameterNames[paramIndices[1]])

            # in case an integer step size for hyper-parameter values is chosen, probability is displayed
            # (probability density otherwise)
            if self.hyperGridConstant[paramIndices[0]]*self.hyperGridConstant[paramIndices[1]] == 1.:
                ax.set_zlabel('probability')
            else:
                ax.set_zlabel('probability density')

        return x, y, marginalDistribution


class ChangepointStudy(HyperStudy):
    """
    This class builds on the HyperStudy-class and the change-point transition model to perform a series of analyses
    with varying change point times. It subsequently computes the average model from all possible change points and
    creates a probability distribution of change point times. It supports any number of change-points and arbitarily
    combined models.
    """
    def __init__(self):
        super(ChangepointStudy, self).__init__()

        # store all possible combinations of change-points (even the ones that are assigned a probability of zero),
        # to reconstruct change-point distribution after analysis
        self.allHyperGridValues = []
        self.mask = []  # mask to select valid change-point combinations

        self.userDefinedGrid = False  # needed to ensure that user-defined hyper-grid is not overwritten by fit-method
        self.hyperGridBackup = []  # needed to reconstruct hyperGrid attribute in the case of break-point model
        print('  --> Change-point analysis')

    def setHyperGrid(self, hyperGrid=[], tBoundaries=[]):
        """
        This method over-rides the corresponding method of the HyperStudy-class. While the class ChangepointStudy
        automatically iterates over all possible combinations of change-points, it is possible to provide an additional
        grid of hyper-parameter values to be analyzed by the fit-method, as in the hyper-study. Furthermore, boundary
        values for the time steps to consider in the change-point study can be defined.

        Args:
            hyperGrid(list): List of lists with each containing the name of a hyper-parameter together with a lower and
                upper boundary as well as a number of steps in between, or a list containing the name of a
                hyper-parameter together with a list of discrete values to fit.
                Example: hyperGrid = [['sigma', 0, 1, 20], ['log10pMin', [-7, -4, -1]]
            tBoundaries(list): A list of lists, each of which contains a lower and upper integer boundary for a change-
                point. This can be set for large data sets, in case the change-point should only be looked for in a
                specific range.
                Example (two change/break-points): tBoundaries = [[0, 20], [80, 100]]
        """
        if len(self.rawData) == 0:
            raise ConfigurationError("Data has to be loaded before calling 'setHyperGrid' in a change-point study.")
        if not self.observationModel:
            raise ConfigurationError("Observation model has to be loaded before calling 'setHyperGrid' in a "
                                     "change-point study.")
        if not self.transitionModel:
            raise ConfigurationError("Transition model has to be loaded before calling 'setHyperGrid' in a change-point"
                                     " study.")

        # format data/timestamps once, so number of data segments is known
        self.formattedData = movingWindow(self.rawData, self.observationModel.segmentLength)
        self.formattedTimestamps = self.rawTimestamps[self.observationModel.segmentLength - 1:]

        # check for 'tChange' hyper-parameters in transition model
        hyperParameterNames = list(flatten(self.unpackHyperParameters(self.transitionModel)))
        nChangepoint = hyperParameterNames.count('tChange')

        # check for 'tBreak' hyper-parameter in transition model
        nBreakpoint = 0
        if hyperParameterNames.count('tBreak') > 1:
            raise NotImplementedError("Multiple instances of SerialTransition models are currently not supported by "
                                      "ChangepointStudy.")

        if hyperParameterNames.count('tBreak') == 1:
            temp = deepcopy(self.selectedHyperParameters)  # store selected hyper-parameters to restore later
            self.selectedHyperParameters = ['tBreak']
            nBreakpoint = len(self.unpackSelectedHyperParameters())
            self.selectedHyperParameters = temp

        if nChangepoint == 0 and nBreakpoint == 0:
            raise ConfigurationError('No change-points or break-points detected in transition model. Check transition '
                                     'model.')

        # using both types is not supported at the moment
        if nChangepoint > 0 and nBreakpoint > 0:
            raise NotImplementedError('Detected both change-points (Changepoint transition model) and break-points '
                                      '(SerialTransitionModel). Currently, only one type is supported in a single '
                                      'transition model.')

        # create hyperGrid in the case of change-points
        if nChangepoint > 0:
            print('+ Detected {} change-point(s) in transition model.'.format(nChangepoint))
            if hyperGrid:
                print('+ {} additional hyper-parameter(s) specified for rastering:'.format(len(hyperGrid)))
                print('  {}'.format([n for n, l, u, s in hyperGrid]))

            # build custom hyper-grid of change-point values (have to be ordered) +
            # standard hyper-grid for other hyper-parameters
            if tBoundaries:  # custom boundaries
                self.hyperGrid = []
                for b in tBoundaries:
                    mask = (self.formattedTimestamps >= b[0])*(self.formattedTimestamps <= b[1])
                    self.hyperGrid += [['tChange', self.formattedTimestamps[mask]]]
                self.hyperGrid += hyperGrid
            else:  # all possible combinations
                self.hyperGrid = [['tChange', self.formattedTimestamps[:-1]]]*nChangepoint + hyperGrid

            # create tuples of hyper-grid parameter values and determine grid spacing in each dimension
            temp = []
            self.hyperGridConstant = []
            for x in self.hyperGrid:
                if len(x) == 2:  # format ['sigma', [0, 0.1, 0.2, 0.3]]
                    temp.append(x[1])
                    d = np.array(x[1])[1:] - np.array(x[1])[:-1]
                    dd = d[1:] - d[:-1]
                    if np.all(dd < 10**-10):  # for equally spaced values, set difference as grid-constant
                        self.hyperGridConstant.append(d[0])
                    else:  # for irregularly spaced values (e.g. categorical), set grid-constant to 1
                        self.hyperGridConstant.append(1)
                elif len(x) == 4:  # format ['sigma', 0, 0.3, 4]
                    name, lower, upper, steps = x
                    temp.append(np.linspace(lower, upper, steps))
                    self.hyperGridConstant.append(np.abs(upper - lower) / (float(steps) - 1))
                else:
                    self.hyperGrid = []
                    self.hyperGridConstant = []

                    raise ConfigurationError('Wrong hyper-grid format: {}. Use either '
                                             'hyperGrid=[["sigma", 0, 1, 3], ...] or [["sigma", [0, 0.5, 1]], ...]'
                                             .format(x))

            temp = np.meshgrid(*temp, indexing='ij')
            self.allHyperGridValues = np.array([t.ravel() for t in temp]).T

            # only accept if change-point values are ordered (and not equal)
            self.mask = np.array([all(x[i] < x[i+1] for i in range(nChangepoint-1)) for x in self.allHyperGridValues],
                                 dtype=bool)
            self.hyperGridValues = self.allHyperGridValues[self.mask]

        # create hyper-grid in the case of break-points
        if nBreakpoint > 0:
            print('+ Detected {} break-point(s) in transition model.'.format(nBreakpoint))
            if hyperGrid:
                print('+ Additional {} hyper-parameter(s) specified for rastering:'.format(len(hyperGrid)))
                print('  {}'.format([x[0] for x in hyperGrid]))

            # build custom hyper-grid of change-point values (have to be ordered) +
            # standard hyper-grid for other hyper-parameters
            if tBoundaries:  # custom boundaries
                self.hyperGrid = []
                for b in tBoundaries:
                    mask = (self.formattedTimestamps >= b[0]) * (self.formattedTimestamps <= b[1])
                    self.hyperGrid += [['tBreak', self.formattedTimestamps[mask]]]
                self.hyperGrid += hyperGrid
            else:  # all possible combinations
                self.hyperGrid = [['tBreak', self.formattedTimestamps[:-1]]]*nBreakpoint + hyperGrid

            # create tuples of hyper-grid parameter values and determine grid spacing in each dimension
            temp = []
            self.hyperGridConstant = []
            for x in self.hyperGrid:
                if len(x) == 2:  # format ['sigma', [0, 0.1, 0.2, 0.3]]
                    temp.append(x[1])
                    d = np.array(x[1])[1:] - np.array(x[1])[:-1]
                    dd = d[1:] - d[:-1]
                    if np.all(dd < 10 ** -10):  # for equally spaced values, set difference as grid-constant
                        self.hyperGridConstant.append(d[0])
                    else:  # for irregularly spaced values (e.g. categorical), set grid-constant to 1
                        self.hyperGridConstant.append(1)
                elif len(x) == 4:  # format ['sigma', 0, 0.3, 4]
                    name, lower, upper, steps = x
                    temp.append(np.linspace(lower, upper, steps))
                    self.hyperGridConstant.append(np.abs(upper - lower) / (float(steps) - 1))
                else:
                    self.hyperGrid = []
                    self.hyperGridConstant = []

                    raise ConfigurationError('Wrong hyper-grid format: {}. Use either '
                                             'hyperGrid=[["sigma", 0, 1, 3], ...] or [["sigma", [0, 0.5, 1]], ...]'
                                             .format(x))

            temp = np.meshgrid(*temp, indexing='ij')
            self.allHyperGridValues = np.array([t.ravel() for t in temp]).T

            # only accept if change-point values are ordered (and not equal)
            self.mask = np.array([all(x[i] < x[i+1] for i in range(nBreakpoint-1)) for x in self.allHyperGridValues],
                                 dtype=bool)
            self.hyperGridValues = self.allHyperGridValues[self.mask]

            # redefine self.hyperGrid, such that('tBreak' only occurs once (is passed as list)
            self.hyperGridBackup = deepcopy(self.hyperGrid)
            self.hyperGrid = [['tBreak', self.formattedTimestamps[:-1], len(self.formattedTimestamps)-1]] + hyperGrid

        if (not hyperGrid) and (not tBoundaries):
            self.userDefinedGrid = False  # prevents fit method from overwriting user-defined hyper-grid
        else:
            self.userDefinedGrid = True

    def fit(self, forwardOnly=False, evidenceOnly=False, silent=False, nJobs=1):
        """
        This method over-rides the corresponding method of the HyperStudy-class. It runs the algorithm for all possible
        combinations of change-points (and possible scans a range of values for other hyper-parameters). The posterior
        sequence represents the average model of all analyses. Posterior mean values are computed from this average
        model.

        Args:
            forwardOnly(bool): If set to True, the fitting process is terminated after the forward pass. The resulting
                posterior distributions are so-called "filtering distributions" which - at each time step -
                only incorporate the information of past data points. This option thus emulates an online
                analysis.
            evidenceOnly(bool): If set to True, only forward pass is run and evidence is calculated. In contrast to the
                forwardOnly option, no posterior mean values are computed and no posterior distributions are stored.
            silent(bool): If set to True, reduced output is generated by the fitting method.
            nJobs(int): Number of processes to employ. Multiprocessing is based on the 'pathos' module.
        """
        # create hyper-grid, if not done by user
        if not self.userDefinedGrid:
            self.setHyperGrid()

        # call fit method of hyper-study
        HyperStudy.fit(self,
                       forwardOnly=forwardOnly,
                       evidenceOnly=evidenceOnly,
                       silent=silent,
                       nJobs=nJobs)

        # for break-points, self.hyperGrid has to be restored to original value after fitting
        # (containing multiple 'tBreak', for proper plotting)
        if self.hyperGridBackup:
            self.hyperGrid = self.hyperGridBackup

        # for proper plotting, hyperGridValues must include all possible combinations of hyper-parameter values. We
        # therefore have to include invalid combinations and assign the probability zero to them.
        temp = np.zeros(len(self.allHyperGridValues))
        temp[self.mask] = self.hyperParameterDistribution
        self.hyperParameterDistribution = temp

        temp = np.zeros(len(self.allHyperGridValues))
        temp[self.mask] = self.hyperPriorValues
        self.hyperPriorValues = temp

    def getChangepointDistribution(self, idx=0, plot=False, **kwargs):
        """
        Computes a marginalized change-point distribution with respect to the specific change-point passed by index
        (first change-point of the transition model: idx=0) and optionally creates a bar chart.

        Args:
            idx(int): Index of the change-point to be analyzed (default: 0 (first change-point))
            plot(bool): If True, a bar chart of the distribution is created
            **kwargs: All further keyword-arguments are passed to the bar-plot (see matplotlib documentation)

        Returns:
            ndarray, ndarray: The first array contains the change-point times, the second one the corresponding
                probability values
        """
        x, marginalDistribution = HyperStudy.getHyperParameterDistribution(self, param=idx, plot=plot, **kwargs)
        if plot:
            plt.xlabel('change-point #{}'.format(idx+1))

        return x, marginalDistribution

    def getBreakpointDistribution(self, idx=0, plot=False, **kwargs):
        """
        Computes a marginalized break-point distribution with respect to the specific change-point passed by index
        (first break-point of the transition model: idx=0) and optionally creates a bar chart.

        Args:
            idx(int): Index of the break-point to be analyzed (default: 0 (first break-point))
            plot(bool): If True, a bar chart of the distribution is created
            **kwargs: All further keyword-arguments are passed to the bar-plot (see matplotlib documentation)

        Returns:
            ndarray, ndarray: The first array contains the break-point times, the second one the corresponding
                probability values
        """
        x, marginalDistribution = HyperStudy.getHyperParameterDistribution(self, param=idx, plot=plot, **kwargs)
        if plot:
            plt.xlabel('break-point #{}'.format(idx+1))

        return x, marginalDistribution

    def getJointChangepointDistribution(self, indices=[0, 1], plot=False, figure=None, subplot=111, **kwargs):
        """
        Computes a joint change-point distribution (of two change-points). The distribution is marginalized with respect
        to the change-points passed by their indices. Note that the optional 3D plot can only be included in an existing
        plot by passing a figure object and subplot specification.

        Args:
            indices(list): List of two indices of change-points to display; default: [0, 1]
                (first and second change-point of the transition model)
            plot(bool): If True, a 3D-bar chart of the distribution is created
            figure: In case the plot is supposed to be part of an existing figure, it can be passed to the method.
                By default, a new figure is created.
            subplot: Characterization of subplot alignment, as in matplotlib. Default: 111
            **kwargs: all further keyword-arguments are passed to the bar3d-plot (see matplotlib documentation)

        Returns:
            ndarray, ndarray, ndarray: The first and second array contains the change-point times, the third one the
                corresponding probability (density) values
        """
        x, y, marginalDistribution = HyperStudy.getJointHyperParameterDistribution(self,
                                                                                   params=indices,
                                                                                   plot=plot,
                                                                                   figure=figure,
                                                                                   subplot=subplot, **kwargs)
        if plot:
            plt.xlabel('change-point #{}'.format(indices[0]+1))
            plt.ylabel('change-point #{}'.format(indices[1]+1))

        return x, y, marginalDistribution

    def getJointBreakpointDistribution(self, indices=[0, 1], plot=False, figure=None, subplot=111, **kwargs):
        """
        Computes a joint break-point distribution (of two break-points). The distribution is marginalized with respect
        to the break-points passed by their indices. Note that the optional 3D plot can only be included in an existing
        plot by passing a figure object and subplot specification.

        Args:
            indices(list): List of two indices of break-points to display; default: [0, 1]
                (first and second break-point of the transition model)
            plot(bool): If True, a 3D-bar chart of the distribution is created
            figure: In case the plot is supposed to be part of an existing figure, it can be passed to the method.
                By default, a new figure is created.
            subplot: Characterization of subplot alignment, as in matplotlib. Default: 111
            **kwargs: all further keyword-arguments are passed to the bar3d-plot (see matplotlib documentation)

        Returns:
            ndarray, ndarray, ndarray: The first and second array contains the break-point times, the third one the
                corresponding probability (density) values
        """
        x, y, marginalDistribution = HyperStudy.getJointHyperParameterDistribution(self,
                                                                                   params=indices,
                                                                                   plot=plot,
                                                                                   figure=figure,
                                                                                   subplot=subplot,
                                                                                   **kwargs)
        if plot:
            plt.xlabel('break-point #{}'.format(indices[0]+1))
            plt.ylabel('break-point #{}'.format(indices[1]+1))

        return x, y, marginalDistribution

    def getDurationDistribution(self, indices=[0, 1], plot=False, **kwargs):
        """
        Computes the distribution of the number of time steps between two change/break-points. This distribution of
        duration is created from the joint distribution of the two specified change/break-points.

        Args:
            indices(list): List of two indices of change/break-points to display; default: [0, 1]
                (first and second change/break-point of the transition model)
            plot(bool): If True, a bar chart of the distribution is created
            **kwargs: All further keyword-arguments are passed to the bar-plot (see matplotlib documentation)

        Returns:
            ndarray, ndarray: The first array contains the number of time steps, the second one the corresponding
                probability values.
        """
        hyperParameterNames = [x[0] for x in self.hyperGrid]

        # check if exactly two indices are provided
        if not len(indices) == 2:
            raise PostProcessingError('Exactly two change/break-points have to be specified ([0, 1]: first two '
                                      'change/break-points).')

        axesToMarginalize = list(range(len(hyperParameterNames)))
        for p in indices:
            axesToMarginalize.remove(p)

        values = self.hyperGridValues[:, indices].T
        duration = np.unique(values[1] - values[0])  # get all possible differences between time points
        durationDistribution = np.zeros(len(duration))  # initialize array for distribution

        # loop over all hyper-grid points and collect probabilities for different durations
        for i, values in enumerate(self.allHyperGridValues[:, indices]):
            if values[1] > values[0]:
                # get matching index in duration (rounding needed because of finite precision)
                idx = np.where(duration.round(10) == (values[1]-values[0]).round(10))[0][0]
                durationDistribution[idx] += self.hyperParameterDistribution[i]

        # properly normalize duration distribution
        durationDistribution /= np.sum(durationDistribution)

        if plot:
            plt.bar(duration, durationDistribution, align='center', width=duration[0], **kwargs)

            plt.xlabel('duration between point #{} and #{} (in time steps)'.format(indices[0]+1, indices[1]+1))
            plt.ylabel('probability')

        return duration, durationDistribution


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
