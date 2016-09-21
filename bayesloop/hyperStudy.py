#!/usr/bin/env python
"""
This file introduces an extension to the basic Study-class that allows to compute the distribution of hyper-parameters.
"""

from __future__ import division, print_function
from .study import *
from .preprocessing import *
from scipy.misc import logsumexp
from scipy.misc import factorial
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
import sympy.abc as abc
from sympy import lambdify
import sympy.stats
from sympy.stats import density
from collections import Iterable


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
