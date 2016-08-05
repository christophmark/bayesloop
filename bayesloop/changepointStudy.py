#!/usr/bin/env python
"""
This file introduces an extension to the basic Study-class which builds on the change-point transition model.
"""

from __future__ import division, print_function
import numpy as np
from copy import deepcopy
from .hyperStudy import *
from .preprocessing import *
from .helper import flatten


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
            hyperGrid: List of lists with each containing the name of a hyper-parameter together with a lower and upper
                boundary as well as a number of steps in between, or a list containing the name of a hyper-parameter
                together with a list of discrete values to fit.
                Example: hyperGrid = [['sigma', 0, 1, 20], ['log10pMin', [-7, -4, -1]]
            tBoundaries: A list of lists, each of which contains a lower and upper integer boundary for a change-point.
                This can be set for large data sets, in case the change-point should only be looked for in a specific
                range.
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
            forwardOnly: If set to True, the fitting process is terminated after the forward pass. The resulting
                posterior distributions are so-called "filtering distributions" which - at each time step -
                only incorporate the information of past data points. This option thus emulates an online
                analysis.
            evidenceOnly: If set to True, only forward pass is run and evidence is calculated. In contrast to the
                forwardOnly option, no posterior mean values are computed and no posterior distributions are stored.
            silent: If set to True, reduced output is generated by the fitting method.
            nJobs: Number of processes to employ. Multiprocessing is based on the 'pathos' module.
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
            idx: Index of the change-point to be analyzed (default: 0 (first change-point))
            plot: If True, a bar chart of the distribution is created
            **kwargs: All further keyword-arguments are passed to the bar-plot (see matplotlib documentation)

        Returns:
            Two numpy arrays. The first array contains the change-point times, the second one the corresponding
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
            idx: Index of the break-point to be analyzed (default: 0 (first break-point))
            plot: If True, a bar chart of the distribution is created
            **kwargs: All further keyword-arguments are passed to the bar-plot (see matplotlib documentation)

        Returns:
            Two numpy arrays. The first array contains the break-point times, the second one the corresponding
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
            indices: List of two indices of change-points to display; default: [0, 1]
                (first and second change-point of the transition model)
            plot: If True, a 3D-bar chart of the distribution is created
            figure: In case the plot is supposed to be part of an existing figure, it can be passed to the method.
                By default, a new figure is created.
            subplot: Characterization of subplot alignment, as in matplotlib. Default: 111
            **kwargs: all further keyword-arguments are passed to the bar3d-plot (see matplotlib documentation)

        Returns:
            Three numpy arrays. The first and second array contains the change-point times, the third one the
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
            indices: List of two indices of break-points to display; default: [0, 1]
                (first and second break-point of the transition model)
            plot: If True, a 3D-bar chart of the distribution is created
            figure: In case the plot is supposed to be part of an existing figure, it can be passed to the method.
                By default, a new figure is created.
            subplot: Characterization of subplot alignment, as in matplotlib. Default: 111
            **kwargs: all further keyword-arguments are passed to the bar3d-plot (see matplotlib documentation)

        Returns:
            Three numpy arrays. The first and second array contains the break-point times, the third one the
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
            indices: List of two indices of change/break-points to display; default: [0, 1]
                (first and second change/break-point of the transition model)
            plot: If True, a bar chart of the distribution is created
            **kwargs: All further keyword-arguments are passed to the bar-plot (see matplotlib documentation)

        Returns:
            Two numpy arrays. The first array contains the number of time steps, the second one the corresponding
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
