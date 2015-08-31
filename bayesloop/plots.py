#!/usr/bin/env python
"""
This file introduces plotting functions to visualize analyses carried out with bayesloop.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plotParameterEvolution(study, param=0, xLower=None, xUpper=None, color='b'):
    """
    Plots a series of marginal posterior distributions corresponding to a single model parameter, together with the
    posterior mean values.

    Parameters:
        study - An instance of the basic or an extended study-class that is used to analyze the data that is to be
            visualized.

        param - parameter name or index of parameter to display; default: 0 (first model parameter)

        color - color from which a light colormap is created

    Returns:
        None
    """

    if type(param) is int:
        paramIndex = param
    elif type(param) is str:
        for i, name in enumerate(study.observationModel.parameterNames):
            if name == param:
                paramIndex = i
    else:
        print 'ERROR: Wrong parameter format. Specify parameter via name or index.'
        return

    axesToMarginalize = range(1, len(study.observationModel.parameterNames) + 1)  # axis 0 is time
    axesToMarginalize.remove(paramIndex + 1)

    marginalPosteriorSequence = np.squeeze(np.apply_over_axes(np.sum, study.posteriorSequence, axesToMarginalize))

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
               extent=[xLower, xUpper - 1] + study.boundaries[paramIndex],
               aspect='auto')

    plt.plot(np.arange(xLower, xUpper), study.posteriorMeanValues[paramIndex], c='k', lw=1.5)

    plt.ylim(study.boundaries[paramIndex])
    plt.ylabel(study.observationModel.parameterNames[paramIndex])
    plt.xlabel('time step')


