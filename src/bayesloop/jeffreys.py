#!/usr/bin/env python
"""
This file provides a single function that uses SymPy to determine the Jeffreys prior of an arbitrary probability
distribution defined within SymPy.
"""

from __future__ import division, print_function
import numpy as np
import sympy.abc as abc
from sympy.stats import density
from sympy import Symbol, Matrix, simplify, diff, integrate, summation, lambdify
from sympy import ln, sqrt
from .helper import free_symbols
from .exceptions import ConfigurationError, PostProcessingError


def get_jeffreys_prior(rv):
    """
    Uses SymPy to determine the Jeffreys prior of a random variable analytically.

    Args:
        rv: SymPy RandomSymbol, corresponding to a probability distribution

    Returns:
        list: List containing Jeffreys prior in symbolic form and corresponding lambda function

    Example:
        rate = Symbol('rate', positive=True)
        rv = stats.Exponential('exponential', rate)
        print get_jeffreys_prior(rv)

        >>> (1/rate, <function <lambda> at 0x0000000007F79AC8>)
    """

    # get support of random variable
    try:
        support = rv._sorted_args[0].distribution.set  # SymPy version <=1.0
    except AttributeError:
        support = rv._sorted_args[1].distribution.set  # SymPy version >=1.1

    # get list of free parameters
    parameters = free_symbols(rv)
    x = abc.x

    # symbolic probability density function
    sym_pdf = density(rv)(x)

    # compute Fisher information matrix
    dim = len(parameters)
    G = Matrix.zeros(dim, dim)

    func = summation if support.is_iterable else integrate
    for i in range(0, dim):
        for j in range(0, dim):
            G[i, j] = func(simplify(sym_pdf *
                                    diff(ln(sym_pdf), parameters[i]) *
                                    diff(ln(sym_pdf), parameters[j])),
                           (x, support.inf, support.sup))

    # symbolic Jeffreys prior
    sym_jeff = simplify(sqrt(G.det()))

    # check if computed Jeffreys prior is equal to 0 (happens e.g. for Cauchy distribution)
    if sym_jeff == 0:
        raise Exception('Jeffreys prior could be computed correctly.')

    # return symbolic Jeffreys prior and corresponding lambda function
    return sym_jeff, lambdify(parameters, sym_jeff, 'numpy')


def compute_jeffreys_prior_ar1(study, t=1):
    """
    This function encodes the Jeffreys prior for the AR1 process as derived by Harald Uhlig in the work "On Jeffreys
    prior when using the exact likelihood function." (Econometric Theory 10 (1994): 633-633. Equation 31). Note that
    only the case of abs(r) < 1 (stationary process) is implemented at the moment.

    Args:
        study: Instance of the Study class that this prior is added to
        t(int): Time step that this prior is computed for (t=1 means that the data point at index 0 will be used to
            compute it)

    Returns:
        Array with prior probabilities.
    """
    if str(study.observation_model) == 'Autoregressive process of first order (AR1)':
        r, s = study.grid
    elif str(study.observation_model) == 'Scaled autoregressive process of first order (AR1)':
        r, s = study.grid
        s = s*np.sqrt(1 - r**2.)
    else:
        raise ConfigurationError('Jeffreys prior for autoregressive process can only be used with AR1 and ScaledAR1 '
                                 'models.')

    # if abs(rho) >= 1., this prior cannot be used
    if np.any(np.abs(r) >= 1.):
        raise ConfigurationError('Jeffreys prior for auto-regressive process is only implemented for stationary '
                                 'processes. Values abs(r) >= 1 are not allowed for this implementation of the prior.')

    if len(study.raw_data) == 0:
        raise ConfigurationError('Data must be loaded before computing the Jeffreys prior for the autoregressive '
                                 'process.')

    d0 = study.raw_data[t-1]  # first observation is accounted for in the prior
    n = len(study.raw_data)  # number of data points

    prior = (1/s**2.)*np.exp(-d0**2.*(1-r**2.)/(2*s**2.))*(4*(r**2.)/(1-r**2.)+2*(n+1))**.5
    prior /= np.sum(prior)
    return prior
