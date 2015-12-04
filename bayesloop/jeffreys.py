#!/usr/bin/env python
"""
This file provides a single function that uses SymPy to determine the Jeffreys prior of an arbitrary probability
distribution defined within SymPy.
"""

import sympy.abc as abc
from sympy.stats import density
from sympy import Symbol, Matrix, simplify, diff, integrate, summation, lambdify
from sympy import ln, sqrt

def getJeffreysPrior(rv):
    """
    Uses SymPy to determine the Jeffreys prior of a random variable analytically.

    Parameters:
        rv - SymPy RandomSymbol, corresponding to a probability distribution

    Returns:
        List, containing Jeffreys prior in symbolic form and corresponding lambda function

    Example:
        rate = Symbol('rate', positive=True)
        rv = stats.Exponential('exponential', rate)
        print getJeffreysPrior(rv)

        >>> (1/rate, <function <lambda> at 0x0000000007F79AC8>)
    """

    # get support of random variable
    support = rv._sorted_args[0].distribution.set

    # get list of free parameters
    parameters = list(rv._sorted_args[0].distribution.free_symbols)
    x = abc.x

    # symbolic probability density function
    symPDF = density(rv)(x)

    # compute Fisher information matrix
    dim = len(parameters)
    G = Matrix.zeros(dim, dim)

    func = summation if support.is_iterable else integrate
    for i in range(0, dim):
        for j in range(0, dim):
            G[i, j] = func(simplify(symPDF *
                                    diff(ln(symPDF), parameters[i]) *
                                    diff(ln(symPDF), parameters[j])),
                           (x, support.inf, support.sup))

    # symbolic Jeffreys prior
    symJeff = simplify(sqrt(G.det()))

    # return symbolic Jeffreys prior and corresponding lambda function
    return symJeff, lambdify(parameters, symJeff, 'numpy')
