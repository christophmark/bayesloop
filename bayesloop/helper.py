#!/usr/bin/env python
"""
This file includes basic helper functions.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.colors as colors


def assignNestedItem(lst, index, value):
    """
    Assign a value to an item of an arbitrarily nested list. The list is manipulated inplace.

    Args:
        lst(list): nested list to assign value to
        index(list): list of indices defining the position of the item to set
        value: value to assign to list item
    """
    x = lst
    for i in index[:-1]:
        x = x[i]
    x[index[-1]] = value


def recursiveIndex(nestedList, query):
    """
    Find index of element (first occurrence) in an arbitrarily nested list.

    Args:
        nestedList(list): list object to search in
        query: target element to find

    Returns:
        list: Position indices
    """
    for index, element in enumerate(nestedList):
        if isinstance(element, (list, tuple)):
            path = recursiveIndex(element, query)
            if path:
                return [index] + path
        if element == query:
            return [index]
    return []


def flatten(lst):
    """
    Flatten arbitrarily nested list. Returns a generator object.

    Args:
        lst(list): list to flatten

    Returns:
        Generator object for flattened list (simply call list(flatten(lst)) to get the result as a list).
    """
    for i in lst:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def createColormap(color, min_factor=1.0, max_factor=0.95):
    """
    Creates colormap with range 0-1 from white to arbitrary color.

    Args:
        color: Matplotlib-readable color representation. Examples: 'g', '#00FFFF', '0.5', [0.1, 0.5, 0.9]
        min_factor(float): Float in the range 0-1, specifying the gray-scale color of the minimal plot value.
        max_factor(float): Float in the range 0-1, multiplication factor of 'color' argument for maximal plot value.

    Returns:
        Colormap object to be used by matplotlib-functions
    """
    rgb = colors.colorConverter.to_rgb(color)
    cdict = {'red':   [(0.0, min_factor, min_factor),
                       (1.0, max_factor*rgb[0], max_factor*rgb[0])],

             'green': [(0.0, min_factor, min_factor),
                       (1.0, max_factor*rgb[1], max_factor*rgb[1])],

             'blue':  [(0.0, min_factor, min_factor),
                       (1.0, max_factor*rgb[2], max_factor*rgb[2])]}

    return colors.LinearSegmentedColormap('custom', cdict)


def oint(start, stop, num):
    """
    Returns evenly spaced numbers over a specified interval. The interval boundaries are NOT included, i.e. the interval
    is an open one. Mainly used for parameter values of the low-level (observation) model, to avoid singularities in the
    likelihood function.

    Args:
        start(scalar): Starting value of the sequence
        stop(scalar): End value of the sequence
        num(int): Number of evenly spaced points within the interval.

    Returns:
        ndarray: Array of evenly spaced numbers from the specified open interval.
    """
    return np.linspace(start, stop, num+2)[1:-1]


def cint(start, stop, num):
    """
    Returns evenly spaced numbers over a specified interval. The interval boundaries are included, i.e. the interval is
    a closed one. Mainly used for hyper-parameter values of the high-level (transition) model.

    Args:
        start(scalar): Starting value of the sequence
        stop(scalar): End value of the sequence
        num(int): Number of evenly spaced points within the interval.

    Returns:
        ndarray: Array of evenly spaced numbers from the specified closed interval.
    """
    return np.linspace(start, stop, num)

def freeSymbols(rv):
    """
    Extracts the free symbols/parameters of a probability distribution from a SymPy random variable, independent of the
    SymPy version.

    Note: In SymPy version <=1.0, the attribute "distribution" was found in rv._sorted_args[0].distribution, while
    as of version 1.1, it is found in rv._sorted_args[1].distribution.

    Args:
        rv: SymPy random variable

    Returns:
        Free symbols of a SymPy random variable
    """

    try:
        symbols = rv._sorted_args[0].distribution.free_symbols
    except AttributeError:
        symbols = rv._sorted_args[1].distribution.free_symbols

    return list(symbols)
