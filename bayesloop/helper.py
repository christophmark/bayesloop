#!/usr/bin/env python
"""
This file includes basic helper functions.
"""

import matplotlib.colors as colors

def assignNestedItem(lst, index, value):
    """
    Assign a value to an item of an arbitrarily nested list. The list is manipulated inplace.

    Parameters:
        lst - list object
        index - list of indices
        value - value to assign to list item

    Returns:
        None
    """
    x = lst
    for i in index[:-1]:
        x = x[i]
    x[index[-1]] = value

def recursiveIndex(lst, target):
    """
    Find index of element (first occurrence) in an arbitrarily nested list.
    (Source: http://stackoverflow.com/questions/24419487/find-index-of-nested-item-in-python)

    Parameters:
        lst - list object
        target - target element to find

    Returns:
        Index as list
    """
    for index, item in enumerate(lst):
        if item == target:
            return [index]
        if isinstance(item, basestring):
            return []
        try:
            path = recursiveIndex(item, target)
        except TypeError:
            pass
        else:
            if path:
                return [index] + path
    return []

def flatten(lst):
    """
    Flatten arbitrarily nested list. Returns a generator object.
    (Source: http://stackoverflow.com/questions/10823877/
             what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python)

    Parameters:
        lst - list object

    Returns:
        Generator object for flattened list (simply call list(flatten(lst)) to get the result as a list).
    """
    for i in lst:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in flatten(i):
                yield j
        else:
            yield i

def create_colormap(color, min_factor=1.0, max_factor=0.95):
    """
    Creates colormap with range 0-1 from white to arbitrary color.

    Parameters:
        color - Matplotlib-readable color representation. Examples: 'g', '#00FFFF', '0.5', [0.1, 0.5, 0.9]
        min_factor - Float in the range 0-1, specifying the gray-scale color of the minimal plot value.
        max_factor - Float in the range 0-1, multiplication factor of 'color' argument for maximal plot value.

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
