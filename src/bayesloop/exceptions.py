#!/usr/bin/env python
"""
This file defines custom exceptions for bayesloop.
"""


class ConfigurationError(Exception):
    """
    Raised if some part of the configuration of a study instance is not consistent, e.g. non-existent parameter names
    are set to be optimized or the shape of a custom prior distribution does not fit the grid size.
    """


class PostProcessingError(Exception):
    """
    Raised if function for post-processing fails, e.g. plotParameterEvolution() or getHyperParameterDistribution().
    """
