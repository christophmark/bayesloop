#!/usr/bin/env python
"""
This file includes functions for saving and loading instances of Study objects using dill.
"""

def save(filename, study):
    """
    Save an instance of a bayesloop study class to file.

    Parameters:
        filename - string
        study - instance of study class (Study, RasterStudy, etc.)

    Returns:
        None
    """
    try:
        import dill
    except:
        print "! The module 'dill' is needed to save study instances as file."
        return

    with open(filename,'wb') as f:
        dill.dump(study, f, protocol=dill.HIGHEST_PROTOCOL)
    print '+ Successfully saved current study.'

def load(filename):
    """
    Load a instance of a bayesloop study class that was saved using the bayesloop.save() function.

    Parameters:
        filename - string

    Returns:
        study instance
    """
    try:
        import dill
    except:
        print "! The module 'dill' is needed to load study instances from file."
        return

    with open(filename,'rb') as f:
        S = dill.load(f)
    print '+ Successfully loaded study.'

    return S
