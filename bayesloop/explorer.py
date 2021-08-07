#!/usr/bin/env python
"""
Automatic model explorer.
"""

from __future__ import print_function, division
import warnings
import numpy as np
import inspect
import sympy.stats
from sympy import Symbol
import sympy.abc as abc
from sympy import lambdify
from sympy.stats import density as sympy_density
from scipy.special import iv, factorial
from scipy.optimize import minimize
from itertools import product
from tqdm import tqdm

from bayesloop.helper import freeSymbols, oint
from bayesloop.observationModels import SymPy


class Explorer:
    """
    TODO
    """
    def __init__(self, data):
        self.data = np.array(data)

        self.sympyDistributions = []
        self.sympyModels = []

        self.boundaries = []
        self.sigma = []

        self.observationModels = []

    def createSympyModels(self):
        """
        TODO
        """
        # "Interface"-functions of the sympy.stats module should not be considered to be a model
        interface = ['ContinuousRV', 'E', 'FiniteRV', 'P', 'cdf', 'cmoment', 'correlation', 'covariance', 'density',
                     'dependent', 'given', 'independent', 'moment', 'pspace', 'random_symbols', 'sample', 'sample_iter',
                     'sampling_density', 'skewness', 'smoment', 'std', 'variance', 'where']

        self.sympyDistributions = [m for m in inspect.getmembers(sympy.stats, inspect.isfunction)
                                   if m[0] not in interface]

        # to create a sympy-model, we need to get the parameter specifications from the doc-strings first
        self.sympyModels = []
        for name, dist in self.sympyDistributions:
            # get parameter names
            paramNames = inspect.getfullargspec(dist)[0][1:]

            # get doc-string
            doc = inspect.getdoc(dist)

            # extract part of doc-string that describes the type and domain of the parameters
            s1 = 'Parameters\n=========='
            s2 = 'Returns\n======='
            try:
                start = doc.index(s1) + len(s1)
                end = doc.index(s2)
            except ValueError:
                # this model has no parameters and is therefore dismissed
                continue

            paramDoc = doc[start:end]

            # get descriptions of individual parameters
            paramDescriptions = paramDoc.split('\n')
            paramDescriptions = [line.lower() for line in paramDescriptions if len(line) > 0]

            # collect type (integer/real) and domain (positive/full) of model parameters
            properties = []
            for paramDescription in paramDescriptions:
                properties.append([])

                # identify type (integer/real)
                if 'integer' in paramDescription:
                    properties[-1].append(1)
                else:
                    properties[-1].append(0)

                # check if parameter is > 0
                if ('> 0' in paramDescription) or \
                        ('positive' in paramDescription) or \
                        ('\\in \\left(0, \\infty\\right)' in paramDescription):
                    properties[-1].append(1)
                else:
                    properties[-1].append(0)

            # Create sympy-model by first creating symbols for all parameters and then passing those symbols to the
            # probability distribution object.
            model = None
            success = False
            while not success:
                try:
                    # create symbols according to parameter specifications
                    symbols = []
                    for property, param_name in zip(properties, paramNames):
                        symbols.append(Symbol(param_name, integer=bool(property[0]), positive=bool(property[1])))

                    # create model
                    model = dist(name, *symbols)
                    success = True

                # something is wrong with parameter specifications
                except ValueError as e:
                    error = str(e)

                    # Sympy may complains that a parameter is not defined as positive-only. This sometimes happens
                    # because the doc-string is complete.
                    if 'positive' in error:
                        p = error.split(' ')[0]
                        idx = paramNames.index(p)
                        properties[idx][1] = 1
                    # If it is a different error, we dismiss the model
                    else:
                        break
                # if some parameter has a special domain (like [0, 1]), we have to dismiss the model, as we cannot
                # create sympy-symbols with custom domains (only positive/negative/full is possible)
                except TypeError as e:
                    break

            if model is not None:
                x = abc.x
                symDensity = sympy_density(model)(x)
                rvParams = freeSymbols(model)

                lambdaDensity = lambdify([x] + rvParams, symDensity,
                                         modules=['numpy', {'factorial': factorial, 'besseli': iv}])

                self.sympyModels.append([model, lambdaDensity, paramNames, len(rvParams), properties])

        print('Created {} SymPy models from {} available distributions.'.format(len(self.sympyModels),
                                                                                len(self.sympyDistributions)))

    def _llh(self, density, data, params):
        value = np.sum(np.log(density(data, *params)))
        return value

    def _boundary(self, density, data, res, idx, sign):
        llh_value = -res.fun
        target = llh_value - 10 * np.log(10.)
        eps = sign * 0.1
        params = list(res.x)
        diff = res.fun - target

        max_iter = 500
        iter = 0
        while np.abs(diff) > 10 ** -3:
            counter = 0

            if diff > 0:
                while diff > 0:
                    counter += 1
                    if counter == 50:
                        eps *= 3.
                        counter = 0

                    params[idx] += eps * np.abs(res.x[idx])
                    diff = self._llh(density, data, params) - target

                    if not np.isfinite(diff):
                        params = list(res.x)
                        diff = self._llh(density, data, params) - target
                        break

            else:
                while diff < 0:
                    counter += 1
                    if counter == 50:
                        eps *= 3.
                        counter = 0

                    params[idx] -= eps * np.abs(res.x[idx])
                    diff = self._llh(density, data, params) - target

                    if not np.isfinite(diff):
                        params = list(res.x)
                        diff = self._llh(density, data, params) - target
                        break

            eps *= 0.5
            iter += 1
            if iter == max_iter:
                break

        return params[idx]

    def _boundaries(self, density, data, res):
        b = []
        for i in range(len(res.x)):
            b.append([self._boundary(density, data, res, i, -1), self._boundary(density, data, res, i, 1)])

        return b

    def computeParameterBoundaries(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            dataBatches = [self.data[i * 10:i * 10 + 10] for i in range(len(self.data) // 10)]

            self.boundaries = []
            self.sigma = []

            self.observationModels = []

            print('Trying to derive appropriate parameter boundaries...')
            counter = 0
            for model, density, names, n, prop in tqdm(self.sympyModels):
                # for now, only models with continuous parameters are allowed
                if np.any(np.array(prop)[:, 0]):
                    continue

                MLEs = []
                boundaries = []

                for batch in dataBatches:
                    try:
                        x0_candidates = [np.nanmean(batch), np.nanstd(batch), 1.]

                        results = []
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            for p in product(x0_candidates, repeat=n):
                                res = minimize(lambda params: -self._llh(density, batch, params), p, method='SLSQP',
                                               options={'maxiter': 100, 'ftol': 1e-2 / np.mean(batch),
                                                        'eps': np.mean(batch) * 1.4901161193847656e-08})
                                results.append(res)

                        nllh_values = [res.fun for res in results]
                        nllh_argmin = np.nanargmin(nllh_values)

                        res = results[nllh_argmin]
                        MLEs.append(res.x)

                        boundaries.append(self._boundaries(density, batch, res))
                    except:
                        pass

                MLEs = np.array(MLEs)
                boundaries = np.array(boundaries)

                print(model)
                #print(boundaries)

                try:
                    sigma = np.nanstd(MLEs, axis=0)
                    finalBoundaries = np.array([np.nanmin(boundaries[:, :, 0], axis=0), np.nanmax(boundaries[:, :, 1], axis=0)]).T
                    print(finalBoundaries)
                    if np.any(np.isnan(finalBoundaries)):
                        raise ValueError

                    if len(finalBoundaries) == 1:
                        gridSize = 1000
                    elif len(finalBoundaries) == 2:
                        gridSize = 200
                    elif len(finalBoundaries) == 3:
                        gridSize = 50
                    elif len(finalBoundaries) > 3:
                        gridSize = 20

                    x = []
                    for name, bound in zip(names, finalBoundaries):
                        x.append(name)
                        x.append(oint(bound[0], bound[1], gridSize))

                    L = SymPy(model, *x, determineJeffreysPrior=False)
                    print(L)
                    self.observationModels.append(L)

                    self.sigma.append(sigma)
                    self.boundaries.append(finalBoundaries)
                except:
                    self.sigma.append(None)
                    self.boundaries.append(None)

                if self.sigma[-1] is not None and self.boundaries[-1] is not None:
                    counter += 1

            print('Found parameter boundaries for {} out of {} SymPy models.'.format(counter, len(self.sympyModels)))

    def createObservationModels(self):
        """
        TODO
        """
        self.observationModels = []

        print(self.boundaries)

        for model, sigma, boundaries in zip(self.sympyModels, self.sigma, self.boundaries):
            print(model)
            print(boundaries)
            if sigma is not None and boundaries is not None:
                sympyModel, lambdaDensity, paramNames, n, properties = model

                if len(paramNames) == 1:
                    gridSize = 1000
                elif len(paramNames) == 2:
                    gridSize = 200
                elif len(paramNames) == 3:
                    gridSize = 50
                elif len(paramNames) > 3:
                    gridSize = 20

                x = []
                for name, bound in zip(paramNames, boundaries):
                    x.append(name)
                    x.append(oint(bound[0], bound[1], gridSize))

                L = SymPy(sympyModel, *x, determineJeffreysPrior=False)

                self.observationModels.append(L)
                # print(L.name)
                # print(paramNames)
                # print(boundaries)
                # print('-----')

data = np.array([5, 4, 1, 0, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4,
                 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0,
                 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0,
                 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 3, 3, 0,
                 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])

E = Explorer(data)
E.createSympyModels()

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")

E.computeParameterBoundaries()
print('---------------------')
#E.createObservationModels()
print(E.observationModels)

from bayesloop import Study
from bayesloop.transitionModels import Static

names = []
log10evi = []

for i, L in enumerate(E.observationModels):
    S = Study()
    S.loadData(data)

    try:
        S.set(L, Static())
        S.fit()

        names.append(L.name)
        log10evi.append(S.log10Evidence)
    except:
        pass

print(names)
print(log10evi)
