#!/usr/bin/env python
"""
Since one might not only be interested in the individual (hyper-)parameters of a bayesloop study, but also in arbitrary
arithmetic combinations of one or more (hyper-)parameters, a parser is needed to compute probability values or
distributions for those derived parameters.
"""

from __future__ import print_function, division
import pyparsing as pp
import re
import operator
import numpy as np
import scipy.special as sp
from tqdm import tqdm, tqdm_notebook
from .exceptions import ConfigurationError


class Parameter(np.ndarray):
    """
    Behaves like a Numpy array, but features additional attributes. This allows us to apply arithmetic operations to
    the grid of parameter values while keeping track of the corresponding probability grid and the parameter's origin.
    """
    def __new__(cls, values, prob, name=None, time=None, study=None):
        obj = np.asarray(values).view(cls)
        obj.prob = prob
        obj.name = name
        obj.time = time
        obj.study = study
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.prob = getattr(obj, 'prob', None)
        self.name = getattr(obj, 'name', None)
        self.time = getattr(obj, 'time', None)
        self.study = getattr(obj, 'study', None)


class HyperParameter(Parameter):
    """
    Behaves like a Numpy array, but features additional attributes. This allows us to apply arithmetic operations to
    the grid of hyper-parameter values while keeping track of the corresponding probability grid and the
    hyper-parameter's origin.
    """
    pass


class Parser:
    """
    Computes derived probability values and distributions based on arithmetic operations of (hyper-)parameters.

    Args:
        studies: One or more bayesloop study instances. All (hyper-)parameters in the specified study object(s) will be
            available to the parser.

    Example:
    ::
        S = bl.Study()
        ...
        P = bl.Parser(S)
        P('sqrt(rate@1910) > 1.')

    """
    def __init__(self, *studies):
        # import all parameter names
        self.studies = studies
        if len(self.studies) == 0:
            raise ConfigurationError('Parser instance takes at least one Study instance as argument.')

        self.names = []
        for study in studies:
            self.names.extend(study.observationModel.parameterNames)

            try:
                # OnlineStudy: loop over all transition models
                for names in study.hyperParameterNames:
                    self.names.extend(names)
            except AttributeError:
                try:
                    # Hyper/ChangepointStudy: only one transition model
                    self.names.extend(study.flatHyperParameterNames)
                except AttributeError:
                    pass

        if not len(np.unique(self.names)) == len(self.names):
            raise ConfigurationError('Specified study objects contain duplicate parameter names.')

        # define arithmetic operators
        self.arith = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv, '^': operator.pow}

        # initialize symbols for parsing
        parameter = pp.oneOf(self.names)
        point = pp.Literal(".")
        e = pp.CaselessLiteral("E")
        fnumber = pp.Combine(pp.Word("+-" + pp.nums, pp.nums) +
                             pp.Optional(point + pp.Optional(pp.Word(pp.nums))) +
                             pp.Optional(e + pp.Word("+-" + pp.nums, pp.nums)))

        # initialize list of all numpy functions, remove functions that collide with (hyper-)parameter names
        self.functions = dir(np)
        for name in self.names:
            try:
                self.functions.remove(name)
                print('! WARNING: Numpy function "{}" will not be available in parser, as it collides with '
                      '(hyper-)parameter names.'.format(name))
            except ValueError:
                pass

        # initialize operators for parsing
        funcop = pp.oneOf(self.functions)
        atop = pp.Literal('@')
        expop = pp.Literal('^')
        signop = pp.oneOf('+ -')
        multop = pp.oneOf('* /')
        plusop = pp.oneOf('+ -')

        # minimal symbol
        atom = (parameter | fnumber)

        # expression based on operator precedence
        self.expr = pp.operatorPrecedence(atom, [(funcop, 1, pp.opAssoc.RIGHT),
                                                 (atop, 2, pp.opAssoc.LEFT),
                                                 (expop, 2, pp.opAssoc.RIGHT),
                                                 (signop, 1, pp.opAssoc.RIGHT),
                                                 (multop, 2, pp.opAssoc.LEFT),
                                                 (plusop, 2, pp.opAssoc.LEFT)])

    def _evaluate(self, parsedString):
        """
        Recursive function to evaluate nested mathematical operations on (Hyper)Parameter instances.

        Args:
            parsedString(list): nested list generated from query by parser

        Returns:
            Derived Parameter instance
        """
        # cases like "3*3*2" are split into "(3*3)*2"
        if len(parsedString) > 3:
            while len(parsedString) > 3:
                if parsedString[0] in self.functions:
                    parsedString = [parsedString[:2]] + parsedString[2:]
                else:
                    parsedString = [parsedString[:3]] + parsedString[3:]

        result = []
        for e in parsedString:
            if isinstance(e, list):
                # unary minus: "-4" --> "(-1)*4"
                if len(e) == 2 and e[0] == '-':
                    e = ['-1', '*', e[1]]

                # unary plus: "+4" --> "1*4"
                elif len(e) == 2 and e[0] == '+':
                    e = ['1', '*', e[1]]

                # numpy function
                elif len(e) == 2 and isinstance(e[0], str):
                    e = [e[0], 'func', e[1]]

                # recursion
                result.append(self._evaluate(e))
            else:
                result.append(e)
        result = self._operation(result[1], result[0], result[2])
        return result

    def _convert(self, string):
        """
        Converts string in query to either a Parameter instance, a Numpy function, a scipy.special function or
        a float number.

        Args:
            string(str): string to convert

        Returns:
            Parameter instance, function or float
        """
        if string in self.names:
            param = [p for p in self.parameters if p.name == string][0]
            return param.copy()
        elif isinstance(string, str) and (string in dir(np)) and callable(getattr(np, string)):
            return getattr(np, string)
        elif isinstance(string, str) and (string in dir(sp)) and callable(getattr(sp, string)):
            return getattr(sp, string)
        else:
            return float(string)

    def _operation(self, symbol, a, b):
        """
        Handles arithmetic operations and selection of time steps for (hyper-)parameters.

        Args:
            symbol(str): operator symbol (one of '+-*/^@' or 'func')
            a: Parameter/HyperParameter instance, or number, or numpy function name
            b: Parameter/HyperParameter instance, or number

        Returns:
            Derived Parameter/HyperParameter instance, or number
        """
        if isinstance(a, str):
            a = self._convert(a)
        if isinstance(b, str):
            b = self._convert(b)

        # time operation
        if symbol == '@':
            if (type(a) == Parameter or (type(a) == HyperParameter and len(a.prob.shape) == 2)) and \
                    not (type(b) == Parameter or type(b) == HyperParameter):
                timeIndex = list(a.study.formattedTimestamps).index(b)
                a.prob = a.prob[timeIndex]
                a.time = b
                return a

        # numpy function
        if symbol == 'func':
            return a(b)

        # arithmetic operation
        elif symbol in self.arith.keys():
            # only perform arithmetic operations on parameters if timestamp is defined by "@" operator or
            # global time "t=..."
            if type(a) == Parameter and a.name != '_derived' and a.time is None:
                raise ConfigurationError('No timestamp defined for parameter "{}"'.format(a.name))
            if type(b) == Parameter and b.name != '_derived' and b.time is None:
                raise ConfigurationError('No timestamp defined for parameter "{}"'.format(b.name))

            # check if hyper-parameters from OnlineStudy instances have a defined time step
            if type(a) == HyperParameter and len(a.prob.shape) == 2 and a.time is None:
                raise ConfigurationError('No timestamp defined for hyper-parameter "{}"'.format(a.name))
            if type(b) == HyperParameter and len(b.prob.shape) == 2 and b.time is None:
                raise ConfigurationError('No timestamp defined for hyper-parameter "{}"'.format(b.name))

            # compute compound distribution of two (hyper-)parameters
            if (type(a) == Parameter and type(b) == Parameter and (not (a.study is b.study) or
                                                                       (a.study is None and b.study is None) or
                                                                       (a.name == b.name and not (a.time == b.time)))) or \
                    (type(a) == HyperParameter and type(b) == HyperParameter and (not (a.study is b.study) or
                                                                                      (a.study is None and b.study is None))) or \
                    ((type(a) == HyperParameter) and (type(b) == Parameter) or
                             (type(b) == HyperParameter) and (type(a) == Parameter)):

                valueTuples = np.array(np.meshgrid(a, b)).T.reshape(-1, 2)
                values = self.arith[symbol](valueTuples[:, 0], valueTuples[:, 1])

                prob = np.prod(np.array(np.meshgrid(a.prob, b.prob)).T.reshape(-1, 2), axis=1)
                prob /= np.sum(prob)
                return Parameter(values, prob, name='_derived')  # derived objects are always "parameters"

            # apply operator directly if compound distribution is not needed
            else:
                return self.arith[symbol](a, b)

    def __call__(self, query, t=None, silent=False):
        self.parameters = []

        # load parameter values, probabilities
        if t is None:
            for study in self.studies:
                # check for OnlineStudy
                storeHistory = -1
                try:
                    storeHistory = study.storeHistory
                except AttributeError:
                    pass

                if storeHistory == -1 or storeHistory == 1:
                    names = study.observationModel.parameterNames
                    for i, name in enumerate(names):
                        index = study.observationModel.parameterNames.index(name)
                        self.parameters.append(Parameter(np.ravel(study.grid[index]),
                                                         np.array([np.ravel(post) for post in study.posteriorSequence]),
                                                         name=name,
                                                         study=study))
                else:
                    names = study.observationModel.parameterNames
                    for i, name in enumerate(names):
                        index = study.observationModel.parameterNames.index(name)
                        self.parameters.append(Parameter(np.ravel(study.grid[index]),
                                                         np.ravel(study.marginalizedPosterior),
                                                         name=name,
                                                         time=study.formattedTimestamps[-1],
                                                         study=study))
        else:
            # compute index of timestamp
            timeIndex = list(self.studies[0].formattedTimestamps).index(t)

            for study in self.studies:
                names = study.observationModel.parameterNames
                for i, name in enumerate(names):
                    index = study.observationModel.parameterNames.index(name)
                    self.parameters.append(Parameter(np.ravel(study.grid[index]),
                                                     np.ravel(study.posteriorSequence[timeIndex]),
                                                     name=name,
                                                     time=t,
                                                     study=study))

        # load hyper-parameter values, probabilities
        for study in self.studies:
            # check for OnlineStudy
            try:
                allNames = study.hyperParameterNames

                # loop over different transition models
                for j, names in enumerate(allNames):
                    # loop over hyper-parameters in transition model
                    for i, name in enumerate(names):
                        index = study._getHyperParameterIndex(study.transitionModels[j], name)

                        if t is None:
                            if study.storeHistory:
                                # extract sequence of only one hyper-parameter
                                hps = []
                                for x in study.hyperParameterSequence:
                                    dist = x[j]/np.sum(x[j])
                                    hps.append(dist)
                                hps = np.array(hps)

                                self.parameters.append(HyperParameter(study.hyperParameterValues[j][:, index],
                                                                      hps,
                                                                      name=name,
                                                                      study=study))
                            else:
                                dist = study.hyperParameterDistribution[j]/np.sum(study.hyperParameterDistribution[j])
                                self.parameters.append(HyperParameter(study.hyperParameterValues[j][:, index],
                                                                      dist,
                                                                      name=name,
                                                                      time=study.formattedTimestamps[-1],
                                                                      study=study))
                        else:
                            if study.storeHistory:
                                # compute index of timestamp
                                timeIndex = list(self.studies[0].formattedTimestamps).index(t)

                                dist = study.hyperParameterSequence[timeIndex][j] / \
                                       np.sum(study.hyperParameterSequence[timeIndex][j])

                                self.parameters.append(HyperParameter(study.hyperParameterValues[j][:, index],
                                                                      dist,
                                                                      name=name,
                                                                      time=t,
                                                                      study=study))
                            else:
                                raise ConfigurationError('OnlineStudy instance is not configured to store history, '
                                                         'cannot access t={}.'.format(t))

            except AttributeError:
                # check for Hyper/ChangepointStudy, i.e. whether study type supports hyper-parameter inference
                try:
                    names = study.flatHyperParameterNames

                    for i, name in enumerate(names):
                        index = study._getHyperParameterIndex(study.transitionModel, name)

                        # probability values
                        normedDist = study.hyperParameterDistribution / np.sum(study.hyperParameterDistribution)

                        # hyper-parameter values
                        try:
                            values = study.allHyperGridValues  # Changepoint-Study
                        except AttributeError:
                            values = study.hyperGridValues  # Hyper-Study

                        self.parameters.append(HyperParameter(values[:, index],
                                                              normedDist,
                                                              name=name,
                                                              study=study))
                except AttributeError:
                    # do not try to access hyper-parameters of basic Study class
                    continue

        # reduce equation
        splitQuery = re.split('>=|<=|==|>|<', query)
        if len(splitQuery) == 1:
            reducedQuery = query
        elif len(splitQuery) == 2:
            # last arithmetic may be omitted in some cases if right side is appended to the left, needs to come first
            #reducedQuery = '-'.join(splitQuery)
            reducedQuery = '-1*('+splitQuery[1]+')+'+splitQuery[0]
        else:
            raise ConfigurationError('Use exactly one operator out of (<, >, <=, >=, ==) to obtain probability value, '
                                     'or none to obtain derived distribution.')

        # evaluate left side
        parsedString = self.expr.parseString(reducedQuery).asList()[0]
        derivedParameter = self._evaluate(parsedString)

        # if no relational operator in query, compute derived distribution
        if len(splitQuery) == 1:
            dmin = np.amin(derivedParameter)
            dmax = np.amax(derivedParameter)

            # bin size is chosen as maximal difference between two derived values
            nBins = int((dmax-dmin)/(np.amax(np.diff(np.sort(derivedParameter)))))
            bins = np.linspace(dmin, dmax, nBins)
            binnedValues = bins[:-1] + (bins[1]-bins[0])
            binnedProbs = []

            if not silent:
                print('+ Computing distribution: {}'.format(query))
                # first assume jupyter notebook and try to use tqdm-widget,
                # if it fails, use normal tqdm-progressbar
                try:
                    it = tqdm_notebook(zip(bins[:-1], bins[1:]), total=len(binnedValues))
                except:
                    it = tqdm(zip(bins[:-1], bins[1:]), total=len(binnedValues))
            else:
                it = zip(bins[:-1], bins[1:])

            for lower, upper in it:
                binnedProbs.append(np.sum(derivedParameter.prob[(derivedParameter >= lower) * (derivedParameter < upper)]))
            binnedProbs = np.array(binnedProbs)

            return binnedValues, binnedProbs

        # if relational operator in query, compute probability value
        elif len(splitQuery) == 2:
            # assign operator
            if '>=' in query:
                op = operator.ge
            elif '>' in query:
                op = operator.gt
            elif '<=' in query:
                op = operator.le
            elif '<' in query:
                op = operator.lt
            elif '==' in query:
                op = operator.eq

            # compute probability
            mask = op(derivedParameter, 0.)
            p = np.sum(derivedParameter.prob[mask])

            if not silent:
                print('P({}) = {}'.format(query, p))
            return p

        else:
            raise ConfigurationError('More than one relational operator found in query.')
