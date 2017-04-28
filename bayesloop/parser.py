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
from tqdm import tqdm, tqdm_notebook
from .core import OnlineStudy
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
        self.names = []

        for study in studies:
            if isinstance(study, OnlineStudy):
                raise NotImplementedError('Parser does not support OnlineStudy instances.')

            self.names.extend(study.observationModel.parameterNames)

            try:
                self.names.extend(study.flatHyperParameterNames)
            except AttributeError:
                pass

        if not len(np.unique(self.names)) == len(self.names):
            raise ConfigurationError('Specified study objects contain duplicate parameter names.')

        # define arithmetic operators
        self.arith = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv, '^': operator.pow}

        # initialize expression stack
        self.exprStack = []

        # initialize symbols for parsing
        parameter = pp.oneOf(self.names)
        point = pp.Literal(".")
        e = pp.CaselessLiteral("E")
        fnumber = pp.Combine(pp.Word("+-" + pp.nums, pp.nums) +
                             pp.Optional(point + pp.Optional(pp.Word(pp.nums))) +
                             pp.Optional(e + pp.Word("+-" + pp.nums, pp.nums)))
        ident = pp.Word(pp.alphas, pp.alphas + pp.nums + "_$")

        plus   = pp.Literal("+")
        minus  = pp.Literal("-")
        mult   = pp.Literal("*")
        div    = pp.Literal("/")
        at     = pp.Literal("@")
        lpar   = pp.Literal("(").suppress()
        rpar   = pp.Literal(")").suppress()
        addop  = plus | minus
        multop = mult | div
        expop  = pp.Literal("^")

        # initialize operator handling
        expr = pp.Forward()
        atom = (pp.Optional("-") + (parameter | fnumber | ident + lpar + expr + rpar).setParseAction(self._push) |
                (lpar + expr.suppress() + rpar)).setParseAction(self._pushUM)

        factor = pp.Forward()
        factor2 = pp.Forward()

        factor << atom + pp.ZeroOrMore((at + factor).setParseAction(self._push))
        term = factor + pp.ZeroOrMore((expop + factor).setParseAction(self._push))
        factor2 << term + pp.ZeroOrMore((multop + factor).setParseAction(self._push))
        expr << factor2 + pp.ZeroOrMore((addop + term).setParseAction(self._push))

        # initialize Backus-Naur form
        self.bnf = expr

    def _operation(self, symbol, a, b):
        """
        Handles arithmetic operations and selection of time steps for (hyper-)parameters.

        Args:
            symbol(str): operator symbol (one of '+-*/^@')
            a: Parameter/HyperParameter instance, or number
            b: Parameter/HyperParameter instance, or number

        Returns:
            Derived arameter/HyperParameter instance, or number
        """
        # time operation
        if symbol == '@':
            if type(a) == Parameter and not (type(b) == Parameter or type(b) == HyperParameter):
                timeIndex = list(a.study.formattedTimestamps).index(b)
                a.prob = a.prob[timeIndex]
                a.time = b
                return a

        # arithmetic operation
        elif symbol in self.arith.keys():
            # only perform arithmetic operations on parameters if timestamp is defined by "@" operator or
            # global time "t=..."
            if type(a) == Parameter and a.name != '_derived' and a.time is None:
                raise ConfigurationError('No timestamp defined for parameter "{}"'.format(a.name))
            if type(b) == Parameter and b.name != '_derived' and b.time is None:
                raise ConfigurationError('No timestamp defined for parameter "{}"'.format(b.name))

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

    def _push(self, strg, loc, toks):
        self.exprStack.append(toks[0])

    def _pushUM(self, strg, loc, toks):
        if toks and toks[0] == '-':
            self.exprStack.append('unary -')

    def _evaluateStack(self, s):
        op = s.pop()
        if op in self.names:
            param = [p for p in self.parameters if p.name == op][0]
            return param.copy()
        if op == 'unary -':
            return -self._evaluateStack(s)
        if op in "+-*/^@":
            op2 = self._evaluateStack(s)
            op1 = self._evaluateStack(s)
            return self._operation(op, op1, op2)
        elif isinstance(op, str) and (op in dir(np)) and callable(getattr(np, op)):
            return getattr(np, op)(self._evaluateStack(s))
        elif op[0].isalpha():
            return op
        else:
            return float(op)

    def __call__(self, query, t=None, silent=False):
        # load parameter values, probabilities
        if t is None:
            self.parameters = []
            for study in self.studies:
                names = study.observationModel.parameterNames
                for i, name in enumerate(names):
                    index = study.observationModel.parameterNames.index(name)
                    self.parameters.append(Parameter(np.ravel(study.grid[index]),
                                                     np.array([np.ravel(post) for post in study.posteriorSequence]),
                                                     name=name,
                                                     study=study))
        else:
            # compute index of timestamp
            timeIndex = list(self.studies[0].formattedTimestamps).index(t)

            self.parameters = []
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
            # check whether study type supports hyper-parameter inference
            try:
                names = study.flatHyperParameterNames
            except AttributeError:
                continue

            for i, name in enumerate(names):
                index = study._getHyperParameterIndex(study.transitionModel, name)
                normedDist = study.hyperParameterDistribution / np.sum(study.hyperParameterDistribution)
                self.parameters.append(HyperParameter(study.hyperGridValues[:, index],
                                                      normedDist,
                                                      name=name,
                                                      study=study))

        # reduce equation
        splitQuery = re.split('>=|<=|==|>|<', query)
        if len(splitQuery) == 1:
            reducedQuery = query
        elif len(splitQuery) == 2:
            reducedQuery = '-'.join(splitQuery)
        else:
            raise ConfigurationError('Use exactly one operator out of (<, >, <=, >=, ==) to obtain probability value, '
                                     'or none to obtain derived distribution.')

        # evaluate left side
        self.exprStack = []
        self.bnf.parseString(reducedQuery)
        derivedParameter = self._evaluateStack(self.exprStack[:])

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
