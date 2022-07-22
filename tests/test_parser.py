#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np


class TestParameterParsing:
    def test_inequality(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 50)))
        S.setTM(bl.tm.Static())
        S.fit()

        S2 = bl.Study()
        S2.loadData(np.array([1, 2, 3, 4, 5]))
        S2.setOM(bl.om.Poisson('rate2', bl.oint(0, 6, 50)))
        S2.setTM(bl.tm.GaussianRandomWalk('sigma', 0.2, target='rate2'))
        S2.fit()

        P = bl.Parser(S, S2)
        P('log(rate2*2*1.2) + 4 + rate^2 > 20', t=3)
        np.testing.assert_almost_equal(P('log(rate2@1*2*1.2) + 4 + rate@2^2 > 20'), 0.19606860326174191, decimal=5,
                                       err_msg='Erroneous parsing result for inequality.')
        np.testing.assert_almost_equal(P('log(rate2*2*1.2) + 4 + rate^2 > 20', t=3), 0.19772797081330246, decimal=5,
                                       err_msg='Erroneous parsing result for inequality with fixed timestamp.')

    def test_distribution(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 50)))
        S.setTM(bl.tm.Static())
        S.fit()

        S2 = bl.Study()
        S2.loadData(np.array([1, 2, 3, 4, 5]))
        S2.setOM(bl.om.Poisson('rate2', bl.oint(0, 6, 50)))
        S2.setTM(bl.tm.GaussianRandomWalk('sigma', 0.2, target='rate2'))
        S2.fit()

        P = bl.Parser(S, S2)
        x, p = P('log(rate2@1*2*1.2)+ 4 + rate@2^2')
        np.testing.assert_allclose(p[100:105],
                                   [0.00732 , 0.007495, 0.005775, 0.003511, 0.003949],
                                   rtol=1e-03, err_msg='Erroneous derived probability distribution.')


class TestHyperParameterParsing:
    def test_statichyperparameter(self):
        S = bl.HyperStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 50)))
        S.setTM(bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 5), target='rate'))
        S.fit()

        p = S.eval('exp(0.99*log(sigma))+1 > 1.1')

        np.testing.assert_almost_equal(p, 0.60696006616644793, decimal=5,
                                       err_msg='Erroneous parsing result for inequality.')

    def test_dynamichyperparameter(self):
        S = bl.OnlineStudy(storeHistory=True)
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 50)))
        S.add('gradual', bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 5), target='rate'))
        S.add('static', bl.tm.Static())

        for d in np.arange(5):
            S.step(d)

        p = S.eval('exp(0.99*log(sigma@2))+1 > 1.1')

        np.testing.assert_almost_equal(p, 0.61228433813735061, decimal=5,
                                       err_msg='Erroneous parsing result for inequality.')

        S = bl.OnlineStudy(storeHistory=False)
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 50)))
        S.add('gradual', bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 5), target='rate'))
        S.add('static', bl.tm.Static())

        for d in np.arange(3):
            S.step(d)

        p = S.eval('exp(0.99*log(sigma))+1 > 1.1')

        np.testing.assert_almost_equal(p, 0.61228433813735061, decimal=5,
                                       err_msg='Erroneous parsing result for inequality.')
