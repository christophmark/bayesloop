#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np


class TestBuiltin:
    def test_static(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.372209708143769, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_deterministic(self):
        S = bl.HyperStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        def linear(t, a=[1, 2]):
            return 0.5 + 0.2*a*t

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.Deterministic(linear, target='rate')
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -9.4050089375418136, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_gaussianrandomwalk(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.GaussianRandomWalk('sigma', 0.2, target='rate')
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.323144246611964, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_alphastablerandomwalk(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.AlphaStableRandomWalk('c', 0.2, 'alpha', 1.5, target='rate')
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.122384638661309, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_changepoint(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.ChangePoint('t_change', 2)
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.070975044765181, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_regimeswitch(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.RegimeSwitch('p_min', -3)
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.372866559561402, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_independent(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.Independent()
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -11.087360077190617, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_notequal(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.NotEqual('p_min', -3)
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.569099863134156, decimal=5,
                                       err_msg='Erroneous log-evidence value.')


class TestNested:
    def test_nested(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.SerialTransitionModel(
            bl.tm.Static(),
            bl.tm.ChangePoint('t_change', 1),
            bl.tm.CombinedTransitionModel(
                bl.tm.GaussianRandomWalk('sigma', 0.2, target='rate'),
                bl.tm.RegimeSwitch('p_min', -3)
            ),
            bl.tm.BreakPoint('t_break', 3),
            bl.tm.Independent()
        )
        S.set(L, T)

        S.fit()

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -10.446556976602032, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
