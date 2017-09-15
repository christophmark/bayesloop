#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np
import matplotlib.pyplot as plt


class TestPlot:
    def test_plot_study(self):
        S = bl.Study()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.Static()
        S.set(L, T)

        S.fit()

        S.plot('rate')
        plt.close('all')

        S.plot('rate', t=2)
        plt.close('all')

    def test_plot_hyperstudy(self):
        S = bl.HyperStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 5), target='rate')
        S.set(L, T)

        S.fit()

        S.plot('rate')
        plt.close('all')

        S.plot('rate', t=2)
        plt.close('all')

        S.plot('sigma')
        plt.close('all')

    def test_plot_changepointstudy(self):
        S = bl.ChangepointStudy()
        S.loadData(np.array([1, 2, 3, 4, 5]))

        L = bl.om.Poisson('rate', bl.oint(0, 6, 100))
        T = bl.tm.SerialTransitionModel(bl.tm.Static(),
                                        bl.tm.ChangePoint('t1', 'all'),
                                        bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 3), target='rate'),
                                        bl.tm.ChangePoint('t2', 'all'),
                                        bl.tm.Static())
        S.set(L, T)

        S.fit()

        S.plot('rate')
        plt.close('all')

        S.plot('rate', t=2)
        plt.close('all')

        S.plot('sigma')
        plt.close('all')

        S.getDD(['t1', 't2'], plot=True)
        plt.close('all')

    def test_plot_onlinestudy(self):
        S = bl.OnlineStudy(storeHistory=True)
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 50)))
        S.add('gradual', bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 5), target='rate'))
        S.add('static', bl.tm.Static())

        for d in np.arange(5):
            S.step(d)

        S.plot('rate')
        plt.close('all')

        S.plot('rate', t=2)
        plt.close('all')

        S.plot('sigma')
        plt.close('all')

        S.plot('sigma', t=2)
        plt.close('all')

        S.plot('gradual')
        plt.close('all')

        S.plot('gradual', local=True)
        plt.close('all')
