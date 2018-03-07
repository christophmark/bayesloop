#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np
import sympy.stats as stats


class TestTwoParameterModel:
    def test_step_set1TM_0hp(self):
        # carry out fit
        S = bl.OnlineStudy(storeHistory=True)
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.setTM(bl.tm.Static())

        data = np.array([1, 2, 3, 4, 5])
        for d in data:
            S.step(d)

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.0053811, 0.38690331, 0.16329865, 0.04887604, 0.01334921],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [0.96310103, 1.5065597, 2.00218465, 2.500366, 3.],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -16.1946904707, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_step_add2TM_2hp_prior_hyperpriors_TMprior(self):
        # carry out fit
        S = bl.OnlineStudy(storeHistory=True)
        S.setOM(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1./s))

        T1 = bl.tm.CombinedTransitionModel(bl.tm.GaussianRandomWalk('s1', [0.25, 0.5],
                                                                    target='mean',
                                                                    prior=stats.Exponential('e', 0.5)),
                                           bl.tm.GaussianRandomWalk('s2', bl.cint(0, 0.2, 2),
                                                                    target='sigma',
                                                                    prior=np.array([0.2, 0.8]))
                                           )

        T2 = bl.tm.Independent()

        S.addTransitionModel('T1', T1)
        S.addTransitionModel('T2', T2)

        S.setTransitionModelPrior([0.9, 0.1])

        data = np.array([1, 2, 3, 4, 5])
        for d in data:
            S.step(d)

        # test transition model distributions
        np.testing.assert_allclose(S.getCurrentTransitionModelDistribution(local=False)[1],
                                   [0.49402616, 0.50597384],
                                   rtol=1e-05, err_msg='Erroneous transition model probabilities.')

        np.testing.assert_allclose(S.getCurrentTransitionModelDistribution(local=True)[1],
                                   [0.81739495, 0.18260505],
                                   rtol=1e-05, err_msg='Erroneous local transition model probabilities.')

        # test hyper-parameter distributions
        np.testing.assert_allclose(S.getCurrentHyperParameterDistribution('s2')[1],
                                   [0.19047162, 0.80952838],
                                   rtol=1e-05, err_msg='Erroneous hyper-parameter distribution.')

        # test parameter distributions
        np.testing.assert_allclose(S.getParameterDistributions('mean', density=False)[1][:, 5],
                                   [0.05825921, 0.20129444, 0.07273516, 0.02125759, 0.0039255],
                                   rtol=1e-05, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.getParameterMeanValues('mean'),
                                   [1.0771838, 1.71494272, 2.45992376, 3.34160617, 4.39337253],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.logEvidence, -9.46900822686, decimal=5,
                                       err_msg='Erroneous log-evidence value.')
