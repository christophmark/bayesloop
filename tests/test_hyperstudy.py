#!/usr/bin/env python

from __future__ import print_function, division
import bayesloop as bl
import numpy as np
import sympy.stats as stats


def make_cache_test_hyperstudy():
    S = bl.HyperStudy()
    S.load_data(np.array([1, 2, 3, 2, 1]), silent=True)
    S.set_observation_model(bl.om.Gaussian('mean', bl.cint(0, 5, 18), 'sigma', bl.oint(0, 2, 18),
                           prior=lambda m, s: 1/s**3), silent=True)
    S.set_transition_model(bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 3), target='mean'), silent=True)
    return S


def test_fit_likelihood_cache_matches_uncached():
    uncached = make_cache_test_hyperstudy()
    uncached.fit(silent=True, cache_likelihoods=False)

    cached = make_cache_test_hyperstudy()
    cached.fit(silent=True, cache_likelihoods=True)

    np.testing.assert_allclose(cached.log_evidence, uncached.log_evidence)
    np.testing.assert_allclose(cached.local_evidence, uncached.local_evidence)
    np.testing.assert_allclose(cached.posterior_sequence, uncached.posterior_sequence)
    np.testing.assert_allclose(cached.posterior_mean_values, uncached.posterior_mean_values)
    np.testing.assert_allclose(cached.hyper_parameter_distribution, uncached.hyper_parameter_distribution)
    np.testing.assert_allclose(cached.log_evidence_list, uncached.log_evidence_list)


def test_evidence_only_likelihood_cache_matches_uncached():
    uncached = make_cache_test_hyperstudy()
    uncached.fit(evidence_only=True, silent=True, cache_likelihoods=False)

    cached = make_cache_test_hyperstudy()
    cached.fit(evidence_only=True, silent=True, cache_likelihoods=True)

    np.testing.assert_allclose(cached.log_evidence, uncached.log_evidence)
    np.testing.assert_allclose(cached.local_evidence, uncached.local_evidence)
    np.testing.assert_allclose(cached.hyper_parameter_distribution, uncached.hyper_parameter_distribution)
    np.testing.assert_allclose(cached.log_evidence_list, uncached.log_evidence_list)


def test_joblib_parallel_fit_matches_single_process():
    single_process = make_cache_test_hyperstudy()
    single_process.fit(silent=True, n_jobs=1)

    parallel = make_cache_test_hyperstudy()
    parallel.fit(silent=True, n_jobs=2)

    np.testing.assert_allclose(parallel.log_evidence, single_process.log_evidence)
    np.testing.assert_allclose(parallel.local_evidence, single_process.local_evidence)
    np.testing.assert_allclose(parallel.posterior_sequence, single_process.posterior_sequence)
    np.testing.assert_allclose(parallel.posterior_mean_values, single_process.posterior_mean_values)
    np.testing.assert_allclose(parallel.hyper_parameter_distribution, single_process.hyper_parameter_distribution)
    np.testing.assert_allclose(parallel.log_evidence_list, single_process.log_evidence_list)


class TestTwoParameterModel:
    def test_fit_0hp(self):
        # carry out fit (this test is designed to fall back on the fit method of the Study class)
        S = bl.HyperStudy()
        S.load_data(np.array([1, 2, 3, 4, 5]))
        S.set_observation_model(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.set_transition_model(bl.tm.Static())
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.get_parameter_distributions('mean', density=False)[1][:, 5],
                                   [0.013349, 0.013349, 0.013349, 0.013349, 0.013349],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.get_parameter_mean_values('mean'),
                                   [3., 3., 3., 3., 3.],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -16.1946904707, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

    def test_fit_1hp(self):
        # carry out fit
        S = bl.HyperStudy()
        S.load_data(np.array([1, 2, 3, 4, 5]))
        S.set_observation_model(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.set_transition_model(bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 2), target='mean'))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.get_parameter_distributions('mean', density=False)[1][:, 5],
                                   [0.017242, 0.014581, 0.012691, 0.011705, 0.011586],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.get_parameter_mean_values('mean'),
                                   [2.92089 , 2.952597, 3.      , 3.047403, 3.07911 ],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -16.0629517262, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.get_hyper_parameter_distribution('sigma')
        print(np.array([x, p]))
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0., 0.2], [0.43828499, 0.56171501]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')

    def test_fit_2hp(self):
        # carry out fit
        S = bl.HyperStudy()
        S.load_data(np.array([1, 2, 3, 4, 5]))
        S.set_observation_model(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))

        T = bl.tm.CombinedTransitionModel(bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 2), target='mean'),
                                          bl.tm.RegimeSwitch('log10p_min', [-3, -1]))

        S.set_transition_model(T)
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.get_parameter_distributions('mean', density=False)[1][:, 5],
                                   [0.005589, 0.112966, 0.04335 , 0.00976 , 0.002909],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.get_parameter_mean_values('mean'),
                                   [0.963756, 2.105838, 2.837739, 3.734359, 4.595412],
                                   rtol=1e-05, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -10.7601875492, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.get_hyper_parameter_distribution('sigma')
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0., 0.2], [0.48943645, 0.51056355]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')

        # test joint hyper-parameter distribution
        x, y, p = S.get_joint_hyper_parameter_distribution(['log10p_min', 'sigma'])
        np.testing.assert_allclose(np.array([x, y]),
                                   [[-3., -1.], [0., 0.2]],
                                   rtol=1e-05, err_msg='Erroneous parameter values in joint hyper-parameter '
                                                       'distribution.')

        np.testing.assert_allclose(p,
                                   [[0.00701834, 0.0075608], [0.48241812, 0.50300274]],
                                   rtol=1e-05, err_msg='Erroneous probability values in joint hyper-parameter '
                                                       'distribution.')

    def test_fit_hyperprior_array(self):
        # carry out fit
        S = bl.HyperStudy()
        S.load_data(np.array([1, 2, 3, 4, 5]))
        S.set_observation_model(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.set_transition_model(bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 2), target='mean', prior=np.array([0.2, 0.8])))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.get_parameter_distributions('mean', density=False)[1][:, 5],
                                   [0.019149, 0.015184, 0.012369, 0.0109  , 0.010722],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.get_parameter_mean_values('mean'),
                                   [2.882151, 2.929385, 3.      , 3.070615, 3.117849],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -15.9915077133, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.get_hyper_parameter_distribution('sigma')
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0., 0.2], [0.16322581, 0.83677419]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')

    def test_fit_hyperprior_function(self):
        # carry out fit
        S = bl.HyperStudy()
        S.load_data(np.array([1, 2, 3, 4, 5]))
        S.set_observation_model(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.set_transition_model(bl.tm.GaussianRandomWalk('sigma', bl.cint(0.1, 0.3, 2), target='mean', prior=lambda s: 1./s))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.get_parameter_distributions('mean', density=False)[1][:, 5],
                                   [0.025476, 0.015577, 0.012088, 0.010889, 0.010749],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.get_parameter_mean_values('mean'),
                                   [2.858477, 2.915795, 3.      , 3.084205, 3.141523],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -15.9898700147, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.get_hyper_parameter_distribution('sigma')
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0.1, 0.3], [0.61609973, 0.38390027]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')

    def test_fit_hyperprior_sympy(self):
        # carry out fit
        S = bl.HyperStudy()
        S.load_data(np.array([1, 2, 3, 4, 5]))
        S.set_observation_model(bl.om.Gaussian('mean', bl.cint(0, 6, 20), 'sigma', bl.oint(0, 2, 20), prior=lambda m, s: 1/s**3))
        S.set_transition_model(bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 2), target='mean', prior=stats.Exponential('e', 1.)))
        S.fit()

        # test parameter distributions
        np.testing.assert_allclose(S.get_parameter_distributions('mean', density=False)[1][:, 5],
                                   [0.016898, 0.014472, 0.012749, 0.011851, 0.011742],
                                   rtol=1e-04, err_msg='Erroneous posterior distribution values.')

        # test parameter mean values
        np.testing.assert_allclose(S.get_parameter_mean_values('mean'),
                                   [2.927888, 2.95679 , 3.      , 3.04321 , 3.072112],
                                   rtol=1e-04, err_msg='Erroneous posterior mean values.')

        # test model evidence value
        np.testing.assert_almost_equal(S.log_evidence, -17.0866290887, decimal=5,
                                       err_msg='Erroneous log-evidence value.')

        # test hyper-parameter distribution
        x, p = S.get_hyper_parameter_distribution('sigma')
        np.testing.assert_allclose(np.array([x, p]),
                                   [[0., 0.2], [0.487971, 0.512029]],
                                   rtol=1e-05, err_msg='Erroneous values in hyper-parameter distribution.')
