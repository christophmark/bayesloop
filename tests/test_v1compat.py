#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
import pytest
import scipy.special
import scipy.stats

import bayesloop as bl
import bayesloop.v1compat  # noqa: F401  (activates the 1.x API)
from bayesloop.v1compat import _camel


DATA = np.array([1, 2, 1, 3, 2, 4], dtype=float)


def _fitted_study():
    S = bl.Study(silent=True)
    S.load_data(DATA, silent=True)
    S.set_observation_model(bl.om.Poisson('rate', bl.oint(0, 6, 100)), silent=True)
    S.set_transition_model(bl.tm.GaussianRandomWalk('sigma', 0.1, target='rate'), silent=True)
    S.fit(silent=True)
    return S


def test_legacy_study_workflow_matches_v2():
    reference = _fitted_study()

    S = bl.Study(silent=True)
    with pytest.warns(DeprecationWarning):
        S.loadData(DATA, silent=True)
    with pytest.warns(DeprecationWarning):
        S.setOM(bl.om.Poisson('rate', bl.oint(0, 6, 100)), silent=True)
    with pytest.warns(DeprecationWarning):
        S.setTM(bl.tm.GaussianRandomWalk('sigma', 0.1, target='rate'), silent=True)
    S.fit(silent=True)

    with pytest.warns(DeprecationWarning):
        assert S.logEvidence == S.log_evidence
    with pytest.warns(DeprecationWarning):
        np.testing.assert_allclose(S.log10Evidence, S.log_evidence / np.log(10))
    np.testing.assert_allclose(S.log_evidence, reference.log_evidence)

    with pytest.warns(DeprecationWarning):
        legacy_dist = S.getPD(2, 'rate')
    np.testing.assert_allclose(legacy_dist, reference.get_parameter_distribution(2, 'rate'))
    with pytest.warns(DeprecationWarning):
        np.testing.assert_allclose(S.getParameterMeanValues('rate'), reference.get_parameter_mean_values('rate'))


def test_legacy_kwargs_on_stable_method_names():
    S = bl.Study(silent=True)
    S.load_data(DATA, silent=True)
    S.set(bl.om.Poisson('rate', bl.oint(0, 6, 100)),
          bl.tm.GaussianRandomWalk('sigma', 0.1, target='rate'), silent=True)
    with pytest.warns(DeprecationWarning):
        S.fit(evidenceOnly=True, silent=True)
    assert np.isfinite(S.log_evidence)

    with pytest.raises(TypeError, match='forwardOnly'):
        S.fit(forwardOnly=True, forward_only=True, silent=True)


def test_legacy_hyperstudy_workflow():
    S = bl.HyperStudy(silent=True)
    S.load_data(DATA, silent=True)
    S.set(bl.om.Poisson('rate', bl.oint(0, 6, 50)),
          bl.tm.GaussianRandomWalk('sigma', bl.cint(0, 0.2, 2), target='rate'), silent=True)
    with pytest.warns(DeprecationWarning):
        S.fit(silent=True, nJobs=1)
    with pytest.warns(DeprecationWarning):
        legacy_dist = S.getHPD('sigma')
    np.testing.assert_allclose(legacy_dist, S.get_hyper_parameter_distribution('sigma'))


def test_legacy_onlinestudy_workflow():
    with pytest.warns(DeprecationWarning):
        S = bl.OnlineStudy(storeHistory=True, silent=True)
    assert S.store_history is True

    S.set_observation_model(bl.om.Gaussian('mean', bl.cint(0, 6, 10), 'sigma', bl.oint(0, 2, 10)), silent=True)
    with pytest.warns(DeprecationWarning):
        S.addTM('static', bl.tm.Static())
    for d in DATA:
        S.step(d)

    with pytest.warns(DeprecationWarning):
        legacy_dist = S.getCPD('mean')
    np.testing.assert_allclose(legacy_dist, S.get_current_parameter_distribution('mean'))


def test_legacy_attribute_assignment_redirects():
    S = bl.Study(silent=True)
    with pytest.warns(DeprecationWarning):
        S.rawData = np.array([1.0, 2.0])
    np.testing.assert_array_equal(S.raw_data, [1.0, 2.0])
    assert 'rawData' not in S.__dict__


def test_legacy_module_names():
    import bayesloop.observationModels as om_v1
    import bayesloop.transitionModels as tm_v1
    from bayesloop.fileIO import load as load_v1

    assert om_v1 is bl.observation_models
    assert tm_v1 is bl.transition_models
    assert om_v1.Poisson is bl.om.Poisson
    assert load_v1 is bl.load


def test_legacy_function_aliases():
    assert bl.getJeffreysPrior.__wrapped__ is bl.jeffreys.get_jeffreys_prior
    assert bl.computeJeffreysPriorAR1.__wrapped__ is bl.jeffreys.compute_jeffreys_prior_ar1
    from bayesloop.preprocessing import movingWindow
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(movingWindow(np.arange(4), 2),
                                      bl.preprocessing.moving_window(np.arange(4), 2))


def test_legacy_constructor_kwargs():
    with pytest.warns(DeprecationWarning):
        L = bl.om.SciPy(scipy.stats.poisson, 'mu', bl.oint(0, 7, 100), fixedParameters={'loc': 0})
    assert L.fixed_parameter_dict == {'loc': 0}


def test_custom_v1_transition_model_matches_static():
    class V1Static(bl.tm.TransitionModel):
        def __init__(self):
            self.study = None
            self.latticeConstant = None
            self.hyperParameterNames = []
            self.hyperParameterValues = []
            self.prior = None
            self.tOffset = 0

        def __str__(self):
            return 'V1-style static model'

        def computeForwardPrior(self, posterior, t):
            return posterior

        def computeBackwardPrior(self, posterior, t):
            return posterior

    def run(transition_model):
        S = bl.Study(silent=True)
        S.load_data(DATA, silent=True)
        S.set_observation_model(bl.om.Poisson('rate', bl.oint(0, 6, 100)), silent=True)
        S.set_transition_model(transition_model, silent=True)
        S.fit(silent=True)
        return S

    np.testing.assert_allclose(run(V1Static()).log_evidence, run(bl.tm.Static()).log_evidence)


def test_custom_v1_observation_model_matches_poisson():
    class V1Poisson(bl.om.ObservationModel):
        def __init__(self, name, value):
            self.name = 'V1-style Poisson'
            self.segmentLength = 1
            self.parameterNames = [name]
            self.parameterValues = [value]
            self.prior = None
            self.multiplyLikelihoods = True

        def pdf(self, grid, dataSegment):
            return (grid[0] ** dataSegment[0]) * np.exp(-grid[0]) / scipy.special.factorial(dataSegment[0])

    def run(observation_model):
        S = bl.Study(silent=True)
        S.load_data(DATA, silent=True)
        S.set_observation_model(observation_model, silent=True)
        S.set_transition_model(bl.tm.Static(), silent=True)
        S.fit(silent=True)
        return S

    reference = run(bl.om.Poisson('rate', bl.oint(0, 6, 100), prior=None))
    legacy = run(V1Poisson('rate', bl.oint(0, 6, 100)))
    np.testing.assert_allclose(legacy.log_evidence, reference.log_evidence)


def test_removed_features_raise_informative_errors():
    S = bl.Study(silent=True)
    with pytest.raises(AttributeError, match='Parser'):
        S.eval('rate > 1')
    with pytest.raises(NotImplementedError, match='Parser'):
        bl.Parser(S)


def test_load_migrates_v1_attribute_names(tmp_path):
    S = _fitted_study()
    log_evidence = S.log_evidence

    # forge a 1.x-style study file: camelCase attribute names on the study and its models
    for obj in (S, S.observation_model, S.transition_model):
        for key in [k for k in obj.__dict__ if not k.startswith('_')]:
            camel = _camel(key)
            if camel != key:
                obj.__dict__[camel] = obj.__dict__.pop(key)
    assert 'rawData' in S.__dict__

    filename = str(tmp_path / 'study_v1.bl')
    bl.save(filename, S)
    S2 = bl.load(filename)

    assert 'raw_data' in S2.__dict__
    assert 'rawData' not in S2.__dict__
    assert 'lattice_constant' in S2.transition_model.__dict__
    np.testing.assert_allclose(S2.log_evidence, log_evidence)
    # post-processing works on the migrated study
    assert len(S2.get_parameter_distribution(2, 'rate')) == 2
