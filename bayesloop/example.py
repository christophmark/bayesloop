# -*- coding: utf-8 -*-
"""
example file
"""

import bayesloop as bl
import matplotlib.pyplot as plt

newStudy = bl.Study()
newStudy.loadExampleData()

M = bl.Poisson()
newStudy.setObservationModel(M)

newStudy.setGridSize([1000])
newStudy.setBoundaries([[0, 6]])

K = bl.GaussianRandomWalk(sigma=0.3)
#K = bl.Static()
newStudy.setTransitionModel(K)


newStudy.fit()

plt.plot(newStudy.posteriorMeanValues[0])
plt.show()