# -*- coding: utf-8 -*-
"""
Sample analysis file used for debugging and illustration.

The analysis investigates the number of coal mining disasters in the UK and how they changed over time due to increasing
safety regulations.
"""

import bayesloop as bl
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

cpal = sns.color_palette()

# configure new study
# -------------------
disasterStudy = bl.Study()  # create study-object
disasterStudy.loadExampleData()  # load example data (number of mining disasters per year)

M = bl.Poisson()  # number of disasters per year is modelled by a poisson distribution
disasterStudy.setObservationModel(M)

disasterStudy.setGridSize([1000])  # configure parameter grid
disasterStudy.setBoundaries([[0, 6]])  # set parameter boundaries

# function for plotting results
# -----------------------------
def plotResults(study):
    plt.imshow(study.posteriorSequence.T,
               origin=0,
               cmap=sns.light_palette(cpal[0], as_cmap=True),
               extent=[1851, 1962] + study.boundaries[0],
               aspect='auto')

    plt.plot(np.arange(1851, 1962), disasterStudy.posteriorMeanValues[0], c='k', lw=1.5)
    plt.bar(np.arange(1851, 1962), disasterStudy.rawData, alpha=.4, facecolor='r', lw=0)

    plt.xlim((1851, 1962))
    plt.ylim([-0.1, 6.1])

# run analysis using different transition models
# ----------------------------------------------
log10EvidenceList = []  # keep track of evidence

gs = gridspec.GridSpec(3, 4)  # subplot alignment
gs.update(left=0.06, right=0.995, bottom=0.1, top=0.995, hspace=0., wspace=0.)

fig = plt.figure(figsize=[7, 4])
fig.text(0.4, 0.01, 'Year')
fig.text(0.8, 0.12, 'log10-evidence')
fig.text(0.01, 0.75, 'No. of disasters per year', rotation='vertical')

# first assumption: static rate of disasters
# ------------------------------------------
K = bl.Static()
disasterStudy.setTransitionModel(K)

disasterStudy.fit()  # fit this model
log10EvidenceList.append(disasterStudy.logEvidence / np.log(10))

plt.subplot(gs[0, :3])  # fill subplot
plotResults(disasterStudy)
plt.xticks(fontsize=12)
plt.yticks([1, 3, 5], fontsize=12)

# second assumption: gradual parameter variations with small rate
# ---------------------------------------------------------------
K = bl.GaussianRandomWalk(sigma=0.2)
disasterStudy.setTransitionModel(K)

disasterStudy.fit()  # fit this model
log10EvidenceList.append(disasterStudy.logEvidence / np.log(10))

plt.subplot(gs[1, :3])  # fill subplot
plotResults(disasterStudy)
plt.xticks(fontsize=12)
plt.yticks([1, 3, 5], fontsize=12)

# third assumption: gradual parameter variations with large rate
# --------------------------------------------------------------
K = bl.GaussianRandomWalk(sigma=0.4)
disasterStudy.setTransitionModel(K)

disasterStudy.fit()  # fit this model
log10EvidenceList.append(disasterStudy.logEvidence / np.log(10))

plt.subplot(gs[2, :3])  # fill subplot
plotResults(disasterStudy)
plt.xticks(fontsize=12)
plt.yticks([1, 3, 5], fontsize=12)

# log10-evidence subplot
# ----------------------
plt.subplot(gs[:, 3])
plt.plot(log10EvidenceList[::-1], np.arange(len(log10EvidenceList)), c=cpal[1], lw=2)
plt.scatter(log10EvidenceList[::-1], np.arange(len(log10EvidenceList)), facecolor=cpal[1], s=100, lw=0)
plt.ylim([-.5, len(log10EvidenceList) - 1 + .5])
plt.xticks([])
plt.yticks([])
plt.grid('off')
ax = plt.gca()
ax.set_axis_bgcolor((237 / 255., 241 / 255., 247 / 255.))

plt.show()


