#!/usr/bin/env python
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

    plt.plot(np.arange(1851, 1962), study.posteriorMeanValues[0], c='k', lw=1.5)
    plt.bar(np.arange(1851, 1962), study.rawData, alpha=.4, facecolor='r', lw=0)

    plt.xlim((1851, 1962))
    plt.ylim([-0.1, 6.1])

# run analysis using different transition models
# ----------------------------------------------
log10EvidenceList = []  # keep track of evidence
localEvidenceList = []  # keep track of local evidence

n = 7
gs = gridspec.GridSpec(n, n+1)  # subplot alignment
gs.update(left=0.4/n, right=0.99, bottom=0.4/n, top=0.95, hspace=0., wspace=0.)

fig = plt.figure(figsize=[7.5, 9])
fig.text(0.5*n/(n+1.0), 0.01, 'Year')
fig.text(0.01, 0.6, 'No. of disasters per year', rotation='vertical')

# first assumption: static rate of disasters
# ------------------------------------------
K = bl.Static()
disasterStudy.setTransitionModel(K)

disasterStudy.fit()  # fit this model
log10EvidenceList.append(disasterStudy.logEvidence / np.log(10))
localEvidenceList.append(disasterStudy.localEvidence)

plt.subplot(gs[0, :n])  # fill subplot
plotResults(disasterStudy)
plt.xticks(fontsize=12)
plt.yticks([1, 3, 5], fontsize=12)

# 2nd assumption: gradual parameter variations with small rate
# ------------------------------------------------------------
K = bl.GaussianRandomWalk(sigma=0.2)
disasterStudy.setTransitionModel(K)

disasterStudy.fit()  # fit this model
log10EvidenceList.append(disasterStudy.logEvidence / np.log(10))
localEvidenceList.append(disasterStudy.localEvidence)

plt.subplot(gs[1, :n])  # fill subplot
plotResults(disasterStudy)
plt.xticks(fontsize=12)
plt.yticks([1, 3, 5], fontsize=12)

# 3rd assumption: gradual parameter variations with large rate
# ------------------------------------------------------------
K = bl.GaussianRandomWalk(sigma=0.4)
disasterStudy.setTransitionModel(K)

disasterStudy.fit()  # fit this model
log10EvidenceList.append(disasterStudy.logEvidence / np.log(10))
localEvidenceList.append(disasterStudy.localEvidence)

plt.subplot(gs[2, :n])  # fill subplot
plotResults(disasterStudy)
plt.xticks(fontsize=12)
plt.yticks([1, 3, 5], fontsize=12)

# 4th assumption: change point model
# ----------------------------------
K = bl.ChangePoint(tChange=40)
disasterStudy.setTransitionModel(K)

disasterStudy.fit()  # fit this model
log10EvidenceList.append(disasterStudy.logEvidence / np.log(10))
localEvidenceList.append(disasterStudy.localEvidence)

plt.subplot(gs[3, :n])  # fill subplot
plotResults(disasterStudy)
plt.xticks(fontsize=12)
plt.yticks([1, 3, 5], fontsize=12)

# 5th assumption: complete change point analysis
# ----------------------------------------------
cpStudy = bl.changepointStudy()
cpStudy.loadExampleData()

M = bl.Poisson()
cpStudy.setObservationModel(M)

cpStudy.setGridSize([1000])
cpStudy.setBoundaries([[0, 6]])

cpStudy.fit()
log10EvidenceList.append(cpStudy.logEvidence / np.log(10))
localEvidenceList.append(cpStudy.localEvidence)

plt.subplot(gs[4, :n])  # fill subplot
plotResults(cpStudy)
plt.xticks(fontsize=12)
plt.yticks([1, 3, 5], fontsize=12)

# 6th assumption: regime-switching model
# ----------------------------------
K = bl.RegimeSwitch(pMin=10**-7)
disasterStudy.setTransitionModel(K)

disasterStudy.fit()  # fit this model
log10EvidenceList.append(disasterStudy.logEvidence / np.log(10))
localEvidenceList.append(disasterStudy.localEvidence)

plt.subplot(gs[5, :n])  # fill subplot
plotResults(disasterStudy)
plt.xticks(fontsize=12)
plt.yticks([1, 3, 5], fontsize=12)

# 7th assumption: combined transition model
# ----------------------------------
K = bl.CombinedTransitionModel(bl.GaussianRandomWalk(sigma=0.05), bl.RegimeSwitch(pMin=10**-7))
disasterStudy.setTransitionModel(K)

disasterStudy.fit()  # fit this model
log10EvidenceList.append(disasterStudy.logEvidence / np.log(10))
localEvidenceList.append(disasterStudy.localEvidence)

plt.subplot(gs[6, :n])  # fill subplot
plotResults(disasterStudy)
plt.xticks(fontsize=12)
plt.yticks([1, 3, 5], fontsize=12)

# log10-evidence subplot
# ----------------------
plt.subplot(gs[:, n])
plt.title('log10-evidence')
plt.plot(log10EvidenceList[::-1], np.arange(len(log10EvidenceList)), c=cpal[1], lw=2)
plt.scatter(log10EvidenceList[::-1], np.arange(len(log10EvidenceList)), facecolor=cpal[1], s=100, lw=0)
plt.ylim([-.5, len(log10EvidenceList) - 1 + .5])
plt.xticks([])
plt.yticks([])
plt.grid('off')
ax = plt.gca()
ax.set_axis_bgcolor((237 / 255., 241 / 255., 247 / 255.))

# Add inset with change-point distribution
# ----------------------------------------
ax_inset = fig.add_axes([0.6, 0.375, 0.20, 0.06])
plt.bar(np.arange(1851, 1962)[30:49], cpStudy.changepointDistribution[30:49], color=cpal[1], lw=0)
plt.yticks([])
ax_inset.set_axis_bgcolor((204/255.,220/255.,214/255.))

fig2 = plt.figure(figsize=[7, 6])
for i, localEvidence in enumerate(localEvidenceList):
    plt.plot(np.arange(1851, 1962), localEvidence, label=str(i)+' --- log10-evidence = '+str(log10EvidenceList[i]), lw=2)
    plt.title('Local evidence')
    plt.legend()

plt.show()


