"""
Printing and plotting the dictionaries of bias and variances prodced by mx_symmetry.py
"""

import cloudpickle as cpickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

"""
Load bias and variances.

MAKE SURE that `n', `mc_reps', `delta1_list', `delta2_list' are the same here as in mc_symmetry.py.
"""
n  = 250
mc_reps = 1000
# discrepancies in the symmetry assumptions
delta1_list = np.arange(-0.6, 0.2, 0.05) # P(D_A(11)=1) - P(D_B(11)=1)
delta2_list = np.arange(-0.6, 0.6, 0.05)  # P(D_A(11)=1, D_A(10)=0) - P(D_B(11)=1, D_B(01)=0)
ddelta1, ddelta2 = np.meshgrid(delta1_list, delta2_list)

with open(r'./biases_reps%d_n%d.p' % (mc_reps, n), 'rb') as myfile:
    biases = cpickle.load(myfile)
    print(biases)

with open(r'./variances_reps%d_n%d.p' % (mc_reps, n), 'rb') as myfile:
    variances = cpickle.load(myfile)
    print(variances)

"""
Plots
"""
pgf_with_rc_fonts = {'font.family': 'serif', 'pgf.texsystem': 'pdflatex'}
mpl.rcParams.update(pgf_with_rc_fonts)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
fig, axes = plt.subplots(4, 4, sharex='all', sharey='all')
# axis labels only for outer plots
for ax in axes[3, :]:
    ax.set(xlabel='$d_1$')
for ax in axes[:, 0]:
    ax.set(ylabel='$d_2$')
# loop through all thetas
for theta_idx in range(4):
    # -- A bias
    cp = axes[theta_idx, 0].contourf(ddelta1, ddelta2, biases['A']['theta' + str(theta_idx + 1)])
    cb = fig.colorbar(cp, ax=axes[theta_idx, 0], shrink=1, aspect=15)
    cb.set_ticks([np.nanmin(biases['A']['theta' + str(theta_idx + 1)]), 0.,
                  np.nanmax(biases['A']['theta' + str(theta_idx + 1)])])
    cb.set_ticklabels(['%.1f' % np.nanmin(biases['A']['theta' + str(theta_idx + 1)]),
                       '%.1f' % 0.,
                       '%.1f' % np.nanmax(biases['A']['theta' + str(theta_idx + 1)])])
    axes[theta_idx, 0].set_title('Bias($\\hat{\\theta}^A_%s$)' % str(theta_idx + 1))
    # -- A var
    cp = axes[theta_idx, 1].contourf(ddelta1, ddelta2, variances['A']['theta' + str(theta_idx + 1)])
    cb = fig.colorbar(cp, ax=axes[theta_idx, 1], shrink=1, aspect=15)
    cb.set_ticks(cb.get_clim())
    cb.set_ticklabels(['%.1f' % v for v in cb.get_clim()])
    #cb.set_ticklabels(['%.1f' % 0., '%.1f' % np.nanmax(variances['A']['theta' + str(theta_idx + 1)])])
    axes[theta_idx, 1].set_title('Var($\\hat{\\theta}^A_%s$)' % str(theta_idx + 1))
    # -- B bias
    cp = axes[theta_idx, 2].contourf(ddelta1, ddelta2, biases['B']['theta' + str(theta_idx + 1)])
    cb = fig.colorbar(cp, ax=axes[theta_idx, 2], shrink=1, aspect=15)
    cb.set_ticks([np.nanmin(biases['B']['theta' + str(theta_idx + 1)]), 0,
                  np.nanmax(biases['B']['theta' + str(theta_idx + 1)])])
    cb.set_ticklabels(['%.1f' % np.nanmin(biases['B']['theta' + str(theta_idx + 1)]),
                       '%.1f' % 0,
                       '%.1f' % np.nanmax(biases['B']['theta' + str(theta_idx + 1)])])
    axes[theta_idx, 2].set_title('Bias($\\hat{\\theta}^B_%s$)' % str(theta_idx + 1))
    # -- B var
    cp = axes[theta_idx, 3].contourf(ddelta1, ddelta2, variances['B']['theta' + str(theta_idx + 1)])
    cb = fig.colorbar(cp, ax=axes[theta_idx, 3], shrink=1, aspect=15)
    #cb.set_clim([0., np.nanmax(variances['B']['theta' + str(theta_idx + 1)])])
    cb.set_ticks(cb.get_clim())
    cb.set_ticklabels(['%.1f' % v for v in cb.get_clim()])
    #cb.set_ticks([0., np.nanmax(variances['B']['theta' + str(theta_idx + 1)])])
    #cb.set_ticklabels(['%.1f' % 0., '%.1f' % np.nanmax(variances['B']['theta' + str(theta_idx + 1)])])
    axes[theta_idx, 3].set_title('Var($\\hat{\\theta}^B_%s$)' % str(theta_idx + 1))
plt.tight_layout()
plt.subplots_adjust(wspace=0.44)
plt.savefig('../latex/mc_symmetry_reps%d_n%d.pgf' %(mc_reps, n), bbox_inches='tight')
plt.show()


