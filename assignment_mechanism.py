"""
MA Thesis. Examines whether individual level effect_idx.effect_idx.d. treatment assignment mechanism converges to 'classical'
factorial design assignment mechanism when pairs are randomly assigned to exactly one of the four groups:
(treatment, treatment), (no treatment, treatment), (treatment, no treatment), (no treatment, no treatment).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcdefaults()
pgf_with_rc_fonts = {'font.family': 'serif', 'pgf.texsystem': 'pdflatex'}
mpl.rcParams.update(pgf_with_rc_fonts)
mpl.rcParams['text.usetex'] = True

def round_own(u, threshold):
    """
    Rounds a number between 0 and 1 to 0 or 1 as specified by the threshold

    Parameters
    ----------
    :param u: scalar
        a real number between 0 and 1
    :param threshold: scalar
        threshold value

    Returns
    -------
    :return: 0 if u<threshold, 1 if u>=threshold
    """
    if u < threshold:
        return 0
    else:
        return 1

show_histogram = False  # if True, plots the MC-distribution for the share of pairs in the 4 groups
np.random.seed([20180311])
reps = 1000
P = 0.5  # Prob(assigned to treatment)
# z_a.z_a.d. treatment assignment with Prob(assigned to treatment)=Prob(z=1)=P. n = number of pairs
for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 300, 500]:
    print('\n------------------------- n = %d -----------------------------' % n)
    # monte carlo
    groupcount_00 = np.zeros(reps)
    groupcount_10 = np.zeros(reps)
    groupcount_01 = np.zeros(reps)
    groupcount_11 = np.zeros(reps)
    for rep in range(reps):
        member_a = [round_own(np.random.uniform(low=0.0, high=1.0), 1 - P) for _ in range(n)]  # member j
        member_b = [round_own(np.random.uniform(low=0.0, high=1.0), 1 - P) for _ in range(n)]  # member j'
        # check the proportion if pairs in each group
        for z_a, z_b in zip(member_a, member_b):
            if z_a == 0 and z_b == 0:
                groupcount_00[rep] += 1
            if z_a == 1 and z_b == 0:
                groupcount_10[rep] += 1
            if z_a == 0 and z_b == 1:
                groupcount_01[rep] += 1
            if z_a == 1 and z_b == 1:
                groupcount_11[rep] +=1
    # for a given sample size print proportion averaged across all Monte Carlo mc_reps
    print('Proportion [Var] (number) of couples in:')
    print('--- group 00: %.4f [%.4f] (%.4f)' %
          (np.mean(groupcount_00 / n), np.var(groupcount_00 / n),  np.mean(groupcount_00)))
    print('--- group 10: %.4f [%.4f] (%.4f)' %
          (np.mean(groupcount_10 / n), np.var(groupcount_10 / n),  np.mean(groupcount_10)))
    print('--- group 01: %.4f [%.4f] (%.4f)' %
          (np.mean(groupcount_01 / n), np.var(groupcount_01 / n),  np.mean(groupcount_01)))
    print('--- group 11: %.4f [%.4f] (%.4f)' %
          (np.mean(groupcount_11 / n), np.var(groupcount_11 / n),  np.mean(groupcount_11)))
    # plot histogram of MC-distribution
    if show_histogram:
        nbins = int(reps / 10)
        f, axes = plt.subplots(4, 1)
        axes = axes.ravel()
        axes[0].hist(groupcount_00 / n, nbins)
        axes[0].set_title('Group (0,0), n=%d' % n)
        axes[1].hist(groupcount_10 / n, nbins)
        axes[1].set_title('Group (1,0), n=%d' % n)
        axes[2].hist(groupcount_01 / n, nbins)
        axes[2].set_title('Group (0,1), n=%d' % n)
        axes[3].hist(groupcount_11 / n, nbins)
        axes[3].set_title('Group (1,1), n=%d' % n)
        plt.suptitle('MC-distribution for the share of pairs in the 4 group')
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show(block=False)