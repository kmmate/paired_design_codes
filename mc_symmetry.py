"""
Thesis.
Monte Carlo simulation to explore what happens to the consistency if the symmetry assumptions are violated.
"""

import numpy as np
import cloudpickle as cpickle
from mc_distributions import PmfD, PdfY
from paired_design import PairedDesign


def bias_and_variance(reps, delta1, delta2, n, p, dyadic_design, pdfy):
    """
    Compute the Bias=E[theta_asymmetric-theta_symmetric] and the Var[theta_asymmetric] to check how off are the
     estimates when the symmetry assumptions are violated.

    Parameters
    ----------
    :param reps: int
        Number of Monte Carlo repetitions.
    :param delta1: scalar
        P(D_A(11)=1) - P(D_B(11)=1)
    :param delta2: scalar
        P(D_A(11)=1, D_A(10)=0) - P(D_B(11)=1, D_B(01)=0)
    :param n: int
        Sample size.
    :param p: scalar in (0,1)
        Probability of treatment encouragement = P(unit is treated)
    :param dyadic_design: DyadicDesign
        Instance of DyadicDesign
    :param pdfy: PdfY
        Instance of the conditional probability distribution of potential outcomes.

    Returns
    -------
    :return: two dictionaries: bias and variance, each with key A and key B
    """
    print('\n Running MC: n = %d, delta1 = %.4f, delta2 = %.4f' % (n, delta1, delta2))
    # distribution of potential treament participations
    fD = {'symmetric': PmfD(0, 0),  # symmetry assumptions hold
          'notsymmetric': PmfD(delta1, delta2)  # symmetry assumptions violated
          }
    fY = pdfy
    # preallocation to store the estimated values for each Monte Carlo rep
    teA_mc_cumsum_symmetric = np.zeros(4)  # from the symmetric version only the mean is needed to compute Bias
    teA_mc_notsymmetric = np.zeros((reps, 4))
    teB_mc_cumsum_symmetric = np.zeros(4)  # from the symmetric version only the mean is needed to compute Bias
    teB_mc_notsymmetric = np.zeros((reps, 4))
    teA = {}
    teB = {}
    for rep in range(reps):
        # treatment encouragement
        zA, zB = dyadic_design.generate_encouragement(npairs=n, p=p)
        # compute the treatment effects when the symmetry assumptions hold and when not
        for symmetry_status in ['symmetric', 'notsymmetric']:
            # treatment participation
            d_sample = fD[symmetry_status].sampler(n)
            dA_10 = np.array([int(d[0]) for d in d_sample])
            dA_11 = np.array([int(d[1]) for d in d_sample])
            dB_01 = np.array([int(d[2]) for d in d_sample])
            dB_11 = np.array([int(d[3]) for d in d_sample])
            # observed participation under One-sided noncompliance
            dA = zA * zB * (dA_11 - dA_10) + zA * dA_10
            dB = zA * zB * (dB_11 - dB_01) + zB * dB_01
            y_sample = fY.sampler(d_sample)  # independent and conditionally identical sample
            yA_00 = np.array([y[0] for y in y_sample])
            yA_10 = np.array([y[1] for y in y_sample])
            yA_01 = np.array([y[2] for y in y_sample])
            yA_11 = np.array([y[3] for y in y_sample])
            yB_00 = np.array([y[4] for y in y_sample])
            yB_10 = np.array([y[5] for y in y_sample])
            yB_01 = np.array([y[6] for y in y_sample])
            yB_11 = np.array([y[7] for y in y_sample])
            # observed outcome
            yA = dA * dB * yA_11 + dA * (1 - dB) * yA_10 + (1 - dA) * dB * yA_01 + (1 - dA) * (1 - dB) * yA_00
            yB = dA * dB * yB_11 + dA * (1 - dB) * yB_10 + (1 - dA) * dB * yB_01 + (1 - dA) * (1 - dB) * yB_00
            # treatment effects
            teA[symmetry_status] = design.treatment_effect_a(yA, dA, dB)
            teB[symmetry_status] = design.treatment_effect_b(yB, dA, dB)
        teA_mc_notsymmetric[rep, :] = teA['notsymmetric']
        teB_mc_notsymmetric[rep, :] = teB['notsymmetric']
        teA_mc_cumsum_symmetric += teA['symmetric']
        teB_mc_cumsum_symmetric += teB['symmetric']
    bias = {'A': np.mean(teA_mc_notsymmetric, 0) - teA_mc_cumsum_symmetric / reps,
            'B': np.mean(teB_mc_notsymmetric, 0) - teB_mc_cumsum_symmetric / reps
            }
    variance = {'A': np.var(teA_mc_notsymmetric, 0),
                'B': np.var(teB_mc_notsymmetric, 0)
                }
    return bias, variance

np.random.seed([20180503])  # for reproducibility
n = 250
mc_reps = 1000 # Monte Carlo repetitions
assignment_probability = 0.5
design = PairedDesign()
# conditional distribution of potential outcomes
covY_sq = abs(np.random.randn(8, 8))
covY = covY_sq.T.dot(covY_sq)
fY = PdfY(covY)
print('The covariance matrix of potential outcomes: \n', covY)
# discrepancies in the symmetry assumptions
delta1_list = np.arange(-0.6, 0.2, 0.05) # P(D_A(11)=1) - P(D_B(11)=1)
delta2_list = np.arange(-0.6, 0.6, 0.05)  # P(D_A(11)=1, D_A(10)=0) - P(D_B(11)=1, D_B(01)=0)
ddelta1, ddelta2 = np.meshgrid(delta1_list, delta2_list)
biases = {'A': {'theta1': [], 'theta2': [], 'theta3': [], 'theta4': []},
          'B': {'theta1': [], 'theta2': [], 'theta3': [], 'theta4': []}}
variances = {'A': {'theta1': [], 'theta2': [], 'theta3': [], 'theta4': []},
             'B': {'theta1': [], 'theta2': [], 'theta3': [], 'theta4': []}}
# compute the bias and the variance for each combination of (delta1, delta2)
for idx in range(len(delta1_list) * len(delta2_list)):
    delta1 = ddelta1.flatten()[idx]
    delta2 = ddelta2.flatten()[idx]
    bias = {}
    variance = {}
    try:
        bias, variance = bias_and_variance(reps=mc_reps, delta1=delta1, delta2=delta2, n=n, p=assignment_probability,
                                           dyadic_design=design, pdfy=fY)
    # catch cases when delta1 and delta2 produce an invalid (not non-negative) pmf for treatment participation
    except ValueError:
        print('Invalid (negative) pmf of treatment participations for given deltas. NaNs returned.')
        bias['A'], bias['B'] = np.full(4, np.nan), np.full(4, np.nan)
        variance['A'], variance['B'] = np.full(4, np.nan), np.full(4, np.nan)
    for member in ['A', 'B']:
        for theta, k in zip(['theta1', 'theta2', 'theta3', 'theta4'], range(4)):
            biases[member][theta].append(bias[member][k])
            variances[member][theta].append(variance[member][k])
# reshaping
for member in ['A', 'B']:
    for theta, k in zip(['theta1', 'theta2', 'theta3', 'theta4'], range(4)):
        biases[member][theta] = np.array(biases[member][theta]).reshape(len(delta2_list), len(delta1_list))
        variances[member][theta] = np.array(variances[member][theta]).reshape(len(delta2_list), len(delta1_list))

# save resulting dictionaries to file
with open(r'./biases_reps%d_n%d.p' %(mc_reps, n), 'wb') as myfile:
    cpickle.dump(biases, myfile)
with open(r'./variances_reps%d_n%d.p' %(mc_reps, n), 'wb') as myfile:
    cpickle.dump(variances, myfile)
