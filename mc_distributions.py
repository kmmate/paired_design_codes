"""
Thesis.
Distribution of potential treatment assignments and potential outcomes for use with the MC simulation
"""

import numpy as np
from scipy.stats import multivariate_normal

class PmfD:
    """
    Probability distribution of the random vector of potential treatment participation:
    [D_A(10), D_A(11), D_B(01), D_B(11)], which follows a Multivariate Bernoulli distribution.

    Note: D_A(10) and D_B(10) stand for the potential participations when Z_A=1 and Z_B=0, and
          D_A(01) and D_B(01) stand for the potential participations when Z_A=0 and Z_B=1.
    """
    def __init__(self, delta1, delta2):
        """
        Setting the joint probabilities of the probability mass function of the potentials treatment participation
         as a function of the extent of the violation of the symmetry assumptions.

        Parameters
        ----------
        :param delta1: scalar
            Discrepancy in first symmetry assumption = P(D_A(11)=1) - P(D_B(11)=1)
        :param delta2: scalar
            Discrepancy in second symmetry assumption = P(D_A(11)=1, D_A(10)=0) - P(D_B(11)=1, D_B(01)=0)
        """
        # free-to-choose joint probabilities
        weights = {'0000': 8,
                   '0100': 4,
                   '0010': 0,
                   '1100': 6,
                   '0110': 0,
                   '0011': 3,
                   '1010': 0,
                   '0101': 5,
                   '1001': 0,
                   '1011': 0,
                   '1101': 8,
                   '1111': 10}
        # quasi-normalisation with deltas not taken into account
        quasinorm_factor = 1 / (sum(weights.values()) + weights['0100'] + weights['1100'] - weights['0011'] +
                                + weights['1100'] - weights['0011'] + weights['1101'])
        quasinorm_weights = {i: weights[i] * quasinorm_factor for i in weights}
        # joint probabilities determined by free-to-choose ones and the discrepancies
        quasinorm_weights['0001'] = -delta1 + quasinorm_weights['0100'] + quasinorm_weights['1100'] -\
                                    quasinorm_weights['0011']
        quasinorm_weights['0111'] = delta2 - delta1 + quasinorm_weights['1100'] - quasinorm_weights['0011'] +\
                                    quasinorm_weights['1101']
        # normalise with deltas taken into account
        norm_factor = 1 / sum(quasinorm_weights.values())
        normalised_weights = {i: quasinorm_weights[i] * norm_factor for i in quasinorm_weights}
        self.pmf = normalised_weights

    def sampler(self, samplesize):
        """
        I.effect_idx.d. sampling from the pmf of the potential treatment participation vector:
        [D_A(10), D_A(11), D_B(01), D_B(11)]

        Parameters
        ----------
        :param samplesize: int
            Sample size

        Returns
        -------
        :return: (samplesize x 4) array of the generated sample
        """
        sample = np.random.choice(list(self.pmf.keys()), size=samplesize, replace=True, p=list(self.pmf.values()))
        return sample


class PdfY:
    """
    Probability distribution of the random vector of potential outcomes:
    [Y_A(00), Y_A(10), Y_A(01), Y_A(11), Y_B(00), Y_B(10), Y_B(01), Y_B(11)] conditional on the potential treatment
    participations. The conditional distribution is a Multivariate Normal distribution with
    'potential treatment participations'-specific mean-vectors and a positive-def covariance matrix, independent of
    the potential treatment participations.

    Note: Y_A(10) and Y_B(10) stand for the potential outcomes when D_A=1 and D_B=0, and
          Y_A(01) and Y_B(01) stand for the potential outcomes when D_A=0 and D_B=1.
    """
    def __init__(self, cov):
        """
        Specifying the distribution of the random vector of potential outcomes conditional on the potential treatment
        participations.

        Parameters
        ----------
        :param cov: 8 x 8 array
            The positive semidefinite covariance matrix, independent of the potential treatment participations.
        """
        self.cov = cov
        self.mu = {'0000': [0, 0, 0, 0, 1, 1, 1, 1],
                   '0100': [0, 0, 0, 1, 1, 1, 1, 0],
                   '0001': [0, 0, 0, -1, 1, 1, 1, 2],
                   '1100': [0, 1, 0, 1, 1, 1, 0, 0],
                   '0011': [0, -1, 0, -1, 1, 1, 2, 2],
                   '0101': [0, 0, 0, 4, 1, 1, 1, 5],
                   '0111': [0, 0, 3, 6, 1, 1, 0, 7],
                   '1101': [0, -1, 0, 6, 1, 3, 1, 7],
                   '1111': [0, 5, 6, 8, 1, 4, 8, 10]
                   }

    def sampler(self, d_sample):
        """
        Draws independent and (conditionally) identically distributed samples from the distribution of the
        potential outcomes conditional on the potential treatment participations.

        Parameters
        ----------
        :param d_sample: list of strings
            Each entry in the list is a value of the vector of the potential participations in string format which
            PmfD.sampler(samplesize) returns.

        Returns
        -------
        :return: array-like
            random sample of size (samplesize x 8)
        """
        normal_sample = np.random.multivariate_normal(np.zeros(8), self.cov, len(d_sample))
        sample = [z + self.mu[d] for z, d in zip(normal_sample, d_sample)]
        return sample

