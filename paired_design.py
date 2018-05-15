"""
Thesis.
Paired 2-by-2 Factorial Design with two identification strategies.
"""

import numpy as np
import _warnings


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


class PairedDesign:
    """
    Paired Experimental Design.
    """
    def __init__(self, z_a=None, z_b=None):
        """
        Initialising the design.

        If the experimenter already assigned the treatment, it can be given as an argument.

        Parameters
        ----------
        :param z_a array-like
            Treatment encouragement belonging to members-A as specified by the experimenter.
        :param z_b array-like
            Treatment encouragement belonging to member-B as specified by the experimenter.

        Examples
        --------
        >>> design = PairedDesign()
        """
        self.z_a = z_a
        self.z_b = z_b
        if (z_a is not None) and (z_b is not None):
            if len(z_a) != len(z_b):
                raise ValueError('The length of the treatment encouragement vector for member-A and member-B must be'
                                 ' the same.')
            self.n = len(z_a)

    def generate_encouragement(self, npairs, p=0.5):
        """
        Assign treatments independently to each individual in the sample.

        WARNING: If called, overwrites encouragement specified by the experimenter at initialisation.

        Parameters
        ----------
        :param npairs: int
            Number of pairs.
        :param p: scalar in (0,1)
            Probability that any given individual is encouraged to take the treatment.
            The default value 0.5 is highly recommended.

        Returns
        -------
        :return: z_a, z_b
            two array of length npairs with treatment encouragement indicated with 1, not-encouragement with 0
            z_a: encouragement belonging to members-A, z_b: encouragement belonging to members-B

        Examples
        --------
        >>> design = PairedDesign()
        >>> number_of_pairs = 314
        >>> z_a, z_b = design.generate_encouragement(number_of_pairs)

        """
        self.n = npairs
        z_a = np.array(list(map(lambda x: round_own(x, 1 - p), np.random.uniform(low=0.0, high=1.0, size=self.n))))
        z_b = np.array(list(map(lambda x: round_own(x, 1 - p), np.random.uniform(low=0.0, high=1.0, size=self.n))))
        # overwrite treatment encouragements
        self.z_a = z_a
        self.z_b = z_b
        return z_a, z_b

    def treatment_effect_a(self, y_a, d_a, d_b, fullsample=True):
        """
        Estimating the baseline level and the treatment effects for member-A

        Parameters
        ----------
        :param y_a: array-like
            Observed outcome of members-A
        :param d_a: array-like
            Observed treatment participation of members-A
        :param d_b: array-like
            Observed treatment participation of members-B
        :param fullsample: bool
            Whether to use the full-sample identification strategy of Theorem 2. If False, the half-sample
            identification strategy of Theorem 1 is used.

        Returns
        -------
        :return: array
            If `fullsample' is True (see Theorem 2), the return is a four-long array with the entries:
                [0]: estimated expected baseline level of member-A
                [1]: estimated average effect of own treatment (treatment-A) on member-A in the subpopulation of
                     complier member-A's
                [2]: estimated average effect of partners' treatment (treatment-B) on member-A in the subpopulation
                     of member-A's with complier member-B partners
                [3] estimated spillover-like effect of member-A.
            If `fullsample' is False (see Theorem 1) the return is a three-long array with the entries [0]-[2] above.

        Examples
        --------
        >>> design = PairedDesign()
        >>> number_of_pairs = 314
        >>> z_a, z_b = design.generate_encouragement(number_of_pairs)
        >>> d_a, d_b, y_a = ...  # give measured data here
        >>> thetahat_a = design.treatment_effect_a(y_a, d_a, d_b, fullsample=True)  # averge effects for member-A
        """
        if self.z_a is None:
            raise ValueError('Treatment encouragement for members-A has to be specified either by the user or'
                             ' by calling the generate_encouragement method.')
        if self.z_b is None:
            raise ValueError('Treatment encouragement for members-B has to be specified either by the user or'
                             ' by calling the generate_encouragement method.')

        y_a = np.array(y_a)
        z_a = np.array(self.z_a)
        z_b = np.array(self.z_b)
        d_a = np.array(d_a)
        d_b = np.array(d_b)
        # full-sample identification strategy
        if fullsample:
            zmatrix_a = np.array([np.ones(self.n), z_a, z_b, z_a * z_b]).T
            dmatrix_a = np.array([np.ones(self.n), d_a, d_b, d_a * d_b]).T
            try:
                thetahat_a = np.linalg.inv(zmatrix_a.T.dot(dmatrix_a)).dot(zmatrix_a.T.dot(y_a))
            except np.linalg.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    _warnings.warn('Singular matrix: insufficient variation in data. Effects cannot be calculated,'
                                  ' np.nans are returned instead')
                    thetahat_a = np.full(4, np.nan)
        # half-sample identification strategy
        else:
            thetahat_a = np.zeros(3)
            # baseline and own effect from Half-1: use only (z_a=1 and z_b=0) or (z_a=0 and z_b=0), ie. only z_b=0
            y_a_half1 = y_a[z_b == 0]
            z_a_half1 = z_a[z_b == 0]
            d_a_half1 = d_a[z_b == 0]
            zmatrix_a_half1 = np.array([np.ones(len(y_a_half1)), z_a_half1]).T
            dmatrix_a_half1 = np.array([np.ones(len(y_a_half1)), d_a_half1]).T
            try:
                thetahat_a[0:2] = np.linalg.inv(zmatrix_a_half1.T.dot(dmatrix_a_half1)).dot(
                    zmatrix_a_half1.T.dot(y_a_half1))
            except np.linalg.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    _warnings.warn('Singular matrix: insufficient variation in data. Effects cannot be calculated,'
                                  ' np.nans are returned instead.')
                    thetahat_a[0:2] = np.full(2, np.nan)
            # spouse effect from Half-2: use only (z_a=0 and z_b=1) or (z_a=0 and z_b=0), ie. only z_a=0
            y_a_half2 = y_a[z_a == 0]
            z_b_half2 = z_b[z_a == 0]
            d_b_half2 = d_b[z_a == 0]
            zmatrix_b_half2 = np.array([np.ones(len(y_a_half2)), z_b_half2]).T
            dmatrix_b_half2 = np.array([np.ones(len(y_a_half2)), d_b_half2]).T
            #thetahat_a[2] = np.linalg.lstsq(zmatrix_b_half2.T.dot(dmatrix_b_half2), zmatrix_b_half2.T.dot(y_a_half2),
             #                            rcond=None)[0][1]
            try:
                thetahat_a[2] = np.linalg.inv(zmatrix_b_half2.T.dot(dmatrix_b_half2)).dot(
                    zmatrix_b_half2.T.dot(y_a_half2))[1]
            except np.linalg.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    _warnings.warn('Singular matrix: insufficient variation in data. Effects cannot be calculated,'
                                  ' np.nans are returned instead')
                    thetahat_a[2] = np.nan
        return thetahat_a

    def treatment_effect_b(self, y_b, d_a, d_b, fullsample=True):
        """
        Estimating the baseline level and the treatment effects of Theorem 2 for member-A

        Parameters
        ----------
        :param y_b: array-like
            Observed outcome of members-A
        :param d_a: array-like
            Observed treatment participation of members-A
        :param d_b: array-like
            Observed treatment participation of members-B
        :param fullsample: bool
            Whether to use the full-sample identification strategy of Theorem 2. If False, the half-sample
            identification strategy of Theorem 1 is used.

        Returns
        -------
        :return: array
            If `fullsample' is True (see Theorem 2), the return is a four-long array with the entries:
                [0]: estimated expected baseline level of member-B
                [1]: estimated average effect of own treatment (treatment-B) on member-B in the subpopulation of
                     complier member-B's
                [2]: estimated average effect of partners' treatment (treatment-A) on member-B in the subpopulation of
                     member-B's with complier member-A partners
                [3]  estimated spillover-like effect of member-B.
            If `fullsample' is False (see Theorem 1) the return is a three-long array with the entries [0]-[2] above.

        Examples
        --------
        >>> design = PairedDesign()
        >>> number_of_pairs = 314
        >>> z_a, z_b = design.generate_encouragement(number_of_pairs)
        >>> d_a, d_b, y_b = ...  # give measured data here
        >>> thetahat_b = design.treatment_effect_a(y_b, d_a, d_b, fullsample=True)  # average effects for member-B
        """
        if self.z_a is None:
            raise ValueError('Treatment encouragement for members-A has to be specified either by the user or'
                             ' by calling the generate_encouragement method.')
        if self.z_b is None:
            raise ValueError('Treatment encouragement for members-B has to be specified either by the user or'
                             ' by calling the generate_encouragement method.')

        y_b = np.array(y_b)
        z_a = np.array(self.z_a)
        z_b = np.array(self.z_b)
        d_a = np.array(d_a)
        d_b = np.array(d_b)
        if fullsample:
            dmatrix_b = np.array([np.ones(self.n), d_b, d_a, d_a * d_b]).T
            zmatrix_b = np.array([np.ones(self.n), z_b, z_a, z_a * z_b]).T
            try:
                thetahat_b = np.linalg.inv(zmatrix_b.T.dot(dmatrix_b)).dot(zmatrix_b.T.dot(y_b))
            except np.linalg.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    _warnings.warn('Singular matrix: insufficient variation in data. Effects cannot be calculated,'
                                  ' nans are returned instead')
                    thetahat_b = np.full(4, np.nan)
        # half-sample identification strategy
        else:
            thetahat_b = np.zeros(3)
            # baseline own effect from Half-2: use only (z_a=0 and z_b=1) or (z_a=0 and z_b=0), ie. only z_a=0
            y_b_half2 = y_b[z_a == 0]
            z_b_half2 = z_b[z_a == 0]
            d_b_half2 = d_b[z_a == 0]
            zmatrix_b_half2 = np.array([np.ones(len(y_b_half2)), z_b_half2]).T
            dmatrix_b_half2 = np.array([np.ones(len(y_b_half2)), d_b_half2]).T
            try:
                thetahat_b[0:2] = np.linalg.inv(zmatrix_b_half2.T.dot(dmatrix_b_half2)).dot(
                    zmatrix_b_half2.T.dot(y_b_half2))
            except np.linalg.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    _warnings.warn('Singular matrix: insufficient variation in data. Effects cannot be calculated,'
                                  ' nans are returned instead')
                    thetahat_b[0:2] = np.full(2, np.nan)
            # spouse effect from Half-1: use only (z_a=1 and z_b=0) or (z_a=0 and z_b=0), ie. only z_b=0
            y_b_half1 = y_b[z_b == 0]
            z_a_half1 = z_a[z_b == 0]
            d_a_half1 = d_a[z_b == 0]
            zmatrix_a_half1 = np.array([np.ones(len(y_b_half1)), z_a_half1]).T
            dmatrix_a_half1 = np.array([np.ones(len(y_b_half1)), d_a_half1]).T
            try:
                thetahat_b[2] = np.linalg.inv(zmatrix_a_half1.T.dot(dmatrix_a_half1)).dot(
                    zmatrix_a_half1.T.dot(y_b_half1))[1]
            except np.linalg.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    _warnings.warn('Singular matrix: insufficient variation in data. Effects cannot be calculated,'
                                  ' nans are returned instead')
                    thetahat_b[2] = np.nan
        return thetahat_b