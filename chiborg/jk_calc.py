import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.sparse import block_diag
from scipy.integrate import quad_vec
from scipy.special import comb
from scipy.linalg import LinAlgError
from collections import namedtuple
from itertools import combinations, chain
from more_itertools import set_partitions
from more_itertools.recipes import powerset
import copy
import warnings

gauss_prior = namedtuple("gauss_prior", ["mean", "std"])
multi_gauss_prior = namedtuple("multi_gauss_prior", ["mean", "cov"])


class jk_calc():

    def __init__(self, jk_data, bp_prior_mean=1, bp_prior_std=0.5,
                 bias_prior_mean=0, bias_prior_std=10, bias_prior_corr=1,
                 hyp_prior=None, analytic=True, mode='diagonal'):
        """
        Class for containing jackknife parameters and doing calculations of
        various test statistics.

        Parameters:
            jk_data: A jk_data object that holds the data and covariance
                with which to work.
            bp_pior_mean: Mean parameter for the bandpower prior
            bp_prior_std: Standard deviation for bandpower prior
            bias_prior_mean: Mean parameter for bias prior
            bias_prior_std: Standard deviation for bias prior
            bias_prior_corr: Correlation coefficient for bias prior in
                correlated hypothesis
            hyp_prior: Prior probability of each hypothesis, in the order
                (null, uncorrelated bias, correlated bias)
            analytic: Whether to use analytic result for likelihood computation
        """
        if mode not in ['full', 'diagonal', 'ternary']:
            raise ValueError("mode must be 'full', 'diagonal', ternary")

        self.mode = mode
        self.jk_data = copy.deepcopy(jk_data)
        if np.amin(bias_prior_std) > 1e4 * np.amin(self.jk_data.std):
            warnings.warn("Bias prior is sufficiently large compared to error"
                          " bar to produce floating point roundoff errors. The"
                          " likelihoods may be untrustworthy and this is an "
                          " unnecessarily wide prior.")

        self.bp_prior = gauss_prior(bp_prior_mean, bp_prior_std)
        self.num_hyp = self.get_num_hyp()
        bias_prior_mean_vec, bias_prior_cov = self._get_bias_mean_cov(bias_prior_mean,
                                                                      bias_prior_std,
                                                                      bias_prior_corr)
        if hyp_prior is None:  # Default to flat
            self.hyp_prior = np.ones(self.num_hyp) / self.num_hyp
        elif not np.isclose(np.sum(hyp_prior), 1):
            raise ValueError("hyp_prior does not sum close to 1, which can result in faulty normalization.")
        elif len(hyp_prior) != self.num_hyp:
            raise ValueError("hyp_prior length does not match hypothesis set length. Check mode keyword.")
        else:
            self.hyp_prior = hyp_prior
        self.bias_prior = multi_gauss_prior(bias_prior_mean_vec, bias_prior_cov)
        self.analytic = analytic
        self.noise_cov = self._get_noise_cov()

        self.like, self.entropy = self.get_like()
        self.sum_entropy = self.hyp_prior @ self.entropy
        self.evid = self.get_evidence()
        self.post = self.get_post()

    def get_num_hyp(self):
        """
        Fun fact: these are called bell numbers in the full case. For N
        bandpowers, we actually want the N+1th bell number, which is the total
        number of ways of partitioning a set with N+1 elements.
        """
        if self.mode == 'full':
            M = self.jk_data.num_pow + 1
            B = np.zeros(M + 1, dtype=int)
            B[0] = 1  # NEED THE SEED
            for n in range(M):
                for k in range(n + 1):
                    B[n + 1] += comb(n, k, exact=True) * B[k]

            num_hyp = B[M]
        elif self.mode == 'diagonal':
            num_hyp = 2**(self.jk_data.num_pow)
        else:
            num_hyp = 3
        return(num_hyp)

    def get_like(self):
        """
        Get the likelihoods for each of the null hypotheses.
        """

        like = np.zeros([self.num_hyp, self.jk_data.num_draw])
        entropy = np.zeros(self.num_hyp)
        for hyp_ind in range(self.num_hyp):
            if self.analytic:
                like[hyp_ind], entropy[hyp_ind] = self._get_like_analytic(hyp_ind)
            else:
                like[hyp_ind] = self._get_like_num(hyp_ind)

        return(like, entropy)

    def _get_bias_mean_cov(self, bias_prior_mean, bias_prior_std, bias_prior_corr):
        if not hasattr(bias_prior_mean, "__iter__"):
            bias_prior_mean = np.repeat(bias_prior_mean, 4)
        if not hasattr(bias_prior_std, "__iter__"):
            bias_prior_std = np.repeat(bias_prior_std, 4)
        bias_cov_shape = [self.num_hyp, self.jk_data.num_pow, self.jk_data.num_pow]
        bias_cov = np.zeros(bias_cov_shape)
        bias_mean = np.zeros([self.num_hyp, self.jk_data.num_pow])

        ###
        # The matrices need to be transitive - this should be all of them. #
        ###
        diag_val = bias_prior_std**2
        hyp_ind = 0
        if self.mode in ["full", "diagonal"]:
            for diag_on in powerset(range(self.jk_data.num_pow)):
                N_on = len(diag_on)
                if N_on == 0:  # Null hypothesis - all 0 cov. matrix
                    hyp_ind += 1
                elif self.mode == "full":
                    parts = set_partitions(diag_on)  # Set of partitionings
                    for part in parts:  # Loop over partitionings
                        bias_cov[hyp_ind, diag_on, diag_on] = diag_val[np.array(diag_on)]
                        for sub_part in part:  # Loop over compartments to correlate them
                            off_diags = combinations(sub_part, 2)  # Get off-diagonal indices for this compartment
                            for pair in off_diags:  # Fill off-diagonal indices for this compartment
                                off_diag_val = bias_prior_corr * bias_prior_std[pair[0]] * bias_prior_std[pair[1]]
                                bias_cov[hyp_ind, pair[0], pair[1]] = off_diag_val
                                bias_cov[hyp_ind, pair[1], pair[0]] = off_diag_val
                        bias_mean[hyp_ind, diag_on] = bias_prior_mean[np.array(diag_on)]
                        hyp_ind += 1
                else:  # Mode must be diagonal
                    bias_cov[hyp_ind, diag_on, diag_on] = diag_val[np.array(diag_on)]
                    bias_mean[hyp_ind, diag_on] = bias_prior_mean[np.array(diag_on)]
                    hyp_ind += 1
        else:
            diag_inds = np.arange(self.jk_data.num_pow)
            bias_cov[1] = diag_val * np.eye(self.jk_data.num_pow)
            bias_cov[2] = (1 - bias_prior_corr) * bias_cov[1] + bias_prior_corr * np.outer(bias_prior_std, bias_prior_std)
            for hyp_ind in [1, 2]:
                bias_mean[hyp_ind] = bias_prior_mean

        return(bias_mean, bias_cov)

    def _get_noise_cov(self):
        # Assuming a scalar
        if hasattr(self.jk_data.std, "__iter__"):  # Assume a vector
            noise_cov = np.diag(self.jk_data.std**2)
        else:
            noise_cov = np.diag(np.repeat(self.jk_data.std**2, self.jk_data.num_pow))
        return(noise_cov)

    def _get_mod_var_cov_sum_inv(self, hyp_ind):
        cov_sum = self.noise_cov + self.bias_prior.cov[hyp_ind]
        cov_sum_inv = np.linalg.inv(cov_sum)
        mod_var = 1 / np.sum(cov_sum_inv)

        return(mod_var, cov_sum_inv, cov_sum)

    def _get_middle_cov(self, mod_var):
        if self.bp_prior.std == 0:
            prec_sum = np.inf
        else:
            prec_sum = 1 / mod_var + 1 / self.bp_prior.std**2
        middle_C = np.ones([self.jk_data.num_pow, self.jk_data.num_pow]) / prec_sum
        return(middle_C)

    def _get_like_analytic(self, hyp_ind):

        mod_var, cov_sum_inv, _ = self._get_mod_var_cov_sum_inv(hyp_ind)
        mu_prime = self.bias_prior.mean[hyp_ind] + np.repeat(self.bp_prior.mean, self.jk_data.num_pow)
        middle_C = self._get_middle_cov(mod_var)

        cov_inv_adjust = cov_sum_inv @ middle_C @ cov_sum_inv
        C_prime = np.linalg.inv(cov_sum_inv - cov_inv_adjust)
        like = multivariate_normal(mean=mu_prime, cov=C_prime).pdf(self.jk_data.bp_draws)
        entropy = multivariate_normal(mean=mu_prime, cov=C_prime).entropy() / np.log(2)

        return(like, entropy)

    def _get_integr(self, hyp_ind):

        _, _, cov_sum = self._get_mod_var_cov_sum_inv(hyp_ind)

        def integrand(x):
            gauss_1 = multivariate_normal.pdf(self.jk_data.bp_draws - self.bias_prior.mean[hyp_ind],
                                              mean=x * np.ones(self.jk_data.num_pow),
                                              cov=cov_sum)
            gauss_2 = norm.pdf(x, loc=self.bp_prior.mean, scale=self.bp_prior.std)

            return(gauss_1 * gauss_2)

        return(integrand)

    def _get_like_num(self, hyp_ind):

        integrand_func = self._get_integr(hyp_ind)

        integral = quad_vec(integrand_func, -np.inf, np.inf)[0]

        return(integral)

    def get_evidence(self):
        evid = self.hyp_prior @ self.like
        return(evid)

    def get_post(self):
        # Transpose to make shapes conform to numpy broadcasting
        post = (self.like.T * self.hyp_prior).T / self.evid
        return(post)

    def gen_bp_mix(self, num_draw):
        """
        Generate a mixture of bandpower objects in accordance with the priors.

        Args:
            num_draw: How many bandpowers to simulate per hypothesis
        """
        bp_list = []
        mean = np.random.normal(loc=self.bp_prior.mean, scale=self.bp_prior.std,
                                size=[num_draw, self.num_hyp, self.jk_data.num_pow])
        bias = np.random.multivariate_normal(mean=self.bias_prior.mean,
                                             cov=block_diag(self.bias_prior.cov),
                                             size=num_draw).reshape([num_draw, self.num_hyp, self.jk_data.num_pow])
        std = self.jk_data.std
        for hyp in range(self.num_hyp):
            bp = bandpower(mean=mean[:, hyp, :], std=std, num_draw=num_draw,
                           bias=bias[:, hyp, :], num_pow=self.jk_data.num_pow)
            bp_list.append(bp)

        return(bp_list)
