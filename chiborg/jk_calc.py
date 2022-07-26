self.jk_hyp.tm_prior.params["loc"]import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.sparse import block_diag
from scipy.integrate import quad_vec
from scipy.linalg import LinAlgError
import copy
import warnings


class jk_calc():

    def __init__(self, jk_data, jk_hyp, analytic=True):
        """
        Class for containing jackknife parameters and doing calculations of
        various test statistics.

        Parameters:
            jk_data: A jk_data object that holds the data and covariance
                with which to work.
            jk_hyp: jk_hyp object that holds useful parameters about test
                hypotheses
            analytic: Whether to use analytic result for likelihood computation
        """
        self.jk_data = copy.deepcopy(jk_data)
        self.jk_hyp = copy.deepcopy(jk_hyp)

        jk_hyp_std_min = np.amin(np.diag(np.sqrt(self.jk_hyp.bias_prior_cov)))
        if jk_hyp_std_min > 1e4 * np.amin(self.jk_data.std):
            warnings.warn("Bias prior is sufficiently large compared to error"
                          " bar to produce floating point roundoff errors. The"
                          " likelihoods may be untrustworthy and this is an "
                          " unnecessarily wide prior.")

        self.analytic = analytic
        if self.analytic and (self.jk_hyp.tm_prior.func != norm):
            raise ValueError("Must use normal prior if asking for analytic "
                             "marginalization.")
        self.noise_cov = self._get_noise_cov()

        self.like, self.entropy = self.get_like()
        self.sum_entropy = self.jk_hyp.hyp_prior @ self.entropy
        self.evid = self.get_evidence()
        self.post = self.get_post()

    def get_like(self):
        """
        Get the likelihoods for each of the hypotheses.
        """

        like = np.zeros([self.jk_hyp.num_hyp, self.jk_data.num_draw])
        entropy = np.zeros(self.jk_hyp.num_hyp)
        for hyp_ind in range(self.jk_hyp.num_hyp):
            if self.analytic:
                like[hyp_ind], entropy[hyp_ind] = self._get_like_analytic(hyp_ind)
            else:
                like[hyp_ind] = self._get_like_num(hyp_ind)

        return(like, entropy)

    def _get_noise_cov(self):
        # Assuming a scalar
        if hasattr(self.jk_data.std, "__iter__"):  # Assume a vector
            noise_cov = np.diag(self.jk_data.std**2)
        else:
            noise_cov = np.diag(np.repeat(self.jk_data.std**2, self.jk_data.num_dat))
        return(noise_cov)

    def _get_mod_var_cov_sum_inv(self, hyp_ind):
        cov_sum = self.noise_cov + self.jk_hyp.bias_prior.cov[hyp_ind]
        cov_sum_inv = np.linalg.inv(cov_sum)
        mod_var = 1 / np.sum(cov_sum_inv)

        return(mod_var, cov_sum_inv, cov_sum)

    def _get_middle_cov(self, mod_var):
        if self.jk_hyp.tm_prior.params["scale"] == 0:
            prec_sum = np.inf
        else:
            prec_sum = 1 / mod_var + 1 / self.jk_hyp.tm_prior.params["scale"]**2
        middle_C = np.ones([self.jk_data.num_dat, self.jk_data.num_dat]) / prec_sum
        return(middle_C)

    def _get_like_analytic(self, hyp_ind):

        mod_var, cov_sum_inv, _ = self._get_mod_var_cov_sum_inv(hyp_ind)
        mu_prime = self.jk_hyp.bias_prior.mean[hyp_ind] + np.repeat(self.jk_hyp.tm_prior.params["loc"], self.jk_data.num_dat)
        middle_C = self._get_middle_cov(mod_var)

        cov_inv_adjust = cov_sum_inv @ middle_C @ cov_sum_inv
        C_prime = np.linalg.inv(cov_sum_inv - cov_inv_adjust)
        like = multivariate_normal(mean=mu_prime, cov=C_prime).pdf(self.jk_data.bp_draws)
        entropy = multivariate_normal(mean=mu_prime, cov=C_prime).entropy() / np.log(2)

        return(like, entropy)

    def _get_integr(self, hyp_ind):

        _, _, cov_sum = self._get_mod_var_cov_sum_inv(hyp_ind)

        def integrand(x):
            gauss_1 = multivariate_normal.pdf(self.jk_data.bp_draws - self.jk_hyp.bias_prior.mean[hyp_ind],
                                              mean=x * np.ones(self.jk_data.num_dat),
                                              cov=cov_sum)
            gauss_2 = self.jk_hyp.tm_prior.func(x, **self.jk_hyp.tm_prior.params)

            return(gauss_1 * gauss_2)

        return(integrand)

    def _get_like_num(self, hyp_ind):

        integrand_func = self._get_integr(hyp_ind)

        integral, err, info = quad_vec(integrand_func,
                                       self.jk_hyp.tm_prior.bounds,
                                       full_output=True)
        if not info["success"]:
            warnings.warn("Numerical integration flagged as unsuccessful. "
                          "Results may be untrustworthy.")

        return(integral)

    def get_evidence(self):
        evid = self.jk_hyp.hyp_prior @ self.like
        return(evid)

    def get_post(self):
        # Transpose to make shapes conform to numpy broadcasting
        post = (self.like.T * self.jk_hyp.hyp_prior).T / self.evid
        return(post)

    def gen_bp_mix(self, num_draw):
        """
        Generate a mixture of bandpower objects in accordance with the priors.

        Args:
            num_draw: How many bandpowers to simulate per hypothesis
        """
        bp_list = []
        mean = np.random.normal(loc=self.jk_hyp.tm_prior.params["loc"], scale=self.jk_hyp.tm_prior.params["scale"],
                                size=[num_draw, self.jk_hyp.num_hyp, self.jk_data.num_dat])
        bias = np.random.multivariate_normal(mean=self.jk_hyp.bias_prior.mean,
                                             cov=block_diag(self.jk_hyp.bias_prior.cov),
                                             size=num_draw).reshape([num_draw, self.jk_hyp.num_hyp, self.jk_data.num_dat])
        std = self.jk_data.std
        for hyp in range(self.jk_hyp.num_hyp):
            bp = bandpower(mean=mean[:, hyp, :], std=std, num_draw=num_draw,
                           bias=bias[:, hyp, :], num_dat=self.jk_data.num_dat)
            bp_list.append(bp)

        return(bp_list)
