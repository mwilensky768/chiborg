import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.sparse import block_diag
from scipy.integrate import quad_vec
from scipy.linalg import LinAlgError
import copy
import warnings


class jk_calc():

    def __init__(self, jk_hyp, analytic=True):
        """
        Class for containing jackknife parameters and doing calculations of
        various test statistics.

        Parameters:
            jk_hyp: jk_hyp object that holds useful parameters about test
                hypotheses (and implicitly a jk_data object)
            analytic: Whether to use analytic result for likelihood computation
        """

        self.jk_hyp = copy.deepcopy(jk_hyp)

        jk_hyp_cov_min = np.amin(np.diag(self.jk_hyp.bias_prior.cov[-1]))
        jk_data_cov_min = np.amin(np.diag(self.jk_hyp.jk_data.noise_cov))
        if jk_hyp_cov_min > 1e8 * jk_data_cov_min:
            warnings.warn("Bias prior is sufficiently large compared to error"
                          " bar to produce floating point roundoff errors. The"
                          " likelihoods may be untrustworthy and this is an "
                          " unnecessarily wide prior.")

        self.analytic = analytic
        gauss = (self.jk_hyp.tm_prior.func.__func__ == norm.pdf.__func__)
        if self.analytic and (not gauss):
            raise ValueError("Must use normal prior if asking for analytic "
                             "marginalization.")

        self.like, self.marg_mean, self.marg_cov, self.entropy = self.get_like()
        self.sum_entropy = self.jk_hyp.hyp_prior @ self.entropy
        self.evid = self.get_evidence()
        self.post = self.get_post()

    def get_like(self):
        """
        Get the likelihoods for each of the hypotheses.
        """

        like = np.zeros([self.jk_hyp.num_hyp, self.jk_hyp.jk_data.num_draw])
        entropy = np.zeros(self.jk_hyp.num_hyp)
        marg_mean = np.zeros([self.jk_hyp.num_hyp,
                              self.jk_hyp.jk_data.num_dat])
        marg_cov = np.zeros([self.jk_hyp.num_hyp,
                             self.jk_hyp.jk_data.num_dat,
                             self.jk_hyp.jk_data.num_dat])

        for hyp_ind in range(self.jk_hyp.num_hyp):
            if self.analytic:
                like[hyp_ind], entropy[hyp_ind], marg_mean[hyp_ind], marg_cov[hyp_ind] = \
                    self._get_like_analytic(hyp_ind)
            else:
                like[hyp_ind] = self._get_like_num(hyp_ind)

        return(like, marg_mean, marg_cov, entropy)

    def _get_mod_var_cov_sum_inv(self, hyp_ind):
        cov_sum = self.jk_hyp.jk_data.noise_cov + self.jk_hyp.bias_prior.cov[hyp_ind]
        cov_sum_inv = np.linalg.inv(cov_sum)
        mod_var = 1 / np.sum(cov_sum_inv)

        return(mod_var, cov_sum_inv, cov_sum)

    def _get_middle_cov(self, mod_var):
        if self.jk_hyp.tm_prior.params["scale"] == 0:
            prec_sum = np.inf
        else:
            prec_sum = 1 / mod_var + 1 / self.jk_hyp.tm_prior.params["scale"]**2
        middle_C = np.ones([self.jk_hyp.jk_data.num_dat, self.jk_hyp.jk_data.num_dat]) / prec_sum
        return(middle_C)

    def _get_like_analytic(self, hyp_ind):

        mod_var, cov_sum_inv, _ = self._get_mod_var_cov_sum_inv(hyp_ind)
        mu_tm = np.full(self.jk_hyp.jk_data.num_dat, self.jk_hyp.tm_prior.params["loc"])
        marg_mean = self.jk_hyp.bias_prior.mean[hyp_ind] + mu_tm
        middle_C = self._get_middle_cov(mod_var)

        cov_inv_adjust = cov_sum_inv @ middle_C @ cov_sum_inv
        marg_cov = np.linalg.inv(cov_sum_inv - cov_inv_adjust)
        like = multivariate_normal(mean=mu_prime, cov=C_prime).pdf(self.jk_hyp.jk_data.data_draws)
        entropy = multivariate_normal(mean=mu_prime, cov=C_prime).entropy() / np.log(2)

        return(like, marg_mean, marg_cov, entropy)

    def _get_integr(self, hyp_ind):

        _, _, cov_sum = self._get_mod_var_cov_sum_inv(hyp_ind)

        def integrand(x):
            gauss_1_arg = self.jk_hyp.jk_data.data_draws - self.jk_hyp.bias_prior.mean[hyp_ind]
            gauss_1 = multivariate_normal.pdf(gauss_1_arg,
                                              mean=np.full(self.jk_hyp.jk_data.num_dat, x),
                                              cov=cov_sum)
            gauss_2 = self.jk_hyp.tm_prior.func(x, **self.jk_hyp.tm_prior.params)

            return(gauss_1 * gauss_2)

        return(integrand)

    def _get_like_num(self, hyp_ind):

        integrand_func = self._get_integr(hyp_ind)

        integral, err, info = quad_vec(integrand_func,
                                       *self.jk_hyp.tm_prior.bounds,
                                       full_output=True)
        if not info.success:
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

    def gen_data_mix(self, num_draw):
        """
        Generate jk_data, jk_hyp, and jk_calc objects whose data draws are a
        mixture from the marginal likelihoods of the hypotheses described by
        this object's k_hyp object. Somewhat memory intensive in exchange for a
        bit of speed. Must be in analytic mode.

        Args:
            num_draw (int): How many total draws are desired for the mixture.
        """
        if not self.analytic:
            raise ValueError("Must be in analytic mode to generate data mixture.")

        # Make the mixed up means
        sim_mean = np.random.normal(loc=self.jk_hyp.tm_prior.params["loc"],
                                    scale=self.jk_hyp.tm_prior.params["scale"],
                                    size=(num_draw, self.jk_hyp.jk_data.num_dat))
        # Make the mixed up biases
        all_bias_draws = [ np.random.multivariate_normal(mean=self.jk_hyp.bias_prior.mean[hyp_ind],
                                                         cov=self.jk_hyp.bias_prior.cov[hyp_ind],
                                                         size=num_draw).T # Have to transpose to get broadcasting right later
                           for hyp_ind in range(self.jk_hyp.num_hyp) ]
        # Do a bunch of multinomial trials to see which component of the mixture should be taken
        trial = np.random.multinomial(1,
                                      self.jk_hyp.hyp_prior,
                                      size=num_draw).argmax(axis=1)
        mix_bias = np.choose(trial, all_bias_draws).T # Have to transpose to get shape right in jk_data

        jkd = jk_data(simulate=True, sim_mean=sim_mean, sim_bias=mix_bias,
                      noise_cov=self.jk_hyp.jk_data.noise_cov,
                      num_dat=self.jk_hyp.jk_data.num_dat, num_draw=num_draw)
        jkh = jk_hyp(jkd, self.jk_hyp.bias_prior.mean,
                     self.jk_hyp.bias_prior.cov, self.jk_hyp.tm_prior,
                     self.jk_hyp.hyp_prior, mode="manual")
        jkc = self.jkc(jkh)

        return(jkd, jkh, jkc)
