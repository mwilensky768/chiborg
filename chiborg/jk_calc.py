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
        gauss = self.jk_hyp.tm_prior.name.lower() == "gaussian"
        if self.analytic and (not gauss):
            raise ValueError("Must use Gaussian prior if asking for analytic "
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
                like[hyp_ind], marg_mean[hyp_ind], marg_cov[hyp_ind], entropy[hyp_ind] = \
                    self.get_like_analytic(hyp_ind)
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

    def get_like_analytic(self, hyp_ind, calc_entropy=True):
        """
        Get the analytic likelihood for a hypothesis assuming all quantities are
        Gaussian. This is now implemented differently than expressed in the
        original paper by taking account of the fact that each hypothesis is a 
        Gaussian linear model. Also calculates the entropy of the marginal
        distribution of the data.

        Parameters:
            hyp_ind (int):
                The index for the hypothesis in question.
        
        Returns:
            like (float): 
                The marginal likelihood of the hypothesis in question.
            marg_mean (array):
                The marginal mean of the data according to the hypothesis.
            marg_cov (array):
                The marginal covariance of the data according to the hypothesis.
            entropy (float):
                The entropy, in bits, of the marginal distribution of the data.
        """

        model_Dmatr = np.hstack((np.ones(self.jk_hyp.jk_data.num_dat), np.eye(self.jk_hyp.jk_data.num_dat)))
        model_mean = np.vstack((np.atleast_2d(self.jk_hyp.tm_prior.params["loc"]), self.jk_hyp.bias_prior.mean[hyp_ind]))

        cov_offdiag = np.zeros([1, self.jk_hyp.jk_data.num_dat])
        model_cov = np.block([[np.atleast_2d(self.jk_hyp.tm_prior.params["scale"]**2), cov_offdiag], 
                              [cov_offdiag.T, self.jk_hyp.bias_prior.cov[hyp_ind]]])
        
        marg_mean = model_Dmatr @ model_mean
        marg_cov = self.jk_hyp.jk_data.noise_cov + model_Dmatr @ model_cov @ model_Dmatr.T

        like = multivariate_normal(mean=marg_mean, cov=marg_cov).pdf(self.jk_hyp.jk_data.data_draws)
        entropy = multivariate_normal(mean=marg_mean, cov=marg_cov).entropy() / np.log(2)

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
