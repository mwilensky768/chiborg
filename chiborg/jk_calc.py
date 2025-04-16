import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import quad_vec
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
                like[hyp_ind] = self.get_like_num(hyp_ind)

        return like, marg_mean, marg_cov, entropy

    def get_like_analytic(self, hyp_ind):
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

        model_Dmatr = np.hstack((np.ones(self.jk_hyp.jk_data.num_dat)[:, None],
                                 np.eye(self.jk_hyp.jk_data.num_dat)))
        model_mean = np.vstack((np.atleast_2d(self.jk_hyp.tm_prior.params["loc"]),
                                self.jk_hyp.bias_prior.mean[hyp_ind][:, None]))

        cov_offdiag = np.zeros([1, self.jk_hyp.jk_data.num_dat])
        model_cov = np.block([[np.atleast_2d(self.jk_hyp.tm_prior.params["scale"]**2), cov_offdiag], 
                              [cov_offdiag.T, self.jk_hyp.bias_prior.cov[hyp_ind]]])
        
        # Index marg_mean on the axis of length 1 so broadcasting isn't a problem
        marg_mean = (model_Dmatr @ model_mean)[:, 0]
        marg_cov = self.jk_hyp.jk_data.noise_cov + model_Dmatr @ model_cov @ model_Dmatr.T
        
        mn_rv = multivariate_normal(mean=marg_mean, cov=marg_cov)
        like = mn_rv.pdf(self.jk_hyp.jk_data.data_draws)
        entropy = mn_rv.entropy() / np.log(2)

        return like, marg_mean, marg_cov, entropy

    def get_integr(self, hyp_ind):
        """
        Generate a function that evaluates the conditional likelihood for the
        underlying true mean multiplied by the prior for the true mean 
        (for a given hypothesis). Used in numerical evaluation of the marginal 
        likelihood in the case that a non-Gaussian prior is used.

        Parameters:
            hyp_ind (int):
                The index for the hypothesis in question.
        
        Returns:
            integrand (function):
                The desired function.
        """

        cov_sum = self.jk_hyp.jk_data.noise_cov + self.jk_hyp.bias_prior.cov[hyp_ind]

        def integrand(x):
            cond_mean = np.full(self.jk_hyp.jk_data.num_dat, x) + self.jk_hyp.bias_prior.mean[hyp_ind]
            gauss_1 = multivariate_normal.pdf(self.jk_hyp.jk_data.data_draws,
                                              mean=cond_mean,
                                              cov=cov_sum)
            gauss_2 = self.jk_hyp.tm_prior.func(x, **self.jk_hyp.tm_prior.params)

            return gauss_1 * gauss_2

        return integrand

    def get_like_num(self, hyp_ind):
        """
        Get the marginal likelihood of the hypothesis using quadrature integration.

        Parameters:
            hyp_ind (int):
                The index of the hypothesis in question.
        
        Returns:
            integral (float):
                The marginal likelihood of the data under this hypothesis.
        """

        integrand_func = self.get_integr(hyp_ind)

        integral, err, info = quad_vec(integrand_func,
                                       *self.jk_hyp.tm_prior.bounds,
                                       full_output=True)
        if not info.success:
            warnings.warn("Numerical integration flagged as unsuccessful. "
                          "Results may be untrustworthy.")

        return integral

    def get_evidence(self):
        """
        Gets the evidence for the entire mixture model (the denominator in the
        model comparison framework).

        Returns:
            evid (float):
                The normalizing constant in the model comparison problem.
        """
        evid = self.jk_hyp.hyp_prior @ self.like
        return evid

    def get_post(self):
        """
        Gets the posterior over the hypotheses.

        Returns:
            post (array):
                The posterior probability for each hypothesis.
        """
        # Transpose to make shapes conform to numpy broadcasting
        post = (self.like.T * self.jk_hyp.hyp_prior).T / self.evid
        return post
