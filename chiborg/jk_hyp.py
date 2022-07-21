from collections import namedtuple

gauss_prior = namedtuple("gauss_prior", ["mean", "std"])
multi_gauss_prior = namedtuple("multi_gauss_prior", ["mean", "cov"])

class jk_hyp():

    def __init__(mode="diagonal", tm_mean=None, tm_cov=None, bias_mean=None,
                 bias_cov=None):
        valid_modes = ["diagonal", "partition", "manual"]
        if mode not in :
            raise ValueError(f"mode keyword must be one of {valid_modes}")
        self.mode = mode
        self.num_hyp = self.get_num_hyp


        self.bp_prior = gauss_prior(tm_mean, tm_std)
        bias_prior_mean_vec, bias_prior_cov = self._get_bias_mean_cov(bias_prior_mean,
                                                                      bias_prior_std,
                                                                      bias_prior_corr)
        self.bias_prior = multi_gauss_prior(bias_prior_mean_vec, bias_prior_cov)

        jk_hyp_std_min = np.amin(np.diag(np.sqrt(self.jk_hyp.bias_prior_cov)))
        if jk_hyp_std_min > 1e4 * np.amin(self.jk_data.std):
            warnings.warn("Bias prior is sufficiently large compared to error"
                          " bar to produce floating point roundoff errors. The"
                          " likelihoods may be untrustworthy and this is an "
                          " unnecessarily wide prior.")

    def get_num_hyp(self):
        """
        Calculate the number of hypotheses based on the mode.
        """
        if self.mode == 'partition': # Calculate the (N + 1)th Bell Number
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
            num_hyp = len(hyp_cov)
        return(num_hyp)

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
