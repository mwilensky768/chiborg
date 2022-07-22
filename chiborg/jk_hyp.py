from collections import namedtuple
import copy

gauss_prior = namedtuple("gauss_prior", ["mean", "std"])
multi_gauss_prior = namedtuple("multi_gauss_prior", ["mean", "cov"])

class jk_hyp():

    def __init__(jk_data, bias_mean, bias_cov, tm_mean, tm_std, mode="diagonal"):
        """
        Args:
            jk_data: A jk_data object containing the data for the hypothesis
                test.
            bias_mean: If mode is 'diagonal' or 'partition', must use a vector.
                The values represent the mean parameter of the bias prior for
                any hypothesis in which that bias is active (detailed in the
                paper). If mode is 'manual', then this must be an array with
                first dimension of length equal to number of hypotheses
                considered, and second dimension equal to the number of data.
            bias_cov: Covariances for the bias priors of the hypotheses in
                consideration. If mode is 'diagonal' or 'partition', must use a
                vector. The values represent the variances of the priors. If
                mode is 'manual', then must supply a sequence of covariance
                matrices for each hypothesis in consideration.
            tm_mean: The mean of the prior for the 'true mean' parameter, i.e.
                what the data would be concentrated around in the absence of
                bias.
            tm_std: The standard deviation of the prior for the
                'true mean parameter'.
            mode: Which set of hypotheses to use. Valid options are 'diagonal',
                'partition', and 'manual'. The first two are detailed in the
                paper, and are essentially summaries of the set of bias
                covariance matrices in consideration. The third indicates that
                the user will supply the specific means and covariances of the
                bias hypotheses.
        """
        valid_modes = ["diagonal", "partition", "manual"]
        if mode not in valid_modes:
            raise ValueError(f"mode keyword must be one of {valid_modes}")
        self.mode = mode
        self.jk_data = copy.deepcopy(self.jk_data)
        self.num_hyp = self.get_num_hyp()

        self.bp_prior = gauss_prior(tm_mean, tm_std)
        if self.mode != "manual":
            bias_mean, bias_cov = self._get_bias_mean_cov(bias_mean,
                                                          bias_cov)
        self.bias_prior = multi_gauss_prior(bias_prior_mean_vec, bias_prior_cov)

    def get_num_hyp(self):
        """
        Calculate the number of hypotheses based on the mode.
        """
        if self.mode == 'partition': # Calculate the (N + 1)th Bell Number
            M = self.jk_data.num_dat + 1
            B = np.zeros(M + 1, dtype=int)
            B[0] = 1  # NEED THE SEED
            for n in range(M):
                for k in range(n + 1):
                    B[n + 1] += comb(n, k, exact=True) * B[k]
            num_hyp = B[M]
        elif self.mode == 'diagonal':
            num_hyp = 2**(self.jk_data.num_dat)
        else:
            num_hyp = self.jk_data.num_dat # Will be manual, so this object will have len
        return(num_hyp)

    def _get_bias_mean_cov(self, bias_mean, bias_cov):
        if self.mode == "manual":
            bias_mean =
        bias_cov_shape = [self.num_hyp, self.jk_data.num_pow, self.jk_data.num_pow]
        bias_cov = np.zeros(bias_cov_shape)
        bias_mean = np.zeros([self.num_hyp, self.jk_data.num_pow])

        ###
        # The matrices need to be transitive - this should be all of them. #
        ###
        diag_val = bias_prior_std**2
        hyp_ind = 0
        if self.mode in ["partition", "diagonal"]:
            for diag_on in powerset(range(self.jk_data.num_pow)):
                N_on = len(diag_on)
                if N_on == 0:  # Null hypothesis - all 0 cov. matrix
                    hyp_ind += 1
                elif self.mode == "partition":
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
