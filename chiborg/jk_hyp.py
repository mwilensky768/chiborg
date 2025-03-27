from collections import namedtuple
import copy
from scipy.special import comb
from itertools import combinations
from more_itertools import set_partitions
from more_itertools.recipes import powerset
from scipy.stats import norm
import numpy as np
import warnings

tm_prior = namedtuple("tm_prior", ["func", "bounds", "params", "name"])
multi_gauss_prior = namedtuple("multi_gauss_prior", ["mean", "cov"])

class HypContainer:

    def __init__(self, data_container, bias_mean, bias_cov,
                 tmp=tm_prior(norm.pdf, [-np.inf, np.inf],
                              {"loc": 0, "scale": 0}, "gaussian"),
                 hyp_prior=None, mode="diagonal"):
        """
        Args:
            data_container: 
                A DataContainer or SimContainer object 
            bias_mean: 
                If mode is 'diagonal' or 'partition', must use a vector.
                The values represent the mean parameter of the bias prior for
                any hypothesis in which that bias is active (detailed in the
                paper). If mode is 'manual', then this must be an array with
                first dimension of length equal to number of hypotheses
                considered, and second dimension equal to the number of data.
            bias_cov: 
                Covariances for the bias priors of the hypotheses in
                consideration. If mode is 'diagonal' or 'partition', must use a
                vector. The values represent the variances of the priors. If
                mode is 'manual', then must supply a sequence of covariance
                matrices for each hypothesis in consideration.
            tmp: 
                tm_prior namedtuple containing a function to be evaluated for
                numerical marginalization over the true mean prior as well as
                the integration bounds and any parameters that the prior has.
                It also has a name attribute that describes the type of prior.
                For proper interaction with the jk_calc class in analytic mode,
                a Gaussian prior must be used with the name 'Gaussian', as in
                the default argument case.
            mode: 
                Which set of hypotheses to use. Valid options are 'diagonal',
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
        self.data_container = copy.deepcopy(data_container)
        self.num_hyp = self.get_num_hyp(bias_cov)

        if self.mode != "manual":
            if not isinstance(bias_mean, np.ndarray):
                warnings.warn("Casting bias_mean to array.")
                bias_mean = np.array(bias_mean)
            if not isinstance(bias_cov, np.ndarray):
                warnings.warn("Casting bias_cov to array.")
                bias_cov = np.array(bias_cov)
            bias_mean, bias_cov = self.get_bias_mean_cov(bias_mean,
                                                          bias_cov)
        self.bias_prior = multi_gauss_prior(bias_mean, bias_cov)
        self.tm_prior = tmp

        if hyp_prior is None:  # Default to flat
            self.hyp_prior = np.ones(self.num_hyp) / self.num_hyp
        elif not np.isclose(np.sum(hyp_prior), 1):
            raise ValueError("hyp_prior does not sum close to 1, which can "
                             "result in faulty normalization.")
        elif len(hyp_prior) != self.num_hyp:
            raise ValueError("hyp_prior length does not match hypothesis set "
                             "length. Check mode keyword.")
        else:
            self.hyp_prior = hyp_prior

    def get_num_hyp(self, bias_cov):
        """
        Calculate the number of hypotheses based on the mode.

        Args:
            bias_cov: bias prior covariance submitted to constructor. Only used
                to set num_hyp in manual mode.
        """
        if self.mode == 'partition': # Calculate the (N + 1)th Bell Number
            M = self.data_container.num_dat + 1
            B = np.zeros(M + 1, dtype=int)
            B[0] = 1  # NEED THE SEED
            for n in range(M):
                for k in range(n + 1):
                    B[n + 1] += comb(n, k, exact=True) * B[k]
            num_hyp = B[M]
        elif self.mode == 'diagonal':
            num_hyp = 2**(self.data_container.num_dat)
        else:
            num_hyp = len(bias_cov)
        return(num_hyp)

    def get_bias_mean_cov(self, bias_mean, bias_cov):
        """
        Automatically construct means and covs for class of hypotheses specified
        by the mode attribute.

        Args:
            bias_mean: A 1d array of means, each component corresponding to a
                datum in the jk_data attribute
            bias_cov: A 1d array of variances, each component corresponding to a
                datum in the jk_data attribute. Should be the variances of the
                bias prior, not the data itself.
        Returns:
            bias_mean_final: Array of bias mean vectors. One for each hypothesis.
            bias_cov_final: Array of covariance matrices for the bias priors.
                One for each hypothesis.
        """
        bias_cov_shape = [self.num_hyp, self.data_container.num_dat, self.data_container.num_dat]
        bias_cov_final = np.zeros(bias_cov_shape)
        bias_mean_final = np.zeros([self.num_hyp, self.data_container.num_dat])

        ###
        # The matrices need to be transitive - this should be all of them. #
        ###
        hyp_ind = 0
        for diag_on in powerset(range(self.data_container.num_dat)):
            N_on = len(diag_on)
            if N_on == 0:  # Null hypothesis - all 0 cov. matrix
                hyp_ind += 1
            elif self.mode == "partition":
                parts = set_partitions(diag_on)  # Set of partitionings
                for part in parts:  # Loop over partitionings
                    bias_cov_final[hyp_ind, diag_on, diag_on] = bias_cov[np.array(diag_on)]
                    for sub_part in part:  # Loop over compartments to correlate them
                        off_diags = combinations(sub_part, 2)  # Get off-diagonal indices for this compartment
                        for pair in off_diags:  # Fill off-diagonal indices for this compartment
                            off_diag_val = np.sqrt(bias_cov[pair[0]] * bias_cov[pair[1]])
                            bias_cov_final[hyp_ind, pair[0], pair[1]] = off_diag_val
                            bias_cov_final[hyp_ind, pair[1], pair[0]] = off_diag_val
                    bias_mean_final[hyp_ind, diag_on] = bias_mean[np.array(diag_on)]
                    hyp_ind += 1
            else:  # Mode must be diagonal
                bias_cov_final[hyp_ind, diag_on, diag_on] = bias_cov[np.array(diag_on)]
                bias_mean_final[hyp_ind, diag_on] = bias_mean[np.array(diag_on)]
                hyp_ind += 1


        return(bias_mean_final, bias_cov_final)
    

class ModelContainer:

    def __init__(self, dmatr, prior_mean, prior_cov,
                 bias_mean, bias_cov):
        """
        A class that contains parameters associated with a Gaussian linear model.

        Parameters:
            dmatr (array):
                A design matrix of shape (dat_len, num_params).
            model_mean (array):
                Mean of the model when the bias is turned on.
        """
        
        self.dmatr = dmatr
        self.num_params = dmatr.shape[1]
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.bias_mean = bias_mean
        self.bias_cov = bias_cov

        self.prior_mean_data = self.dmatr @ self.prior_mean
        self.bias_mean_data = self.dmatr @ self.bias_mean
        self.prior_cov_data = self.dmatr @ self.prior_cov @ self.dmatr.T
        self.bias_cov_data = self.dmatr @ self.bias_cov @ self.dmatr.T
        
