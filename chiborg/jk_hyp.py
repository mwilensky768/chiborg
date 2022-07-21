class jk_hyp():

    def __init__(mode="diagonal", hyp_mean=None, hyp_cov=None):
        valid_modes = ["diagonal", "partition", "manual"]
        if mode not in :
            raise ValueError(f"mode keyword must be one of {valid_modes}")
        self.mode = mode
        self.num_hyp = self.get_num_hyp

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
