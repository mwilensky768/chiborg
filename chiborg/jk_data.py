import numpy as np
from scipy.linalg import cholesky

class jk_data():

    def __init__(self, simulate=False, sim_mean=np.zeros(2), noise_cov=np.eye(2),
                 sim_bias=np.zeros(2), num_dat=2, num_draw=1,
                 meas_dat=None):
        """
        Container for holding data and covariance (i.e. what is conditioned on
        formalism)

        Parameters:
            simulate: Whether to simulate data draws
            sim_mean: The unbiased mean of the data draws
            noise_cov: The data covariance
            sim_bias: An optional bias vector to add to the mean
            num_dat: Number of data to draw in a single group
            num_draw: Number of draws to do of length num_dat
            meas_dat: Some measured data, if not simulating.

        Attributes:
            data_draws: An array of draws from a gaussian with parameters
                described by the class parameters, OR an array of measured
                bandpowers.
        """

        neither = ((meas_dat is None) and (not simulate))
        both = ((meas_dat is not None) and simulate)
        if neither:
            raise ValueError("User must supply measured bandpowers if simulate is False")
        elif both:
            raise ValueError("Measured data have been supplied and simulate has "
                             "been set to True. User must either supply "
                             "measured bandpowers or simulate them.")

        if not isinstance(num_draw, int):
            warnings.warn("Casting num_draw parameter as an integer")
            num_samp = int(num_draw)

        if not isinstance(num_dat, int):
            warnings.warn("Casting num_dat parameter as an integer")
            num_dat = int(num_dat)

        self.num_dat = num_dat
        self.num_draw = num_draw


        self.sim_mean = sim_mean
        self.noise_cov= noise_cov
        self.sim_bias = sim_bias

        if simulate:
            self.dat_shape = (num_draw, num_dat)
            self.simulated = True
            self.data_draws = self.sim_draws()
        else:
            self.simulated = False
            if self.num_draw > 1:
                warnings.warn("num_draw must be set to 1 for measured data."
                              "Changing this parameter now.")
                self.num_draw = 1
            meas_dat_arr = np.array(meas_dat)
            if self.num_dat != len(meas_dat_arr):
                warnings.warn("num_dat must be equal to length of supplied data."
                              " Resetting this parameter.")
                self.num_dat = len(meas_dat_arr)
            self.data_draws = meas_dat_arr[np.newaxis, :]
            self.dat_shape = (num_draw, num_dat)
            shape_match = self.data_draws.shape == self.dat_shape
            if not shape_match:
                raise ValueError("User must supply 1-dimensional input for "
                                 "meas_dat of length num_dat.")

    def sim_draws(self):
        """
        Simulate draws according to the error covariance, mean, and bias. Uses
        a standard multivariate gaussian sampling technique using Cholesky
        factorization for the covariance matrix.
        """
        std_gauss = np.random.normal(size=[self.num_draw, self.num_dat])
        cho = cholesky(self.noise_cov, lower=False)
        draws = std_gauss@cho + self.sim_mean + self.sim_bias

        return(draws)
