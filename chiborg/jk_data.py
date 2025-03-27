import numpy as np
from scipy.linalg import cholesky
import warnings

class DataContainer:

    def __init__(self, dat, noise_cov, dmatr=None):
        """
        Container for holding data and covariance (i.e. what is conditioned on
        formalism)

        Parameters:
            meas_dat: 
                Some measured data. Each datum can be vector-valued where each
                of the num_dat measurements are of length dat_len. In other words,
                this can be a 2d array of shape (num_dat, dat_len).
            noise_cov: 
                The data covariance, shape (num_dat, dat_len, num_dat, dat_len)
            dmatr:
                A design matrix for a linear model. Otherwise just assume the OG
                identity design matrix. Shape (dat_len, num_params).
        """
        if not isinstance(dat, np.ndarray):
            warnings.warn("Casting data to numpy array.")
            dat = np.array(dat)
        if dat.ndim == 1:
            self.dat = np.atleast_2d(dat).T # Interpret the dat_len as 1
        else:
            self.dat = dat
        self.num_dat, self.dat_len = self.dat.shape
        self.noise_cov= noise_cov
        cov_shape_correct = self.noise_cov.shape == 2 * self.dat.shape
        assert cov_shape_correct, "Reshape supplied covariance so that it is 4 dimensional, inserting axes of length 1 where appropriate."
    

class SimContainer:
    
    def __init__(self, num_draw, dat_len, mean, noise_cov, bias):
        """
        Class for performing and containing simulations.

        Parameters:
            num_draw (int):
                Number of simulated draws to take.
            dat_len (int):
                Vector length of data to draw.
            mean (array):
                True mean of the simulated draws, length dat_len
            noise_cov (array):
                Noise covariance of the draws, shape (dat_len, dat_len)
            bias (array):
                Bias to assign to each of the simulated draws.
            
        """
        if not isinstance(num_draw, int):
            warnings.warn("Casting num_draw parameter as an integer")
            num_draw = int(num_draw)

        if not isinstance(dat_len, int):
            warnings.warn("Casting num_dat parameter as an integer")
            dat_len = int(dat_len)

        self.num_draw = num_draw
        self.dat_len = dat_len
        self.dat_shape = (self.num_draw, self.dat_len)
        self.simulated = True
        self.noise_cov = noise_cov
        self.mean = mean
        self.bias = bias

        self.dat = self.sim_draws()

    def sim_draws(self):
        """
        Simulate draws according to the error covariance, mean, and bias. Uses
        a standard multivariate gaussian sampling technique using Cholesky
        factorization for the covariance matrix.
        """
        std_gauss = np.random.normal(size=[self.dat_len, self.num_dat])
        cho = cholesky(self.noise_cov, lower=False)
        draws = std_gauss@cho + self.mean + self.bias

        return(draws)

