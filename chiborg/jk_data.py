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
                identity design matrix.

        Attributes:
            data_draws: An array of draws from a gaussian with parameters
                described by the class parameters, OR an array of measured
                bandpowers.
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
            
        if dmatr is None:
            dmatr = np.eye(self.dat_len)
        self.dmatr = dmatr
    

class SimContainer:
    
    def __init__(self, num_draw, num_dat, mean, noise_cov, bias):
        if not isinstance(num_draw, int):
            warnings.warn("Casting num_draw parameter as an integer")
            num_draw = int(num_draw)

        if not isinstance(num_dat, int):
            warnings.warn("Casting num_dat parameter as an integer")
            num_dat = int(num_dat)

        self.num_draw = num_draw
        self.num_dat = num_dat
        self.dat_shape = (self.num_draw, self.num_dat)
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
        std_gauss = np.random.normal(size=[self.num_draw, self.num_dat])
        cho = cholesky(self.noise_cov, lower=False)
        draws = std_gauss@cho + self.mean + self.bias

        return(draws)

