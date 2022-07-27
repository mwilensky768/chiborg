import numpy as np

class jk_data():

    def __init__(self, simulate=True, sim_mean=np.zeros(2), sim_cov=np.eye(2),
                 sim_bias=np.zeros(2), num_dat=2, num_draw=int(1e6),
                 meas_dat=None):
        """
        Container for holding data and covariance (i.e. what is conditioned on
        formalism)

        Parameters:
            simulate: Whether to simulate data draws
            sim_mean: The unbiased mean of the data draws
            sim_cov: The data covariance
            sim_bias: An optional bias vector to add to the mean
            num_dat: Number of data to draw in a single group
            num_draw: Number of draws to do of length num_dat
            meas_dat: Some measured data, if not simulating.

        Attributes:
            data_draws: An array of draws from a gaussian with parameters
                described by the class parameters, OR an array of measured
                bandpowers.
        """

        neither = ((bp_meas is None) and (not simulate))
        both = ((bp_meas is not None) and simulate)
        if neither:
            raise ValueError("User must supply measured bandpowers if simulate is False")
        elif both:
            raise ValueError("Measured bandpowers have been supplied and simulate has been set to True."
                             "User must either supply measured bandpowers or simulate them.")

        if not isinstance(num_draw, int):
            warnings.warn("Casting num_draw parameter as an integer")
            num_samp = int(num_draw)

        if not isinstance(num_dat, int):
            warnings.warn("Casting num_dat parameter as an integer")
            num_dat = int(num_dat)

        if "__iter__" in dir(std):
            if not isinstance(std, np.ndarray):
                warnings.warn("Casting std parameter as an array")
                std = np.array(std)

        self.num_dat = num_dat
        self.num_draw = num_draw
        self.dat_shape = (num_draw, num_dat)

        self.mean = mean
        self.std = std
        self.bias = bias

        if simulate:
            self.simulated = True
            self.data_draws = np.random.normal(loc=self.mean + self.bias,
                                               scale=self.std,
                                               size=self.dat_shape)
        else:
            self.simulated = False
            if self.num_draw > 1:
                raise ValueError("num_draw must be set to 1 for measured bandpowers.")
            bp_meas_arr = np.array(bp_meas)
            self.data_draws = bp_meas_arr[np.newaxis, :]
            shape_match = self.bp_draws.shape == self.dat_shape
            if not shape_match:
                raise ValueError("User must supply 1-dimensional input for bp_meas of length num_dat.")
