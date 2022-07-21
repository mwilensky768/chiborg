import numpy as np

class jk_data():

    def __init__(self, simulate=True, mean=1, std=1, bias=0, num_pow=4,
                 num_draw=int(1e6), bp_meas=None):
        """
        Container for holding data and covariance (i.e. what is conditioned on)

        Parameters:
            simulate: Whether to simulate bandpower draws
            mean: The unbiased mean of the bandpower draws
            std: The standard deviation of the bandpower draws
            bias: An optional bias to add to the mean
            num_pow: Number of bandpowers to draw in a single group
            num_draw: Number of draws to do of length num_pow
            bp_meas: Some measured bandpowers, if not simulating.

        Attributes:
            bp_draws: An array of draws from a gaussian with parameters
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

        if not isinstance(num_pow, int):
            warnings.warn("Casting num_pow parameter as an integer")
            num_pow = int(num_pow)

        if "__iter__" in dir(std):
            if not isinstance(std, np.ndarray):
                warnings.warn("Casting std parameter as an array")
                std = np.array(std)

        self.num_pow = num_pow
        self.num_draw = num_draw
        self.dat_shape = (num_draw, num_pow)

        self.mean = mean
        self.std = std
        self.bias = bias

        if simulate:
            self.simulated = True
            self.bp_draws = np.random.normal(loc=self.mean + self.bias,
                                             scale=self.std,
                                             size=self.dat_shape)
        else:
            self.simulated = False
            if self.num_draw > 1:
                raise ValueError("num_draw must be set to 1 for measured bandpowers.")
            bp_meas_arr = np.array(bp_meas)
            self.bp_draws = bp_meas_arr[np.newaxis, :]
            shape_match = self.bp_draws.shape == self.dat_shape
            if not shape_match:
                raise ValueError("User must supply 1-dimensional input for bp_meas of length num_pow.")