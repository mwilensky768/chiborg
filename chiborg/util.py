import numpy as np
from chiborg import jk_data, jk_hyp, jk_calc

def gen_data_mix(jkc, num_draw):
    """
    Generate jk_data, jk_hyp, and jk_calc objects whose data draws are a
    mixture from the marginal likelihoods of the hypotheses described by
    the jkc object's jk_hyp object. Somewhat memory intensive in exchange for a
    bit of speed. Must be in analytic mode.

    Args:
        jkc: A jk_calc object from which to generate the mixture
        num_draw (int): How many total draws are desired for the mixture.
    """
    if not jkc.analytic:
        raise ValueError("Must be in analytic mode to generate data mixture.")

    # Make the mixed up means
    sim_mean = np.random.normal(loc=jkc.jk_hyp.tm_prior.params["loc"],
                                scale=jkc.jk_hyp.tm_prior.params["scale"],
                                size=(num_draw, jkc.jk_hyp.jk_data.num_dat))
    # Make the mixed up biases
    all_bias_draws = [ np.random.multivariate_normal(mean=jkc.jk_hyp.bias_prior.mean[hyp_ind],
                                                     cov=jkc.jk_hyp.bias_prior.cov[hyp_ind],
                                                     size=num_draw).T # Have to transpose to get broadcasting right later
                          for hyp_ind in range(jkc.jk_hyp.num_hyp) ]
    # Do a bunch of multinomial trials to see which component of the mixture should be taken
    trial = np.random.multinomial(1,
                                  jkc.jk_hyp.hyp_prior,
                                  size=num_draw).argmax(axis=1)
    mix_bias = np.choose(trial, all_bias_draws).T # Have to transpose to get shape right in jk_data

    jkd_mix = jk_data(simulate=True, sim_mean=sim_mean, sim_bias=mix_bias,
                      noise_cov=jkc.jk_hyp.jk_data.noise_cov,
                      num_dat=jkc.jk_hyp.jk_data.num_dat, num_draw=num_draw)
    jkh_mix = jk_hyp(jkd, jkc.jk_hyp.bias_prior.mean,
                     jkc.jk_hyp.bias_prior.cov, jkc.jk_hyp.tm_prior,
                     jkc.jk_hyp.hyp_prior, mode="manual")
    jkc_mix = jk_calc(jkh)

    return(jkd_mix, jkh_mix, jkc_mix)

def get_mut_info(jkc):
    """
    Gets the mutual information b/w hypotheses and data set that corresponds to
    those hypotheses. Used to evaluate the distinguishability of the hypotheses.

    Args:
        jkc: A jk_calc object from which to generate the mixture.
    """

    jkd_mix, jkh_mix, jkc_mix = gen_data_mix(jkc)
    logs = -np.where(jkc.evid > 0, np.log2(jkc.evid), 0)
    S_d = np.mean(logs)
    I = S_d - jkc_mix.sum_entropy

    return(I)
