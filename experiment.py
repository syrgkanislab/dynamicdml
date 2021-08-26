import numpy as np
from snmm import get_linear_model_reg, get_linear_multimodel_reg
from snmm import get_model_reg, get_multimodel_reg
from snmm import get_poly_model_reg, get_poly_multimodel_reg
from econml.utilities import cross_product
from snmm import SNMMDynamicDML
from snmm import gen_data
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
import pandas as pd
from joblib import Parallel, delayed
import joblib
from collections import OrderedDict
import os
import argparse
from blip import BlipSpec, SimpleHeteroBlipSpec, SimpleBlipSpec, true_param_parse

def pi(t, X, T):
    return np.ones(T[t].shape)


def experiment(*, n_periods, n_units, n_treatments,
                n_x, s_x, s_t,
                hetero_strenth, n_hetero_vars, autoreg,
                instance_seed, sample_seed, verbose,
                max_poly_degree, high_dim):
    m = n_periods
    y, X, T, true_effect_params, _ = gen_data(n_periods=m, n_units=n_units,
                                              n_treatments=n_treatments,
                                              n_x=n_x, s_x=s_x, s_t=s_t,
                                              hetero_strenth=hetero_strenth,
                                              n_hetero_vars=n_hetero_vars,
                                              autoreg=autoreg,
                                              instance_seed=instance_seed,
                                              sample_seed=sample_seed)
    if n_hetero_vars == 0:
        X['het'] = X[0]

    #######################
    # High dim blip
    #######################

    bs = BlipSpec().fit(X, T)
    phi = bs.phi
    phi_names = bs.phi_names

    true_quantities = true_param_parse(X, T, true_effect_params, n_hetero_vars, n_periods, phi, phi_names)
    true_params, true_params_sel, true_policy, true_policy_delta, true_opt_policy, true_opt_policy_delta = true_quantities

    # model_reg_fn = lambda X, y: get_model_reg(X, y, degrees=[1])
    # multimodel_reg_fn = lambda X, y: get_multimodel_reg(X, y, degrees=[1])
    # model_reg_fn = get_linear_model_reg
    # multimodel_reg_fn = get_linear_multimodel_reg
    model_reg_fn = lambda X, y: get_poly_model_reg(X, y, degree=1, interaction_only=True)
    multimodel_reg_fn = lambda X, y: get_poly_multimodel_reg(X, y, degree=1, interaction_only=True)

    if high_dim:
        est = SNMMDynamicDML(m=m, phi=phi, phi_names_fn=phi_names,
                            model_reg_fn=model_reg_fn,
                            model_final_fn=lambda: LassoCV(fit_intercept=False),
                            verbose=verbose)
        est.fit(X, T, y, pi)

        ##################################
        # Post selection low dim blip
        ##################################
        sig = {}
        for t in range(m):
            sig[t] = np.abs(est.psi_[t]) > 0.01
            sig[t][:T[t].shape[1]] = True # always

        def phi_sub(t, X, T, Tt):
            return phi(t, X, T, Tt)[:, sig[t]]

        def phi_names_sub(t):
            return np.array(phi_names(t))[sig[t]]

        est_sub = SNMMDynamicDML(m=m, phi=phi_sub, phi_names_fn=phi_names_sub,
                            model_reg_fn=lambda X, y: get_model_reg(X, y, degrees=list(np.arange(1, max_poly_degree + 1)), verbose=verbose-2),
                            model_final_fn=lambda: LinearRegression(fit_intercept=False),
                            verbose=verbose)
        est_sub.fit(X, T, y, pi)
        est_sub.fit_opt(X, T, y, beta=1.5)

    ##################################
    # Oracle low dim blip
    ##################################
    oracle_blip = SimpleHeteroBlipSpec(n_hetero_vars).fit(X, T)

    est_low = SNMMDynamicDML(m=m, phi=oracle_blip.phi, phi_names_fn=oracle_blip.phi_names,
                         model_reg_fn=lambda X, y: get_model_reg(X, y, degrees=list(np.arange(1, max_poly_degree + 1)), verbose=verbose-2),
                         model_final_fn=lambda: LinearRegression(fit_intercept=False),
                         verbose=verbose)
    est_low.fit(X, T, y, pi)
    est_low.fit_opt(X, T, y, beta=1.5)

    results = {}
    results['true'] = {'params': true_params,
                       'pival': true_policy,
                       'pidelta': true_policy_delta,
                       'optval': true_opt_policy,
                       'optdelta': true_opt_policy_delta}
    if high_dim:
        results['reg'] = {'pival': est.policy_value_,
                        'pidelta': est.policy_delta_simple_,
                        'params': {t: est.param_summary(t).summary_frame() for t in range(m)}}
        results['ols'] = {'pival': est_sub.policy_value_,
                        'pidelta': est_sub.policy_delta_simple_,
                        'params': {t: est_sub.param_summary(t).summary_frame() for t in range(m)},
                        'optval': est_sub.opt_policy_value_,
                        'optdelta': est_sub.opt_policy_delta_simple_,
                        'optparams': {t: est_sub.opt_param_summary(t).summary_frame() for t in range(m)}}
    results['oracle_ols'] = {'pival': est_low.policy_value_,
                             'pidelta': est_low.policy_delta_simple_,
                             'params': {t: est_low.param_summary(t).summary_frame() for t in range(m)},
                             'optval': est_low.opt_policy_value_,
                             'optdelta': est_low.opt_policy_delta_simple_,
                             'optparams': {t: est_low.opt_param_summary(t).summary_frame() for t in range(m)}}
    return results


def main(*, n_periods, n_units, n_treatments,
         n_x, s_x, s_t, n_instances, inst_seed, n_samples, sample_seed, max_poly_degree, high_dim,
         hetero_strenth=2.0, n_hetero_vars=0, autoreg=1.0, verbose=0):

    res = []
    for instance_seed in np.arange(inst_seed, inst_seed + n_instances):
        res.append(Parallel(n_jobs=-1, verbose=1)(delayed(experiment)(n_periods=n_periods,
                                                                      n_units=n_units,
                                                                      n_treatments=n_treatments,
                                                                      n_x=n_x, s_x=s_x, s_t=s_t,
                                                                      hetero_strenth=hetero_strenth,
                                                                      n_hetero_vars=n_hetero_vars,
                                                                      autoreg=autoreg,
                                                                      instance_seed=instance_seed,
                                                                      sample_seed=sample_seed,
                                                                      max_poly_degree=max_poly_degree,
                                                                      high_dim=high_dim,
                                                                      verbose=verbose)
                                            for sample_seed in np.arange(sample_seed, sample_seed + n_samples)))
    return res


def amlexperiment(n_instances, inst_seed, n_samples, sample_seed, n_x, n_hetero_vars, n_units,
                  max_poly_degree, high_dim):
    n_periods = 2
    res = main(n_periods=n_periods, n_units=n_units, n_treatments=1,
            n_x=n_x, s_x=2, s_t=2,
            n_hetero_vars=n_hetero_vars,
            n_instances=n_instances,
            inst_seed=inst_seed,
            n_samples=n_samples,
            sample_seed=sample_seed,
            max_poly_degree=max_poly_degree,
            high_dim=high_dim,
            verbose=0)
    file = f'n_ins_{n_instances}_start_{inst_seed}_n_sam_{n_samples}_start_{sample_seed}_n_hetero_vars_{n_hetero_vars}_n_units_{n_units}_n_x_{n_x}.jbl'
    joblib.dump(res, os.path.join(os.environ['AMLT_OUTPUT_DIR'], file))

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n_x", "--n_x", type=int)
    parser.add_argument("-n_hetero_vars", "--n_hetero_vars", type=int)
    parser.add_argument("-n_units", "--n_units", type=int)
    parser.add_argument("-n_instances", "--n_instances", type=int)
    parser.add_argument("-inst_start_seed", "--inst_start_seed", type=int)
    parser.add_argument("-n_samples", "--n_samples", type=int)
    parser.add_argument("-sample_start_seed", "--sample_start_seed", type=int)
    parser.add_argument("-max_poly_degree", "--max_poly_degree", type=int)
    parser.add_argument("-high_dim", "--high_dim", type=int)
    args = parser.parse_args()
    amlexperiment(args.n_instances, args.inst_start_seed,
                  args.n_samples, args.sample_start_seed,
                  args.n_x, args.n_hetero_vars, args.n_units,
                  args.max_poly_degree, args.high_dim)