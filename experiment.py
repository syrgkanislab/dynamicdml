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


def experiment(*, n_periods, n_units, n_treatments,
                n_x, s_x, s_t,
                hetero_strenth=.5, n_hetero_vars=0, autoreg=1.0,
                instance_seed=None, sample_seed=None, verbose=0):
    m = n_periods
    y, X, T, true_effect_params = gen_data(n_periods=m, n_units=n_units,
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

    het_cols11 = list(X[1].columns)
    het_cols10 = list(X[0].columns)
    het_cols0 = list(X[0].columns)
    def phi(t, X, T, Tt):
        if t == 1:
            return np.hstack([Tt,
                            cross_product(Tt, T[t-1]),
                            cross_product(Tt, X[t][het_cols11].values),
                            cross_product(Tt, X[t-1][het_cols10].values),
                            cross_product(Tt, T[t-1], X[t][het_cols11].values),
                            cross_product(Tt, T[t-1], X[t-1][het_cols10].values)
                            ])
        elif t==0:
            return np.hstack([Tt, cross_product(Tt, X[t][het_cols0].values)])
        raise AttributeError("Not valid")

    def phi_names(t):
        if t == 1:
            return ([f't2[{x}]' for x in range(T[1].shape[1])] +
                    [f't2[{x}]*t1[{y}]' for y in range(T[0].shape[1]) for x in range(T[1].shape[1])] +
                    [f't2[{x}]*x2[{y}]' for y in het_cols11 for x in range(T[1].shape[1])] + 
                    [f't2[{x}]*x1[{y}]' for y in het_cols10 for x in range(T[1].shape[1])] +
                    [f't2[{x}]*t1[{y}]*x2[{z}]' for z in het_cols11 for y in range(T[0].shape[1])
                                            for x in range(T[1].shape[1])] + 
                    [f't2[{x}]*t1[{y}]*x1[{z}]' for z in het_cols10 for y in range(T[0].shape[1])
                                            for x in range(T[1].shape[1])]
                )
        elif t == 0:
            return ([f't1[{x}]' for x in range(T[1].shape[1])] + 
                    [f't1[{x}]*x1[{y}]' for y in het_cols0 for x in range(T[1].shape[1])])
        raise AttributeError("Not valid")

    def pi(t, X, T):
        return np.ones(T[t].shape)

    true_params = {}
    true_params_sel = {}
    true_policy = 0
    true_policy_delta = 0
    true_opt_policy = 0
    true_opt_policy_delta = 0
    for t in range(m):
        true_params[t] = OrderedDict()
        for name in phi_names(t):
            true_params[t][name] = 0
        for i in range(T[t].shape[1]):
            true_params[t][f't{t+1}[{i}]'] = true_effect_params[t][i][0]
            for j in range(n_hetero_vars):
                true_params[t][f't{t+1}[{i}]*x1[x{j}]'] = true_effect_params[t][i][1 + j]
        true_params_sel[t] = np.array([v for _, v in true_params[t].items()])
        true1 = np.mean(np.sum(true_params_sel[t] * phi(t, X, T, np.ones(T[t].shape)), axis=1))
        true0 = np.mean(np.sum(true_params_sel[t] * phi(t, X, T, np.zeros(T[t].shape)), axis=1))
        delta = true1 - true0
        true_policy += true1
        true_policy_delta += delta
        true_opt_policy += true1 * (delta >= 0) + true0 * (delta < 0)
        true_opt_policy_delta += delta * (delta >= 0)

    # model_reg_fn = lambda X, y: get_model_reg(X, y, degrees=[1])
    # multimodel_reg_fn = lambda X, y: get_multimodel_reg(X, y, degrees=[1])
    # model_reg_fn = get_linear_model_reg
    # multimodel_reg_fn = get_linear_multimodel_reg
    model_reg_fn = lambda X, y: get_poly_model_reg(X, y, degree=1, interaction_only=True)
    multimodel_reg_fn = lambda X, y: get_poly_multimodel_reg(X, y, degree=1, interaction_only=True)

    est = SNMMDynamicDML(m=m, phi=phi, phi_names_fn=phi_names,
                        model_reg_fn=model_reg_fn,
                        model_final_fn=lambda: LassoCV(),
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
                         model_reg_fn=model_reg_fn, #lambda X, y: get_model_reg(X, y, degrees=[1], verbose=verbose-2),
                         model_final_fn=lambda: LinearRegression(),
                         verbose=verbose)
    est_sub.fit(X, T, y, pi)
    est_sub.fit_opt(X, T, y)

    ##################################
    # Oracle low dim blip
    ##################################
    oracle_sig = {}
    for t in range(m):
        oracle_sig[t] = np.abs(true_params_sel[t]) > 0.0

    def phi_oracle(t, X, T, Tt):
        return phi(t, X, T, Tt)[:, oracle_sig[t]]

    def phi_names_oracle(t):
        return np.array(phi_names(t))[oracle_sig[t]]

    est_low = SNMMDynamicDML(m=m, phi=phi_oracle, phi_names_fn=phi_names_oracle,
                         model_reg_fn=model_reg_fn, #lambda X, y: get_model_reg(X, y, degrees=[1], verbose=verbose-2),
                         model_final_fn=lambda: LinearRegression(),
                         verbose=verbose)
    est_low.fit(X, T, y, pi)
    est_low.fit_opt(X, T, y)

    results = {}
    results['true'] = {'params': true_params,
                       'pival': true_policy,
                       'pidelta': true_policy_delta,
                       'optval': true_opt_policy,
                       'optdelta': true_opt_policy_delta}
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
         n_x, s_x, s_t, n_instances, n_samples,
         hetero_strenth=.5, n_hetero_vars=0, autoreg=1.0, verbose=0):

    res = []
    for instance_seed in range(n_instances):
        res.append(Parallel(n_jobs=-1, verbose=1)(delayed(experiment)(n_periods=n_periods,
                                                                      n_units=n_units,
                                                                      n_treatments=n_treatments,
                                                                      n_x=n_x, s_x=s_x, s_t=s_t,
                                                                      hetero_strenth=hetero_strenth,
                                                                      n_hetero_vars=n_hetero_vars,
                                                                      autoreg=autoreg,
                                                                      instance_seed=instance_seed,
                                                                      sample_seed=sample_seed,
                                                                      verbose=verbose)
                                            for sample_seed in range(n_samples)))
    return res

def all_experiments():
    n_instances = 10
    n_samples = 100
    n_periods = 2
    for n_x in [20, 50, 100]:
        for n_hetero_vars in [0, 1, 2]:
            for n_units in [1000, 10000]:
                res = main(n_periods=n_periods, n_units=n_units, n_treatments=1,
                        n_x=n_x, s_x=2, s_t=2,
                        n_hetero_vars=n_hetero_vars,
                        n_instances=n_instances,
                        n_samples=n_samples)
                file = f'n_ins_{n_instances}_n_sam_{n_samples}_n_hetero_vars_{n_hetero_vars}_n_units_{n_units}_n_x_{n_x}.jbl'
                joblib.dump(res, file)

if __name__=="__main__":
    all_experiments()