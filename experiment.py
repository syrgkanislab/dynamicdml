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

#########################
# High dim blip spec
#########################

class BlipSpec:

    def fit(self, X, T):
        self.n_treatments = {}
        for t, treatvec in T.items():
            self.n_treatments[t] = treatvec.shape[1]
        self.Xcols = {}
        for t, Xdf in X.items():
            self.Xcols[t] = list(Xdf.columns)
        return self

    def phi(self, t, X, T, Tt):
        if t >= 1:
            return np.hstack([Tt,
                            cross_product(Tt, T[t-1]),
                            cross_product(Tt, X[t].values),
                            cross_product(Tt, X[0].values),
                            cross_product(Tt, T[t-1], X[t].values),
                            cross_product(Tt, T[t-1], X[0].values)
                            ])
        elif t==0:
            return np.hstack([Tt, cross_product(Tt, X[t].values)])
        raise AttributeError("Not valid")

    def phi_names(self, t):
        if t >= 1:
            return ([f't[{x}]' for x in range(self.n_treatments[t])] +
                    [f't[{x}]*lagt[{y}]' for y in range(self.n_treatments[t-1]) for x in range(self.n_treatments[t])] +
                    [f't[{x}]*x[{y}]' for y in list(self.Xcols[t]) for x in range(self.n_treatments[t])] + 
                    [f't[{x}]*x0[{y}]' for y in list(self.Xcols[0]) for x in range(self.n_treatments[t])] +
                    [f't[{x}]*lagt[{y}]*x[{z}]' for z in list(self.Xcols[t]) for y in range(self.n_treatments[t-1])
                                            for x in range(self.n_treatments[t])] + 
                    [f't[{x}]*lagt[{y}]*x0[{z}]' for z in list(self.Xcols[0]) for y in range(self.n_treatments[t])
                                            for x in range(self.n_treatments[t])]
                )
        elif t == 0:
            return ([f't[{x}]' for x in range(self.n_treatments[t])] + 
                    [f't[{x}]*x0[{y}]' for y in list(self.Xcols[t]) for x in range(self.n_treatments[t])])
        raise AttributeError("Not valid")

def true_param_parse(X, T, true_effect_params, n_hetero_vars, m, phi, phi_names):
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
            true_params[t][f't[{i}]'] = true_effect_params[t][i][0]
            for j in range(n_hetero_vars):
                true_params[t][f't[{i}]*x0[x{j}]'] = true_effect_params[t][i][1 + j]
        true_params_sel[t] = np.array([v for _, v in true_params[t].items()])
        true1 = np.mean(phi(t, X, T, np.ones(T[t].shape)) @ true_params_sel[t])
        true0 = np.mean(phi(t, X, T, np.zeros(T[t].shape)) @ true_params_sel[t])
        delta = true1 - true0
        true_policy += true1
        true_policy_delta += delta
        true_opt_policy += true1 * (delta >= 0) + true0 * (delta < 0)
        true_opt_policy_delta += delta * (delta >= 0)
    return true_params, true_params_sel, true_policy, true_policy_delta, true_opt_policy, true_opt_policy_delta

def pi(t, X, T):
    return np.ones(T[t].shape)


def experiment(*, n_periods, n_units, n_treatments,
                n_x, s_x, s_t,
                hetero_strenth, n_hetero_vars, autoreg,
                instance_seed, sample_seed, verbose):
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
                         model_reg_fn=lambda X, y: get_model_reg(X, y, degrees=[2], verbose=verbose-2),
                         model_final_fn=lambda: LinearRegression(fit_intercept=False),
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
                         model_reg_fn=lambda X, y: get_model_reg(X, y, degrees=[2], verbose=verbose-2),
                         model_final_fn=lambda: LinearRegression(fit_intercept=False),
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
         n_x, s_x, s_t, n_instances, inst_seed, n_samples, sample_seed,
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
                                                                      verbose=verbose)
                                            for sample_seed in np.arange(sample_seed, sample_seed + n_samples)))
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

def amlexperiment(n_instances, inst_seed, n_samples, sample_seed, n_x, n_hetero_vars, n_units):
    n_periods = 2
    res = main(n_periods=n_periods, n_units=n_units, n_treatments=1,
            n_x=n_x, s_x=2, s_t=2,
            n_hetero_vars=n_hetero_vars,
            n_instances=n_instances,
            inst_seed=inst_seed,
            n_samples=n_samples,
            sample_seed=sample_seed,
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
    args = parser.parse_args()
    amlexperiment(args.n_instances, args.inst_start_seed,
                  args.n_samples, args.sample_start_seed,
                  args.n_x, args.n_hetero_vars, args.n_units)