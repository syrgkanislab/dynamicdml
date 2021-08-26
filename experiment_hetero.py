import numpy as np
from snmm import get_linear_model_reg, get_linear_multimodel_reg
from snmm import get_model_reg, get_multimodel_reg
from snmm import get_poly_model_reg, get_poly_multimodel_reg
from econml.utilities import cross_product
from snmm import gen_data
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
import pandas as pd
from joblib import Parallel, delayed
import joblib
from collections import OrderedDict
import os
import argparse
from econml.grf import CausalForest
from sklearn.linear_model import LinearRegression
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn.ensemble import RandomForestRegressor
from hetero_utils import WeightedModelFinal, LinearModelFinal, Ensemble, GCV
from snmm import HeteroSNMMDynamicDML
import copy
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from blip import SimpleBlipSpec, true_param_parse

def pi(t, X, T):
    return np.ones(T[t].shape)

FUNCTIONS = {'abs': np.abs,
             'linear': lambda x: x,
             'square': lambda x: x**2/10,
             'step': lambda x: x >= 0,
             '3dpoly': lambda x: x**3/20 + x**2/20,
             'steplinear': lambda x: (x >= 0) * x}

def get_metrics(X, het_est, dgp, m, n_hetero_vars, store_preds):
    metrics = {}
    for t in range(m):
        metrics[t] = {}
        pred = het_est.dynamic_effects(X['het'])[t][:, 0]
        true = dgp.hetero_effect_fn(t, X['het'].values[:, :n_hetero_vars]).flatten()
        metrics[t]['rmse'] = np.sqrt(np.mean((pred - true)**2))
        metrics[t]['mape'] = np.mean(np.abs(pred - true) / (np.abs(true) + 1e-12))
        metrics[t]['mae'] = np.mean(np.abs(pred - true))
        if store_preds:
            metrics[t]['xhet'] = X['het'].values[:, :n_hetero_vars]
            metrics[t]['true'] = true
            metrics[t]['pred'] = pred
    return metrics

def experiment(*, n_periods, n_units,
                n_x, s_x, s_t,
                hetero_strenth, n_hetero_vars, autoreg,
                instance_seed, sample_seed, verbose,
                max_poly_degree, nonlin_fn, store_preds=False):
    m = n_periods
    n_treatments = 1
    y, X, T, true_effect_params, dgp = gen_data(n_periods=m, n_units=n_units,
                                              n_treatments=n_treatments,
                                              n_x=n_x, s_x=s_x, s_t=s_t,
                                              hetero_strenth=hetero_strenth,
                                              n_hetero_vars=n_hetero_vars,
                                              autoreg=autoreg,
                                              nonlin_fn=FUNCTIONS[nonlin_fn],
                                              instance_seed=instance_seed,
                                              sample_seed=sample_seed)
    X['het'] = X[0].copy()

    results = {}

    oracle_blip = SimpleBlipSpec().fit(X, T)

    cf_gen = lambda ms, md, mvl, fr: lambda: CausalForest(n_estimators=1000,
                                                    max_depth=md,
                                                    min_samples_leaf=ms,
                                                    min_balancedness_tol=0.45,
                                                    max_samples=fr,
                                                    inference=False,
                                                    min_var_fraction_leaf=mvl,
                                                    min_var_leaf_on_val=True,
                                                    random_state=123)
    linear_gen = lambda: LinearModelFinal(StatsModelsLinearRegression(fit_intercept=False),
                                        lambda x: x)
    lasso_gen = lambda: LinearModelFinal(LassoCV(fit_intercept=False, random_state=123),
                                        lambda x: x)
    polylasso_gen = lambda: LinearModelFinal(LassoCV(fit_intercept=False, random_state=123),
                                             lambda x: Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),
                                                                 ('sc', StandardScaler())]).fit_transform(x))
    rf_gen = lambda ms, md: lambda: WeightedModelFinal(RandomForestRegressor(n_estimators=100,
                                                                    min_samples_leaf=ms,
                                                                    max_depth=md,
                                                                    random_state=123))
    lgbm_gen = lambda lr, md: lambda: WeightedModelFinal(lgb.LGBMRegressor(num_leaves=32, min_child_samples=40,
                                                                    learning_rate=lr,
                                                                    max_depth=md,
                                                                    random_state=123))

    model_gens = [(f'cf{it}', cf_gen(ms, md, mvl, fr))
                  for it, (ms, md, mvl, fr) in enumerate([(40, 3, None, .45),
                                                          (40, 3, None, .8),
                                                          (40, 5, None, .8),
                                                          (40, 5, 0.01, .8)])] 
    model_gens += [('ols', linear_gen), ('lassocv', lasso_gen), ('2dlassocv', polylasso_gen)]
    model_gens += [(f'rf{it}', rf_gen(ms, md))
                   for it, (ms, md) in enumerate([(40, 3), (40, 5)])]
    model_gens += [(f'lgbm{it}', lgbm_gen(lr, md))
                   for it, (lr, md) in enumerate([(0.01, 1), (0.1, 1), (0.01, 3), (0.1, 3)])]

    gcv_gen = lambda: GCV(model_gens=model_gens, ensemble=False)

    het_est = HeteroSNMMDynamicDML(m=m, phi=oracle_blip.phi, phi_names_fn=oracle_blip.phi_names,
                                model_reg_fn=lambda X, y: get_model_reg(X, y, degrees=list(np.arange(1, max_poly_degree + 1)), verbose=verbose-2),
                                model_final_fn=gcv_gen,
                                verbose=verbose)
    het_est.fit(X, T, y, pi)

    results['maxcv'] = get_metrics(X, het_est, dgp, n_periods, n_hetero_vars, store_preds)
    for t in range(n_periods):
        results['maxcv'][t]['model'] = het_est.models_[t].model_.__repr__()

    het_est.model_final_fn = lambda: GCV(model_gens=model_gens, ensemble=True, beta=1000)
    het_est.fit_final()

    results['softmaxcv'] = get_metrics(X, het_est, dgp, n_periods, n_hetero_vars, store_preds)

    for name, mgen in model_gens:
        het_est.model_final_fn = mgen
        het_est.fit_final()
        results[name] = get_metrics(X, het_est, dgp, n_periods, n_hetero_vars, store_preds)
    return results


def main(*, n_periods, n_units, n_treatments,
         n_x, s_x, s_t, n_instances, inst_seed, n_samples, sample_seed, max_poly_degree, nonlin_fn,
         hetero_strenth=2.0, n_hetero_vars=1, autoreg=1.0, verbose=0):

    res = []
    for instance_seed in np.arange(inst_seed, inst_seed + n_instances):
        res.append(Parallel(n_jobs=-1, verbose=1)(delayed(experiment)(n_periods=n_periods,
                                                                      n_units=n_units,
                                                                      n_x=n_x, s_x=s_x, s_t=s_t,
                                                                      hetero_strenth=hetero_strenth,
                                                                      n_hetero_vars=n_hetero_vars,
                                                                      autoreg=autoreg,
                                                                      instance_seed=instance_seed,
                                                                      sample_seed=sample_seed,
                                                                      max_poly_degree=max_poly_degree,
                                                                      nonlin_fn=nonlin_fn,
                                                                      verbose=verbose)
                                            for sample_seed in np.arange(sample_seed, sample_seed + n_samples)))
    return res


def amlexperiment(n_instances, inst_seed, n_samples, sample_seed, n_x, n_hetero_vars, n_units,
                  max_poly_degree, nonlin_fn):
    n_periods = 2
    res = main(n_periods=n_periods, n_units=n_units, n_treatments=1,
            n_x=n_x, s_x=2, s_t=2,
            n_hetero_vars=n_hetero_vars,
            n_instances=n_instances,
            inst_seed=inst_seed,
            n_samples=n_samples,
            sample_seed=sample_seed,
            max_poly_degree=max_poly_degree,
            nonlin_fn=nonlin_fn,
            verbose=0)
    file = f'nonlin_fn_{nonlin_fn}_n_ins_{n_instances}_start_{inst_seed}_n_sam_{n_samples}_start_{sample_seed}_n_hetero_vars_{n_hetero_vars}_n_units_{n_units}_n_x_{n_x}.jbl'
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
    parser.add_argument("-nonlin_fn", "--nonlin_fn", type=str)
    args = parser.parse_args()
    amlexperiment(args.n_instances, args.inst_start_seed,
                  args.n_samples, args.sample_start_seed,
                  args.n_x, args.n_hetero_vars, args.n_units,
                  args.max_poly_degree, args.nonlin_fn)