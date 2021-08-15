from inspect import Attribute
import numpy as np
import pandas as pd
from sklearn.linear_model import (MultiTaskLasso, MultiTaskLassoCV, Lasso, LassoCV, 
                                  LogisticRegression, LinearRegression)
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from econml.sklearn_extensions.model_selection import GridSearchCVList
from sklearn.model_selection import cross_val_predict, KFold
from econml.grf import CausalForest
from econml.utilities import cross_product, check_inputs
from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from econml.inference import NormalInferenceResults
from dgp import DynamicPanelDGP
from copy import deepcopy

def get_linear_model_reg(X, y):
    est = LassoCV(cv=3).fit(X, y)
    return Lasso(alpha = est.alpha_)

def get_linear_multimodel_reg(X, y):
    est = MultiTaskLassoCV(cv=3).fit(X, y)
    return MultiTaskLasso(alpha = est.alpha_)

def get_poly_model_reg(X, y, *, degree, interaction_only):
    Xtr = Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False,
                             interaction_only=interaction_only)),
                    ('sc', StandardScaler())]).fit_transform(X)
    est = LassoCV(cv=3).fit(Xtr, y)
    return Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False,
                             interaction_only=interaction_only)),
                    ('sc', StandardScaler()),
                    ('ls', Lasso(alpha = est.alpha_))])

def get_poly_multimodel_reg(X, y, *, degree, interaction_only):
    Xtr = Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False,
                             interaction_only=interaction_only)),
                    ('sc', StandardScaler())]).fit_transform(X)
    est = MultiTaskLassoCV(cv=3).fit(Xtr, y)
    return Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False,
                             interaction_only=interaction_only)),
                    ('sc', StandardScaler()),
                    ('ls', MultiTaskLasso(alpha = est.alpha_))])

def get_model_reg(X, y, *, degrees=[1, 2], verbose=0):
    if len(y.shape) == 2 and y.shape[1] == 1:
        y = y.ravel()
    model = GridSearchCVList([Pipeline([('poly', PolynomialFeatures(include_bias=False,
                                                                   interaction_only=True)),
                                        ('sc', StandardScaler()),
                                        ('ls', Lasso())]),
                              RandomForestRegressor(n_estimators=100, min_samples_leaf=20, max_depth=3),
                              lgb.LGBMRegressor(num_leaves=32)],
                             param_grid_list=[{'poly__degree': degrees, 'ls__alpha': np.logspace(-4, 2, 20)},
                                              {'min_weight_fraction_leaf': [.01, .1]},
                                              {'learning_rate': [0.1, 0.3], 'max_depth': [3, 5]}],
                             cv=3,
                             scoring='r2',
                             verbose=verbose)
    return model.fit(X, y).best_estimator_

def get_multimodel_reg(X, y, *, degrees=[1, 2], verbose=0):
    model = GridSearchCVList([Pipeline([('poly', PolynomialFeatures(include_bias=False,
                                                                   interaction_only=True)),
                                        ('sc', StandardScaler()),
                                        ('ls', MultiTaskLasso())]),
                              RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_depth=5)],
                             param_grid_list=[{'poly__degree': degrees, 'ls__alpha': np.logspace(-4, 2, 20)},
                                              {'min_weight_fraction_leaf': [.01, .1]}],
                             cv=3,
                             scoring='r2',
                             verbose=verbose)
    return model.fit(X, y).best_estimator_

##############################################
# Linear Structural Nested Mean Models
##############################################

def fit_nuisances(y, X, T, m, phi, pi, get_model_reg, get_multimodel_reg, multitask=True, verbose=1):
    yres = {}
    Qres = {}
    for t in range(m):
        if verbose > 0:
            print(f'Info set: {t}')
        It = np.hstack([X[i] for i in range(t + 1)] + [T[i] for i in range(t)])
        model_reg = get_model_reg(It, y)
        if verbose > 1:
            print('model_y:', model_reg)
        yres[t] = y - cross_val_predict(model_reg, It, y).reshape(y.shape)
        if verbose > 0:
            print('r2score_y', 1 - np.mean(yres[t]**2) / np.var(y))
        Qres[t] = {}
        for j in np.arange(t, m):
            if verbose > 0:
                print(f'Treatment: {j}')
            Qjt = phi(j, X, T, T[j]) - (phi(j, X, T, pi(j, X, T)) if j > t else 0)
            if multitask:
                multimodel_reg = get_multimodel_reg(It, Qjt)
                if verbose > 1:
                    print('model_Qjt:', multimodel_reg)
                Qres[t][j] = Qjt - cross_val_predict(multimodel_reg, It, Qjt).reshape(Qjt.shape)
                if verbose > 0:
                    print('r2score_Qjt', 1 - np.mean(Qres[t][j]**2, axis=0) / np.var(Qjt, axis=0))
            else:
                Qres[t][j] = 1.0 * Qjt.copy()
                for k in range(Qjt.shape[1]):
                    model_reg = get_model_reg(It, Qjt[:, k])
                    if verbose > 1:
                        print(f'model_Qjt[{k}]:', model_reg)
                    Qres[t][j][:, k] -= cross_val_predict(model_reg, It, Qjt[:, k]).reshape(Qjt.shape[0])
                    if verbose > 0:
                        print(f'r2score_Qjt[{k}]', 1 - np.mean(Qres[t][j][:, k]**2) / np.var(Qjt[:, k]))
    return yres, Qres

def fit_final(yres, Qres, m, get_model_final=lambda: LinearRegression()):
    psi = {}
    Yadj = {}
    res = {}
    for t in np.arange(0, m)[::-1]:
        Yadj[t] = yres[t].copy()
        Yadj[t] -= np.sum([np.dot(Qres[t][j], psi[j]) for j in np.arange(t + 1, m)], axis=0)
        psi[t] = get_model_final().fit(Qres[t][t], Yadj[t]).coef_
        res[t] = Yadj[t] - np.dot(Qres[t][t], psi[t])
    return Yadj, psi, res

def fit_cov(Qres, psi, res, m):
    nparams = np.sum([Qres[t][t].shape[1] for t in range(m)])
    nsamples = Qres[0][0].shape[0]
    inds = [0] + list(np.cumsum([Qres[t][t].shape[1] for t in range(m)])) + [nparams]
    J = np.zeros((nparams, nparams))
    Sigma = np.zeros((nparams, nparams))

    for t in range(m):
        for j in range(m):
            Stj = (Qres[t][t] * res[t].reshape(-1, 1)).T @ (Qres[j][j] * res[j].reshape(-1, 1))
            Stj /= nsamples
            Sigma[inds[t]:inds[t+1]][:, inds[j]:inds[j+1]] = Stj
            Sigma[inds[j]:inds[j+1]][:, inds[t]:inds[t+1]] = Stj.T
            if j >= t:
                Jtj = Qres[t][t].T @ Qres[t][j] / nsamples
                J[inds[t]:inds[t+1]][:, inds[j]:inds[j+1]] = Jtj

    invJ = np.linalg.pinv(J)
    cov = invJ @ Sigma @ invJ.T
    stderr = np.sqrt(np.diag(cov) / nsamples)
    stderr_dict = {j: stderr[inds[j]:inds[j+1]] for j in range(m)}
    return cov, stderr, invJ, stderr_dict

def fit_policy(y, X, T, Qres, psi, res, invJ, phi, pi, m):
    Q = {j: phi(j, X, T, T[j]) - phi(j, X, T, pi(j, X, T)) for j in range(m)}
    point = y - np.sum([np.dot(Q[j], psi[j]) for j in range(m)], axis=0)
    est = np.mean(point)
    inf = point - est
    mom = np.hstack([(Qres[t][t] * res[t].reshape(-1, 1)) for t in range(m)])
    inf += mom @ invJ.T @ np.mean(np.hstack([Q[j] for j in range(m)]), axis=0)
    var = np.mean(inf**2)
    return est, var, np.sqrt(var / y.shape[0])

def fit_policy_delta_simple(X, T, Qres, psi, res, invJ, phi, pi, pi_base, m):
    Q = {j: phi(j, X, T, pi_base(j, X, T)) - phi(j, X, T, pi(j, X, T)) for j in range(m)}
    point = - np.sum([np.dot(Q[j], psi[j]) for j in range(m)], axis=0)
    est = np.mean(point)
    inf = point - est
    mom = np.hstack([(Qres[t][t] * res[t].reshape(-1, 1)) for t in range(m)])
    inf += mom @ invJ.T @ np.mean(np.hstack([Q[j] for j in range(m)]), axis=0)
    var = np.mean(inf**2)
    return est, var, np.sqrt(var / X[0].shape[0])

def fit_policy_delta(X, T, phi,
                     Qres, psi, res, invJ, pi, 
                     Qres_base, psi_base, res_base, invJ_base, pi_base, m):
    Q_base = {j: phi(j, X, T, pi_base(j, X, T)) for j in range(m)}
    Q_target = {j: phi(j, X, T, pi(j, X, T)) for j in range(m)}
    point = np.sum([np.dot(Q_target[j], psi[j]) for j in range(m)], axis=0)
    point -= np.sum([np.dot(Q_base[j], psi_base[j]) for j in range(m)], axis=0)
    est = np.mean(point)
    inf = point - est
    mom_base = np.hstack([(Qres_base[t][t] * res_base[t].reshape(-1, 1)) for t in range(m)])
    inf += mom_base @ invJ_base.T @ np.mean(np.hstack([Q_base[j] for j in range(m)]), axis=0)
    mom_target = np.hstack([(Qres[t][t] * res[t].reshape(-1, 1)) for t in range(m)])
    inf -= mom_target @ invJ.T @ np.mean(np.hstack([Q_target[j] for j in range(m)]), axis=0)
    var = np.mean(inf**2)
    return est, var, np.sqrt(var / X[0].shape[0])


##############################################
# Linear SNMMs with Heterogeneous Parameters
##############################################

class LinearModelFinal:

    def __init__(self, linear_model, phi):
        self.linear_model = clone(linear_model)
        self.phi = phi

    def fit(self, X, T, y):
        y, T, X, _ = check_inputs(y, T, X, W=None, multi_output_T=True, multi_output_Y=True)
        self.d_t = T.shape[1]
        X = np.hstack([np.ones((X.shape[0], 1)), self.phi(X)])
        self.linear_model.fit(cross_product(T, X), y)
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=False, ensure_min_features=0)
        X = np.hstack([np.ones((X.shape[0], 1)), self.phi(X)])
        output = np.zeros((X.shape[0], self.d_t))
        for i in range(self.d_t):
            indicator = np.zeros((X.shape[0], self.d_t))
            indicator[:, i] = 1
            output[:, i] = self.linear_model.predict(cross_product(indicator, X))
        return output


def fit_heterogeneous_final(yres, Qres, X, m, get_model_final=lambda: CausalForest()):
    psi = {}
    Yadj = {}
    res = {}
    models = {}
    for t in np.arange(0, m)[::-1]:
        Yadj[t] = yres[t].copy()
        Yadj[t] -= np.sum([np.sum(Qres[t][j] * models[j].predict(X['het']), axis=1)
                           for j in np.arange(t + 1, m)], axis=0)
        models[t] = get_model_final().fit(X['het'], Qres[t][t], Yadj[t])
        res[t] = Yadj[t] - np.sum(Qres[t][t] * models[t].predict(X['het']), axis=1)
    return Yadj, models, res

def fit_policy_heterogeneous(y, X, T, models, phi, pi, m):
    Q = {j: phi(j, X, T, T[j]) - phi(j, X, T, pi(j, X, T)) for j in range(m)}
    point = y - np.sum([np.sum(Q[j] * models[j].predict(X['het']), axis=1) for j in range(m)], axis=0)
    est = np.mean(point)
    inf = point - est
    var = np.mean(inf**2) / X[0].shape[0]
    if hasattr(models[0], 'predict_projection_var'):
        meanQ = {j: np.mean(Q[j], axis=0).reshape(1, -1) for j in range(m)}
        var += np.sum([np.mean(models[j].predict_projection_var(X['het'],
                                                                projector=np.tile(meanQ[j], (X['het'].shape[0], 1))),
                            axis=0)
                    for j in range(m)])
    return est, var, np.sqrt(var)

def fit_policy_delta_simple_heterogeneous(X, T, models, phi, pi, pi_base, m):
    Q = {j: phi(j, X, T, pi_base(j, X, T)) - phi(j, X, T, pi(j, X, T)) for j in range(m)}
    point = - np.sum([np.sum(Q[j] * models[j].predict(X['het']), axis=1) for j in range(m)], axis=0)
    est = np.mean(point)
    inf = point - est
    var = np.mean(inf**2) / X[0].shape[0]
    if hasattr(models[0], 'predict_projection_var'):
        meanQ = {j: np.mean(Q[j], axis=0).reshape(1, -1) for j in range(m)}
        var += np.sum([np.mean(models[j].predict_projection_var(X['het'],
                                                                projector=np.tile(meanQ[j], (X['het'].shape[0], 1))),
                            axis=0)
                    for j in range(m)])
    return est, var, np.sqrt(var)


def fit_policy_delta_heterogeneous(X, T, phi,
                                   models, pi, 
                                   models_base, pi_base, m):
    Q_base = {j: phi(j, X, T, pi_base(j, X, T)) for j in range(m)}
    Q_target = {j: phi(j, X, T, pi(j, X, T)) for j in range(m)}
    point = np.sum([np.sum(Q_target[j] * models[j].predict(X['het']), axis=1) for j in range(m)], axis=0)
    point -= np.sum([np.sum(Q_base[j] * models_base[j].predict(X['het']), axis=1) for j in range(m)], axis=0)
    est = np.mean(point)
    inf = point - est
    var = np.mean(inf**2) / X[0].shape[0]
    if hasattr(models[0], 'predict_projection_var'):
        meanQ = {j: np.mean(Q_target[j], axis=0).reshape(1, -1) for j in range(m)}
        var += np.sum([np.mean(models[j].predict_projection_var(X['het'],
                                                                projector=np.tile(meanQ[j], (X['het'].shape[0], 1))),
                            axis=0)
                    for j in range(m)])
    if hasattr(models_base[0], 'predict_projection_var'):
        meanQ = {j: np.mean(Q_base[j], axis=0).reshape(1, -1) for j in range(m)}
        var += np.sum([np.mean(models_base[j].predict_projection_var(X['het'],
                                                                projector=np.tile(meanQ[j], (X['het'].shape[0], 1))),
                            axis=0)
                    for j in range(m)])
    return est, var, np.sqrt(var)


##############################################
# Linear SNMMs: Optimal Dynamic Regime
##############################################

def pi_opt(j, X, T, phi, psi):
    out0 = np.dot(phi(j, X, T, np.zeros(T[j].shape)), psi[j])
    out1 = np.dot(phi(j, X, T, np.ones(T[j].shape)), psi[j])
    return np.argmax(np.hstack([out0.reshape(-1, 1), out1.reshape(-1, 1)]), axis=1).reshape(T[j].shape)

def fit_opt(y, X, T, m, phi,
            get_model_reg, get_multimodel_reg, multitask=True,
            get_model_final=lambda: LinearRegression(),
            verbose=1):
    yres = {}
    Qres = {}
    psi = {}
    Yadj = {}
    res = {}
    for t in np.arange(0, m)[::-1]:
        if verbose > 0:
            print(f'Info set: {t}')
        It = np.hstack([X[i] for i in range(t + 1)] + [T[i] for i in range(t)])
        model_reg = get_model_reg(It, y)
        if verbose > 1:
            print('model_y:', model_reg)
        yres[t] = y - cross_val_predict(model_reg, It, y).reshape(y.shape)
        if verbose > 0:
            print('r2score_y', 1 - np.mean(yres[t]**2) / np.var(y))
        Qres[t] = {}
        for j in np.arange(t, m):
            if verbose > 0:
                print(f'Treatment: {j}')
            Qjt = phi(j, X, T, T[j]) - (phi(j, X, T, pi_opt(j, X, T, phi, psi)) if j > t else 0)
            if multitask:
                multimodel_reg = get_multimodel_reg(It, Qjt)
                if verbose > 1:
                    print('model_Qjt:', multimodel_reg)
                Qres[t][j] = Qjt - cross_val_predict(multimodel_reg, It, Qjt).reshape(Qjt.shape)
                if verbose > 0:
                    print('r2score_Qjt', 1 - np.mean(Qres[t][j]**2, axis=0) / np.var(Qjt, axis=0))
            else:
                Qres[t][j] = 1.0 * Qjt.copy()
                for k in range(Qjt.shape[1]):
                    model_reg = get_model_reg(It, Qjt[:, k])
                    if verbose > 1:
                        print(f'model_Qjt[{k}]:', model_reg)
                    Qres[t][j][:, k] -= cross_val_predict(model_reg, It, Qjt[:, k]).reshape(Qjt.shape[0])
                    if verbose > 0:
                        print(f'r2score_Qjt[{k}]', 1 - np.mean(Qres[t][j][:, k]**2) / np.var(Qjt[:, k]))

        Yadj[t] = yres[t].copy()
        Yadj[t] -= np.sum([np.dot(Qres[t][j], psi[j]) for j in np.arange(t + 1, m)], axis=0)
        psi[t] = get_model_final().fit(Qres[t][t], Yadj[t]).coef_
        res[t] = Yadj[t] - np.dot(Qres[t][t], psi[t])
    return Yadj, psi, res, yres, Qres


######################################################################
# Linear SNMMs: Optimal Dynamic Regime with Heterogeneous Parameters
######################################################################

def pi_opt_heterogeneous(j, X, T, phi, models):
    psi_het = models[j].predict(X['het'])
    out0 = np.sum(phi(j, X, T, np.zeros(T[j].shape)) * psi_het, axis=1)
    out1 = np.sum(phi(j, X, T, np.ones(T[j].shape)) * psi_het, axis=1)
    return np.argmax(np.hstack([out0.reshape(-1, 1), out1.reshape(-1, 1)]), axis=1).reshape(T[j].shape)

def fit_opt_heterogeneous(y, X, T, m, phi,
                          get_model_reg, get_multimodel_reg, multitask=True,
                          get_model_final=lambda: CausalForest(),
                          verbose=1):
    yres = {}
    Qres = {}
    models = {}
    Yadj = {}
    res = {}
    for t in np.arange(0, m)[::-1]:
        if verbose > 0:
            print(f'Info set: {t}')
        It = np.hstack([X[i] for i in range(t + 1)] + [T[i] for i in range(t)])
        model_reg = get_model_reg(It, y)
        if verbose > 1:
            print('model_y:', model_reg)
        yres[t] = y - cross_val_predict(model_reg, It, y).reshape(y.shape)
        Qres[t] = {}
        for j in np.arange(t, m):
            if verbose > 0:
                print(f'Treatment: {j}')
            Qjt = phi(j, X, T, T[j]) 
            Qjt -= (phi(j, X, T, pi_opt_heterogeneous(j, X, T, phi, models)) if j > t else 0)
            if multitask:
                multimodel_reg = get_multimodel_reg(It, Qjt)
                if verbose > 1:
                    print('model_Qjt:', multimodel_reg)
                Qres[t][j] = Qjt - cross_val_predict(multimodel_reg, It, Qjt).reshape(Qjt.shape)
            else:
                Qres[t][j] = 1.0 * Qjt.copy()
                for k in range(Qjt.shape[1]):
                    model_reg = get_model_reg(It, Qjt[:, k])
                    if verbose > 1:
                        print(f'model_Qjt[{k}]:', model_reg)
                    Qres[t][j][:, k] -= cross_val_predict(model_reg, It, Qjt[:, k]).reshape(Qjt.shape[0])
                    if verbose > 0:
                        print(f'r2score_Qjt[{k}]', 1 - np.mean(Qres[t][j][:, k]**2) / np.var(Qjt[:, k]))

        Yadj[t] = yres[t].copy()
        Yadj[t] -= np.sum([np.sum(Qres[t][j] * models[j].predict(X['het']), axis=1)
                           for j in np.arange(t + 1, m)], axis=0)
        models[t] = get_model_final().fit(X['het'], Qres[t][t], Yadj[t])
        res[t] = Yadj[t] - np.sum(Qres[t][t] * models[t].predict(X['het']), axis=1)
    return Yadj, models, res, yres, Qres


#############################################
# Estimators
#############################################

def pi_base(j, X, T):
    return np.zeros(T[j].shape)

class SNMMDynamicDML:

    def __init__(self, *, m, phi, phi_names_fn, model_reg_fn,
                 multimodel_reg_fn=None, multitask=False,
                 model_final_fn,
                 verbose=1):
        self.m = m
        self.phi = phi
        self.phi_names_fn = phi_names_fn
        self.model_reg_fn = model_reg_fn
        self.multimodel_reg_fn = multimodel_reg_fn
        self.multitask = multitask
        self.model_final_fn = model_final_fn
        self.verbose = verbose
    
    def fit_nuisances(self, X, T, y, pi):
        self.pi_ = pi
        self.X_ = deepcopy(X)
        self.T_ = deepcopy(T)
        self.y_ = deepcopy(y)
        self.yres_, self.Qres_ = fit_nuisances(y, X, T, self.m, self.phi, pi,
                                        self.model_reg_fn, self.multimodel_reg_fn,
                                        multitask=self.multitask, verbose=self.verbose)
        return self

    def fit_final(self):
        self.Yadj_, self.psi_, self.res_ = fit_final(self.yres_, self.Qres_, self.m, self.model_final_fn)
        self.cov_, self.stderr_, self.invJ_, self.stderr_dict_ = fit_cov(self.Qres_, self.psi_, self.res_, self.m)
        self.piest_, self.pivar_, self.pistderr_ = fit_policy(self.y_, self.X_, self.T_,
                                                              self.Qres_, self.psi_, self.res_,
                                                              self.invJ_, self.phi, self.pi_, self.m)

        delta_res = fit_policy_delta_simple(self.X_, self.T_, self.Qres_, self.psi_, self.res_,
                                            self.invJ_, self.phi, self.pi_, pi_base, self.m)
        self.delta_piest_, self.delta_pivar_, self.delta_pistderr_ = delta_res

        return self

    def fit(self, X, T, y, pi):
        return self.fit_nuisances(X, T, y, pi).fit_final()

    def param_summary(self, t, *, coef_thr=-np.inf):
        sig = np.abs(self.psi_[t]) > coef_thr
        return NormalInferenceResults(1, 1, self.psi_[t][sig],
                                      self.stderr_dict_[t][sig],
                                      None, 'coefficient',
                                      feature_names=np.array(self.phi_names_fn(t))[sig])
    
    @property
    def policy_value_(self):
        return self.piest_, self.pistderr_
    
    @property
    def policy_delta_simple_(self):
        return self.delta_piest_, self.delta_pistderr_

    def fit_base(self):
        self.est_base_ = SNMMDynamicDML(m=self.m, phi=self.phi, phi_names_fn=self.phi_names_fn,
                                        model_reg_fn=self.model_reg_fn,
                                        multimodel_reg_fn=self.multimodel_reg_fn, multitask=self.multitask,
                                        model_final_fn=self.model_final_fn,
                                        verbose=self.verbose)
        self.est_base_.fit(self.X_, self.T_, self.y_, pi_base)
        return self

    def policy_delta_complex(self):
        if not hasattr(self, 'est_base_'):
            raise AttributeError('Call `self.fit_base()` before calling this method.')
        piest, _, pistderr = fit_policy_delta(self.X_, self.T_, self.phi,
                                              self.Qres_, self.psi_, self.res_, self.invJ_, self.pi_, 
                                              self.est_base_.Qres_, self.est_base_.psi_, self.est_base_.res_,
                                              self.est_base_.invJ_, self.est_base_.pi_, self.m)
        return piest, pistderr

    def pi_star(self, j, X, T):
        return pi_opt(j, X, T, self.phi, self.psi_opt_)

    def fit_opt(self, X, T, y):
        self.opt_X_ = deepcopy(X)
        self.opt_T_ = deepcopy(T)
        self.opt_y_ = deepcopy(y)
        _, psi_opt, res_opt, _, Qres_opt = fit_opt(y, X, T, self.m, self.phi,
                                           self.model_reg_fn, self.multimodel_reg_fn, multitask=self.multitask,
                                           get_model_final=self.model_final_fn,
                                           verbose=self.verbose)
        cov_opt, _, invJ_opt, stderr_dict_opt = fit_cov(Qres_opt, psi_opt, res_opt, self.m)
        self.psi_opt_, self.res_opt_, self.Qres_opt_ = psi_opt, res_opt, Qres_opt
        self.cov_opt_, self.invJ_opt_, self.stderr_opt_ = cov_opt, invJ_opt, stderr_dict_opt

        self.opt_piest_, _, self.opt_pistderr_ = fit_policy(y, X, T,
                                                            Qres_opt, psi_opt, res_opt, invJ_opt,
                                                            self.phi, self.pi_star, self.m)
        self.delta_opt_piest_, _, self.delta_opt_pistderr_ = fit_policy_delta_simple(X, T,
                                                                                     Qres_opt, psi_opt, res_opt, invJ_opt,
                                                                                     self.phi, self.pi_star, pi_base,
                                                                                     self.m)
        return self
    
    def opt_param_summary(self, t, *, coef_thr=-np.inf):
        sig = np.abs(self.psi_opt_[t]) > coef_thr
        return NormalInferenceResults(1, 1, self.psi_opt_[t][sig],
                                      self.stderr_opt_[t][sig],
                                      None, 'coefficient',
                                      feature_names=np.array(self.phi_names_fn(t))[sig])

    @property
    def opt_policy_value_(self):
        return self.opt_piest_, self.opt_pistderr_
    
    @property
    def opt_policy_delta_simple_(self):
        return self.delta_opt_piest_, self.delta_opt_pistderr_

    def opt_policy_delta_complex(self):
        if not hasattr(self, 'est_base_'):
            raise AttributeError('Call `self.fit_base()` before calling this method.')
        piest, _, pistderr = fit_policy_delta(self.opt_X_, self.opt_T_, self.phi,
                                              self.Qres_opt_, self.psi_opt_, self.res_opt_, self.invJ_opt_, self.pi_star, 
                                              self.est_base_.Qres_, self.est_base_.psi_, self.est_base_.res_,
                                              self.est_base_.invJ_, self.est_base_.pi_, self.m)
        return piest, pistderr



class HeteroSNMMDynamicDML(SNMMDynamicDML):

    def fit_final(self):
        self.Yadj_, self.models_, self.res_ = fit_heterogeneous_final(self.yres_, self.Qres_, self.X_,
                                                                      self.m, self.model_final_fn)
        self.piest_, _, self.pistderr_ = fit_policy_heterogeneous(self.y_, self.X_, self.T_, self.models_,
                                                                  self.phi, self.pi_, self.m)
        self.delta_piest_, _, self.delta_pistderr_ = fit_policy_delta_simple_heterogeneous(self.X_, self.T_,
                                                                                           self.models_, self.phi,
                                                                                           self.pi_, pi_base, self.m)
        return self

    def param_summary(self, t, *, coef_thr=-np.inf):
        if not hasattr(self.models_[0], 'linear_model'):
            raise AttributeError("Method available only when final model is `LinearModelFinal`")
        
        xnames = ['1'] + list(self.X_['het'].columns)
        feat_names = np.array([f'{x}*{y}' for y in xnames for x in self.phi_names_fn(t)])
        stderr = np.zeros(self.models_[t].linear_model.coef_.shape)
        if hasattr(self.models_[t].linear_model, 'coef_stderr_'):
            stderr = self.models_[t].linear_model.coef_stderr_
        sig = np.abs(self.models_[t].linear_model.coef_) >= coef_thr
        return NormalInferenceResults(1, 1,
                                      self.models_[t].linear_model.coef_[sig],
                                      stderr[sig],
                                      None, 'coefficient',
                                      feature_names=feat_names[sig])

    def feature_importances_(self, t):
        return pd.DataFrame({'name': self.X_['het'].columns, 
                            'importance': self.models_[t].feature_importances_})

    @property
    def policy_value_(self):
        return self.piest_, self.pistderr_

    @property
    def policy_delta_simple_(self):
        return self.delta_piest_, self.delta_pistderr_

    def fit_base(self):
        self.est_base_ = HeteroSNMMDynamicDML(m=self.m, phi=self.phi, phi_names_fn=self.phi_names_fn,
                                        model_reg_fn=self.model_reg_fn,
                                        multimodel_reg_fn=self.multimodel_reg_fn, multitask=self.multitask,
                                        model_final_fn=self.model_final_fn)
        self.est_base_.fit(self.X_, self.T_, self.y_, pi_base)
        return self

    def policy_delta_complex(self):
        if not hasattr(self, 'est_base_'):
            raise AttributeError('Call `self.fit_base()` before calling this method.')
            
        piest, _, pistderr = fit_policy_delta_heterogeneous(self.X_, self.T_, self.phi,
                                                            self.models_, self.pi_,
                                                            self.est_base_.models_, self.est_base_.pi_,
                                                            self.m)
        return piest, pistderr

    def pi_star(self, j, X, T):
        return pi_opt_heterogeneous(j, X, T, self.phi, self.models_)

    def fit_opt(self, X, T, y):
        self.opt_X_ = deepcopy(X)
        self.opt_T_ = deepcopy(T)
        self.opt_y_ = deepcopy(y)
        _, self.opt_models_, _, _, _ = fit_opt_heterogeneous(y, X, T, self.m, self.phi,
                                                             self.model_reg_fn, self.multimodel_reg_fn,
                                                             multitask=self.multitask,
                                                             get_model_final=self.model_final_fn,
                                                             verbose=self.verbose)

        self.opt_piest_, _, self.opt_pistderr_ = fit_policy_heterogeneous(y, X, T, self.opt_models_,
                                                                          self.phi, self.pi_star, self.m)
        self.delta_opt_piest_, _, self.delta_opt_pistderr_ = fit_policy_delta_simple_heterogeneous(self.X_, self.T_,
                                                                                           self.opt_models_, self.phi,
                                                                                           self.pi_star, pi_base, self.m)
        return self

    def opt_param_summary(self, t, *, coef_thr=-np.inf):
        if not hasattr(self.opt_models_[0], 'linear_model'):
            raise AttributeError("Method available only when final model is `LinearModelFinal`")
        
        xnames = ['1'] + list(self.X_['het'].columns)
        feat_names = np.array([f'{x}*{y}' for y in xnames for x in self.phi_names_fn(t)])
        stderr = np.zeros(self.opt_models_[t].linear_model.coef_.shape)
        if hasattr(self.opt_models_[t].linear_model, 'coef_stderr_'):
            stderr = self.opt_models_[t].linear_model.coef_stderr_
        sig = np.abs(self.opt_models_[t].linear_model.coef_) >= coef_thr
        return NormalInferenceResults(1, 1,
                                      self.opt_models_[t].linear_model.coef_[sig],
                                      stderr[sig],
                                      None, 'coefficient',
                                      feature_names=feat_names[sig])
    
    def opt_feature_importances_(self, t):
        return pd.DataFrame({'name': self.X_['het'].columns, 
                            'importance': self.opt_models_[t].feature_importances_})

    @property
    def opt_policy_value_(self):
        return self.opt_piest_, self.opt_pistderr_
    
    @property
    def opt_policy_delta_simple_(self):
        return self.delta_opt_piest_, self.delta_opt_pistderr_

    def opt_policy_delta_complex(self):
        if not hasattr(self, 'est_base_'):
            raise AttributeError('Call `self.fit_base()` before calling this method.')
        piest, _, pistderr = fit_policy_delta_heterogeneous(self.opt_X_, self.opt_T_, self.phi,
                                                            self.opt_models_, self.pi_star,
                                                            self.est_base_.models_, self.est_base_.pi_, self.m)
        return piest, pistderr


###########################
# Data generation
###########################

def gen_data(*, n_periods, n_units, n_treatments, n_x, s_x, s_t,
             hetero_strenth=0, n_hetero_vars=0, autoreg=1.0,
             instance_seed=None, sample_seed=None):
    hetero_inds = np.arange(n_x - n_hetero_vars, n_x) if n_hetero_vars > 0 else None
    dgp = DynamicPanelDGP(n_periods, n_treatments, n_x)
    dgp.create_instance(s_x, hetero_strength=hetero_strenth, hetero_inds=hetero_inds,
                        random_seed=instance_seed,
                        autoreg=autoreg)
    Y, T, X, W, groups = dgp.observational_data(n_units, s_t=s_t, random_seed=sample_seed)

    if n_hetero_vars > 0:
        true_effect_inds = []
        for t in range(n_treatments):
            true_effect_inds += [t * (1 + n_x)] + (list(t * (1 + n_x) + 1 + hetero_inds)
                                                if len(hetero_inds)>0 else [])
        true_effect_params = dgp.true_hetero_effect[:, true_effect_inds]
        true_effect_params = true_effect_params.reshape((n_treatments*n_periods, 1 + hetero_inds.shape[0]))
    else:
        true_effect_params = dgp.true_effect
    true_effect_params = true_effect_params.reshape(n_periods, n_treatments, n_hetero_vars + 1)

    Y = Y.reshape((-1, n_periods))
    T = T.reshape((-1, n_periods, T.shape[1]))
    if X is not None:
        X = X.reshape((-1, n_periods, X.shape[1]))
    W = W.reshape((-1, n_periods, W.shape[1]))

    if X is not None:
        x0_cols = [f'x{i}' for i in range(X.shape[-1])] + [f'w{i}' for i in range(W.shape[-1])]
        x1_cols = [f'x{i}' for i in range(X.shape[-1])] + [f'w{i}' for i in range(W.shape[-1])]

        y = Y[:, -1]
        X = {0: pd.DataFrame(np.hstack([X[:, 0, :], W[:, 0, :]]), columns=x0_cols),
            1: pd.DataFrame(np.hstack([X[:, 1, :], W[:, 1, :]]), columns=x1_cols),
            'het': pd.DataFrame(np.hstack([X[:, 0, :]]), columns=[f'x{i}' for i in range(X.shape[-1])])}
    else:
        x0_cols = [f'w{i}' for i in range(W.shape[-1])]
        x1_cols = [f'w{i}' for i in range(W.shape[-1])]
        X = {0: pd.DataFrame(W[:, 0, :], columns=x0_cols),
             1: pd.DataFrame(W[:, 1, :], columns=x1_cols)}
    
    T = {0: T[:, 0, :], 1: T[:, 1, :]}
    y = Y[:, -1]
    return y, X, T, true_effect_params

