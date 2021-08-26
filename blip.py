
import numpy as np
from collections import OrderedDict
from econml.utilities import cross_product

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

class SimpleHeteroBlipSpec(BlipSpec):

    def __init__(self, n_hetero_vars):
        self.n_hetero_vars = n_hetero_vars

    def phi(self, t, X, T, Tt):
        if t >= 0:
            return np.hstack([Tt, cross_product(Tt, X[0].values[:, :self.n_hetero_vars])])
        raise AttributeError("Not valid")

    def phi_names(self, t):
        if t >= 0:
            return ([f't[{x}]' for x in range(self.n_treatments[t])] + 
                    [f't[{x}]*x0[{y}]' for y in list(self.Xcols[t][:self.n_hetero_vars]) for x in range(self.n_treatments[t])])
        raise AttributeError("Not valid")


class SimpleBlipSpec(BlipSpec):

    def phi(self, t, X, T, Tt):
        if t >= 0:
            return np.hstack([Tt])
        raise AttributeError("Not valid")

    def phi_names(self, t):
        if t >= 0:
            return [f't[{x}]' for x in range(self.n_treatments[t])]
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
