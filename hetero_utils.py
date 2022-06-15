from sklearn.base import clone
import numpy as np
from sklearn.utils import check_array
from econml.utilities import cross_product, check_inputs
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed
from scipy.special import softmax

class LinearModelFinal:

    def __init__(self, linear_model, phi, fit_cate_intercept=True):
        self.linear_model = clone(linear_model)
        self.phi = phi
        self.fit_cate_intercept = fit_cate_intercept

    def fit(self, X, T, y):
        y, T, X, _ = check_inputs(y, T, X, W=None, multi_output_T=True, multi_output_Y=True)
        self.d_t = T.shape[1]
        if self.fit_cate_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), self.phi(X)])
        else:
            X = self.phi(X)
        self.linear_model.fit(cross_product(T, X), y)
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=False, ensure_min_features=0)
        if self.fit_cate_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), self.phi(X)])
        else:
            X = self.phi(X)
        output = np.zeros((X.shape[0], self.d_t))
        for i in range(self.d_t):
            indicator = np.zeros((X.shape[0], self.d_t))
            indicator[:, i] = 1
            output[:, i] = self.linear_model.predict(cross_product(indicator, X))
        return output
    
    @property
    def coef_(self,):
        return self.linear_model.coef_

    def __repr__(self):
        return self.linear_model.__repr__()

class WeightedModelFinal:
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, T, y):
        if len(T.shape) > 1 and T.shape[1] > 1:
            raise AttributeError('Method not available for multiple treatments')
        self.d_t_ = T.shape[1:]
        weights = T.flatten() + 1e-8
        self.model_ = clone(self.model, safe=False).fit(X, y/weights, sample_weight=weights**2)
        return self
    
    def predict(self, X):
        return self.model_.predict(X).reshape((X.shape[0],) + self.d_t_)
    
    @property
    def feature_importances_(self,):
        return self.model_.feature_importances_

    def __repr__(self):
        return self.model.__repr__()


class Wrapper(BaseEstimator):
    
    def __init__(self, *, model, d_t):
        self.model = model
        self.d_t = d_t

    def fit(self, TX, y):
        self.model_ = clone(self.model, safe=False)
        self.model_.fit(TX[:, self.d_t:], TX[:, :self.d_t], y)
        return self

    def predict(self, TX):
        return self.model_.predict(TX[:, self.d_t:]).reshape((-1, self.d_t))

    def score(self, TX, y):
        ypred = np.sum(self.predict(TX) * TX[:, :self.d_t], axis=1)
        return 1 - np.mean((ypred - y.flatten())**2) / np.var(y)

    def __repr__(self):
        return self.model.__repr__()

def fit_model(model, X, T, y):
    return model.fit(X, T, y)

class Ensemble(BaseEstimator):
    
    def __init__(self, *, model_gens, weights):
        self.model_gens = model_gens
        self.weights = weights

    def fit(self, X, T, y):
        self.models_ = Parallel(n_jobs=-1, verbose=0)(delayed(fit_model)(mgen(), X, T, y)
                                                      for _, mgen in self.model_gens)
        return self
    
    def predict(self, X):
        return np.average([mdl.predict(X) for mdl in self.models_], weights=self.weights, axis=0)
    
    def __repr__(self):
        return "\n".join([f'({weight:.3f}, {model.__repr__()})' for weight, model in zip(self.weights, self.models_)])
    
def score_model(model, d_t, TX, y):
    return np.mean(cross_val_score(Wrapper(model=model, d_t=d_t), TX, y, cv=3))

class GCV(BaseEstimator):
    
    def __init__(self, *, model_gens, ensemble, beta=None):
        self.model_gens = model_gens
        self.ensemble = ensemble
        self.beta = beta
    
    def fit(self, X, T, y):
        TX = np.hstack([T, X])
        scores = Parallel(n_jobs=-1, verbose=0)(delayed(score_model)(mgen(), T.shape[1], TX, y)
                                                for _, mgen in self.model_gens)
        self.scores_ = np.array(scores)
        if self.ensemble:
            self.model_ = Ensemble(model_gens=self.model_gens,
                                   weights=softmax(self.beta * self.scores_)).fit(X, T, y)
        else:
            self.model_ = self.model_gens[np.argmax(self.scores_)][1]().fit(X, T, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def __repr__(self):
        return self.model_.__repr__()
