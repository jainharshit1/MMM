#changes have been made in the hill saturation class to include for the xpoint so that the saturation effect can be calculated at one single point instead of the enitre array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np
class HillSaturation(BaseEstimator, TransformerMixin):
    # Smaller Alpha: More C-shaped (quick initial response, fast saturation).
    # Larger Alpha: More S-shaped (slow initial response, rapid increase, gradual saturation).
    # Gamma: Controls the point where the curve starts to bend
    # Smaller Gamma (closer to 0.3): The inflection point occurs earlier, meaning the curve starts to bend and show diminishing returns sooner.
    # Larger Gamma (closer to 1): The inflection point occurs later, meaning the curve stays linear for longer before starting to show diminishing returns.
    # Alpha range : 0.5 and 3.
    # Gamma range : 0.3 and 1.
    def __init__(self, alpha: float = 1.0, gamma: float = 0.5, x_marginal: np.ndarray = None):
        self.alpha = alpha
        self.gamma = gamma
        self.x_marginal = x_marginal

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        return self

    def transform(self, X: np.ndarray, x_point: np.ndarray = None):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        if len(X) == 0:
            raise ValueError("Input array X cannot be empty")

        inflexion = np.dot(np.array([1 - self.gamma, self.gamma]), np.array([np.min(X), np.max(X)]))

        # if self.x_marginal is None:
        #     x_scurve = X**self.alpha / (X**self.alpha + inflexion**self.alpha)
        # else:
        #     x_scurve = self.x_marginal**self.alpha / (self.x_marginal**self.alpha + inflexion**self.alpha)
        if x_point is None:
            if self.x_marginal is None:
                x_scurve = X**self.alpha / (X**self.alpha + inflexion**self.alpha)
            else:
                x_scurve = self.x_marginal**self.alpha / (self.x_marginal**self.alpha + inflexion**self.alpha)
        else:
            x_scurve = x_point**self.alpha / (x_point**self.alpha + inflexion**self.alpha)

        return x_scurve
class ExponentialSaturation(BaseEstimator, TransformerMixin):
    def __init__(self, a=1.):
        self.a = a

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)  # from BaseEstimator
        return self

    def transform(self, X, x_point=None):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)  # from BaseEstimator
        
        if x_point is None:
            return 1 - np.exp(-self.a * X)
        else:
            x_point = np.atleast_2d(x_point)
            return 1 - np.exp(-self.a * x_point)