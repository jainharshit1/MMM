from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from scipy.signal import convolve2d
from scipy.stats import weibull_min
import numpy as np
class AdstockGeometric(BaseEstimator, TransformerMixin):
    def __init__(self, theta: float = 0.5):
        """
        Initialize the AdstockGeometric transformer with a fixed decay rate (theta).

        Parameters:
            An ad-stock of theta = 0.75 means that 75% of the impressions in period 1 were carried over to period 2.
            theta (float): Decay rate for the adstock transformation. Must be between 0 and 1.
        """
        self.theta = theta
        self.is_fitted_ = False  # Attribute to track whether the transformer is fitted

    def fit(self, X, y=None):
        """
        Fit method for the transformer. Checks the input array and validates the theta parameter.

        Parameters:
            X (array-like): The data to transform.
            y (optional): Ignored, exists for compatibility.

        Returns:
            self: Returns the instance itself.
        """
        X = check_array(X, ensure_2d=False)
        if not 0 <= self.theta <= 1:
            raise ValueError("Theta must be between 0 and 1")
        self._check_n_features(X, reset=True)
        self.is_fitted_ = True  # Set to True after fitting
        return self

    def transform(self, X):
        """
        Apply the geometric adstock transformation to the input data and return additional transformation details.

        Parameters:
            X (array-like): The data to transform.

        Returns:
            result (dict): A dictionary containing the original values, transformed values,
                           cumulative decay factors, and total inflation.
        """
        if not self.is_fitted_:
            raise ValueError(
                "This AdstockGeometric instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        X = check_array(X, ensure_2d=False)
        self._check_n_features(X, reset=False)

        n = len(X)
        if n == 0:
            return {}

        x_decayed = np.zeros_like(X)
        if n > 0:
            x_decayed[0] = X[0]
            for i in range(1, n):
                x_decayed[i] = X[i] + self.theta * x_decayed[i - 1]

        theta_vec_cum = np.cumprod(np.concatenate(([1], np.full(n - 1, self.theta))))
        inflation_total = np.sum(x_decayed) / np.sum(X) if np.sum(X) != 0 else 0

        # return {
        #     'original_values': X,
        #     'decayed_values': x_decayed,
        #     'cumulative_decay_factors': theta_vec_cum,
        #     'total_inflation': inflation_total
        # }
        return x_decayed


class AdstockWeibull(BaseEstimator, TransformerMixin):
    def __init__(self, shape=1.0, scale=1.0, adstock_type='cdf', detailed_output=False):
        self.shape = shape
        self.scale = scale
        self.adstock_type = adstock_type
        self.detailed_output = detailed_output
        self.is_fitted_ = False

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=True)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, ensure_2d=True)

        x_bin = np.arange(1, X.shape[0] + 1)
        scale_trans = np.round(np.quantile(np.arange(1, X.shape[0] + 1), self.scale), 0)

        # results = []
        decayed_output = []

        for column in X.T:
            if self.adstock_type.lower() == 'cdf':
                decay = 1 - weibull_min.cdf(x_bin, self.shape, scale=scale_trans)
                theta_vec_cum = np.cumprod(np.insert(decay, 0, 1))
            elif self.adstock_type.lower() == 'pdf':
                decay = weibull_min.pdf(x_bin, self.shape, scale=scale_trans)
                theta_vec_cum = self.normalize(decay)

            x_decayed = np.convolve(column, theta_vec_cum, mode='full')[:len(column)]
            x_imme = column * theta_vec_cum[:len(column)]
            inflation_total = np.sum(x_decayed) / np.sum(column) if np.sum(column) != 0 else 0
            decayed_output.append(x_decayed)
        #     if self.detailed_output:
        #         result = {
        #             'x': column,
        #             'x_decayed': x_decayed,
        #             'thetaVecCum': theta_vec_cum,
        #             'inflation_total': inflation_total,
        #             'x_imme': x_imme
        #         }
        #     else:
        #         result = x_decayed
        #     results.append(result)
        #
        # return results if self.detailed_output else np.array(results).T
        return np.array(decayed_output).T  # Return transposed to match original shape

    @staticmethod
    def normalize(x):
        # Normalize function to scale PDF output from 0 to 1
        return (x - min(x)) / (max(x) - min(x)) if max(x) != min(x) else x

class ExponentialCarryover(BaseEstimator, TransformerMixin):
    def __init__(self, strength=0.5, length=1):
        self.strength = strength
        self.length = length

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self.sliding_window_ = (self.strength ** np.arange(self.length + 1)).reshape(-1, 1)
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        convolution = convolve2d(X, self.sliding_window_)
        if self.length > 0:
            convolution = convolution[: -self.length]
        return convolution