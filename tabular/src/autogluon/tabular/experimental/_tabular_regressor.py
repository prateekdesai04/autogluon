from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from autogluon.core.metrics import Scorer

from .. import TabularPredictor
from ._scikit_mixin import ScikitMixin


class TabularRegressor(BaseEstimator, RegressorMixin, ScikitMixin):
    def __init__(
        self,
        eval_metric: str | Scorer = None,
        time_limit: float = None,
        presets: list[str] | str = None,
        hyperparameters: dict | str = None,
        path: str = None,
        verbosity: int = 2,
        init_args: dict = None,
        fit_args: dict = None,
    ):
        self.eval_metric = eval_metric
        self.time_limit = time_limit
        self.presets = presets
        self.hyperparameters = hyperparameters
        self.path = path
        self.verbosity = verbosity
        self.init_args = init_args
        self.fit_args = fit_args

    def fit(self, X, y):
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)  # Commented out to allow for object dtypes

        self.n_features_in_ = X.shape[1]

        init_args = self._get_init_args(problem_type="regression")
        fit_args = self._get_fit_args()

        self.predictor_ = TabularPredictor(**init_args)

        train_data = self._combine_X_y(X=X, y=y)

        self.predictor_.fit(train_data, **fit_args)

        # Return the regressor
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Inconsistent number of features between fit and predict calls: ({self.n_features_in_}, {X.shape[1]})")

        data = pd.DataFrame(X)
        y_pred = self.predictor_.predict(data=data).to_numpy()
        return y_pred
