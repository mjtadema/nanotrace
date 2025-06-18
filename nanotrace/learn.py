"""
Some convenience functions to make using OneClass models (outlier detection) easier to use.
"""
from __future__ import annotations
__copyright__ = """
Copyright 2025 Matthijs Tadema

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

ocsvm_params = {
    'oneclasssvm__nu': [0.001, 0.01, 0.1, 0.5],
    'oneclasssvm__gamma': ['scale', 0.001, 0.01, 0.1, 0.5]
}

Estimator = type("Estimator", (), {"fit": None})
Predictor = type("Predictor", (), {"decision_function": None})

def gridSearchCVOneClass(estimator: Estimator,
                        Xpos: pd.DataFrame, *, param_grid,
                        cv: int | object=5, scoring: str | Callable='recall',
                        refit=True, **kwargs) -> Predictor:
    """
    Search a grid of parameter combinations for a given estimator.
    Requires the estimator to be a OneClassSVM or at least a Pipeline with
    a OneClassSVM at the end.

    :param estimator: a OneClassSVM or a Pipeline
    :param Xpos: features of the _positive_ class
    :param param_grid: parameter grid
    :param cv: optional fold for crossvalidation
    :param refit: whether to refit or not
    :param scoring: scoring function to use
    :param kwargs: Additional kwargs get passed to GridSearchCV
    :return: optimized predictor
    """
    search = GridSearchCV(estimator, param_grid=param_grid,
                          scoring=scoring, cv=cv, refit=refit, **kwargs)
    # Train this only on the positive samples
    search.fit(Xpos, np.repeat(1.0, len(Xpos)))
    return search


class TunedOneClass:
    """
    Tune the decision threshold for a one class predictor
    """
    def __init__(self, predictor: Predictor, scoring=accuracy_score) -> None:
        """
        Initialized the tuner with a predictor.
        Usually OneClassSVM or a Pipeline that was already pre trained.

        :param predictor: trained predictor
        """
        self.predictor = predictor
        self.scorer = scoring
        self._threshold = None
        self.cv_results_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Tune the decision threshold to optimize the accuracy score

        :param X: features
        :param y: labels
        """
        dec = self.predictor.decision_function(X)

        def f(x):
            y_pred = np.where(self.predictor.decision_function(X) > x, 1.0, -1.0)
            return 1 - self.scorer(y, y_pred)

        t = minimize_scalar(f, bounds=(dec.min(), dec.max())).x
        self._threshold = t
        logger.info(f"optimal threshold: {t}")

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise Exception("first run .fit(X,Y)")
        return self._threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict labels using the tuned threshold
        :param X: features
        :return: labels
        """
        return np.where(self.predictor.decision_function(X) > self.threshold, 1.0, -1.0)