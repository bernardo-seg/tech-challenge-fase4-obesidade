import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MtransGrouper(BaseEstimator, TransformerMixin):
    """Agrupa as categorias raras de 'mtrans'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_array = np.asarray(X)
        X_series = pd.Series(X_array.flatten(), name="mtrans")
        mtrans_agrupado = X_series.replace(
            ["moto", "bicicleta", "caminhando"], "outros"
        )
        return mtrans_agrupado.values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return ["mtrans_grouped"]
        return input_features


class CalcGrouper(BaseEstimator, TransformerMixin):
    """Agrupa as categorias raras de 'calc'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_array = np.asarray(X)
        X_series = pd.Series(X_array.flatten(), name="calc")
        sempre_freq = X_series.replace("sempre", "frequentemente")
        return sempre_freq.values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return ["calc_grouped"]
        return input_features


class RoundingTransformer(BaseEstimator, TransformerMixin):
    """Arredonda os dados sintéticos (ex: 2.45 -> 2)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, **kwargs):
        return np.round(X).astype(int)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError(
                "input_features must be provided for RoundingTransformer.get_feature_names_out"
            )
        return input_features
