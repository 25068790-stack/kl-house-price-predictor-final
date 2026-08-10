from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy import sparse


class PortableHousePriceModel:
    """Portable serving version of the frozen final XGBoost workflow."""

    def __init__(self, model_path: str | Path, config_path: str | Path):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)

        self.booster = xgb.Booster()
        self.booster.load_model(self.model_path)

        expected = int(self.config["expected_model_features"])
        if self.booster.num_features() != expected:
            raise RuntimeError(
                f"Model/config mismatch: booster has {self.booster.num_features()} "
                f"features, expected {expected}."
            )

    @staticmethod
    def _is_missing(value) -> bool:
        if value is None:
            return True
        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False

    def _transform_one(self, record: dict):
        x = dict(record)

        # Exact deterministic feature engineering used in the final notebook.
        for raw_name, engineered_name in (
            ("built_up_sqft", "log_built_up_sqft"),
            ("land_area_sqft", "log_land_area_sqft"),
        ):
            value = x.get(raw_name, 0.0)
            if self._is_missing(value):
                value = 0.0
            value = float(value)
            if value < 0:
                raise ValueError("Property area cannot be negative.")
            x[engineered_name] = float(np.log1p(value))

        # Frozen numerical median imputation.
        numerical_columns = self.config["numeric_columns"]
        statistics = np.asarray(
            self.config["numeric_imputer_statistics"], dtype=float
        )

        numerical = np.array(
            [
                np.nan
                if self._is_missing(x.get(column))
                else float(x.get(column))
                for column in numerical_columns
            ],
            dtype=float,
        )

        missing_mask = np.isnan(numerical)
        imputed = numerical.copy()
        imputed[missing_mask] = statistics[missing_mask]

        # Same missing-value indicators fitted in the research pipeline.
        indicator_indices = self.config[
            "numeric_missing_indicator_indices"
        ]
        indicators = missing_mask[indicator_indices].astype(float)

        # Frozen most-frequent categorical imputation + frozen OHE ordering.
        categorical_parts = []

        for column, fallback, categories in zip(
            self.config["categorical_columns"],
            self.config["categorical_imputer_statistics"],
            self.config["one_hot_categories"],
        ):
            value = x.get(column)
            if value is None or str(value).strip() == "":
                value = fallback

            value = str(value)

            one_hot = np.zeros(len(categories), dtype=float)
            try:
                one_hot[categories.index(value)] = 1.0
            except ValueError:
                # Equivalent to OneHotEncoder(handle_unknown="ignore")
                pass

            categorical_parts.append(one_hot)

        dense_vector = np.concatenate(
            [imputed, indicators, *categorical_parts]
        ).astype(np.float32, copy=False)

        expected = int(self.config["expected_model_features"])
        if dense_vector.shape[0] != expected:
            raise RuntimeError(
                f"Prepared {dense_vector.shape[0]} features; expected {expected}."
            )

        # The original ColumnTransformer produced a sparse matrix.
        # Preserving sparse representation is necessary for exact XGBoost parity.
        return sparse.csr_matrix(dense_vector.reshape(1, -1))

    def predict(self, record: dict) -> float:
        matrix = self._transform_one(record)
        log_prediction = float(
            self.booster.predict(xgb.DMatrix(matrix))[0]
        )
        prediction = float(np.expm1(log_prediction))

        if not np.isfinite(prediction) or prediction <= 0:
            raise RuntimeError("The model returned an invalid prediction.")

        return prediction
