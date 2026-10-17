import pandas as pd
from catboost import CatBoostClassifier

from fraud_detection.preprocessing import CATEGORICAL_COLUMNS, TIME_COLUMNS


def train_model(
    features: pd.DataFrame,
    labels: pd.Series,
) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        auto_class_weights='Balanced',
        random_seed=42,
        thread_count=2,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        features,
        labels,
        cat_features=[c + '_cat' for c in CATEGORICAL_COLUMNS] + TIME_COLUMNS,
    )
    return model
