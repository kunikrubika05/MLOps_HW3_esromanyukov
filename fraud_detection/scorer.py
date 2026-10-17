from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier

from fraud_detection.preprocessing import load_state, preprocess


class Scorer:
    def __init__(
        self,
        artifacts: Path = Path('/runtime/models'),
        threshold: float = 0.98,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        self.artifacts = artifacts
        self.threshold = threshold
        self.version = ''
        self.reload()

    def reload(
        self,
    ) -> None:
        version = (self.artifacts / 'latest.txt').read_text().strip()
        if version == self.version:
            return
        directory = self.artifacts / version
        model = CatBoostClassifier()
        model.load_model(directory / 'model.cbm')
        state = load_state(directory / 'preprocessing.json')
        self.model = model
        self.state = state
        self.version = version

    def score(
        self,
        transaction: dict[str, Any],
    ) -> tuple[float, int, str]:
        return self.score_many([transaction])[0]

    def score_many(
        self,
        transactions: list[dict[str, Any]],
    ) -> list[tuple[float, int, str]]:
        self.reload()
        features = preprocess(pd.DataFrame(transactions), self.state)
        scores = self.model.predict_proba(features, thread_count=1)[:, 1]
        return [
            (float(score), int(score > self.threshold), self.version)
            for score in scores
        ]
