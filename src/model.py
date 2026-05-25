from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .fuzzy_confidence import FuzzyDecision, FuzzyReliabilitySystem


AVAILABLE_CLASSIFIERS = ("SVM RBF", "Random Forest", "LDA")


@dataclass
class TrainResult:
    accuracy: float
    report: str
    confusion: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    fuzzy_decisions: list[FuzzyDecision]


@dataclass
class EEGFuzzyClassifier:
    pipeline: Pipeline
    classes: np.ndarray
    noise_min: float
    noise_max: float
    feature_names: list[str]
    used_channels: list[str]
    classifier_name: str = "SVM RBF"

    @staticmethod
    def create(classifier_name: str = "SVM RBF") -> "EEGFuzzyClassifier":
        """Create ML pipeline selected by user.

        Every classifier must support predict_proba(), because the fuzzy module
        uses class probabilities to compute confidence and margin.
        """
        if classifier_name == "SVM RBF":
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            probability=True,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
        elif classifier_name == "Random Forest":
            pipeline = Pipeline(
                steps=[
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=42,
                            class_weight="balanced",
                            min_samples_leaf=2,
                        ),
                    )
                ]
            )
        elif classifier_name == "LDA":
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", LinearDiscriminantAnalysis()),
                ]
            )
        else:
            raise ValueError(
                f"Nieznany klasyfikator: {classifier_name}. "
                f"Dostępne opcje: {', '.join(AVAILABLE_CLASSIFIERS)}."
            )

        return EEGFuzzyClassifier(
            pipeline=pipeline,
            classes=np.array([]),
            noise_min=0.0,
            noise_max=1.0,
            feature_names=[],
            used_channels=[],
            classifier_name=classifier_name,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_ratio: np.ndarray,
        feature_names: list[str],
        used_channels: list[str],
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> TrainResult:
        X_train, X_test, y_train, y_test, noise_train, noise_test = train_test_split(
            X,
            y,
            noise_ratio,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        self.pipeline.fit(X_train, y_train)
        self.classes = self.pipeline.named_steps["classifier"].classes_
        self.feature_names = feature_names
        self.used_channels = used_channels
        self.noise_min = float(np.percentile(noise_train, 5))
        self.noise_max = float(np.percentile(noise_train, 95))

        # Ujednolicenie wyniku: predykcja jest klasą z największym prawdopodobieństwem.
        # Dzięki temu pred=... zgadza się z wartościami pokazywanymi w proba=[...].
        probabilities = self.pipeline.predict_proba(X_test)
        y_pred = self.classes[np.argmax(probabilities, axis=1)]

        fuzzy_decisions = self.fuzzy_evaluate_batch(probabilities, noise_test)

        return TrainResult(
            accuracy=float(accuracy_score(y_test, y_pred)),
            report=classification_report(y_test, y_pred, zero_division=0),
            confusion=confusion_matrix(y_test, y_pred, labels=self.classes),
            y_test=y_test,
            y_pred=y_pred,
            probabilities=probabilities,
            fuzzy_decisions=fuzzy_decisions,
        )

    def _normalize_noise(self, noise_ratio: np.ndarray) -> np.ndarray:
        denom = max(self.noise_max - self.noise_min, 1e-12)
        return np.clip((noise_ratio - self.noise_min) / denom, 0, 1)

    def fuzzy_evaluate_batch(self, probabilities: np.ndarray, noise_ratio: np.ndarray) -> list[FuzzyDecision]:
        fuzzy = FuzzyReliabilitySystem()
        normalized_noise = self._normalize_noise(noise_ratio)
        decisions: list[FuzzyDecision] = []

        for proba, noise in zip(probabilities, normalized_noise):
            confidence = float(np.max(proba))
            if len(proba) >= 2:
                margin = float(abs(proba[0] - proba[1]))
            else:
                margin = 0.0
            decisions.append(fuzzy.evaluate(confidence, margin, float(noise)))

        return decisions

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "EEGFuzzyClassifier":
        return joblib.load(path)
