"""Lightweight machine learning models used by the rescue system."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from random import Random
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Dataset:
    features: List[List[float]]
    tri_labels: List[int]
    sobr_values: List[float]


class FeatureScaler:
    """Per-feature min-max scaler implemented with the standard library."""

    def __init__(self) -> None:
        self._min: List[float] = []
        self._max: List[float] = []

    def fit(self, rows: Iterable[Sequence[float]]) -> None:
        iterator = iter(rows)
        try:
            first = list(next(iterator))
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise ValueError("dataset is empty") from exc

        mins = list(first)
        maxs = list(first)
        for row in iterator:
            for idx, value in enumerate(row):
                if value < mins[idx]:
                    mins[idx] = value
                if value > maxs[idx]:
                    maxs[idx] = value

        self._min = mins
        self._max = maxs

    def transform(self, row: Sequence[float]) -> List[float]:
        if not self._min or not self._max:
            raise RuntimeError("Scaler must be fitted before calling transform().")

        transformed: List[float] = []
        for value, min_value, max_value in zip(row, self._min, self._max):
            if max_value == min_value:
                transformed.append(0.0)
            else:
                transformed.append((value - min_value) / (max_value - min_value))
        return transformed


class _KNNBase:
    def __init__(self, k: int, samples: Iterable[Tuple[List[float], float]]) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self._k = k
        self._samples: List[Tuple[List[float], float]] = [(list(feat), value) for feat, value in samples]

    def _neighbors(self, normalized_features: Sequence[float]) -> List[Tuple[float, float]]:
        distances: List[Tuple[float, float]] = []
        for features, label in self._samples:
            distance = sqrt(sum((f - g) ** 2 for f, g in zip(features, normalized_features)))
            distances.append((distance, label))
        distances.sort(key=lambda item: item[0])
        return distances[: self._k]


class KNNClassifier(_KNNBase):
    def predict(self, normalized_features: Sequence[float]) -> int:
        votes: List[Tuple[float, int]] = []
        for distance, label in self._neighbors(normalized_features):
            votes.append((distance, int(label)))

        tally: dict[int, Tuple[int, float]] = {}
        for distance, label in votes:
            count, cumulative_distance = tally.get(label, (0, 0.0))
            tally[label] = (count + 1, cumulative_distance + distance)

        # Majority vote with a tie-breaker favouring the closest neighbours
        best_label = None
        best_count = -1
        best_distance = float("inf")
        for label, (count, cumulative_distance) in tally.items():
            avg_distance = cumulative_distance / count if count else float("inf")
            if count > best_count or (count == best_count and avg_distance < best_distance):
                best_label = label
                best_count = count
                best_distance = avg_distance

        if best_label is None:  # pragma: no cover - guard
            raise RuntimeError("no neighbours available")
        return best_label


class KNNRegressor(_KNNBase):
    def predict(self, normalized_features: Sequence[float]) -> float:
        neighbours = self._neighbors(normalized_features)
        if not neighbours:  # pragma: no cover - guard
            raise RuntimeError("no neighbours available")
        return sum(label for _, label in neighbours) / len(neighbours)


def load_dataset(csv_path: Path) -> Dataset:
    import csv

    features: List[List[float]] = []
    tri_labels: List[int] = []
    sobr_values: List[float] = []

    with csv_path.open("r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        if headers is None:
            raise ValueError("dataset is empty")

        for row in reader:
            idade = float(row[0])
            fc = float(row[1])
            fr = float(row[2])
            pas = float(row[3])
            spo2 = float(row[4])
            temp = float(row[5])
            pr = float(row[6])
            sg = float(row[7])
            fx = float(row[8])
            queim = float(row[9])
            gcs = float(row[10])
            avpu = float(row[11])
            tri = int(row[12])
            sobr = float(row[13])

            features.append([idade, fc, fr, pas, spo2, temp, pr, sg, fx, queim, gcs, avpu])
            tri_labels.append(tri)
            sobr_values.append(sobr)

    return Dataset(features, tri_labels, sobr_values)


@dataclass
class ModelMetrics:
    tri_accuracy: float
    sobr_mae: float


class VictimModels:
    """Wrapper around the KNN models for triage and survivability."""

    def __init__(self, dataset_path: Path, k_neighbors: int = 5, seed: int = 42) -> None:
        self.dataset = load_dataset(dataset_path)
        self.scaler = FeatureScaler()
        self.scaler.fit(self.dataset.features)

        normalized = [self.scaler.transform(row) for row in self.dataset.features]
        tri_samples = list(zip(normalized, self.dataset.tri_labels))
        sobr_samples = list(zip(normalized, self.dataset.sobr_values))

        self.classifier = KNNClassifier(k_neighbors, tri_samples)
        self.regressor = KNNRegressor(k_neighbors, sobr_samples)
        self.metrics = self._evaluate(k_neighbors, seed)

    def _evaluate(self, k_neighbors: int, seed: int) -> ModelMetrics:
        rng = Random(seed)
        indices = list(range(len(self.dataset.features)))
        rng.shuffle(indices)
        split_index = max(1, int(0.8 * len(indices)))
        train_idx = indices[:split_index]
        test_idx = indices[split_index:]
        if not test_idx:
            test_idx = train_idx

        train_features = [self.dataset.features[i] for i in train_idx]
        train_tri = [self.dataset.tri_labels[i] for i in train_idx]
        train_sobr = [self.dataset.sobr_values[i] for i in train_idx]
        test_features = [self.dataset.features[i] for i in test_idx]
        test_tri = [self.dataset.tri_labels[i] for i in test_idx]
        test_sobr = [self.dataset.sobr_values[i] for i in test_idx]

        scaler = FeatureScaler()
        scaler.fit(train_features)
        train_normalized = [scaler.transform(row) for row in train_features]
        classifier = KNNClassifier(k_neighbors, zip(train_normalized, train_tri))
        regressor = KNNRegressor(k_neighbors, zip(train_normalized, train_sobr))

        total = len(test_features)
        correct = 0
        abs_error = 0.0
        for features, tri_label, sobr_value in zip(test_features, test_tri, test_sobr):
            normalized = scaler.transform(features)
            prediction = classifier.predict(normalized)
            if prediction == tri_label:
                correct += 1
            sobr_prediction = regressor.predict(normalized)
            abs_error += abs(sobr_prediction - sobr_value)

        tri_accuracy = correct / total if total else 0.0
        sobr_mae = abs_error / total if total else 0.0
        return ModelMetrics(tri_accuracy, sobr_mae)

    def predict(self, vital_signals: Sequence[float]) -> Tuple[int, float]:
        # vital_signals structure: [vid, idade, fc, fr, pas, spo2, temp, pr, sg, fx, queim, gcs, avpu]
        if len(vital_signals) < 13:
            raise ValueError("vital_signals vector is incomplete")
        features = vital_signals[1:13]
        normalized = self.scaler.transform(features)
        tri_prediction = self.classifier.predict(normalized)
        sobr_prediction = self.regressor.predict(normalized)
        return tri_prediction, min(max(sobr_prediction, 0.0), 1.0)
