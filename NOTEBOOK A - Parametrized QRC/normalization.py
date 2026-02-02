# src/normalization.py

import numpy as np
from dataclasses import dataclass


@dataclass
class MinMaxNormalizer:

    y_min: float
    y_max: float
    clip: bool = False

    def transform(self, y_raw: np.ndarray) -> np.ndarray:
        y_norm = (y_raw - self.y_min) / (self.y_max - self.y_min)
        if self.clip:
            y_norm = np.clip(y_norm, 0.0, 1.0)
        return y_norm

    def inverse_transform(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * (self.y_max - self.y_min) + self.y_min


def make_minmax_normalizer(
    method: str,
    y_all: np.ndarray,
    T_TRAIN: int | None = None,
    margin_factor: float = 0.0,
    clip: bool | None = None,
) -> MinMaxNormalizer:
  
    method = method.lower()

    if method not in {"train", "full", "enlarged"}:
        raise ValueError(f"Unknown normalization method: {method}")

    if method in {"train", "enlarged"} and T_TRAIN is None:
        raise ValueError("T_TRAIN must be provided for 'train' and 'enlarged' methods.")

    if method == "train":
        y_train = y_all[: T_TRAIN + 1]
        y_min = float(np.min(y_train))
        y_max = float(np.max(y_train))
        if clip is None:
            clip = True  
        return MinMaxNormalizer(y_min=y_min, y_max=y_max, clip=clip)

    if method == "full":
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        if clip is None:
            clip = False  
        return MinMaxNormalizer(y_min=y_min, y_max=y_max, clip=clip)

    if method == "enlarged":
        y_train = y_all[: T_TRAIN + 1]
        y_min_train = float(np.min(y_train))
        y_max_train = float(np.max(y_train))
        span_train = y_max_train - y_min_train

        y_min = y_min_train - margin_factor * span_train
        y_max = y_max_train + margin_factor * span_train

        if clip is None:
            clip = False  

        return MinMaxNormalizer(y_min=y_min, y_max=y_max, clip=clip)
