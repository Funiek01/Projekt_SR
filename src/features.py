from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mne
import numpy as np


@dataclass
class FeatureConfig:
    channels: tuple[str, ...] = (
        "C3",
        "C4",
        "Cz",
        "FC3",
        "FC4",
        "CP3",
        "CP4",
        "C1",
        "C2",
        "C5",
        "C6",
    )
    # Dla motor imagery najbardziej typowe i użyteczne są pasma mu oraz beta.
    # Usunięcie theta zmniejsza liczbę cech i ogranicza ryzyko przeuczenia.
    bands: tuple[tuple[str, float, float], ...] = (
        ("mu", 8.0, 13.0),
        ("beta", 13.0, 30.0),
    )
    # Uproszczony wskaźnik szumu dla modułu fuzzy.
    noise_band: tuple[float, float] = (30.0, 45.0)
    total_band: tuple[float, float] = (4.0, 45.0)


def _choose_channels(epochs: mne.Epochs, preferred: Iterable[str]) -> list[str]:
    available = [ch for ch in preferred if ch in epochs.ch_names]
    return available if available else list(epochs.ch_names)


def _bandpower_fft(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """Return mean power in a frequency band for data shaped epochs x channels x samples."""
    freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / sfreq)
    spectrum = np.abs(np.fft.rfft(data, axis=-1)) ** 2
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.zeros(data.shape[:2])
    return spectrum[..., mask].mean(axis=-1)


def extract_bandpower_features(
    epochs: mne.Epochs,
    config: FeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract log-bandpower features and a simple noise index.

    Returns:
        X: feature matrix, shape n_epochs x n_features
        noise_ratio: one value per epoch, higher means noisier
        feature_names: names for columns in X
        used_channels: selected EEG channel names
    """
    config = config or FeatureConfig()
    used_channels = _choose_channels(epochs, config.channels)
    ep = epochs.copy().pick_channels(used_channels, ordered=True)

    data = ep.get_data()  # n_epochs x n_channels x n_samples
    sfreq = float(ep.info["sfreq"])

    features = []
    feature_names: list[str] = []

    for band_name, fmin, fmax in config.bands:
        power = _bandpower_fft(data, sfreq, fmin, fmax)
        features.append(np.log10(power + 1e-12))
        feature_names.extend([f"{ch}_{band_name}" for ch in used_channels])

    X = np.concatenate(features, axis=1)

    noise_power = _bandpower_fft(data, sfreq, *config.noise_band).mean(axis=1)
    total_power = _bandpower_fft(data, sfreq, *config.total_band).mean(axis=1)
    noise_ratio = noise_power / (total_power + 1e-12)

    return X, noise_ratio, feature_names, used_channels


def labels_from_epochs(epochs: mne.Epochs) -> np.ndarray:
    """Convert MNE epoch event codes to readable class labels."""
    reverse_event_id = {v: k for k, v in epochs.event_id.items()}
    return np.array([reverse_event_id[int(code)] for code in epochs.events[:, -1]])
