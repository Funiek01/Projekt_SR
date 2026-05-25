from __future__ import annotations

from typing import Iterable

import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


DEFAULT_RUNS = [4, 8, 12]  # Motor imagery: left vs right hand


def load_motor_imagery_epochs(
    subjects: Iterable[int],
    runs: Iterable[int] = DEFAULT_RUNS,
    tmin: float = 0.5,
    tmax: float = 3.5,
    l_freq: float = 8.0,
    h_freq: float = 45.0,
) -> mne.Epochs:
    """Load and preprocess PhysioNet EEGBCI motor-imagery epochs.

    The selected runs correspond to imagined left/right fist movement. In these
    runs, annotation T1 means LEFT_HAND and T2 means RIGHT_HAND.
    """
    subjects = list(subjects)
    runs = list(runs)

    if not subjects:
        raise ValueError("Podaj co najmniej jeden numer badanego, np. 1,2,3.")

    raw_files = eegbci.load_data(subjects, runs, update_path=True)

    raws = []
    for file_path in raw_files:
        raw = read_raw_edf(file_path, preload=True, verbose="ERROR")
        eegbci.standardize(raw)
        raws.append(raw)

    raw = concatenate_raws(raws)
    raw.set_montage("standard_1005", on_missing="ignore")
    raw.pick_types(eeg=True, meg=False, stim=False, eog=False)

    # Pasma mu/beta są używane jako cechy, a zakres 30-45 Hz służy do oceny szumu.
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")

    events, event_code = mne.events_from_annotations(raw, verbose="ERROR")
    if "T1" not in event_code or "T2" not in event_code:
        raise RuntimeError(
            "Nie znaleziono oznaczeń T1/T2 w adnotacjach EDF. "
            "Sprawdź, czy używasz przebiegów 4, 8, 12."
        )

    event_id = {
        "LEFT_HAND": event_code["T1"],
        "RIGHT_HAND": event_code["T2"],
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        picks="eeg",
        reject_by_annotation=True,
        verbose="ERROR",
    )

    return epochs
