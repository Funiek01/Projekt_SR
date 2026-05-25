from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from src.data_loader import DEFAULT_RUNS, load_motor_imagery_epochs
from src.features import extract_bandpower_features, labels_from_epochs
from src.model import AVAILABLE_CLASSIFIERS, EEGFuzzyClassifier


class EEGFuzzyApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EEG Motor Imagery — ML + Fuzzy Logic")
        self.geometry("950x700")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.model: EEGFuzzyClassifier | None = None

        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            root,
            text="Klasyfikacja EEG: LEFT_HAND vs RIGHT_HAND z modułem logiki rozmytej",
            font=("Arial", 14, "bold"),
        )
        title.pack(anchor="w", pady=(0, 12))

        form = ttk.LabelFrame(root, text="Ustawienia danych i modelu")
        form.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(form, text="Numery badanych, np. 1,2,3:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.subjects_var = tk.StringVar(value="7")
        ttk.Entry(form, textvariable=self.subjects_var, width=30).grid(row=0, column=1, sticky="w", padx=8, pady=8)

        ttk.Label(form, text="Runs:").grid(row=0, column=2, sticky="w", padx=8, pady=8)
        self.runs_var = tk.StringVar(value=",".join(str(r) for r in DEFAULT_RUNS))
        ttk.Entry(form, textvariable=self.runs_var, width=15).grid(row=0, column=3, sticky="w", padx=8, pady=8)

        ttk.Label(form, text="Klasyfikator:").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.classifier_var = tk.StringVar(value="SVM RBF")
        classifier_combo = ttk.Combobox(
            form,
            textvariable=self.classifier_var,
            values=list(AVAILABLE_CLASSIFIERS),
            state="readonly",
            width=25,
        )
        classifier_combo.grid(row=1, column=1, sticky="w", padx=8, pady=8)

        hint = ttk.Label(
            form,
            text="Finalny wariant testowany: badany 7, runs 4,8,12, cechy mu+beta, filtr 8–45 Hz.",
        )
        hint.grid(row=1, column=2, columnspan=2, sticky="w", padx=8, pady=8)

        buttons = ttk.Frame(root)
        buttons.pack(fill=tk.X, pady=(0, 10))

        self.train_button = ttk.Button(buttons, text="Trenuj model", command=self._start_training)
        self.train_button.pack(side=tk.LEFT, padx=(0, 8))

        self.save_button = ttk.Button(buttons, text="Zapisz model", command=self._save_model, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT)

        info = ttk.LabelFrame(root, text="Wyniki")
        info.pack(fill=tk.BOTH, expand=True)

        self.output = tk.Text(info, wrap=tk.WORD, height=28)
        self.output.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.status_var = tk.StringVar(value="Gotowy")
        status = ttk.Label(root, textvariable=self.status_var)
        status.pack(anchor="w", pady=(6, 0))

    def _parse_int_list(self, value: str, field_name: str) -> list[int]:
        try:
            result = [int(x.strip()) for x in value.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(f"Pole '{field_name}' powinno zawierać liczby oddzielone przecinkami.") from exc
        if not result:
            raise ValueError(f"Pole '{field_name}' nie może być puste.")
        return result

    def _start_training(self) -> None:
        try:
            subjects = self._parse_int_list(self.subjects_var.get(), "Numery badanych")
            runs = self._parse_int_list(self.runs_var.get(), "Runs")
            classifier_name = self.classifier_var.get()
        except ValueError as exc:
            messagebox.showerror("Błąd danych", str(exc))
            return

        self.output.delete("1.0", tk.END)
        self._log("Start treningu...\n")
        self._log(f"Badani: {subjects}\n")
        self._log(f"Runs: {runs}\n")
        self._log(f"Klasyfikator: {classifier_name}\n\n")
        self.train_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self.status_var.set("Trenowanie modelu...")

        thread = threading.Thread(
            target=self._train_worker,
            args=(subjects, runs, classifier_name),
            daemon=True,
        )
        thread.start()

    def _train_worker(self, subjects: list[int], runs: list[int], classifier_name: str) -> None:
        try:
            self._log("1/4 Wczytywanie EDF przez MNE...\n")
            epochs = load_motor_imagery_epochs(subjects=subjects, runs=runs)
            self._log(f"   Liczba epok: {len(epochs)}\n")
            self._log(f"   Kanały EEG: {len(epochs.ch_names)}\n\n")

            self._log("2/4 Ekstrakcja cech pasmowych...\n")
            X, noise_ratio, feature_names, used_channels = extract_bandpower_features(epochs)
            y = labels_from_epochs(epochs)
            self._log(f"   Wymiar macierzy cech: {X.shape}\n")
            self._log(f"   Użyte kanały: {', '.join(used_channels)}\n")
            self._log(f"   Liczba klas: {', '.join(sorted(set(y)))}\n\n")

            self._log(f"3/4 Trening klasyfikatora: {classifier_name}...\n")
            clf = EEGFuzzyClassifier.create(classifier_name)
            result = clf.train(X, y, noise_ratio, feature_names, used_channels)
            self.model = clf

            self._log("4/4 Ocena decyzji przez system rozmyty...\n\n")
            self._log("=== WYNIKI ML ===\n")
            self._log(f"Klasyfikator: {clf.classifier_name}\n")
            self._log(f"Accuracy: {result.accuracy:.3f}\n\n")
            self._log("Classification report:\n")
            self._log(result.report + "\n")
            self._log("Macierz pomyłek, kolejność klas: " + ", ".join(clf.classes) + "\n")
            self._log(str(result.confusion) + "\n\n")

            reliabilities = np.array([d.reliability for d in result.fuzzy_decisions])
            levels = [d.level for d in result.fuzzy_decisions]
            self._log("=== WYNIKI FUZZY ===\n")
            self._log(f"Średnia wiarygodność decyzji: {reliabilities.mean():.1f}/100\n")
            for level in ["PEWNA", "UMIARKOWANA", "NIEPEWNA"]:
                self._log(f"{level}: {levels.count(level)} próbek\n")

            self._log("\nPrzykładowe decyzje testowe:\n")
            for idx in range(min(10, len(result.y_test))):
                proba = result.probabilities[idx]
                fuzzy = result.fuzzy_decisions[idx]
                proba_text = ", ".join(
                    f"{cls}={p:.3f}" for cls, p in zip(clf.classes, proba)
                )
                self._log(
                    f"#{idx + 1}: true={result.y_test[idx]}, pred={result.y_pred[idx]}, "
                    f"proba=[{proba_text}], fuzzy={fuzzy.level}, "
                    f"reliability={fuzzy.reliability:.1f}/100\n"
                )

            self._log("\nGotowe. Model można zapisać przyciskiem 'Zapisz model'.\n")
            self.log_queue.put("__TRAIN_DONE__")
        except Exception:
            self._log("\nWystąpił błąd:\n")
            self._log(traceback.format_exc())
            self.log_queue.put("__TRAIN_FAILED__")

    def _save_model(self) -> None:
        if self.model is None:
            messagebox.showinfo("Brak modelu", "Najpierw wytrenuj model.")
            return

        safe_name = self.model.classifier_name.lower().replace(" ", "_")
        path = filedialog.asksaveasfilename(
            defaultextension=".joblib",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
            initialfile=f"eeg_fuzzy_model_{safe_name}.joblib",
        )
        if not path:
            return
        self.model.save(Path(path))
        messagebox.showinfo("Zapisano", f"Model zapisano do:\n{path}")

    def _log(self, text: str) -> None:
        self.log_queue.put(text)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg == "__TRAIN_DONE__":
                    self.train_button.configure(state=tk.NORMAL)
                    self.save_button.configure(state=tk.NORMAL)
                    self.status_var.set("Gotowe")
                    continue
                if msg == "__TRAIN_FAILED__":
                    self.train_button.configure(state=tk.NORMAL)
                    self.save_button.configure(state=tk.DISABLED)
                    self.status_var.set("Błąd")
                    continue
                self.output.insert(tk.END, msg)
                self.output.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)


if __name__ == "__main__":
    app = EEGFuzzyApp()
    app.mainloop()
