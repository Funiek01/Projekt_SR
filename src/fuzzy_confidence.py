from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


@dataclass
class FuzzyDecision:
    reliability: float
    level: str


class FuzzyReliabilitySystem:
    """Fuzzy module for interpreting ML prediction confidence.

    Inputs:
        confidence: max class probability in [0, 1]
        margin: absolute difference between class probabilities in [0, 1]
        noise: normalized EEG noise index in [0, 1]

    Output:
        reliability: crisp score in [0, 100]
    """

    def __init__(self) -> None:
        confidence = ctrl.Antecedent(np.linspace(0, 1, 101), "confidence")
        margin = ctrl.Antecedent(np.linspace(0, 1, 101), "margin")
        noise = ctrl.Antecedent(np.linspace(0, 1, 101), "noise")
        reliability = ctrl.Consequent(np.linspace(0, 100, 101), "reliability")

        confidence["low"] = fuzz.trimf(confidence.universe, [0.0, 0.0, 0.55])
        confidence["medium"] = fuzz.trimf(confidence.universe, [0.40, 0.60, 0.80])
        confidence["high"] = fuzz.trimf(confidence.universe, [0.65, 1.0, 1.0])

        margin["small"] = fuzz.trimf(margin.universe, [0.0, 0.0, 0.30])
        margin["medium"] = fuzz.trimf(margin.universe, [0.15, 0.45, 0.75])
        margin["large"] = fuzz.trimf(margin.universe, [0.60, 1.0, 1.0])

        noise["low"] = fuzz.trimf(noise.universe, [0.0, 0.0, 0.35])
        noise["medium"] = fuzz.trimf(noise.universe, [0.20, 0.50, 0.80])
        noise["high"] = fuzz.trimf(noise.universe, [0.65, 1.0, 1.0])

        reliability["uncertain"] = fuzz.trimf(reliability.universe, [0, 0, 45])
        reliability["moderate"] = fuzz.trimf(reliability.universe, [30, 55, 80])
        reliability["certain"] = fuzz.trimf(reliability.universe, [65, 100, 100])

        rules = [
            ctrl.Rule(
                confidence["high"] & margin["large"] & noise["low"],
                reliability["certain"],
            ),
            ctrl.Rule(confidence["high"] & noise["low"], reliability["certain"]),
            ctrl.Rule(confidence["high"] & noise["medium"], reliability["moderate"]),
            ctrl.Rule(
                confidence["medium"] & (noise["low"] | noise["medium"]),
                reliability["moderate"],
            ),
            ctrl.Rule(
                confidence["low"] | margin["small"] | noise["high"],
                reliability["uncertain"],
            ),
        ]

        self._system = ctrl.ControlSystem(rules)

    def evaluate(self, confidence_value: float, margin_value: float, noise_value: float) -> FuzzyDecision:
        sim = ctrl.ControlSystemSimulation(self._system)
        sim.input["confidence"] = float(np.clip(confidence_value, 0, 1))
        sim.input["margin"] = float(np.clip(margin_value, 0, 1))
        sim.input["noise"] = float(np.clip(noise_value, 0, 1))
        sim.compute()

        score = float(sim.output.get("reliability", 0.0))
        if score < 40:
            level = "NIEPEWNA"
        elif score < 70:
            level = "UMIARKOWANA"
        else:
            level = "PEWNA"
        return FuzzyDecision(reliability=score, level=level)
