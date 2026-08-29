"""Scenario definitions and normalized inverse sensitivities for Problem 4."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import ModelParameters


@dataclass(frozen=True)
class SensitivityScenario:
    case_id: str
    parameter: str
    scenario: str
    parameter_value: float
    unit: str
    baseline_value: float
    updates: dict[str, float]

    def parameters(self, baseline: ModelParameters) -> ModelParameters:
        return baseline.with_updates(**self.updates)


def scenario_definitions() -> list[SensitivityScenario]:
    """Return required low, baseline, high and discrete sensitivity cases."""

    rows: list[SensitivityScenario] = []

    def add_triplet(
        parameter: str,
        unit: str,
        baseline: float,
        low: float,
        high: float,
        field_name: str,
    ) -> None:
        rows.extend(
            [
                SensitivityScenario(
                    f"{parameter}_low", parameter, "low", low, unit, baseline,
                    {field_name: low},
                ),
                SensitivityScenario(
                    f"{parameter}_baseline", parameter, "baseline", baseline,
                    unit, baseline, {},
                ),
                SensitivityScenario(
                    f"{parameter}_high", parameter, "high", high, unit, baseline,
                    {field_name: high},
                ),
            ]
        )

    add_triplet("M0", "W/m2", 70.0, 63.0, 77.0, "metabolic_rate_W_m2")
    add_triplet("h_in", "W/(m2 K)", 3.0, 2.7, 3.3, "h_in_W_m2K")
    add_triplet("lambda_out", "1", 1.0, 0.9, 1.1, "lambda_out")
    add_triplet("beta_DSC", "K/min", 10.0, 9.0, 11.0, "dsc_scan_rate_K_min")
    rows.extend(
        [
            SensitivityScenario(
                "tau_bl_60", "tau_bl", "60 s", 60.0, "s", 173.0,
                {"blood_flow_time_constant_s": 60.0},
            ),
            SensitivityScenario(
                "tau_bl_173", "tau_bl", "baseline", 173.0, "s", 173.0, {},
            ),
            SensitivityScenario(
                "tau_bl_200", "tau_bl", "200 s", 200.0, "s", 173.0,
                {"blood_flow_time_constant_s": 200.0},
            ),
            SensitivityScenario(
                "Tskin_initial_37", "T_skin_initial", "baseline", 37.0,
                "degC", 37.0, {},
            ),
            SensitivityScenario(
                "Tskin_initial_33p7", "T_skin_initial", "33.7 degC", 33.7,
                "degC", 37.0, {"initial_skin_temperature_C": 33.7},
            ),
        ]
    )
    return rows


def normalized_sensitivity_summary(
    inverse: pd.DataFrame, baseline_lambda: float
) -> pd.DataFrame:
    """Calculate centered dimensionless sensitivities for symmetric parameters."""

    rows: list[dict[str, float | str]] = []
    for parameter in ("M0", "h_in", "lambda_out", "beta_DSC"):
        subset = inverse[inverse["parameter"] == parameter].set_index("scenario")
        low = subset.loc["low"]
        base = subset.loc["baseline"]
        high = subset.loc["high"]
        sensitivity = (
            (float(high["lambda_opt"]) - float(low["lambda_opt"]))
            / (float(high["parameter_value"]) - float(low["parameter_value"]))
            * float(base["baseline_value"])
            / baseline_lambda
        )
        rows.append(
            {
                "parameter": parameter,
                "baseline_value": float(base["baseline_value"]),
                "low_value": float(low["parameter_value"]),
                "high_value": float(high["parameter_value"]),
                "lambda_low": float(low["lambda_opt"]),
                "lambda_baseline": float(base["lambda_opt"]),
                "lambda_high": float(high["lambda_opt"]),
                "normalized_sensitivity_lambda": sensitivity,
                "abs_normalized_sensitivity_lambda": abs(sensitivity),
                "direction_interpretation": (
                    "parameter increase raises required PCM multiplier"
                    if sensitivity > 0.0
                    else "parameter increase lowers required PCM multiplier"
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        "abs_normalized_sensitivity_lambda", ascending=False
    )
    result["rank"] = range(1, len(result) + 1)
    return result.reset_index(drop=True)
