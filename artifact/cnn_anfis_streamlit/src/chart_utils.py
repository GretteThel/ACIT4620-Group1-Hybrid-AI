"""Presentational data for signed ANFIS logit-contribution charts.

This module does not alter inference, model weights, or training/test data.
Do not treat logit contributions as probabilities.
"""
from __future__ import annotations

import math
from typing import Any


def contribution_plot_rows(rule_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return 16 labelled and signed values for a zero-based horizontal chart.

    The original 1-based rule order is preserved. Values are not rounded or
    clipped, including negative values or near-zero floating-point entries.
    """
    if len(rule_table) != 16:
        raise ValueError("Expected exactly 16 original fuzzy rules")
    result: list[dict[str, Any]] = []
    for i, item in enumerate(rule_table, 1):
        name = f"Rule {i}"
        if item.get("Rule") != name:
            raise ValueError(f"Rule order mismatch at {name}")
        value = float(item["Predicted-logit contribution"])
        if not math.isfinite(value):
            raise ValueError("Non-finite signed rule contribution")
        result.append({
            "Rule": name,
            "Contribution": value,
            "Direction": "Positive" if value >= 0 else "Negative",
        })
    return result
