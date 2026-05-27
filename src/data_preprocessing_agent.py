"""
Claude API data-preprocessing agent for the DR pipeline.

Accepts an arbitrary CSV (any column names, any language) and maps it to the
canonical schema required by downstream pipeline steps:

    datetime          — ISO-8601 timestamp
    price             — electricity price (€/MWh)
    load              — total load (MW)
    renewable_total   — total renewable generation (MW)
    wind_generation   — wind generation (MW)   [optional if not present in source]
    solar_generation  — solar generation (MW)  [optional if not present in source]

The agent:
1. Detects column mapping via Claude (handles German/French column names, etc.)
2. Executes pandas code to clean, rename, gap-fill, and flag anomalies
3. Returns a cleaned DataFrame, a structured quality report, and a human-readable briefing

Usage
-----
    from src.data_preprocessing_agent import preprocess_csv

    df_clean, quality_report, briefing = preprocess_csv("path/to/raw.csv")
"""
from __future__ import annotations

import io
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import anthropic
except ImportError as e:
    raise ImportError("pip install anthropic") from e

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

CANONICAL_COLS = [
    "datetime",
    "price",
    "load",
    "renewable_total",
    "wind_generation",
    "solar_generation",
]

REQUIRED_COLS = {"datetime", "price", "load"}

# ---------------------------------------------------------------------------
# System prompt (cached via cache_control)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a data-cleaning expert for electricity market time-series data.

Your job is to map an arbitrary CSV to this canonical schema:
  - datetime          : ISO-8601 timestamp (hourly, UTC preferred)
  - price             : spot electricity price in €/MWh (can be negative)
  - load              : total system load / demand in MW
  - renewable_total   : total renewable generation in MW
  - wind_generation   : wind generation in MW   (optional)
  - solar_generation  : solar generation in MW  (optional)

Rules you MUST follow:
1. Required columns: datetime, price, load. If any cannot be found, report clearly.
2. Optional columns: renewable_total, wind_generation, solar_generation.
   - If wind + solar columns exist but renewable_total does not, compute it as their sum.
   - If none of the renewable columns exist, set them to NaN.
3. Gap filling:
   - Gaps ≤ 6 hours: forward-fill then linear interpolate.
   - Gaps > 6 hours: leave as NaN and add a `gap_flag` boolean column marking those rows.
4. Anomaly detection:
   - Load anomalies: load > 3× its 168-hour rolling mean → add `load_anomaly_flag`.
   - Price anomalies: price < -500 €/MWh → add `price_anomaly_flag`.
5. Always parse datetime robustly (handle mixed formats, timezone offsets).
6. Output only the canonical columns plus any _flag columns — drop all other columns.
7. After cleaning, call `output_result` with the serialized cleaned DataFrame and quality report.

You have access to one tool: `execute_code` to run pandas code on the raw DataFrame.
The DataFrame is already loaded as variable `df` (raw, unmodified).
Run as many `execute_code` calls as needed, then call `output_result` when done.
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "execute_code",
        "description": (
            "Execute a Python/pandas code snippet. "
            "The variable `df` holds the current working DataFrame (starts as the raw CSV). "
            "Assign your result back to `df` to persist changes. "
            "Returns stdout + any exception traceback."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python code. Use `df` as the DataFrame variable.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "output_result",
        "description": (
            "Finalise preprocessing. Call this once when the DataFrame is fully cleaned. "
            "Pass the cleaned DataFrame serialised as JSON-orient='records' and a quality report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "df_json": {
                    "type": "string",
                    "description": "df.to_json(orient='records', date_format='iso')",
                },
                "quality_report": {
                    "type": "object",
                    "description": (
                        "Structured quality report with keys: "
                        "columns_mapped (dict), rows_total (int), rows_missing_before (int), "
                        "rows_filled_interpolated (int), rows_gap_flagged (int), "
                        "load_anomalies (int), price_anomalies (int), issues (list[str])."
                    ),
                },
                "briefing": {
                    "type": "string",
                    "description": (
                        "2–4 sentence human-readable summary of what was done and any warnings."
                    ),
                },
            },
            "required": ["df_json", "quality_report", "briefing"],
        },
    },
]

# ---------------------------------------------------------------------------
# Safe code executor
# ---------------------------------------------------------------------------

def _exec_code(code: str, context: dict[str, Any]) -> str:
    """
    Execute `code` in a restricted namespace that contains only `context` plus
    safe stdlib / data-science imports. Returns combined stdout + any exception.
    """
    import sys

    _builtins_dict = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)  # type: ignore[arg-type]
    _allowed = (
        "len", "range", "enumerate", "zip", "map", "filter",
        "list", "dict", "set", "tuple", "int", "float", "str", "bool",
        "print", "repr", "round", "abs", "min", "max", "sum",
        "isinstance", "type", "hasattr", "getattr", "setattr",
        "sorted", "reversed", "any", "all", "next", "iter",
        "ValueError", "KeyError", "IndexError", "TypeError",
        "AttributeError", "RuntimeError", "StopIteration",
    )
    safe_globals: dict[str, Any] = {
        "__builtins__": {k: _builtins_dict[k] for k in _allowed if k in _builtins_dict},
        "pd": pd,
        "np": np,
        "io": io,
        "json": json,
    }
    # Merge in the DataFrame context
    safe_globals.update(context)

    stdout_capture = io.StringIO()
    import contextlib

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(code, "<agent_code>", "exec"), safe_globals)  # noqa: S102
        # Pull updated `df` back into context
        if "df" in safe_globals:
            context["df"] = safe_globals["df"]
        output = stdout_capture.getvalue()
        return output if output else "(no output)"
    except Exception:
        return traceback.format_exc()


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

def preprocess_csv(
    csv_path: str | Path,
    *,
    model: str = "claude-sonnet-4-6",
    max_iterations: int = 20,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, dict, str]:
    """
    Run the Claude preprocessing agent on an arbitrary CSV file.

    Parameters
    ----------
    csv_path : path to the raw CSV file.
    model    : Claude model to use.
    max_iterations : safety cap on agentic loop iterations.
    api_key  : Anthropic API key (defaults to ANTHROPIC_API_KEY env var).

    Returns
    -------
    df_clean      : cleaned pandas DataFrame with canonical columns.
    quality_report: dict with gap/anomaly/mapping statistics.
    briefing      : human-readable summary string.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Load raw CSV and give the agent a preview
    raw_df = pd.read_csv(csv_path)
    exec_context: dict[str, Any] = {"df": raw_df.copy()}

    preview_lines = [
        f"Shape: {raw_df.shape}",
        f"Columns: {list(raw_df.columns)}",
        "",
        raw_df.head(5).to_string(),
        "",
        "Dtypes:",
        raw_df.dtypes.to_string(),
    ]
    preview = "\n".join(preview_lines)

    user_message = (
        f"Here is the raw CSV I need cleaned.\n\n"
        f"File: {Path(csv_path).name}\n\n"
        f"```\n{preview}\n```\n\n"
        "Please map columns to the canonical schema, clean the data, and call `output_result`."
    )

    messages: list[dict] = [{"role": "user", "content": user_message}]

    # Result placeholders
    result_df: pd.DataFrame | None = None
    quality_report: dict = {}
    briefing: str = ""

    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Agent finished without calling output_result — extract text
            for block in response.content:
                if hasattr(block, "text"):
                    briefing = block.text
            break

        if response.stop_reason != "tool_use":
            break

        # Process tool calls
        tool_results = []
        done = False

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "execute_code":
                code = block.input.get("code", "")
                output = _exec_code(code, exec_context)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )

            elif block.name == "output_result":
                try:
                    df_json = block.input.get("df_json", "[]")
                    records = json.loads(df_json)
                    result_df = pd.DataFrame(records)
                    if "datetime" in result_df.columns:
                        result_df["datetime"] = pd.to_datetime(result_df["datetime"])
                except Exception as exc:
                    result_df = exec_context.get("df", pd.DataFrame())
                    briefing += f"\n[JSON parse error: {exc}]"

                quality_report = block.input.get("quality_report", {})
                briefing = block.input.get("briefing", briefing)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Done.",
                    }
                )
                done = True

        # Feed tool results back
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if done:
            break

    # Fallback: use whatever is in exec_context if agent never called output_result
    if result_df is None:
        result_df = exec_context.get("df", raw_df)
        if not quality_report:
            quality_report = {"issues": ["Agent did not call output_result; returning last DataFrame state."]}
        if not briefing:
            briefing = "Preprocessing completed (no explicit briefing from agent)."

    return result_df, quality_report, briefing
