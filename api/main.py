"""
DR Compass — FastAPI interface layer.

Three core endpoints:
  POST /upload          — upload a raw CSV; optionally preprocess via Claude agent
  POST /run_pipeline    — execute the full 6-step DR pipeline in a background task
  GET  /get_schedule    — return the latest optimized DR schedule

Supporting endpoints:
  GET  /pipeline/status/{job_id}   — poll background job status + logs
  GET  /summary                    — return the cost-savings summary table
  GET  /health                     — liveness check

Start the server:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import io
import json
import shutil
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DR Compass API",
    description="Day-ahead demand-response optimization pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# In-memory job store (replace with Redis for production)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}


def _new_job() -> str:
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "finished_at": None,
        "logs": [],
        "error": None,
    }
    return job_id


def _job_log(job_id: str, msg: str) -> None:
    if job_id in _jobs:
        _jobs[job_id]["logs"].append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")


def _job_done(job_id: str, error: str | None = None) -> None:
    if job_id in _jobs:
        _jobs[job_id]["status"] = "failed" if error else "completed"
        _jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()
        _jobs[job_id]["error"] = error


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    rows: int
    columns: list[str]
    saved_path: str
    preprocessing_status: Literal["skipped", "completed", "failed"]
    quality_report: dict | None = None
    briefing: str | None = None


class PipelineRequest(BaseModel):
    config_overrides: dict | None = None  # optional per-request config patches


class PipelineResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    finished_at: str | None
    logs: list[str]
    error: str | None


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline_sync(job_id: str, config_path: str, overrides: dict | None) -> None:
    """Blocking pipeline execution — run inside a thread pool via asyncio."""
    try:
        _jobs[job_id]["status"] = "running"

        # Apply any per-request config overrides to a temp config file
        cfg_path = Path(config_path)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if overrides:
            _deep_merge(cfg, overrides)
            tmp_cfg = UPLOAD_DIR / f"config_{job_id}.yaml"
            tmp_cfg.write_text(yaml.dump(cfg), encoding="utf-8")
            cfg_path = tmp_cfg

        # Import and run each step (imports are deferred so the server starts fast)
        from scripts.run_01_prepare_data import main as run_01
        from scripts.run_02_features_elasticity_dr import main as run_02
        from scripts.run_03_forecast_calibration_scenarios import main as run_03
        from scripts.run_04_day_ahead_decision import main as run_04
        from scripts.run_05_counterfactuals import main as run_05
        from scripts.run_06_evaluation_dashboard import main as run_06

        steps = [
            ("01 — prepare data",          run_01),
            ("02 — features & elasticity",  run_02),
            ("03 — forecast & scenarios",   run_03),
            ("04 — DR optimization",        run_04),
            ("05 — figures & counterfactuals", run_05),
            ("06 — evaluation dashboard",   run_06),
        ]
        for label, fn in steps:
            _job_log(job_id, f"Starting {label}")
            fn(str(cfg_path))
            _job_log(job_id, f"Completed {label}")

        # Cleanup temp config if created
        if overrides and tmp_cfg.exists():
            tmp_cfg.unlink()

        _job_done(job_id)
        _job_log(job_id, "Pipeline completed successfully.")

    except Exception as exc:
        err = traceback.format_exc()
        _job_done(job_id, error=str(exc))
        _job_log(job_id, f"FAILED: {exc}")


def _deep_merge(base: dict, patch: dict) -> None:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ── /upload ─────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    preprocess: bool = Query(
        default=False,
        description="Run Claude preprocessing agent to map columns to canonical schema",
    ),
) -> UploadResponse:
    """
    Upload a raw CSV file.

    - Saves to `data/uploads/<file_id>_<filename>`.
    - If `preprocess=true`, runs the Claude API preprocessing agent
      (requires ANTHROPIC_API_KEY in environment).
    - Returns file metadata + optional quality report and briefing.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{file.filename}"
    dest = UPLOAD_DIR / safe_name

    contents = await file.read()
    dest.write_bytes(contents)

    # Quick peek to return basic metadata
    try:
        raw_df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc

    preprocess_status: Literal["skipped", "completed", "failed"] = "skipped"
    quality_report: dict | None = None
    briefing: str | None = None

    if preprocess:
        try:
            from src.data_preprocessing_agent import preprocess_csv
            df_clean, quality_report, briefing = preprocess_csv(dest)
            clean_path = UPLOAD_DIR / f"{file_id}_{Path(file.filename).stem}_clean.csv"
            df_clean.to_csv(clean_path, index=False)
            preprocess_status = "completed"
        except Exception as exc:
            preprocess_status = "failed"
            briefing = f"Preprocessing failed: {exc}"

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        rows=len(raw_df),
        columns=list(raw_df.columns),
        saved_path=str(dest.relative_to(PROJECT_ROOT)),
        preprocessing_status=preprocess_status,
        quality_report=quality_report,
        briefing=briefing,
    )


# ── /run_pipeline ────────────────────────────────────────────────────────────

@app.post("/run_pipeline", response_model=PipelineResponse)
async def run_pipeline(
    body: PipelineRequest,
    background_tasks: BackgroundTasks,
) -> PipelineResponse:
    """
    Trigger the full 6-step DR pipeline as a background task.

    Optionally pass `config_overrides` to patch specific config values for this run:

    ```json
    {
      "config_overrides": {
        "data": {"country_code": "FR"},
        "decision": {"flexibility_ratios": [0.05, 0.10]}
      }
    }
    ```

    Returns a `job_id` — poll `/pipeline/status/{job_id}` for progress.
    The pipeline typically takes 90–180 seconds.
    """
    job_id = _new_job()
    background_tasks.add_task(
        asyncio.get_event_loop().run_in_executor,
        None,  # default thread pool
        _run_pipeline_sync,
        job_id,
        str(CONFIG_PATH),
        body.config_overrides,
    )
    return PipelineResponse(
        job_id=job_id,
        status="queued",
        message=f"Pipeline queued. Poll /pipeline/status/{job_id} for progress.",
    )


@app.get("/pipeline/status/{job_id}", response_model=JobStatusResponse)
def pipeline_status(job_id: str) -> JobStatusResponse:
    """Poll the status and logs of a running or completed pipeline job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatusResponse(job_id=job_id, **job)


# ── /get_schedule ─────────────────────────────────────────────────────────────

@app.get("/get_schedule")
def get_schedule(
    fmt: Literal["json", "csv"] = Query(default="json", description="Response format"),
) -> Any:
    """
    Return the latest optimized day-ahead DR schedule.

    Each row is one hour with columns:
    - `target_datetime` — the hour this schedule applies to
    - `q_DA_optimal`    — optimal DR quantity (MW reduction)
    - `price_p10/p50/p90` — scenario price quantiles
    - `action`          — human-readable label (reduce / hold / guardrail_triggered)

    Use `?fmt=csv` to get raw CSV text instead of JSON.
    """
    schedule_path = PROJECT_ROOT / "outputs" / "tables" / "day_ahead_optimal_dr_schedule.csv"
    if not schedule_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No schedule found. Run /run_pipeline first.",
        )

    df = pd.read_csv(schedule_path)

    if fmt == "csv":
        return JSONResponse(
            content={"csv": df.to_csv(index=False)},
            media_type="application/json",
        )

    # Enrich with a human-readable action label if not already present
    if "action" not in df.columns and "q_DA_optimal" in df.columns:
        df["action"] = df["q_DA_optimal"].apply(
            lambda q: "reduce" if q > 1e-3 else "hold"
        )

    # Convert datetime columns to ISO strings for JSON serialisation
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].dt.isoformat()

    return {"schedule": df.to_dict(orient="records"), "n_hours": len(df)}


# ── /summary ─────────────────────────────────────────────────────────────────

@app.get("/summary")
def summary() -> dict:
    """
    Return the cost-savings summary for the latest pipeline run.

    Fields include: avg_price_reduction, total_cost_reduction, constraints_satisfied, etc.
    """
    summary_path = PROJECT_ROOT / "outputs" / "tables" / "optimized_price_vs_observed_summary.csv"
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No summary found. Run /run_pipeline first.",
        )
    df = pd.read_csv(summary_path)
    return {"summary": df.to_dict(orient="records")[0]}
