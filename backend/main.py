"""
FairLens Dashboard - FastAPI Backend
=====================================
Run with: uvicorn main:app --reload --port 8000

CACHING SYSTEM
--------------
Results are saved to ./cache/<session_id>/
  - results.json  : full metrics + config
  - meta.json     : lightweight index entry (name, date, config summary)

A cache_key is computed from the dataset fingerprint + training config.
If a matching key exists, training is skipped and cached results are returned
instantly — no retraining needed.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid, json, hashlib, os, shutil, math
from datetime import datetime
import pandas as pd
import io
import traceback

from fairness_engine import FairnessEngine


def sanitize_floats(obj):
    """
    Recursively walk a dict/list and replace any float nan/inf/-inf with None.
    Python's json.dumps crashes on these; JSON spec has no NaN/Infinity literals.
    Caused by division-by-zero in DI when one demographic group has zero predictions.
    """
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

app = FastAPI(title="FairLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Storage paths ─────────────────────────────────────────────────────────────
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── In-memory stores ──────────────────────────────────────────────────────────
jobs: Dict[str, Any]        = {}   # active training jobs
datasets: Dict[str, Any]    = {}   # uploaded dataframes + fingerprints


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    dataset_id: str
    target_column: str
    protected_attribute: str
    privileged_value: Any
    favorable_value: Any = None        # which target value = positive class (auto-detect if None)
    protected_threshold: Any = None   # numeric threshold for continuous protected attributes
    feature_columns: List[str]
    models: List[str]
    mitigation_steps: List[str]
    smote_variants: List[str]
    test_size: float = 0.3
    session_name: str = ""        # optional human-readable label


# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_cache_key(df_fingerprint: str, config: TrainConfig) -> str:
    """Deterministic hash from dataset content + training config."""
    payload = {
        "fp":         df_fingerprint,
        "target":     config.target_column,
        "protected":  config.protected_attribute,
        "privileged": str(config.privileged_value),
        "features":   sorted(config.feature_columns),
        "models":     sorted(config.models),
        "mitigation": sorted(config.mitigation_steps),
        "smote":      sorted(config.smote_variants),
        "test_size":  config.test_size,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def df_fingerprint(df: pd.DataFrame) -> str:
    """Cheap fingerprint: shape + column names + first/last row hash."""
    sig = f"{df.shape}|{'|'.join(df.columns)}|{pd.util.hash_pandas_object(df.iloc[[0,-1]]).sum()}"
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def find_cached_session(cache_key: str) -> Optional[str]:
    """Return session_id if cache_key matches any saved session, else None."""
    for session_id in os.listdir(CACHE_DIR):
        meta_path = os.path.join(CACHE_DIR, session_id, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("cache_key") == cache_key:
                return session_id
    return None


def save_session(session_id: str, results: dict, config: TrainConfig,
                 cache_key: str, session_name: str):
    """Persist results + metadata to disk."""
    session_dir = os.path.join(CACHE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    with open(os.path.join(session_dir, "results.json"), "w") as f:
        json.dump(results, f)

    meta = {
        "session_id":   session_id,
        "session_name": session_name or f"Session {session_id[:6]}",
        "cache_key":    cache_key,
        "created_at":   datetime.now().isoformat(),
        "dataset_rows": results.get("baseline", {}).get("total_rows", "?"),
        "models":       config.models,
        "mitigation":   config.mitigation_steps,
        "target":       config.target_column,
        "protected":    config.protected_attribute,
        "best_model":   results.get("summary", {}).get("best_model", ""),
        "best_spd":     results.get("summary", {}).get("best_spd_after", None),
        "best_accuracy":results.get("summary", {}).get("best_accuracy", None),
    }
    with open(os.path.join(session_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def load_session_results(session_id: str) -> dict:
    path = os.path.join(CACHE_DIR, session_id, "results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Session {session_id} not found")
    with open(path) as f:
        return json.load(f)


def list_sessions() -> list:
    sessions = []
    for session_id in os.listdir(CACHE_DIR):
        meta_path = os.path.join(CACHE_DIR, session_id, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                sessions.append(json.load(f))
    return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "FairLens API is running"}


# ── Dataset upload ────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV and return column info + preview."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    content = await file.read()
    try:
        # Try UTF-8 first, fall back to latin-1 (handles Adult Census & most datasets)
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        df = pd.read_csv(
            io.StringIO(text),
            na_values=["?", " ?", "? ", "NA", "N/A", ""],
            skipinitialspace=True,   # strip spaces after commas
        )

        # If columns are unnamed (no header), assign generic names
        if all(str(c).startswith("Unnamed") for c in df.columns):
            raise HTTPException(400, "CSV has no header row. Please add column names as the first row.")

        # Strip whitespace from string columns (Adult Census has trailing spaces)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        # Note: missing values are handled by the engine via mode imputation (Step 2).
        # Dropping here would silently discard valid rows (e.g. Adult Census loses 3,620 rows).
        # Only raise if the entire dataframe is empty.
        if len(df) == 0:
            raise HTTPException(400, "CSV is empty. Please check the file.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {str(e)}")

    dataset_id  = str(uuid.uuid4())
    fingerprint = df_fingerprint(df)
    datasets[dataset_id] = {"df": df, "fingerprint": fingerprint, "filename": file.filename}

    columns_info = []
    for col in df.columns:
        unique_vals   = df[col].nunique()
        sample_values = df[col].dropna().unique()[:8].tolist()
        columns_info.append({
            "name":          col,
            "dtype":         str(df[col].dtype),
            "unique_count":  unique_vals,
            "sample_values": [str(v) for v in sample_values],
            "is_numeric":    pd.api.types.is_numeric_dtype(df[col]),
        })

    preview = df.head(5).fillna("").to_dict(orient="records")

    # Dataset size warnings — shown immediately after upload
    upload_warnings = []
    n = len(df)
    if n < 500:
        upload_warnings.append({
            "level": "error",
            "msg":   f"Only {n} rows detected. A minimum of 500 rows is required for reliable results."
        })
    elif n < 2000:
        upload_warnings.append({
            "level": "warning",
            "msg":   f"{n} rows detected. ADB (Adversarial Debiasing) requires at least 2,000 rows for reliable convergence. Consider using only Reweighing + CEO mitigation steps."
        })

    # Class imbalance check (rough — uses first binary-looking column)
    # Just a size warning here; exact imbalance checked after target is configured

    return {
        "dataset_id": dataset_id,
        "filename":   file.filename,
        "shape":      {"rows": len(df), "columns": len(df.columns)},
        "columns":    columns_info,
        "preview":    preview,
        "warnings":   upload_warnings,
    }


# ── Cache check (call before training to see if we can skip) ──────────────────

@app.post("/api/check-cache")
def check_cache(config: TrainConfig):
    """
    Returns {cached: true, session_id, meta} if an identical run exists,
    or {cached: false} if training is needed.
    """
    if config.dataset_id not in datasets:
        return {"cached": False}

    fp          = datasets[config.dataset_id]["fingerprint"]
    cache_key   = make_cache_key(fp, config)
    session_id  = find_cached_session(cache_key)

    if session_id:
        meta_path = os.path.join(CACHE_DIR, session_id, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        return {"cached": True, "session_id": session_id, "meta": meta}

    return {"cached": False}


# ── Training ──────────────────────────────────────────────────────────────────

@app.post("/api/train")
async def start_training(config: TrainConfig, background_tasks: BackgroundTasks):
    """Start a training job. Always trains fresh (cache check is done client-side)."""
    if config.dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found. Please re-upload.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status":       "queued",
        "progress":     0,
        "current_step": "Initialising...",
        "logs":         [],
        "results":      None,
        "session_id":   None,
        "error":        None,
    }

    ds = datasets[config.dataset_id]
    background_tasks.add_task(run_training_job, job_id, ds["df"], ds["fingerprint"], config)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "status":       job["status"],
        "progress":     job["progress"],
        "current_step": job["current_step"],
        "logs":         job["logs"][-20:],
        "session_id":   job.get("session_id"),
    }


@app.get("/api/results/{job_id}")
def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(400, f"Job not complete (status: {job['status']})")
    return job["results"]


# ── Session management ────────────────────────────────────────────────────────

@app.get("/api/sessions")
def get_sessions():
    """List all saved sessions (newest first)."""
    return list_sessions()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """Load full results for a saved session."""
    try:
        return load_session_results(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a saved session from disk."""
    session_dir = os.path.join(CACHE_DIR, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(404, "Session not found")
    shutil.rmtree(session_dir)
    return {"deleted": session_id}


@app.patch("/api/sessions/{session_id}/rename")
def rename_session(session_id: str, body: dict):
    """Rename a session."""
    meta_path = os.path.join(CACHE_DIR, session_id, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(404, "Session not found")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["session_name"] = body.get("name", meta["session_name"])
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND TRAINING JOB
# ─────────────────────────────────────────────────────────────────────────────

async def run_training_job(job_id: str, df: pd.DataFrame,
                            fingerprint: str, config: TrainConfig):
    job = jobs[job_id]

    def log(msg: str):
        job["logs"].append(msg)
        print(f"[Job {job_id[:8]}] {msg}")

    def update(step: str, progress: int):
        job["current_step"] = step
        job["progress"]     = progress
        log(step)

    try:
        job["status"] = "running"
        update("Preparing dataset...", 5)

        engine = FairnessEngine(
            df=df,
            target_column=config.target_column,
            protected_attribute=config.protected_attribute,
            privileged_value=config.privileged_value,
            favorable_value=config.favorable_value,
            protected_threshold=config.protected_threshold,
            feature_columns=config.feature_columns,
            test_size=config.test_size,
        )

        update("Computing baseline bias metrics...", 10)
        baseline = engine.compute_baseline_metrics()
        feature_importance = engine.compute_feature_importance()

        smote_results = {}
        if config.smote_variants:
            update("Running SMOTE experiments...", 20)
            smote_results = engine.run_smote_experiments(
                config.smote_variants,
                progress_callback=lambda s, p: update(s, 20 + int(p * 20))
            )

        mitigation_results = {}
        total_models = len(config.models)
        for i, model_name in enumerate(config.models):
            base_progress = 40 + int(i / total_models * 55)
            update(f"Training {model_name}...", base_progress)
            model_res = engine.run_model_pipeline(
                model_name=model_name,
                mitigation_steps=config.mitigation_steps,
                progress_callback=lambda s, p: update(
                    f"{model_name}: {s}", base_progress + int(p * 55 / total_models)
                )
            )
            mitigation_results[model_name] = model_res

        update("Saving results...", 98)

        results = {
            "config":              config.dict(),
            "baseline":            baseline,
            "feature_importance":  feature_importance,
            "smote_results":       smote_results,
            "mitigation_results":  mitigation_results,
            "summary":             engine.generate_summary(baseline, mitigation_results),
        }

        # Sanitize any nan/inf floats before JSON serialization
        # These arise from division-by-zero in DI when a group has zero predictions
        results = sanitize_floats(results)

        # ── Persist to disk cache ────────────────────────────────────────────
        session_id = str(uuid.uuid4())
        cache_key  = make_cache_key(fingerprint, config)
        # Session name: use user-provided name, then filename, then generic fallback
        sname = (config.session_name
                 or datasets.get(config.dataset_id, {}).get("filename", "")
                 or f"{config.protected_attribute} — {config.target_column}")
        save_session(session_id, results, config, cache_key, sname)

        job["results"]    = results
        job["session_id"] = session_id
        job["status"]     = "complete"
        job["progress"]   = 100
        job["current_step"] = "Done! Results saved."

    except Exception as e:
        job["status"]       = "failed"
        job["error"]        = str(e)
        job["current_step"] = f"Error: {str(e)}"
        log(traceback.format_exc())
