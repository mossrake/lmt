#!/usr/bin/env python3
"""
Language Model Tectonics(TM) — LLM Drift Detection Tool

Detects behavioral drift in language models by comparing daily responses
to a stored baseline using Jaccard distance, then charts the RMS drift
score over a rolling 30-day window.

Usage:
    python tectonics.py run                 # daily run + generate report
    python tectonics.py run --rebaseline    # force new baseline first
    python tectonics.py report              # regenerate chart from stored data
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import to_rgba

__version__ = "0.3.2-beta"

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "tectonics.db"
CONFIG_PATH = SCRIPT_DIR / "config.json"
LOCAL_CONFIG_PATH = SCRIPT_DIR / "config.local.json"
PROMPTS_PATH = SCRIPT_DIR / "prompts.json"
OUTPUT_PATH = SCRIPT_DIR / "tectonics_report.png"
EVIDENCE_PATH = SCRIPT_DIR / "lmt_evidence.json"
SUITE_NAME = "default"
LOOKBACK_DAYS = 30

# Default environment variable per provider, used when config gives no key.
DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
}

# ---------------------------------------------------------------------------
# Provider call functions
# ---------------------------------------------------------------------------

def call_openai(api_key, mcfg, prompt, max_tokens=2048, system_prompt=None):
    """Call direct OpenAI chat completion. Returns (response_text, resolved_version)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
        + [{"role": "user", "content": prompt}]
    kwargs = dict(
        model=mcfg["family"],
        messages=messages,
        max_tokens=max_tokens,
    )
    temp = mcfg.get("temperature", 0)
    if temp is not None:
        kwargs["temperature"] = temp
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    version = resp.model
    return text, version


def call_azure_openai(api_key, mcfg, prompt, max_tokens=2048, system_prompt=None):
    """Call Azure OpenAI chat completion. Returns (response_text, resolved_version)."""
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=mcfg["endpoint"],
        api_key=api_key,
        api_version=mcfg.get("api_version", "2024-10-21"),
    )
    deployment = mcfg.get("deployment", mcfg["family"])
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
        + [{"role": "user", "content": prompt}]
    kwargs = dict(
        model=deployment,
        messages=messages,
        max_tokens=max_tokens,
    )
    temp = mcfg.get("temperature", 0)
    if temp is not None:
        kwargs["temperature"] = temp
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    # Azure returns deployment name in resp.model; system_fingerprint
    # may carry the actual revision — capture both.
    version = resp.model
    if hasattr(resp, "system_fingerprint") and resp.system_fingerprint:
        version = f"{resp.model}@{resp.system_fingerprint}"
    return text, version


def call_anthropic(api_key, mcfg, prompt, max_tokens=2048, system_prompt=None):
    """Call Anthropic messages API. Returns (response_text, resolved_version)."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    kwargs = dict(
        model=mcfg["family"],
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_prompt:
        kwargs["system"] = system_prompt
    temp = mcfg.get("temperature", 0)
    if temp is not None:
        kwargs["temperature"] = temp
    resp = client.messages.create(**kwargs)
    text = "\n".join(
        block.text for block in resp.content if hasattr(block, "text")
    )
    version = resp.model
    return text, version


def call_openai_compatible(api_key, mcfg, prompt, max_tokens=2048, system_prompt=None):
    """
    Call any OpenAI-compatible endpoint (Ollama, vLLM, DeepSeek, Together,
    Groq, LiteLLM, etc.). Requires 'endpoint' in model config.
    Returns (response_text, resolved_version).
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key or "unused",
        base_url=mcfg["endpoint"],
    )
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
        + [{"role": "user", "content": prompt}]
    kwargs = dict(
        model=mcfg.get("deployment", mcfg["family"]),
        messages=messages,
        max_tokens=max_tokens,
    )
    temp = mcfg.get("temperature", 0)
    if temp is not None:
        kwargs["temperature"] = temp
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    version = resp.model or mcfg["family"]
    if hasattr(resp, "system_fingerprint") and resp.system_fingerprint:
        version = f"{version}@{resp.system_fingerprint}"
    return text, version


PROVIDERS = {
    "openai": call_openai,
    "azure_openai": call_azure_openai,
    "anthropic": call_anthropic,
    "openai_compatible": call_openai_compatible,
}

# ---------------------------------------------------------------------------
# Jaccard distance & aggregation
# ---------------------------------------------------------------------------

def tokenize(text):
    """Word-level tokenization: lowercase, split on whitespace."""
    return set(text.lower().split())


def jaccard_distance(text1, text2):
    """1 - Jaccard similarity between two texts (word-level)."""
    s1, s2 = tokenize(text1), tokenize(text2)
    union = s1 | s2
    if not union:
        return 0.0
    return 1.0 - len(s1 & s2) / len(union)


def rms(values):
    """Root-mean-square of a list of floats."""
    if not values:
        return 0.0
    return math.sqrt(sum(v ** 2 for v in values) / len(values))

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS baseline_meta (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT NOT NULL,
            prompt_hash     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS baseline_responses (
            baseline_id     INTEGER NOT NULL,
            model_key       TEXT NOT NULL,
            prompt_id       TEXT NOT NULL,
            response        TEXT NOT NULL,
            version         TEXT,
            baselined_at    TEXT NOT NULL DEFAULT '',
            run_index       INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (baseline_id) REFERENCES baseline_meta(id)
        );

        CREATE TABLE IF NOT EXISTS run_responses (
            run_date        TEXT NOT NULL,
            model_key       TEXT NOT NULL,
            prompt_id       TEXT NOT NULL,
            response        TEXT NOT NULL,
            version         TEXT,
            jaccard_dist    REAL NOT NULL,
            excess          REAL
        );

        CREATE INDEX IF NOT EXISTS idx_run_date
            ON run_responses(run_date);
        CREATE INDEX IF NOT EXISTS idx_baseline_model
            ON baseline_responses(baseline_id, model_key);
    """)
    # Migration for databases created by v0.1 (columns won't exist there).
    for stmt in (
        "ALTER TABLE baseline_responses ADD COLUMN run_index INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE run_responses ADD COLUMN excess REAL",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


def get_active_baseline(conn):
    """Return (id, prompt_hash) of the most recent baseline, or None."""
    row = conn.execute(
        "SELECT id, prompt_hash FROM baseline_meta ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row


def get_baseline_responses(conn, baseline_id, model_key):
    """Return {prompt_id: [response_text, ...]} — all baseline runs per prompt."""
    rows = conn.execute(
        "SELECT prompt_id, response FROM baseline_responses "
        "WHERE baseline_id = ? AND model_key = ? ORDER BY run_index",
        (baseline_id, model_key),
    ).fetchall()
    out = defaultdict(list)
    for pid, text in rows:
        out[pid].append(text)
    return dict(out)


def noise_band(baseline_texts):
    """
    Per-prompt noise band: the mean all-pairs Jaccard distance among a
    prompt's baseline runs. This is the endpoint's self-similarity under its
    own stochasticity — the floor below which "drift" is indistinguishable
    from noise. With a single baseline run the band is 0 (v0.1 behavior).
    """
    if len(baseline_texts) < 2:
        return 0.0
    dists = [
        jaccard_distance(a, b)
        for a, b in itertools.combinations(baseline_texts, 2)
    ]
    return sum(dists) / len(dists)


def cross_distance(text, baseline_texts):
    """Mean Jaccard distance from today's response to each baseline run."""
    if not baseline_texts:
        return 0.0
    return sum(jaccard_distance(text, b) for b in baseline_texts) / len(baseline_texts)


def get_baseline_date(conn, baseline_id):
    """Return the creation date string (YYYY-MM-DD) for a baseline."""
    row = conn.execute(
        "SELECT created_at FROM baseline_meta WHERE id = ?", (baseline_id,)
    ).fetchone()
    return row[0][:10] if row else None


def model_has_baseline(conn, baseline_id, model_key):
    """Check whether a model has any responses stored under this baseline."""
    row = conn.execute(
        "SELECT COUNT(*) FROM baseline_responses "
        "WHERE baseline_id = ? AND model_key = ?",
        (baseline_id, model_key),
    ).fetchone()
    return row[0] > 0


def model_has_run_today(conn, model_key, today):
    """Check whether a model already has run data for today."""
    row = conn.execute(
        "SELECT COUNT(*) FROM run_responses "
        "WHERE run_date = ? AND model_key = ?",
        (today, model_key),
    ).fetchone()
    return row[0] > 0

# ---------------------------------------------------------------------------
# Prompt hashing
# ---------------------------------------------------------------------------

def hash_prompts(prompts):
    """
    SHA-256 prefix of the canonical JSON of the prompt library.

    Empty/absent system prompts are excluded from the hash, so a legacy
    list-form prompts.json hashes identically to v0.3.0 — upgrading does not
    invalidate an existing baseline. A NON-empty system prompt is part of
    the probe stimulus and does change the hash (rebaselining event).
    """
    semantic = [
        {k: v for k, v in p.items() if not (k == "system_prompt" and not v)}
        for p in prompts
    ]
    canonical = json.dumps(semantic, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Config & prompts
# ---------------------------------------------------------------------------

def resolve_api_keys(config):
    """
    Resolve each api_keys entry to an actual secret.

    Resolution order per provider:
      1. Value of the form "env:VARNAME"  -> read from that environment variable
      2. Empty string or missing entry    -> read from the provider's default
         environment variable (see DEFAULT_KEY_ENV)
      3. Any other literal value          -> used as-is (discouraged; keep
         literals only in git-ignored config.local.json, never config.json)
    """
    resolved = {}
    keys = config.get("api_keys", {})
    providers_in_use = {m["provider"] for m in config.get("models", [])}
    for provider in providers_in_use | set(keys):
        raw = keys.get(provider, "")
        if isinstance(raw, str) and raw.startswith("env:"):
            resolved[provider] = os.environ.get(raw[4:], "")
        elif not raw:
            resolved[provider] = os.environ.get(
                DEFAULT_KEY_ENV.get(provider, ""), ""
            )
        else:
            resolved[provider] = raw
            print(
                f"WARNING: literal API key found in config for '{provider}'. "
                f"Prefer \"env:VARNAME\" references or config.local.json "
                f"(git-ignored) so secrets never enter version control."
            )
    config["api_keys"] = resolved
    return config


def load_config():
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found.")
        print("Copy config.example.json to config.json and edit it.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    # Optional local overlay (git-ignored): shallow-merge top-level keys;
    # api_keys and models are replaced wholesale if present.
    if LOCAL_CONFIG_PATH.exists():
        with open(LOCAL_CONFIG_PATH) as f:
            local = json.load(f)
        for k, v in local.items():
            if k == "api_keys" and isinstance(v, dict):
                config.setdefault("api_keys", {}).update(v)
            else:
                config[k] = v
    return resolve_api_keys(config)


def load_prompts():
    """
    Load the probe library. Two accepted forms:

      Legacy list:   [ {"id": ..., "prompt": ...}, ... ]
      Object form:   { "system_prompt": "...",          # optional default
                       "prompts": [ {"id": ..., "prompt": ...,
                                     "system_prompt": "..."   # optional override
                                    }, ... ] }

    Returns a normalized list where each prompt carries its resolved
    "system_prompt" (or None). The system prompt is part of the probe
    stimulus: probe what you deploy. Because the baseline hash covers this
    file's content, changing any prompt OR system prompt is a rebaselining
    event — the next run will re-baseline automatically.
    """
    if not PROMPTS_PATH.exists():
        print(f"ERROR: {PROMPTS_PATH} not found.")
        sys.exit(1)
    with open(PROMPTS_PATH) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        default_sp = raw.get("system_prompt") or None
        prompts = raw.get("prompts", [])
    else:
        default_sp = None
        prompts = raw
    if not prompts:
        print("ERROR: prompts.json contains no prompts.")
        sys.exit(1)
    for p in prompts:
        p["system_prompt"] = p.get("system_prompt") or default_sp
    return prompts


def model_key(m):
    """Stable string key for a model config entry."""
    return f"{m['provider']}/{m['family']}"

# ---------------------------------------------------------------------------
# Validation — cheap ping to verify each model name before a full run
# ---------------------------------------------------------------------------

def validate_models(config):
    """
    Send a trivial prompt to each model to verify the name resolves.
    Returns list of model configs that passed, prints warnings for failures.
    """
    api_keys = config["api_keys"]
    valid = []

    print("Validating model endpoints...")
    for mcfg in config["models"]:
        provider = mcfg["provider"]
        family = mcfg["family"]
        mk = model_key(mcfg)
        api_key = api_keys.get(provider)

        if not api_key:
            print(f"  SKIP  {mk}: no API key for '{provider}'")
            continue

        call_fn = PROVIDERS.get(provider)
        if not call_fn:
            print(f"  SKIP  {mk}: unsupported provider '{provider}'")
            continue

        print(f"  PING  {mk} ... ", end="", flush=True)
        try:
            _, version = call_fn(api_key, mcfg, "Respond with only: OK", max_tokens=16)
            print(f"OK  [{version}]")
            valid.append(mcfg)
        except Exception as exc:
            print(f"FAILED — {exc}")
            print(f"         Check that '{family}' is a valid model name for {provider}.")

    if not valid:
        print("\nERROR: No models passed validation. Check config.json.")
        sys.exit(1)

    print(f"\n{len(valid)}/{len(config['models'])} models validated.\n")
    return valid

# ---------------------------------------------------------------------------
# API execution — run prompts against a single model
# ---------------------------------------------------------------------------

def run_prompts_for_model(api_key, call_fn, mcfg, mk, prompts):
    """
    Send all prompts to one model.
    Returns [(prompt_id, response_text, version), ...].
    """
    results = []
    for p in prompts:
        pid = p["id"]
        print(f"  {mk}  <-  {pid} ... ", end="", flush=True)
        try:
            text, version = call_fn(
                api_key, mcfg, p["prompt"],
                system_prompt=p.get("system_prompt"),
            )
            results.append((pid, text, version))
            print(f"OK  [{version}]")
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append((pid, "", "error"))
    return results

# ---------------------------------------------------------------------------
# Per-model baselining
# ---------------------------------------------------------------------------

def baseline_model(conn, baseline_id, mk, mcfg, prompts, config):
    """
    Baseline a single model: run all prompts k times (config "baseline_runs",
    default 3), store every run under the given baseline_id, and record
    drift=0 for today. Multiple runs establish the per-prompt noise band —
    the endpoint's self-similarity under its own stochasticity.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()
    k = max(1, int(config.get("baseline_runs", 3)))

    provider = mcfg["provider"]
    api_key = config["api_keys"][provider]
    call_fn = PROVIDERS[provider]

    print(f"\n  Baselining {mk} ({k} run{'s' if k != 1 else ''}) ...")
    for run_index in range(k):
        if k > 1:
            print(f"  -- baseline run {run_index + 1}/{k}")
        results = run_prompts_for_model(api_key, call_fn, mcfg, mk, prompts)
        for prompt_id, text, version in results:
            conn.execute(
                "INSERT INTO baseline_responses "
                "(baseline_id, model_key, prompt_id, response, version, "
                " baselined_at, run_index) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (baseline_id, mk, prompt_id, text, version, now_iso, run_index),
            )
            if run_index == 0:
                conn.execute(
                    "INSERT INTO run_responses "
                    "(run_date, model_key, prompt_id, response, version, "
                    " jaccard_dist, excess) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (today, mk, prompt_id, text, version, 0.0, 0.0),
                )

    conn.commit()
    print(f"  Baseline stored for {mk}.")

# ---------------------------------------------------------------------------
# Core commands
# ---------------------------------------------------------------------------

def create_full_baseline(conn, prompts, config, valid_models):
    """
    Create a new baseline_meta entry and baseline every validated model.
    Used on first run, --rebaseline, or prompt-hash change.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt_hash = hash_prompts(prompts)

    print("=" * 60)
    print("CREATING NEW BASELINE")
    print("=" * 60)

    cursor = conn.execute(
        "INSERT INTO baseline_meta (created_at, prompt_hash) VALUES (?, ?)",
        (datetime.now(timezone.utc).isoformat(), prompt_hash),
    )
    baseline_id = cursor.lastrowid

    # Clear any existing run data for today
    conn.execute("DELETE FROM run_responses WHERE run_date = ?", (today,))
    conn.commit()

    for mcfg in valid_models:
        mk = model_key(mcfg)
        baseline_model(conn, baseline_id, mk, mcfg, prompts, config)

    print(f"\nBaseline #{baseline_id} created  (hash: {prompt_hash})")
    return baseline_id


def rebaseline_single(conn, baseline_id, mk, mcfg, prompts, config):
    """
    Re-arm ONE model: replace its baseline rows under the CURRENT baseline
    set with fresh k-run responses (new baselined_at, new noise band) and
    reset today's reading to drift=0. Other models' baselines, dates, and
    history are untouched. This is the post-incident acceptance step: a
    human has reviewed a change and declared the new behavior the new
    normal. It is never done automatically.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn.execute(
        "DELETE FROM baseline_responses WHERE baseline_id = ? AND model_key = ?",
        (baseline_id, mk),
    )
    conn.execute(
        "DELETE FROM run_responses WHERE model_key = ? AND run_date = ?",
        (mk, today),
    )
    conn.commit()
    baseline_model(conn, baseline_id, mk, mcfg, prompts, config)


def daily_run(conn, baseline_id, prompts, config, valid_models):
    """
    For each model: if it has no baseline, baseline it on the fly.
    Otherwise compare today's responses to the baseline.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"DAILY RUN — {today}")
    print("=" * 60)

    for mcfg in valid_models:
        mk = model_key(mcfg)
        provider = mcfg["provider"]

        # --- Per-model baseline check ---
        if not model_has_baseline(conn, baseline_id, mk):
            print(f"\n  New model detected: {mk} — baselining now.")
            baseline_model(conn, baseline_id, mk, mcfg, prompts, config)
            continue  # drift=0 already recorded; nothing to compare yet

        # --- Skip if already ran today ---
        if model_has_run_today(conn, mk, today):
            print(f"\n  {mk}: already ran today, skipping.")
            continue

        # --- Run and compare ---
        api_key = config["api_keys"][provider]
        call_fn = PROVIDERS[provider]
        baseline_map = get_baseline_responses(conn, baseline_id, mk)
        bands = {pid: noise_band(texts) for pid, texts in baseline_map.items()}

        print(f"\n  Comparing {mk} to baseline ...")
        results = run_prompts_for_model(
            api_key, call_fn, mcfg, mk, prompts
        )

        for prompt_id, text, version in results:
            base_texts = baseline_map.get(prompt_id)
            if not base_texts:
                print(f"    WARNING: no baseline for prompt '{prompt_id}', skipping.")
                continue
            dist = cross_distance(text, base_texts)
            excess = max(0.0, dist - bands.get(prompt_id, 0.0))
            conn.execute(
                "INSERT INTO run_responses "
                "(run_date, model_key, prompt_id, response, version, "
                " jaccard_dist, excess) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (today, mk, prompt_id, text, version, dist, excess),
            )

        conn.commit()

    print(f"\nDaily run complete for {today}.")

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(conn, config):
    """
    Render the report as a status grid: one row per model (problems first),
    one value-bearing cell per day colored by zone, per-model baseline date,
    noise band, version-change markers, and a resolved-versions table.

    The grid is rendered FROM the same compute_status() data that
    `status --json` emits — it is the reference consumption of the feed.
    Days shown = config "report_days" (default 15).
    """
    zones_cfg = config.get("drift_zones", DEFAULT_ZONES)
    cons = consequence_join(config)
    days_n = int(config.get("report_days", 15))
    status = compute_status(conn, config, cons, days=days_n)
    if not status:
        print("No run data in the lookback window; nothing to plot.")
        return

    zone_keys = sorted(
        [k for k, v in zones_cfg.items() if "below" in v],
        key=lambda k: zones_cfg[k]["below"],
    ) + [k for k, v in zones_cfg.items() if "above" in v]
    zrank = {k: i for i, k in enumerate(zone_keys)}

    status.sort(key=lambda s: (
        -zrank.get(s["zone"], 0),
        -(s["max_consequence"] or 0),
        s["model"],
    ))

    today = datetime.now(timezone.utc).date()
    day_list = [(today - timedelta(days=days_n - 1 - d)) for d in range(days_n)]
    day_keys = [d.isoformat() for d in day_list]

    def cell_fill(zkey):
        if zrank.get(zkey, 0) == 0:
            return "#ffffff"
        return to_rgba(zones_cfg.get(zkey, {}).get("color", "#888888"), 0.30)

    def fmt(v):
        s = f"{v:.3f}"
        return s[1:] if s.startswith("0") else s

    n = len(status)
    row_h = 0.5
    table_h = 0.34 * (n + 1) + 0.8
    fig_h = 2.0 + n * row_h + 1.1 + table_h
    fig, ax = plt.subplots(figsize=(15.5, fig_h))
    grid_top = n + 2.0
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.6 - table_h / row_h, grid_top)
    ax.axis("off")

    x_model, x_q1, x_base, x_band = 0.005, 0.208, 0.235, 0.302
    x_grid0, x_grid1 = 0.350, 0.992
    cw = (x_grid1 - x_grid0) / days_n

    fig.suptitle("Language Model Tectonics\u2122  (\u03b2)",
                 fontsize=16, fontweight="bold", y=0.985)
    suite_tag = f"suite: {SUITE_NAME}  \u2022  " if SUITE_NAME != "default" else ""
    ax.text(0.5, n + 1.62,
            f"{suite_tag}{days_n}-day drift status grid  \u2022  v{__version__}  \u2022  "
            f"Generated {today.isoformat()}  \u2022  "
            f"values are RMS excess over each model's noise band",
            fontsize=9.5, color="#555", ha="center")

    hy = n + 0.75
    for x, lab in [(x_model, "MODEL"), (x_q1, "Q1"), (x_base, "BASELINED"),
                   (x_band, "BAND")]:
        ax.text(x, hy, lab, fontsize=8.5, fontweight="bold",
                color="#666", va="center")
    shown_month = None
    for d, day in enumerate(day_list):
        ax.text(x_grid0 + (d + 0.5) * cw, hy, day.strftime("%d"),
                fontsize=7.5, color="#666", ha="center", va="center")
        if day.month != shown_month:
            ax.text(x_grid0 + (d + 0.5) * cw, hy + 0.55, day.strftime("%b"),
                    fontsize=7.5, color="#999", ha="center")
            shown_month = day.month

    from matplotlib.patches import Rectangle
    for i, s in enumerate(status):
        y = n - i - 1
        emph = zrank.get(s["zone"], 0) > 0
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y), 1, 1, facecolor="#fafafa",
                                   edgecolor="none", zorder=0))
        ax.text(x_model, y + 0.5, s["model"], fontsize=8.6, va="center",
                fontweight="bold" if emph else "normal")
        c = s["max_consequence"]
        ax.text(x_q1, y + 0.5, str(c) if c else "\u2013",
                fontsize=9.5, va="center")
        bdate = s["baseline_date"] or "?"
        young = bdate != "?" and bdate > day_keys[0]
        ax.text(x_base, y + 0.5, bdate[5:] if bdate != "?" else "?",
                fontsize=8.5, family="monospace",
                color="#b45309" if young else "#666", va="center")
        band = s["noise_band_rms"]
        ax.text(x_band, y + 0.5, fmt(band) if band is not None else "\u2013",
                fontsize=8.5, family="monospace", color="#888", va="center")

        hist = {h["date"]: h for h in s["history"]}
        changes = {v["date"] for v in s["version_changes"]}
        for d, dk in enumerate(day_keys):
            x0 = x_grid0 + d * cw
            h = hist.get(dk)
            if h is None:
                ax.add_patch(Rectangle((x0, y + 0.1), cw * 0.94, 0.8,
                             facecolor="#f3f4f6", edgecolor="#e5e7eb",
                             linewidth=0.5, zorder=2))
                ax.text(x0 + cw * 0.47, y + 0.5, "\u2013", fontsize=7,
                        color="#bbb", ha="center", va="center", zorder=3)
                continue
            zk = h["zone"]
            ax.add_patch(Rectangle((x0, y + 0.1), cw * 0.94, 0.8,
                         facecolor=cell_fill(zk), edgecolor="#e5e7eb",
                         linewidth=0.5, zorder=2))
            ax.text(x0 + cw * 0.47, y + 0.5, fmt(h["rms_excess"]),
                    fontsize=6.9, family="monospace",
                    color="#111" if zrank.get(zk, 0) > 0 else "#666",
                    ha="center", va="center", zorder=3)
            if dk in changes:
                ax.plot([x0 + cw * 0.47], [y + 0.955], marker="v",
                        markersize=4.5, color="#2563eb", zorder=4)

    # Legend
    ly = -1.0
    lx = 0.29
    for k in zone_keys:
        zc = zones_cfg[k]
        lab = zc.get("label", k)
        if "below" in zc:
            lab += f" (< {zc['below']:.2f})".replace("0.", ".")
        else:
            lab += f" (\u2265 {zc.get('above', 0):.2f})".replace("0.", ".")
        ax.add_patch(Rectangle((lx, ly + 0.1), 0.013, 0.5,
                     facecolor=cell_fill(k), edgecolor="#9ca3af",
                     linewidth=0.6))
        ax.text(lx + 0.018, ly + 0.35, lab, fontsize=8.5, va="center")
        lx += 0.02 + 0.008 * len(lab)
    ax.plot([lx + 0.01], [ly + 0.35], marker="v", markersize=4.5,
            color="#2563eb")
    ax.text(lx + 0.02, ly + 0.35, "version changed", fontsize=8.5,
            va="center")
    ax.text(lx + 0.135, ly + 0.35, "\u2013  no data / before baseline",
            fontsize=8.5, color="#888", va="center")
    ax.text(0.005, ly + 0.35, "BAND = model noise floor", fontsize=8,
            color="#888", va="center")

    # Resolved-versions table
    ty = ly - 1.1
    ax.text(0.005, ty, "Resolved Model Versions  (baseline, and dates with a change)",
            fontsize=10.5, fontweight="bold", va="center")
    change_dates = sorted({v["date"] for s in status for v in s["version_changes"]})
    baseline = get_active_baseline(conn)
    col_x = [0.16] + [0.16 + 0.28 + i * 0.20 for i in range(len(change_dates))]
    ax.text(0.16, ty - 0.7, "Baseline", fontsize=8.5, fontweight="bold",
            color="#666", va="center")
    for j, cd in enumerate(change_dates):
        ax.text(col_x[j + 1] if j + 1 < len(col_x) else 0.9, ty - 0.7, cd,
                fontsize=8.5, fontweight="bold", color="#666", va="center")
    for i, s in enumerate(status):
        ry = ty - 1.3 - i * 0.62
        ax.text(0.005, ry, s["model"].split("/", 1)[-1], fontsize=8.5,
                va="center")
        base_ver = ""
        if baseline:
            row = conn.execute(
                "SELECT version FROM baseline_responses WHERE baseline_id = ? "
                "AND model_key = ? AND version IS NOT NULL LIMIT 1",
                (baseline[0], s["model"]),
            ).fetchone()
            base_ver = row[0] if row else ""
        ax.text(0.16, ry, base_ver, fontsize=8, family="monospace",
                color="#444", va="center")
        cmap = {v["date"]: v["version"] for v in s["version_changes"]}
        for j, cd in enumerate(change_dates):
            if cd in cmap:
                ax.text(col_x[j + 1] if j + 1 < len(col_x) else 0.9, ry,
                        cmap[cd], fontsize=8, family="monospace",
                        color="#1d4ed8", va="center")

    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nReport saved -> {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Drift zones
# ---------------------------------------------------------------------------

DEFAULT_ZONES = {
    "stable":      {"below": 0.05, "label": "Stable",            "color": "#22c55e"},
    "drift":       {"below": 0.15, "label": "Drift Detected",    "color": "#eab308"},
    "significant": {"above": 0.15, "label": "Significant Shift", "color": "#ef4444"},
}


def zone_for(score, config):
    """Return (zone_key, label) for a drift score under the configured zones."""
    zones = config.get("drift_zones", DEFAULT_ZONES)
    belows = sorted(
        [(v["below"], k, v.get("label", k)) for k, v in zones.items() if "below" in v]
    )
    for top, key, label in belows:
        if score < top:
            return key, label
    aboves = [(k, v.get("label", k)) for k, v in zones.items() if "above" in v]
    if aboves:
        return aboves[0]
    return "significant", "Significant Shift"


# ---------------------------------------------------------------------------
# Consequence join (Mossrake Language Model Diligence Coverage Manifests)
#
# Consequence enters LMT ONLY through lmd-spec Coverage Manifests — the
# assessed, human-authored artifacts that bind a process (and its Q1
# consequence level) to a model. LMT performs a read-only join and never
# stores or invents consequence itself. If no manifests are provided, all
# consequence-aware behavior is inert.
# ---------------------------------------------------------------------------

def _norm(s):
    import re as _re
    return _re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_manifests(manifests_dir):
    """Load lmd-spec Coverage Manifests from a directory of .json files."""
    out = []
    d = Path(manifests_dir)
    if not d.is_dir():
        print(f"WARNING: manifests directory not found: {d}")
        return out
    for p in sorted(d.glob("*.json")):
        try:
            with open(p) as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: skipping {p.name}: {e}")
            continue
        if "lmd_spec_version" not in m or "consequence" not in m or "subject" not in m:
            print(f"WARNING: skipping {p.name}: not a Coverage Manifest")
            continue
        level = m.get("consequence", {}).get("level")
        if not isinstance(level, int):
            print(f"WARNING: skipping {p.name}: missing consequence.level")
            continue
        out.append((p.name, m))
    return out


def consequence_join(config):
    """
    Join monitored models to consuming processes declared in Coverage
    Manifests. Returns {model_key: {"edges": [...], "max": int|None}} where
    each edge is {"process": str, "level": int, "manifest": filename}.
    """
    join = {model_key(m): {"edges": [], "max": None} for m in config["models"]}
    mdir = config.get("manifests_dir")
    if not mdir:
        return join
    manifests = load_manifests(mdir)
    for mcfg in config["models"]:
        mk = model_key(mcfg)
        fam, prov = _norm(mcfg.get("family")), _norm(mcfg.get("provider"))
        for fname, man in manifests:
            sm = man.get("subject", {}).get("model", {}) or {}
            mid, mprov = _norm(sm.get("model_id")), _norm(sm.get("provider"))
            if not mid or mid != fam:
                continue
            if mprov and prov and mprov != prov and mprov not in prov and prov not in mprov:
                continue
            desc = man["subject"].get("process_description", "").strip()
            join[mk]["edges"].append({
                "process": (desc[:117] + "...") if len(desc) > 120 else desc,
                "level": man["consequence"]["level"],
                "manifest": fname,
            })
        levels = [e["level"] for e in join[mk]["edges"]]
        join[mk]["max"] = max(levels) if levels else None
    return join


def compute_status(conn, config, cons, days=LOOKBACK_DAYS):
    """
    Per-model status for machine consumption: latest reading plus a
    `days`-long history (date, rms_excess, zone, version), the model's noise
    floor, its version changes in the window, and the date the model's OWN
    baseline was established (models added to the config later are baselined
    later; per-model dates make a young baseline visible to consumers).
    """
    baseline = get_active_baseline(conn)
    baseline_id = baseline[0] if baseline else None
    start = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    out = []
    for mcfg in config["models"]:
        mk = model_key(mcfg)
        rows = conn.execute(
            "SELECT run_date, COALESCE(excess, jaccard_dist), version "
            "FROM run_responses WHERE model_key = ? AND run_date >= ? "
            "ORDER BY run_date",
            (mk, start),
        ).fetchall()
        if not rows:
            continue
        by_date = defaultdict(list)
        versions = {}
        for ds, val, ver in rows:
            by_date[ds].append(val)
            if ver:
                versions[ds] = ver

        history, version_changes = [], []
        prev_ver = None
        for ds in sorted(by_date):
            score = rms(by_date[ds])
            zkey, _ = zone_for(score, config)
            ver = versions.get(ds)
            history.append({
                "date": ds,
                "rms_excess": round(score, 4),
                "zone": zkey,
                "version": ver,
            })
            if ver and prev_ver and ver != prev_ver:
                version_changes.append({"date": ds, "version": ver})
            if ver:
                prev_ver = ver

        model_baseline_date = None
        band_rms = None
        if baseline_id is not None:
            row = conn.execute(
                "SELECT MIN(baselined_at) FROM baseline_responses "
                "WHERE baseline_id = ? AND model_key = ?",
                (baseline_id, mk),
            ).fetchone()
            if row and row[0]:
                model_baseline_date = row[0][:10]
            base_map = get_baseline_responses(conn, baseline_id, mk)
            if base_map:
                band_rms = round(rms([noise_band(v) for v in base_map.values()]), 4)

        latest = history[-1]
        zkey, zlabel = zone_for(latest["rms_excess"], config)
        out.append({
            "model": mk,
            "latest_run": latest["date"],
            "rms_excess": latest["rms_excess"],
            "zone": zkey,
            "zone_label": zlabel,
            "version": latest["version"],
            "baseline_date": model_baseline_date,
            "noise_band_rms": band_rms,
            "max_consequence": cons.get(mk, {}).get("max"),
            "consuming_processes": cons.get(mk, {}).get("edges", []),
            "version_changes": version_changes,
            "history": history,
        })
    return out


def print_status(conn, config, as_json=False, days=LOOKBACK_DAYS):
    cons = consequence_join(config)
    status = compute_status(conn, config, cons, days=days)
    if as_json:
        print(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tool": f"Language Model Tectonics v{__version__}",
            "suite": SUITE_NAME,
            "lookback_days": days,
            "models": status,
        }, indent=2))
        return
    if not status:
        print("No run data yet.")
        return
    status.sort(key=lambda s: (-(s["max_consequence"] or 0), s["model"]))
    for s in status:
        c = s["max_consequence"]
        ctag = f"Q1={c}" if c else "Q1=?"
        print(f"{s['model']:<40} {s['rms_excess']:>7.4f}  {s['zone_label']:<18} "
              f"{ctag:<5}  {s['latest_run']}  "
              f"baselined {s['baseline_date'] or '?'}  {s['version'] or ''}")
        for e in s["consuming_processes"]:
            print(f"    L{e['level']}  {e['process']}  [{e['manifest']}]")


def check_alerts(conn, config, floor):
    """
    Alert when a model in a non-stable zone serves a process at or above the
    consequence floor. Models with no manifest coverage (unknown consequence)
    are treated as meeting the floor — unknown stakes are not assumed low.
    Returns the list of alerting statuses.
    """
    cons = consequence_join(config)
    alerts = []
    for s in compute_status(conn, config, cons):
        if s["zone"] == "stable":
            continue
        c = s["max_consequence"]
        if c is None or c >= floor:
            alerts.append(s)
    for s in alerts:
        c = s["max_consequence"]
        print(f"ALERT: {s['model']} is in zone '{s['zone_label']}' "
              f"(RMS excess {s['rms_excess']:.4f}) "
              f"serving {'unknown-consequence' if c is None else f'consequence-{c}'} "
              f"process(es) on {s['latest_run']}.")
    return alerts


# ---------------------------------------------------------------------------
# Evidence emission (Mossrake Language Model Diligence specification)
# ---------------------------------------------------------------------------

def emit_evidence(conn, config):
    """
    Emit one evidence item per monitored model, conformant to the evidence
    structure of the Mossrake Language Model Diligence specification
    (lmd-spec, Coverage Record / Coverage Manifest schemas). Suitable for
    attachment to the Drift (Q3) dimension of a Coverage Manifest.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    baseline = get_active_baseline(conn)
    if not baseline:
        print("No baseline exists — nothing to attest.")
        return
    baseline_id = baseline[0]
    baseline_date = get_baseline_date(conn, baseline_id)

    items = []
    for mcfg in config["models"]:
        mk = model_key(mcfg)
        rows = conn.execute(
            "SELECT run_date, COALESCE(excess, jaccard_dist) FROM run_responses "
            "WHERE model_key = ? ORDER BY run_date",
            (mk,),
        ).fetchall()
        if not rows:
            continue
        by_date = defaultdict(list)
        for ds, val in rows:
            by_date[ds].append(val)
        days = sorted(by_date)
        latest = days[-1]
        latest_rms = rms(by_date[latest])
        items.append({
            "type": "monitoring",
            "description": (
                f"Language Model Tectonics drift monitoring for {mk}"
                + (f" (suite: {SUITE_NAME})" if SUITE_NAME != "default" else "")
                + ": "
                f"{len(days)} daily runs since baseline {baseline_date}; "
                f"latest run {latest} RMS drift score "
                f"{latest_rms:.3f} (excess over baseline self-similarity band); "
                f"scheduled fixed-corpus behavioral comparison at pinned "
                f"parameters."
            ),
            "metrics": {
                "latest_rms_excess": round(latest_rms, 4),
                "days_monitored": len(days),
                "baseline_date": baseline_date,
                "latest_run_date": latest,
                "baseline_runs": max(1, int(config.get("baseline_runs", 3))),
                "prompt_count": len(by_date[latest]),
            },
            "source": f"Language Model Tectonics v{__version__}",
            "produced_at": now_iso,
        })

    with open(EVIDENCE_PATH, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Evidence for {len(items)} model(s) -> {EVIDENCE_PATH}")
    print(
        "Attach items to the 'evidence' array of the drift dimension in an "
        "LMD Coverage Manifest (see github.com/mossrake/lmd-spec)."
    )

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Language Model Tectonics \u2014 LLM drift detection",
    )
    sub = parser.add_subparsers(dest="command")

    def add_suite_args(p):
        p.add_argument(
            "--prompts", metavar="PATH", default=None,
            help="Prompt suite file to use (default: prompts.json next to "
                 "the script). One file per use case; the shipped library "
                 "is the endpoint-generic baseline suite",
        )
        p.add_argument(
            "--db", metavar="PATH", default=None,
            help="Database for this suite (default: derived from the suite "
                 "filename, e.g. invoice.json -> invoice.db; or "
                 "tectonics.db for the default suite). Report and evidence "
                 "filenames are derived the same way",
        )

    def add_manifests_arg(p):
        p.add_argument(
            "--manifests", metavar="DIR", default=None,
            help="Directory of lmd-spec Coverage Manifests (.json) declaring "
                 "which processes consume each model and at what consequence "
                 "level (overrides config \"manifests_dir\")",
        )

    run_p = sub.add_parser("run", help="Execute daily run and generate report")
    run_p.add_argument(
        "--rebaseline", action="store_true",
        help="Force a new baseline before running (all models, unless "
             "scoped with --model)",
    )
    run_p.add_argument(
        "--model", action="append", metavar="MODEL_KEY", default=None,
        help="Scope --rebaseline to one model (provider/family, e.g. "
             "'openai/gpt-4o'); repeatable. Use after a reviewed change is "
             "accepted as the new normal — other models' baselines are "
             "untouched",
    )
    run_p.add_argument(
        "--alert-floor", type=int, metavar="N", default=None,
        help="Exit nonzero (2) if any model in a non-stable zone serves a "
             "process at consequence level N or above; models with no "
             "manifest coverage are treated as meeting the floor",
    )
    add_manifests_arg(run_p)
    add_suite_args(run_p)

    report_p = sub.add_parser("report", help="Regenerate report from existing data")
    add_manifests_arg(report_p)
    add_suite_args(report_p)

    status_p = sub.add_parser(
        "status",
        help="Latest per-model reading joined with consuming-process consequence",
    )
    status_p.add_argument("--json", action="store_true", help="Machine-readable output")
    status_p.add_argument(
        "--days", type=int, default=LOOKBACK_DAYS, metavar="N",
        help=f"History window in days (default {LOOKBACK_DAYS})",
    )
    add_manifests_arg(status_p)
    add_suite_args(status_p)

    evidence_p = sub.add_parser(
        "evidence",
        help="Emit LMD-spec-conformant drift evidence (lmt_evidence.json)",
    )
    add_suite_args(evidence_p)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Suite resolution: a suite is a prompts file with its own database and
    # derived output filenames. Defaults are unchanged (single-suite install).
    global PROMPTS_PATH, DB_PATH, OUTPUT_PATH, EVIDENCE_PATH, SUITE_NAME
    if getattr(args, "prompts", None):
        PROMPTS_PATH = Path(args.prompts)
        stem = PROMPTS_PATH.stem
        SUITE_NAME = stem
        DB_PATH = PROMPTS_PATH.with_name(f"{stem}.db")
        OUTPUT_PATH = PROMPTS_PATH.with_name(f"{stem}_report.png")
        EVIDENCE_PATH = PROMPTS_PATH.with_name(f"{stem}_evidence.json")
    if getattr(args, "db", None):
        DB_PATH = Path(args.db)
        stem = DB_PATH.stem
        OUTPUT_PATH = DB_PATH.with_name(f"{stem}_report.png")
        EVIDENCE_PATH = DB_PATH.with_name(f"{stem}_evidence.json")

    config = load_config()
    if getattr(args, "manifests", None):
        config["manifests_dir"] = args.manifests
    prompts = load_prompts()
    prompt_hash = hash_prompts(prompts)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    try:
        if args.command == "run":
            # Validate all model endpoints first
            valid_models = validate_models(config)

            baseline = get_active_baseline(conn)
            scoped = bool(args.rebaseline and args.model)
            if args.model and not args.rebaseline:
                print("--model only scopes --rebaseline; ignoring.")
            if scoped and (baseline is None or baseline[1] != prompt_hash):
                print("No matching baseline set exists; --model scoping "
                      "requires an intact baseline. Doing a full baseline.")
                scoped = False
            need_full_baseline = (
                (args.rebaseline and not scoped)
                or baseline is None
                or baseline[1] != prompt_hash
            )

            if need_full_baseline:
                if baseline and baseline[1] != prompt_hash:
                    print("Prompt library changed — forcing re-baseline.\n")
                elif args.rebaseline:
                    print("Re-baseline requested.\n")
                else:
                    print("No existing baseline — creating one.\n")
                baseline_id = create_full_baseline(
                    conn, prompts, config, valid_models
                )
            else:
                baseline_id = baseline[0]
                bd = get_baseline_date(conn, baseline_id)
                print(f"Using baseline #{baseline_id} ({bd})\n")
                if scoped:
                    valid_keys = {model_key(m): m for m in valid_models}
                    targets = []
                    for mk in args.model:
                        if mk in valid_keys:
                            targets.append(mk)
                        else:
                            print(f"WARNING: --model '{mk}' not in config "
                                  f"(expected one of: {', '.join(valid_keys)})")
                    for mk in targets:
                        print(f"Re-arming baseline for {mk} only.")
                        rebaseline_single(conn, baseline_id, mk,
                                          valid_keys[mk], prompts, config)
                daily_run(conn, baseline_id, prompts, config, valid_models)

            generate_report(conn, config)

            if args.alert_floor is not None:
                if check_alerts(conn, config, args.alert_floor):
                    sys.exit(2)

        elif args.command == "report":
            generate_report(conn, config)

        elif args.command == "status":
            print_status(conn, config, as_json=args.json, days=args.days)

        elif args.command == "evidence":
            emit_evidence(conn, config)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
