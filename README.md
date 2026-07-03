# Language Model Tectonics™ (β)

A daily drift monitor for language model endpoints. Detects silent behavioral
change by comparing responses against a stored multi-run baseline, scoring
only the movement that exceeds each endpoint's own noise.

**LMT detects and dates change in model endpoints. It does not rank, grade,
or judge them.**

## Why

Model providers update weights, swap versions behind family aliases, adjust
serving infrastructure, and retire models — often without notice. Gateways
and proxies can reroute traffic silently. If your process depends on a model
behaving a certain way, you want to know when that changes. This tool is the
seismograph.

## How it works

LLM outputs are stochastic even at temperature 0, so responses can't simply
be diffed against a baseline — a single-run comparison charts noise as drift.
LMT instead measures each endpoint's similarity *to itself* first:

1. **Baseline** — every prompt in your library runs **k times** (default 3)
   against every configured model. The mean all-pairs Jaccard distance among
   a prompt's baseline runs is its **noise band**: the endpoint's natural
   self-similarity under its own stochasticity.
2. **Daily run** — each prompt runs again. The response's mean Jaccard
   distance to the baseline runs is compared against the noise band. Only the
   **excess** over the band is scored as drift. Same model serving the
   endpoint → excess ≈ 0. A silent swap, reroute, or major update → excess
   jumps, usually dramatically.
3. **Report** — a status grid: one row per model (problems sorted first),
   one value-bearing cell per day colored by severity zone, each model's
   baseline date and noise band, version-change markers, and a
   resolved-versions table (LMT records `response.model` — and
   `system_fingerprint` where available — on every call, so
   provider-attested version changes are dated alongside behavioral ones).
   The grid is rendered from the same data `status --json` emits, so it is
   also the reference for consuming the feed in your own dashboard.

What this catches: silent model swaps, gateway substitution, endpoint
rerouting, major model updates, significant safety-layer or serving changes.
What it does not catch: capability regression that preserves wording style
("same voice, dumber answers"). For consequence-calibrated drift coverage,
see the Language Model Diligence framework below.

## Quick start

```bash
pip install -r requirements.txt

cp config.example.json config.json     # then edit
export OPENAI_API_KEY=...              # keys live in the environment
export ANTHROPIC_API_KEY=...

python tectonics.py run                # first run creates baseline + report
python tectonics.py run                # subsequent runs compare + report
python tectonics.py run --rebaseline   # force new baseline (all models)
python tectonics.py run --rebaseline --model openai/gpt-4o
                                       # re-arm ONE model after a reviewed
                                       # change is accepted as the new normal
python tectonics.py report             # regenerate chart from stored data
python tectonics.py evidence           # emit LMD-conformant drift evidence
```

Output: `tectonics_report.png` (and `lmt_evidence.json` from `evidence`).

Worked input and output samples — example Coverage Manifests, `status --json`
output, and `evidence` output — are in [`examples/`](examples/).

Run it daily from cron or a scheduled task. Changing `prompts.json` forces a
re-baseline automatically (the library is content-hashed).

## API keys

**Never put literal keys in `config.json` — it is for committing; secrets are
not.** Three supported patterns, in recommended order:

1. `"env:VARNAME"` references in `config.json` (see `config.example.json`)
2. Empty/missing entries fall back to standard variables:
   `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
   `OPENAI_COMPATIBLE_API_KEY`
3. A git-ignored `config.local.json` overlay for anything machine-local

`config.json`, `config.local.json`, and `tectonics.db` are all listed in
`.gitignore`. The database stores full model responses — treat it as
sensitive if your prompts are.

## Bring your own prompts

The included `prompts.json` is a 12-prompt demonstration library — enough to
see the tool work, underpowered for confident detection. The tool's power
comes from **your own private prompt library**:

- **Use prompts drawn from your actual workload.** Drift that matters is
  drift on *your* task.
- **Probe under your system prompt.** Real applications wrap the model in a
  system prompt, and that wrapper is usually the deployment's main bounding
  instrument — probing without it measures the model's stock personality,
  not your working environment. `prompts.json` accepts an object form with a
  library-wide `system_prompt` and optional per-prompt overrides:

  ```json
  {
    "system_prompt": "You are the support-draft assistant. Respond only with ...",
    "prompts": [
      {"id": "p01", "prompt": "..."},
      {"id": "p02", "prompt": "...", "system_prompt": "Numbers only."}
    ]
  }
  ```

  The plain list form still works (no system prompt). System prompts are
  part of the probe stimulus and often as sensitive as the probes — the same
  keep-it-private guidance applies.
- **Changing any prompt or system prompt is a rebaselining event.** Scores
  are only comparable against an identical stimulus. The library is
  content-hashed, so the next `run` after an edit re-baselines
  automatically — but plan for it: the day you edit is a day without a
  comparable reading.
- **Keep your library private.** A published corpus can be recognized and
  special-cased by whoever serves the endpoint; a private one cannot. Do not
  commit your production `prompts.json` to a public repository.
- **More prompts, more confidence.** Detection confidence grows with library
  size; 50–200 prompts is a reasonable production range. Prompts with
  deterministic ground truth (math, extraction, format compliance) sharpen
  the signal further.

## Supported providers

- **openai** — direct OpenAI API
- **azure_openai** — Azure OpenAI deployments (`endpoint`, `deployment`,
  `api_version` in the model config)
- **anthropic** — Anthropic Messages API
- **openai_compatible** — anything speaking the OpenAI dialect: Ollama, vLLM,
  LM Studio, DeepSeek, Together, Groq, LiteLLM and other gateways
  (`endpoint` in the model config). This covers on-prem and open-weights
  deployments; point it at your own serving layer.

See `config.example.json` for the shape of each.

## Adapting LMT to your production call

The provider functions in `tectonics.py` are deliberately minimal: model,
messages, `max_tokens`, `temperature`, and an optional system prompt — the
first-order stimulus shapers. Real deployments often set more (`top_p`,
penalties, seeds, stop sequences, `response_format`, provider-specific
options). If your production call does, **edit the provider function to
match it** — the functions are short and self-contained, and this is
supported use, not a workaround. Probe what you deploy.

One rule when you do: **anything you add to the request is part of the
probe stimulus.** Scores are only comparable against an identical stimulus,
so changing a hand-added parameter later is a rebaselining event
(`run --rebaseline`, or `--rebaseline --model <key>` for one model). Prompts
and system prompts live in `prompts.json` and are content-hashed, so edits
there re-baseline automatically — parameters added in code are **not**
hashed, so remembering to re-baseline after changing them is on you.

The statistical core (noise bands, excess scoring, the feed, evidence
emission) never inspects the request; it works identically whatever your
call looks like. Modifying LMT for your own operations needs no permission
and no rename — only *redistributing* a modified version requires a
different name (see NOTICE).

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `baseline_runs` | 3 | Runs per prompt when baselining; establishes the noise band. Higher = tighter band, more API spend. |
| `report_days` | 15 | Days shown in the report grid. |
| `manifests_dir` | (none) | Directory of lmd-spec Coverage Manifests for consequence-aware status, alerting, and chart emphasis. |
| `temperature` (per model) | 0 | Sampling temperature; keep pinned. Set to `null` for models that reject the parameter. |
| `drift_zones` | see example | Chart severity bands (thresholds apply to excess scores). |

## Consequence-aware monitoring

An endpoint LMT monitors may serve several processes at different stakes —
an N×M matrix of drift versus consequence. LMT can join that matrix for you,
but **consequence enters LMT only through lmd-spec Coverage Manifests** —
the assessed, human-authored artifacts that bind a process (and its Q1
consequence level) to a model. There is no native consequence setting in
LMT's config, by design: the assessment is the single source of truth, and
this tool performs a read-only join against it.

Point LMT at a directory of manifests (`"manifests_dir"` in config, or
`--manifests DIR`), and:

- `python tectonics.py status` shows each model's latest reading with its
  consuming processes and their consequence levels; `status --json` emits
  the full joined matrix — including a per-model `history` array
  (`--days N`, default 30), the model's noise floor (`noise_band_rms`),
  its `baseline_date`, and `version_changes` — for ingestion by whatever
  dashboard you already run (Grafana, Datadog, ServiceNow — LMT is the
  sensor, not the pane of glass; presentation is yours).
- `python tectonics.py run --alert-floor 4` exits nonzero when a model in a
  non-stable zone serves a process at consequence level 4 or above — under
  cron with `MAILTO`, a complete alerting system with no infrastructure.
  Models with no manifest coverage are treated as meeting the floor:
  unknown stakes are not assumed low.
- The report chart orders and weights lines by consequence, so the models
  that matter most read first.

Which processes an endpoint serves, and what happens when drift is detected
on a high-consequence one, are human decisions — LMT surfaces the joined
facts and executes the thresholds you configured; it judges nothing.

## Language Model Diligence

LMT is a monitoring mechanism in the sense of the [Language Model
Diligence](https://mossrake.ai/language-model-diligence) framework's Drift
dimension ("if the model or its behavior changed, would you know?").
`python tectonics.py evidence` emits evidence items conformant to the
[Mossrake Language Model Diligence specification](https://github.com/mossrake/lmd-spec),
ready to attach to the drift dimension of a Coverage Manifest. How much drift
coverage a process warrants is a function of its consequence level — that
calibration is the framework's job, not this tool's.

### What running LMT lets you claim

The Drift score in an LMD assessment is always **your claim, made by a
person** — this tool substantiates it; it does not award it. As a guide to
the framework's anchor scale:

- **Occasional manual runs** support a claim of *Informal (3)* — ad hoc
  monitoring exists.
- **Scheduled daily runs** (cron) support a claim of *Monitored (4)* —
  automated checks that would surface significant behavioral changes.
- **Scheduled runs with a private, workload-drawn prompt library and
  configured severity zones as defined acceptance criteria** support a claim
  of *Tested (5)* — automated behavioral regression against defined
  acceptance criteria.

In a facilitated review, the claim is evaluated against the evidence: the
`evidence` output (run cadence, days monitored, corpus size, baseline
discipline) is what makes the number survive scrutiny. Detection scope
matters too — LMT covers behavioral-consistency drift, not capability
regression (see *How it works*), and a Tested claim should be honest about
that boundary.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Language Model Tectonics is published by Mossrake Group, LLC. Language Model
Tectonics™, Mossrake®, and Mossrake Language Model Diligence™ are trademarks
of Mossrake Group, LLC; no trademark rights are granted by this repository.
Truthful, nominative references to this tool by name are ordinary fair use
and do not require permission. Forks and derivative distributions must not
use the Language Model Tectonics name or imply Mossrake origin, certification,
or endorsement.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Feedback: info@mossrake.com

---

© 2026 Mossrake Group, LLC
