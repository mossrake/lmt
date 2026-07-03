# Contributing

Thank you for your interest in Language Model Tectonics.

## Feedback

This repository does not use GitHub Issues. Feedback is welcome by email:
info@mossrake.com.

LMT is in beta, and field reports carry the most weight — particularly:

- False positives: endpoints flagged as drifting that had not changed
  (include the model, the excess scores, and your `baseline_runs` setting)
- Missed changes: known provider updates the chart did not surface
- Provider quirks: endpoints where version capture, temperature pinning, or
  the noise-band methodology behaves unexpectedly
- Corpus design findings: what prompt structures produce the sharpest signal
  on your workloads

## Pull Requests

This repository does not accept unsolicited pull requests.

Language Model Tectonics is stewarded by Mossrake Group, LLC. Maintaining a
single, unambiguous chain of authorship keeps provenance clean and preserves
the licensing structure described in LICENSE and NOTICE. Changes are made by
Mossrake, informed by field reports, and released as versioned revisions.

If you would like to propose a specific change — including new provider
handlers — describe it by email. If a contribution is invited, it will
require a signed contributor agreement assigning or broadly licensing the
contribution to Mossrake Group, LLC before it can be merged.

## Scope

This repository contains the drift monitoring tool. For the Language Model
Diligence framework, the Self-Assessment, or the machine-readable
specification, see [mossrake.ai/language-model-diligence](https://mossrake.ai/language-model-diligence)
and [github.com/mossrake/lmd-spec](https://github.com/mossrake/lmd-spec).

## Contact

Mossrake Group, LLC — info@mossrake.com
