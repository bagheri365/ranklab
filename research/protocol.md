# RankLab Protocol

The authoritative research design is `project_plan.md`.

M0 exists to convert dataset-dependent unknowns into an executable frozen protocol. Primary comparative M1 evaluation must not begin before the M0 exit gate passes.

## Freeze rule

`protocol_frozen_m0.yaml` is the authoritative machine-readable protocol object once its `status` is changed to `FROZEN` and all required fields are populated.

Its SHA-256 is computed over the canonical file bytes **without embedding that hash inside the hashed payload**. Store/reference the hash externally in M1 manifests or adjacent metadata.
