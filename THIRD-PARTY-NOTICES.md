# Third-party inventory

This inventory is part of the source boundary. It distinguishes shipped
runtime inputs from development-only feasibility tools. Exact transitive
artifacts remain governed by the committed `uv.lock` and the Plebian OS native
package closure.

| Input | Use | Licence | Disposition |
| --- | --- | --- | --- |
| Pillow 11.1.0 | image decode/encode adapter | HPND | runtime Python dependency; no code copied |
| ONNX Runtime 1.29.0 | feasibility-only supervised-worker adapter | MIT | optional `spike` group; not a release-wide F118 selection or shipped pin |
| jsonschema 4.25.1 | frozen-contract tests | MIT | development only |
| pytest 8.4.1 | tests | MIT | development only |
| ruff 0.12.9 | source checks | MIT | development only |
| mypy 1.17.1 | type checks | MIT | development only |
| hatchling 1.27.x | wheel/sdist build backend | MIT | build only |

The vendored JSON Schemas and golden protocol fixtures are Kilix-owned
Apache-2.0 material frozen jointly by F108 and F115 on 2026-08-25. Their byte
identities are recorded in `contracts/AUTHORITY-SHA256SUMS`.

No model weight is distributed by this repository. F108 S0 path 2 requires
every model profile to use the F100 user-supplied acquisition flow with a
per-profile DUTS/DIS5K terms decision. A friendly model name or this code
licence is never weight or training-data permission.

