# vendor

`kilix_f108_f115_contracts-0.2.1.dev5-py3-none-any.whl` is the F108/F115
contract carrier: JSON schemas, conformance fixtures and the registry the suite
validates against. It is not published to any index.

It is vendored here rather than referenced through a path outside the
repository. The previous `[tool.uv.sources]` entry pointed at
`../../../research/gpu_terminal/...`, which resolves only when the clone sits
exactly three directories below a parent that also contains that tree. At any
other depth `uv sync --locked` failed with exit 2, so `make setup`,
`make env-check` and therefore `make check` failed on an otherwise green
repository. A lock that only resolves at one filesystem depth is not a lock.

`SHA256SUMS` binds the copy. Verify with:

    cd vendor && sha256sum -c SHA256SUMS
