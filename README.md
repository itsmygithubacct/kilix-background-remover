# Kilix Background Remover

Local-first image cutout provider for Plebian OS / Kilix 0.2.1. The repository
is a clean-room Apache-2.0 implementation. It is local-only during development;
qualification does not authorize a remote, tag, artifact, or publication.

The current implementation line adopts the frozen F108/F115 mask-first JSON
contracts and freezes a deterministic, wholly synthetic image/mask corpus. It
contains a supervised reference worker used to exercise bounded decode,
first-class masks, compositing, cancellation, batching, atomic output, CLI/TUI
and command-free app-bridge behaviour before a release model is selected.

The reference profile is deliberately impossible to mistake for a qualified
model. Normal commands fail closed unless `--reference-profile` is supplied.
No model bytes are mirrored here. Production model selection, F100
user-supplied acquisition, F106 profiles, shared ONNX Runtime packaging and
H1/H2/H3 qualification remain separate release gates.

## Development

The repository requires the release-frozen `uv` 0.12.5 tool.

```sh
uv sync --locked --all-groups
make check
```

Regenerate and verify the owned corpus:

```sh
uv run --frozen python tools/generate_corpus.py --root tests/fixtures/corpus
uv run --frozen pytest tests/test_corpus.py
```

No application path opens a listener or downloads a runtime/model. Product
integration with the Kilix host, `kilix-content`, F106 and Plebian OS is not
present on this branch.

