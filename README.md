# Kilix Background Remover

Local-first image cutout provider for Plebian OS / Kilix 0.2.1. The repository
is a clean-room Apache-2.0 implementation. It is local-only during development;
qualification does not authorize a remote, tag, artifact, or publication.

The current implementation line consumes the candidate-R5 F108/F115 mask-first
`/v2` JSON contracts and freezes a deterministic, wholly synthetic image/mask
corpus. It contains a supervised reference worker used to exercise bounded
decode, first-class masks, compositing, cancellation, batching, atomic output,
offline video, CLI/TUI and command-free app-bridge behaviour before a release
model is selected.

The carrier is pinned exactly to
`kilix-f108-f115-contracts==0.2.1.dev5`, wheel SHA-256
`73ce62f29329d6f1999aa79a765c44476d323641a9e2f7f5b73387735d387e4e`,
and public manifest SHA-256
`803a5661a708b366b1d26884a4cf52d45c71dac58926e8216eb69aa902cbd25c`.
That package is a candidate, not a G5b freeze or acceptance. The F108 return is
conditional on those exact bytes and must be rerun if they change before
freeze.

The reference profile is deliberately impossible to mistake for a qualified
model. Normal commands fail closed unless `--reference-profile` is supplied.
No model bytes are mirrored here. Production model selection, F100
user-supplied acquisition, F106 profiles, shared ONNX Runtime packaging and
H1/H2/H3 qualification remain separate release gates.

## Development

The repository requires the release-frozen `uv` 0.12.5 executable with
SHA-256 `b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46`.
Set `UV` to its absolute path; the gate refuses a version or digest mismatch.

```sh
UV=/absolute/path/to/release-pinned-uv-0.12.5
make setup UV="$UV"
make check UV="$UV"
```

Regenerate and verify the owned corpus:

```sh
"$UV" run --frozen python tools/generate_corpus.py --root tests/fixtures/corpus
"$UV" run --frozen pytest tests/test_corpus.py
```

No application path opens a listener or downloads a runtime/model. Product
integration with the Kilix host, `kilix-content`, F106 and Plebian OS is not
present on this branch.

## Product surfaces

Image decoding runs in a disposable spawned process. The default decode budget
is 30 seconds of CPU time, 30 seconds of wall time and 2 GiB of address space;
only a bounded, metadata-free RGBA PNG crosses back to the persistent worker.
Image and video outputs use verified `0600` sibling staging files and a
no-replace atomic commit.

The offline-video command has an explicit estimate/confirmation join. The first
call produces no destination and returns the confirmation digest:

```sh
kilix-background-remover video INPUT OUTPUT \
  --output-kind matte
```

The second call must repeat the same source and settings and supply that exact
digest:

```sh
kilix-background-remover video INPUT OUTPUT \
  --output-kind matte \
  --confirm-estimate CONFIRMATION_SHA256 \
  --reference-profile
```

All required output profiles are implemented (`6/6`): `transparent-mov`,
`transparent-webm`, `matte`, `composite-image`, `composite-video`, and `gif`.
The composite profiles require `--background-image` or `--background-video`,
respectively. GIF uses a disclosed hard alpha threshold and requires
`--no-audio` when the source has audio. Audio is preserved by default for the
other capable profiles; `--no-audio` explicitly removes it. Temporal smoothing,
scene-cut isolation, batch overlap/resume, raw-frame mode, VFR timestamps and
rotation are handled locally by the fixed `/usr/bin/ffmpeg` adapter.

`kilix_background_remover.editable_mask` is the in-repository pane-4 reference
consumer for F115. `consume_editable_mask_transcript` validates the full
candidate-R5 `/v2` transcript, all request/result identity joins and the exact
bounded gray8 foreground-alpha PNG before `EditableMaskDocument` atomically
attaches one full-source-geometry mask. Accepted cancellation leaves the
document unchanged, and provider-controlled prose is never rendered.
