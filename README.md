# Kilix Background Remover

Local-first image and offline-video cutout provider for Plebian OS / Kilix
0.2.1. The repository is a clean-room Apache-2.0 implementation. Development
is published on the `work/0.2.1-f108` ref. That work-ref publication transports
the exact review subject; it is not stream acceptance and does not authorize a
release branch, tag, public artifact, model, or weight publication.

The current implementation line consumes the candidate-R5 F108/F115 mask-first
`/v2` JSON contracts and freezes a deterministic, wholly synthetic image/mask
corpus. One `BackgroundRemovalProvider` owns the supervised worker used by all
5/5 product surfaces: provider port, CLI, TUI, contained app, and the F115
editable-mask boundary. It exercises bounded decode, first-class masks,
compositing, cancellation, batching, atomic output, and offline video before a
release model is selected.

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

Image header/EXIF inspection and full decoding both run in disposable spawned
processes. The default parser budget is 30 seconds of CPU time, 30 seconds of
wall time and 2 GiB of address space. The long-lived side accepts only a 4 KiB
closed JSON status record and an exact-size, mode-`0600`, metadata-free RGBA
raster; it never unpickles child data or reparses a child-generated image.
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
rotation are handled locally by the fixed `/usr/bin/ffmpeg` adapter. Smoothing
uses a fixed 1 MiB accumulator independent of temporal radius. Before atomic
publication, the staged carrier is decoded again and its authoritative mask,
alpha, or RGB plane is compared with every rendered frame; a metadata-valid but
pixel-wrong encoder result is refused.

The installed wheel exposes 5/5 executables/surfaces:

```text
kilix-background-remover                 image, batch and video CLI
kilix-background-remover-tui             keyboard TUI for image/video requests
kilix-background-remover-app             contained graphical app and headless lifecycle
kilix-background-remover-app-bridge      bounded command-free app bridge
kilix-background-remover-provider        length-framed F115 provider port
```

All 5/5 enter the same provider façade. The per-frame video adapter is lent the
provider's existing supervised worker; it does not create a second inference
path. `kilix-background-remover doctor --json` reports the installed decode
budgets, the 6/6 video kinds and the exact candidate manifest identity. Its
inference working-set control covers each source pixel exactly once in
deterministic row-major 2-D tiles of at most 1,048,576 pixels. The reported
release tiling phases, seam policy and RSS threshold remain unset and the
provider remains explicitly unqualified until the release owner selects them.

Installed qualification code can measure an explicitly selected Linux process
tree without selecting a release ceiling:

```python
import os

from kilix_background_remover.rss import ProcessTreeRssMonitor

with ProcessTreeRssMonitor(os.getpid()) as monitor:
    outcome = provider.run(request)
measurement = monitor.measurement
```

The measurement is sampled aggregate `VmRSS` for the named root and observed
descendants. Provider identity leaves both the release measurement scope and
threshold unset; the mechanism by itself supplies 0/1 D14 release verdicts.

The TUI accepts either a canonical candidate-R5 image request or a fixed-field
video request. It exposes q/Escape cancellation and r retry, and reports the
decode/model/backend/resource policy before work starts:

```sh
kilix-background-remover-tui image-request.json --reference-profile
kilix-background-remover-tui video-request.json \
  --operation video --reference-profile
```

`kilix-background-remover-app` opens the contained graphical client. The same
controller has a display-independent lifecycle form used for installed package
qualification:

```sh
kilix-background-remover-app --message app-message.json --reference-profile
```

The bridge accepts only fixed operations and typed paths. It accepts 0/7 shell
commands, environment variables, model URLs, download URLs, import names,
listener addresses or remote schema URLs.

`kilix_background_remover.editable_mask` is the in-repository pane-4 reference
consumer for F115. `consume_editable_mask_transcript` validates the full
candidate-R5 `/v2` transcript, all request/result identity joins and the exact
bounded gray8 foreground-alpha PNG before `EditableMaskDocument` atomically
attaches one full-source-geometry mask. Accepted cancellation leaves the
document unchanged, and provider-controlled prose is never rendered.
`run_reference_editable_mask_operation` is the executable in-repository harness:
it submits a real composited layer through the supervised provider, retains the
reported threshold/feather settings, validates the resulting transcript and
commits only a sealed, sample-digest-revalidated immutable mask plan.

The separately installed F115 consumer does not import those implementation
modules. It uses `kilix-background-remover-provider`, whose exact byte protocol
is specified in [docs/F115-PROVIDER-PORT-v1.md](docs/F115-PROVIDER-PORT-v1.md).
The port provides discovery, canonical request/progress/terminal transport,
durable cancellation, and owned-session teardown. F108 supplies that port;
the external F108-to-F115 Gate 8 disposition remains F115-owned at 0/1.
