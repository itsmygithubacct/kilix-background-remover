# Kilix Background Remover

A local-first image and offline-video cutout provider for Kilix, with five
command, terminal and app surfaces over one supervised worker.

**Status:** candidate, not part of Plebian-OS. This README describes the
repository's default branch, `work/0.2.1-f108`. The Plebian-OS 0.2.2 release
notes list it (F108, `ce99d1af`) among stream deliverables that do not ship in
the image. The Kilix for Windows preview's WSL app catalog
(`windows/wsl/app-catalog.json` in kilix) lists it as "Background Remover",
pinned to `ce99d1af` (this branch's head), for bounded image inspection only:
no production cutout model is selected, so inference is unavailable there. The
bundled reference profile is a small synthetic feasibility model, not a release
model.

## What it does

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

- Removes an image background to a cutout, mask or composite, singly or as a
  resumable ordered batch.
- Runs offline video jobs to transparent MOV or WebM, matte, composite image or
  video, or GIF, after an explicit estimate and confirmation.
- Decodes every image in a disposable, resource-capped process and commits
  outputs atomically with private permissions.
- Serves the F115 editable-mask consumer through a length-framed provider port.

## Quick start

### Development

The repository requires the release-frozen `uv` 0.12.5 executable with
SHA-256 `b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46`.
Set `UV` to its absolute path; the gate refuses a version or digest mismatch.

```sh
UV=/absolute/path/to/release-pinned-uv-0.12.5
make setup UV="$UV"
make check UV="$UV"
```

`make setup` syncs `--all-groups --all-extras`. Both halves are required: the
suite imports `onnxruntime` from the `spike` extra, so a `--all-groups`-only
sync leaves 30 cases failing on assertions rather than on a missing dependency.
`make check` runs `env-check` first, which refuses with one message naming what
is absent instead of letting the suite fail obscurely.

The `contracts` extra carries `kilix-f108-f115-contracts`, which is **published
to no index** and is supplied locally through `[tool.uv.sources]`. The product
installs and runs without it; contract operations then refuse with
`C-CARRIER-INSTALLED` rather than failing to import.

Regenerate and verify the owned corpus:

```sh
"$UV" run --frozen python tools/generate_corpus.py --root tests/fixtures/corpus
"$UV" run --frozen pytest tests/test_corpus.py
```

No application path opens a listener or downloads a runtime/model. Product
integration with the Kilix host, `kilix-content`, F106 and Plebian OS is not
present on this branch.

### Run

After `make setup`, the five executables are in `.venv/bin`:

```sh
.venv/bin/kilix-background-remover doctor --json
.venv/bin/kilix-background-remover image subject.png \
  --output-dir out --reference-profile
```

## Usage

| Command | What it does |
|---|---|
| `kilix-background-remover image INPUT --output-dir DIR` | One image; `--mask-only`, `--webp`, `--composite-color R,G,B,A` or `--composite-image F`. |
| `kilix-background-remover batch INPUT_DIR --output-dir DIR [--state-dir D]` | An ordered directory batch that can resume. |
| `kilix-background-remover video INPUT OUTPUT --output-kind KIND` | Estimate a video job; repeat with `--confirm-estimate SHA256` to run it. |
| `kilix-background-remover run-contract REQUEST.json` | Run one candidate-R5 `/v2` contract request. |
| `kilix-background-remover measure-contract REQUEST.json --fixture-tier h0\|h1\|h2` | Measure one cold request on an attested capacity fixture. |
| `kilix-background-remover doctor [--json]` | Decode budgets, video kinds and the candidate manifest identity. |
| `kilix-background-remover-tui REQUEST.json [--operation video]` | Keyboard TUI with q/Escape cancel and r retry. |
| `kilix-background-remover-app [--message MSG.json]` | Contained graphical app, or its headless lifecycle form. |
| `kilix-background-remover-app-bridge`, `kilix-background-remover-provider` | The fixed-operation app bridge and the F115 provider port. |

Commands that run the worker fail closed without `--reference-profile` until a
release model is selected.
`KIND` is `transparent-mov`, `transparent-webm`, `matte`, `composite-image`,
`composite-video` or `gif`. Details: [docs/product-surfaces.md](docs/product-surfaces.md).

## How it works

```
CLI   TUI   contained app   app bridge   F115 provider port
  \     \         |             /             /
   BackgroundRemovalProvider: one facade, one supervised worker
        |                                   |
  disposable decode processes         video: /usr/bin/ffmpeg adapter,
  (CPU, wall-time, address caps)      frames lent to the same worker
        |                                   |
  mask -> cutout / composite          re-decode and compare every frame
        \                                   /
         0600 sibling staging -> verified no-replace atomic commit
```

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

## Documentation

- [docs/product-surfaces.md](docs/product-surfaces.md): bounded decode, video
  confirmation and profiles, the five executables, RSS measurement, the
  capacity fixture contract, the TUI, the app, and the F115 mask consumer.
- [docs/F115-PROVIDER-PORT-v1.md](docs/F115-PROVIDER-PORT-v1.md): the exact
  byte protocol of the provider port.
- [contracts/README.md](contracts/README.md): the frozen F108/F115 contracts.
- [tests/fixtures/README.md](tests/fixtures/README.md): the synthetic corpus.
- [evidence/0.2.1-product-return-r1/README.md](evidence/0.2.1-product-return-r1/README.md):
  the recorded product-return evidence.
- [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Related repositories

- [kilix-image-shop](https://github.com/itsmygithubacct/kilix-image-shop#readme):
  the F115 image editor core whose editable-mask operation this provider serves.
- [kilix](https://github.com/itsmygithubacct/kilix#readme): carries the Kilix
  for Windows WSL catalog entry.

## Part of Plebian-OS

Kilix Background Remover is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
It does not ship in Plebian-OS; only the Kilix for Windows preview's WSL
catalog lists it, without a cutout model. The map of every repository and how
it ships is on the
[components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE) (Copyright 2026
itsmygithubacct). Third-party notices are in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
