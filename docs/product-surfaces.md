# Product surfaces

Bounded decoding, the offline-video confirmation join and output profiles, the five installed executables, process-tree RSS measurement, the capacity fixture contract, the TUI, the contained app and bridge, and the F115 editable-mask consumer. Moved from the README.

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
descendants. Every sampled PID is rebound to its Linux proc-stat start time
before and after the RSS read: root PID reuse is refused, and reused or
reparented descendants are omitted. Provider identity leaves both the release
measurement scope and threshold unset; the mechanism by itself supplies 0/1
D14 release verdicts.

The F100-C0 H0/H1/H2 fixture contract is also pinned into provider identity as
3/3 frozen tiers. `measure-contract` first attests every property observable
inside the declared fixture and refuses a mismatched runtime before starting a
window. Every completed window publishes the complete `/proc/loadavg` line at
both boundaries (2/2), including the mandatory H2 pair, plus elapsed monotonic
time and peak process-tree RSS:

```sh
kilix-background-remover measure-contract request.json \
  --fixture-tier h2 --reference-profile > measurement.json
```

That command always reports the bundled feasibility profile as unqualified and
awards 0/1 release-acceptance credit. It does not substitute its 0/4 quality
metrics for the missing licensed corpus, real profile, frozen quality floor, or
release-owner qualification decisions.

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
is specified in [docs/F115-PROVIDER-PORT-v1.md](F115-PROVIDER-PORT-v1.md).
The port provides discovery, canonical request/progress/terminal transport,
durable cancellation, and owned-session teardown. F108 supplies that port;
the external F108-to-F115 Gate 8 disposition remains F115-owned at 0/1.
