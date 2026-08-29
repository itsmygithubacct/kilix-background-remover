# F108 OD-22 conditional candidate-R5 return

Status: **conditional product return; candidate R5 is not frozen and not accepted**.

OD-22 authorizes F108 to return before the G5b freeze only against public
manifest SHA-256
`803a5661a708b366b1d26884a4cf52d45c71dac58926e8216eb69aa902cbd25c`.
This complete return must be regenerated if those candidate bytes change.
It does not waive any of section 10's `10/10` freeze conditions and is not a
self-grade.

## Required return (`5/5`)

1. Outcome ledger: `outcomes.jsonl` plus `summary.json`; independently matched
   `168/168` unique public cases: transcript accept `23/23`, transcript refuse
   `81/81`, pixel/PNG `41/41`, registry `8/8`, and manifest `15/15`.
2. Candidate/carrier pins: `pins.json`; installed public resources `46/46`,
   schemas `12/12`, exact source/public manifests, registry, distribution,
   version, wheel, and sdist identities.
3. Installed-carrier result: `installed-carrier.json`; isolated interpreter
   `1/1`, empty current directory `1/1`, installed product `1/1`, installed
   carrier `1/1`, and lookup dependencies on candidate source/current directory
   `0/2`.
4. Cancellation evidence: `cancellation.json`; forced race outcomes `2/2`,
   durable crash points `2/2`, valid public transcripts `4/4`, committed
   publication after accepted cancellation `0/1`, and committed publication
   when terminal reservation wins `1/1`. Both sides use the production
   `DurableCancellationGate.reserve_terminal` transaction used by the worker.
5. G5a rejection: `g5a-production-rejection.json`; production entry points
   rejected `2/2` with exact refusal `schema:C-WIRE-IDENTITY`, v2 dispatch
   identities `10/10`, and negotiation/fallback paths `0/1`.

There are `6/6` canonical machine records because the outcome artifact has a
JSONL ledger and a separate summary. `EVIDENCE-SHA256SUMS` binds all `6/6`.

## Installed execution identity

- product wheel SHA-256:
  `4f8712c6e53e8f394412ece86a6361001a64a3e64720c69cc345a61c8e5a79e0`
- carrier distribution: `kilix-f108-f115-contracts==0.2.1.dev5`
- carrier wheel SHA-256:
  `73ce62f29329d6f1999aa79a765c44476d323641a9e2f7f5b73387735d387e4e`
- carrier sdist SHA-256:
  `b208503754691076a8f5e92cb7303ca5f3f451905fa759afd0e0606e7017d929`

The PREP8 continuation generator ran from a fresh offline virtual environment
under `python -I`, with an empty current directory. The exact installed product
and carrier wheels were explicit installer inputs. All canonical machine
records reproduced byte-for-byte (`6/6`). The product gate before this
generation passed corpus `3/3`, suite `131/131`, lint `1/1`, formatting
`42/42`, typing `20/20`, and two byte-identical source/wheel builds (`4/4`
artifacts across `2/2` build passes) using pinned uv `0.12.5`.

## Reproduction

Build the product with the release-pinned uv, create a fresh environment,
install the exact product and carrier wheels, change into an empty directory,
then run:

```sh
python -I /home/pleb/gpu_terminal/kilix-apps/kilix-background-remover/tools/generate_conditional_r5_return.py \
  --output-dir /home/pleb/gpu_terminal/kilix-apps/kilix-background-remover/evidence/conditional-r5-return \
  --candidate-source /home/pleb/research/gpu_terminal/f108-f115-g5b-r5 \
  --g5a-fixture /home/pleb/gpu_terminal/kilix-apps/kilix-background-remover/contracts/fixtures/valid/f108-reference-mask-lifecycle.json
```

Provider refusals during generation: none (`0/0` received).
