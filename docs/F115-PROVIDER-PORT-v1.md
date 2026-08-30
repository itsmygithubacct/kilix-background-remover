# F108 installed provider port for the F115 editable-mask consumer

`kilix-background-remover-provider` is F108's installed, command-free provider
port. It gives the F115-owned adapter the required 5/5 capabilities without
importing an F108 implementation module:

1. 1/5 `DISCOVER` returns immutable installed provider/carrier identity.
2. 2/5 `SUBMIT` accepts the exact canonical candidate-R5 request bytes.
3. 3/5 `MESSAGE` returns exact canonical progress and terminal bytes in provider
   order.
4. 4/5 `CANCEL` accepts exact canonical cancellation bytes and
   `CANCEL-OUTCOME` returns the durable linearization outcome bytes.
5. 5/5 `CLOSE` closes only this provider session and its owned worker/resources.

The transport is length-framed stdio. An input header is an uppercase operation,
one ASCII space, a base-10 payload length, and LF. The payload immediately
follows the header and is exactly that many bytes. The 4/4 accepted input
operations are:

```text
DISCOVER 0\n
SUBMIT N\n<exact N-byte canonical request>
CANCEL N\n<exact N-byte canonical cancel request>
CLOSE 0\n
```

The 5/5 output frame kinds are `IDENTITY`, `MESSAGE`, `CANCEL-OUTCOME`,
`PORT-ERROR`, and `CLOSED`. Every non-empty JSON payload is RFC 8785 canonical
bytes plus exactly one LF and is bounded at 2,097,152/2,097,152 bytes. Candidate
request, progress, terminal, cancel and cancel-outcome bytes are neither
pretty-printed nor repaired by the port.

There are 0/7 command-bearing fields: shell command, environment, model URL,
download URL, Python import name, listener address, and remote schema URL. The
process opens 0/1 network listeners. It accepts 1/1 active request per session
and owns 1/1 persistent supervised worker.

The reference profile is untrained and not release qualified. It is enabled only
for the committed synthetic qualification artifact with
`--reference-profile`; ordinary installed invocation fails closed when no
qualified user-supplied profile is present.

This port supplies F108's side of the integration boundary. The external
installed F108-to-F115 round trip and F115 Gate 8 disposition remain F115-owned;
this repository claims 0/1 F115 gate dispositions.
