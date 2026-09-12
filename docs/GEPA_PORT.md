# Standard-library Optimize Anything port

The optimization plugin owns the active GEPA implementation in
`plugins/optimization/gepa/`, its evaluators and commands, and
`plugins/optimization/GEPA_LICENSE`. Installing the optimization package includes
these resources. The engine uses relative package imports and participates in
the same captured plugin generations as other features.

## Upstream provenance and active source

The port derives from GEPA's
[`optimize_anything` launch](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)
and the immutable [`v0.1.0` release](https://github.com/gepa-ai/gepa/releases/tag/v0.1.0),
commit `a79dc2922cdb0308e9de4f6a5bb9b5ec16f05a91`.
The source-bearing launch merge
`5463a1f5a0880a4dd1330f239870bf647122c329` supplied the same `src/gepa` files.

The active source is a typed standard-library port and differs from upstream.
Unused optional adapters, progress/telemetry integrations, code-execution
utilities, and legacy state-schema migration have been removed. The private
callback dispatcher and its event-payload construction have also been removed;
the public optimizer never exposed those optional constructor callbacks.
Stop conditions and optimizer logging remain active. The remaining
implementation lives inside the optimization package. There is no global import
guard or separate top-level GEPA runtime.

`/optimize verify` reports these distinct checks:

| Report field | Meaning |
| --- | --- |
| `source` | Current engine/license file count, byte count and hash; active import audit; `modified_from_upstream: true` |
| `upstream_source` | Recorded provenance of the original upstream source closure |
| `prompts` | Byte and SHA-256 comparisons for the four retained prompt templates |
| `oracle` | Deterministic evaluator/proposal transcript and its expected hash |

The upstream closure record is 39 files, 322,289 bytes, with SHA-256
`41298300b7fe5588afd81e50f07a4e1b49b060a317687e0a351b279a1e758c7d`.
That identifies the original source closure; it is not the current port's hash.
The current hash is calculated from the active engine and license bytes on each
verification run. The aggregate hashes include relative paths, NUL separators,
and file contents.

The four prompt templates and the 2,029-byte transcript oracle remain unchanged.
The expected transcript SHA-256 is
`adc7258f502b2c9a5d52c8d15220f31948414639c8160bf544675f8660ae3c29`.
The oracle uses a canned reflection response and canonical JSON to check one
proposal/selection trajectory, evaluator order, score accounting, lineage, and
result serialization. Preserving that trajectory and the templates does not
establish equivalence for every possible optimizer input.

## Standard-library execution

Reflection is supplied as a callable using the registered provider service.
Evaluation can run sequentially or through a bounded standard-library thread
pool. The active port has no optional third-party integration path.

The verification import audit scans absolute imports in the active engine's
Python files and reports non-standard-library module roots. Proven
`TYPE_CHECKING` branches are excluded from the runtime import inventory, so
static override annotations can use typing extensions without importing them
in the running application. Runtime acceptance
also executes installed optimization commands in isolated Python processes with
site-packages disabled. The audit is a source check; the terminal runs exercise
the command paths described below.

To reproduce those checks through the actual TUI on POSIX:

```bash
python3 -B -S -m tools.build_plugin_catalog
python3 -B -S -m tools.optimization_tui --output ./verification-run-1/optimization
```

The driver opens chat and enters `/optimize verify`, `/optimize demo`,
`/optimize useful-demo`, `/optimize benchmark`, and `/incident demo`. It checks
prompt/oracle parity, the active source/import report, offline improvement,
held-out task results, equivalent sequential/parallel scores and selection,
bounded parallelism, and canonical LF protocol output. It retains each command's
report and the terminal transcript. Use a new output directory for each run.
See [recorded verification results](VERIFICATION.md) for completed runs.

## Optimization commands

The plugin registers `/optimize` and `/incident`. Enter commands in chat:

```text
/optimize demo --output optimized_protocol.txt
/optimize useful-demo --workers 4
/optimize benchmark --workers 4 --cases 12 --delay-ms 25
/incident demo --workers 4 --max-proposals 6 --target 1 --output incident.txt --report incident.json
```

Automation can pass the same command strings to `raychat.py --exec`. Choose the
workspace with `--workspace PATH`; relative output paths resolve there.
Commands run in an isolated process, so double Escape can stop them while the
terminal remains responsive. Offline demos, verification, and deterministic
benchmarks need no API credentials.

The simple demo uses a known evaluator and canned reflection. The useful demo
evaluates file-inspection tasks on separate test cases, reporting model turns
and tool calls as well as success. The scalability benchmark compares sequential
and bounded parallel evaluation while checking equivalent scores and selection.
These deterministic cases validate the optimization machinery and evaluator
wiring; they do not measure a live model's quality.

The incident demo uses disjoint training, validation, and test cases. It proposes
appended instructions while preserving the base protocol, stops at its validation
target or proposal budget, and evaluates the selected candidate on held-out
incidents. Reports include lineage, the validation trajectory, paired test
outcomes, turn counts, and source hashes. A saved candidate is not automatically
applied to chat.

## Hosted optimization

Provider profiles and defaults are configured under
`plugins.settings.optimization` in `raychat.json`. Task and reflection models can
be selected separately, using credential environment-variable names. Inspect
supported arguments from chat:

```text
/optimize live --help
/incident live --help
```

Commands accept provider, URL, model, request options, worker count, proposal
budget, retry policy, and report/output paths. JSON request options must remain
one quoted argument within the plugin command; commands use Python's `shlex`
parsing on all platforms. Entering the slash command directly in chat avoids an
additional operating-system shell quoting layer.

Hosted evaluation uses the registered Chat Completions provider interface and
its request validation. Reports omit API keys. Live runs spend provider credits
and can produce different candidates and scores across repetitions. Measured
improvements apply to the selected cases, model, settings, and evaluation gate.
The [Self-Harness experiment](SELF_HARNESS.md) is another live evaluation path;
its results are separate from the deterministic GEPA oracle.

## Apply a protocol and package the port

Review a candidate and its report before selecting it:

```bash
python3 -B -S raychat.py --protocol-file optimized_protocol.txt --workspace ./workspace
```

The protocol loader accepts a bounded regular UTF-8 file. Optimizers preserve the
base prompt and append learned instructions. Tool validators, approvals, path
confinement, and process policy remain the responsibility of their owning
plugins and the session host.

Build and exercise the extracted release on POSIX, preserving evidence:

```bash
python3 -B -S -m tools.build_portable --smoke --smoke-output ./verification-run-1/release
python3 -B -S -m tools.build_portable --check --no-smoke
```

The portable bundle contains the plugin package and its source. Smoke rebuilds
package archives for comparison, compiles the source, and drives offline TUI
acceptance on POSIX. Windows smoke checks archive/package integrity and source
compilation; it does not claim Windows terminal coverage. See the
[release guide](RELEASE.md) for the complete acceptance sequence.

Files in `docs/verification/artifacts` record the code and evaluation used when each
report was produced. Older reports can contain upstream-identical source hashes
from before this refactor. Use a new TUI run to obtain evidence tied to the
current package; historical hashes are not current-source verification.
