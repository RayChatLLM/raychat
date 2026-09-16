# Self-Harness

This optional plugin is disabled in the default configuration. Ordinary chat can
still modify and recover the running core through the built-in tools described in
[Live core updates](LIVE_CORE.md). Enable Self-Harness explicitly only when its
separate proposal/evaluation loop is desired by removing `self_harness` from the
configuration's `plugins.disabled` list.

The `self_harness` plugin adapts the proposal/validation loop in
[Nano's self_harness.rs](https://github.com/skorotkiewicz/nano-agent/blob/main/src/self_harness.rs)
and the failure-driven, held-in/held-out gate in
[Self-Harness: Harnesses That Improve Themselves](https://arxiv.org/abs/2606.09498).
It uses the existing provider, filesystem operations, process runner, context
hooks, and transactional plugin loader. Everything uses Python's standard library.

This implements the mechanism, not a reproduction of the paper's benchmark
results. Improvement depends on the operator's evaluator and task distribution.

## Run it

From a running chat:

```text
/self-harness --scores -- python3 -B -S evaluator.py
/self-harness --exit-code -- python3 -B -S -m unittest discover -s tests
/self-harness status
```

Arguments are parsed with `shlex` and executed as an argument array, without a
shell. `{workspace}` and `{harness}` arguments expand to the temporary candidate
workspace and its overlay file. Prefer relative script paths so the evaluator
imports the staged implementation. Supply executable arguments explicitly when
using a path containing spaces.

Set `plugins.settings.self_harness.validation_argv` and `validation_mode` in `raychat.json`
to fix a reusable evaluator. Then `/self-harness` or the model's
`{"action":"self_harness"}` runs it. The tool cannot change the evaluator;
it requires the ordinary tool approval policy. The slash command is an explicit
operator request. Proposals use the same configured chat provider/model with no
tools exposed to the proposer.

## Evaluation contract

Score mode is the default. The command must exit zero and print one JSON object:

```json
{
  "held_in": {
    "passed": 7,
    "total": 10,
    "failures": [
      {"signature": ["verifier cause", "causal status", "agent mechanism"], "trace": "failed example A"},
      {"signature": ["verifier cause", "causal status", "agent mechanism"], "trace": "failed example B"}
    ]
  },
  "held_out": {"passed": 8, "total": 10}
}
```

Each split must have a positive integer total and an integer passed count between
zero and total. Baseline/candidate split sizes must stay fixed. A candidate is
eligible only when neither split regresses and at least one improves. Optional
repetitions compare aggregate success rates; repeated evaluations do not count
as distinct failure examples. The evaluator owns test selection and model
configuration and must keep them fixed. Held-out traces and scores never enter
the proposer prompt, including through previous-attempt records.

Exit-code mode follows Nano's simpler rule: every candidate validation run must
exit zero. This mode checks the command's success, not statistical improvement.
Both modes run baseline and candidate in separate temporary copies of the same
workspace snapshot, with command timeouts, bounded output, credential-filtered
environment, and process-tree cleanup. The copies omit symlinks, `.git`, virtual
environments, build output, `.env` files, and this plugin's audit directory.
These are execution directories for trusted local code, not a security sandbox.

## Evidence, proposals, and promotion

The plugin retains bounded recent tool outcomes across sessions in
`.raychat/self-harness/evidence.jsonl`. Recurring groups need at least two
observations. Tool groups identify observed failure cause/action; they do not
claim to establish causality. Evaluator-supplied signatures can provide the
paper's stronger causal diagnosis. Passing actions and recent attempts are
included so the proposer can preserve working behavior and avoid repeated edits.

A proposal is JSON with `rationale`, an exact observed `signature`, a complete
`overlay`, and optional `files` mapping relative plugin filenames to complete
source. Nano's `WHY: ... HARNESS: ... END` overlay format is also supported.
Overlays default to a 4,000-byte cap; plugin patches to 64,000 bytes.

`editable_roots` defaults to `.raychat/plugins`, the workspace's installed
packages. Link a development project there when you intend to permit edits. The core, evaluator, and Self-Harness's own
implementation are outside the default editable surface. New manifest package
plugins in trusted discovery directories can be promoted as well as existing
plugins. Core/TUI/SDK changes still require restarting Python.

`candidate_count` selects 1–8 proposals; `repetitions` selects 1–10 evaluator
runs per baseline/candidate. The best eligible candidate is promoted; compatible
candidate merging is not implemented. Proposals, validation results and decisions
are retained in `.raychat/self-harness/attempts.jsonl`, with the selected candidate
in `candidate.json`. No weights are trained or modified.

`proposal_retries` allows 0–2 syntax/contract repairs (default 1), before any
candidate evaluation. Repair feedback contains no held-out results. The proposer
is told that future sessions see only the deployed overlay, so it must encode
learned rules directly instead of referring to the training evidence.

Promotion checks for intervening edits, atomically replaces each affected file,
and rebuilds the plugin generation at the end of the current operation. Invalid
imports, API registrations, dependencies, or configuration restore the prior
files and keep the prior generation active. Cancellation discards pending
promotion. History and unrelated plugin state remain attached to the same chat.
The active overlay is `.raychat/harness.md`; it contributes to subsequent requests
through a context hook. After editing that file manually, use `/plugins reload`.
External side effects performed by a validator are not transactionally reversible.

The runtime is dynamically extensible independently of this optimizer: an agent
can write a trusted plugin, request a `plugins` reload, finish its turn, and use
the new capability in the next message. See [the plugin API](PLUGINS.md).

## Measure task benefit

From the running terminal chat:

```text
/benchmark-harness --output self-harness-report.json --repetitions 2
```

This optional experiment uses `RAYCHAT_AUTH_TOKEN`, `RAYCHAT_MODEL`, and
`RAYCHAT_BASE_URL`, shared with the main chat. The live optimization and incident
commands use this same configuration for both task and reflection requests;
their request options and timeouts can differ. Offline demonstrations and
benchmarks use local fixtures without provider credentials.

The experiment reuses the optimization plugin's fixed file-task cases and exact
action/artifact verifier: three training cases, two validation cases, and two
test cases evaluated only after candidate selection. One candidate is proposed (with at most one format repair by default);
baseline and candidate use the same model and sampling settings. A local gateway
keeps provider credentials out of the validation process and its logs.

The deployment tasks deliberately omit an organization-specific formatting
policy that held-in verifier feedback supplies. Improvement on these cases is
evidence of useful instruction learning, not general coding improvement. The
report retains individual outcomes, the proposal, acceptance/rejection, request
counts and wall time. Broader efficacy needs representative repository tasks,
independent tests, repeated trials and a comparison that accounts for the added
model calls and evaluation cost.

The TUI driver records acceptance or rejection and the post-selection score.
It accepts `--request-options` for explicit options applied to every paired
evaluation; the benchmark fixes temperature at 0 and output at 8,192 tokens.
Test outcomes never enter the model's proposal prompt. An interrupted or failed
provider run promotes no candidate and is not evidence of improvement.

To reproduce the terminal acceptance with retained evidence (paid provider calls):

```bash
python3 -B -S -m tools.self_harness_tui --output /tmp/raychat-harness-proof --request-options '{"reasoning_effort":"low"}'
```

Evaluator JSON has its own bounded `max_evaluator_bytes` allowance (256 KiB by
default), separate from the short output retained for ordinary command tools.

## Supervised core updates

Interactive sessions submit source proposals to the stable core supervisor. Changes
to `raychat/` and `plugins/` (including Self-Harness itself) become isolated releases
after the fixed quality, test, packaging, and restoration gates pass. The footer
reports activation status; `/update-log` identifies detailed diagnostics. See
[live updates and recovery](LIVE_CORE.md) for retained releases and emergency recovery.
