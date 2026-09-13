# Verification

The current queue/status work is recorded in
[`ui_queue_status_report.json`](verification/artifacts/ui_queue_status_report.json).
It covers SDK 4 push status, command completion, automatic mouse copying,
transactional FIFO queue editing in root and workflow-child chats, real launcher
use, package upgrades, and the 50-child collective task. Older reports below
retain their original source identities and describe earlier releases.

Run the new composer acceptance directly with:

```bash
python3 -B -S -m tools.composer_tui --output /tmp/raychat-composer-check
```

The portable smoke gate also includes this driver. It verifies keyboard queue
navigation and editing, dispatch suspension, draft restoration, command
completion without execution, exact clipboard bytes on mouse release, footer
feedback without saved-history changes, and independent workflow-child chats.


Verification requires the complete unittest suite, strict typing and lint/format
checks, plus actual RayChat terminal interaction with keyboard, mouse and screen
observations. [Final functional QA](verification/artifacts/reorganization_report.json)
ran 592 tests with one Windows-only skip, passed all thirteen packaged terminal
drivers and configured static checks, and completed the 50-child collective task.
The [evidence index](verification/README.md) identifies the tested source, the
static-only import correction, and separately retained earlier live experiments.

The checked-in drivers launch `python -B -S` in isolated workspaces and homes. They observe model requests and filesystem/process
effects; they do not invoke feature or session APIs from the driver.

## Reproduce

Use a fresh output directory for each run:

```bash
python3 -m tools.build_plugin_catalog
.venv/bin/python tools/check_types.py
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
python3 -B -S -m unittest discover -s tests -v
python3 -B -S -m tools.bare_tui --output /tmp/raychat-bare
python3 -B -S -m tools.accept_tui --output /tmp/raychat-interaction
python3 -B -S -m tools.features_tui --output /tmp/raychat-features
python3 -B -S -m tools.startup_tui --output /tmp/raychat-startup
python3 -B -S -m tools.profile_upgrade_tui --output /tmp/raychat-upgrades
python3 -B -S -m tools.persistence_tui --output /tmp/persistence
python3 -B -S -m tools.package_download_tui --output /tmp/package-download
python3 -B -S -m tools.adversarial_agents_tui --output /tmp/raychat-agent-edges
python3 -B -S -m tools.ui_stress_tui --output /tmp/raychat-ui-stress
python3 -B -S -m tools.collective_tui --output /tmp/raychat-collective --agents 50 --parallel 8
python3 -B -S -m tools.optimization_tui --output /tmp/raychat-optimization
python3 -B -S -m tools.plugin_guide_tui --output /tmp/raychat-plugin-guide
python3 -B -S -m tools.reload_race_tui --output /tmp/raychat-reload-race
```

Install the pinned development tools as described in [the README](../README.md).
Mypy and Ruff are development-only dependencies; the harness, plugins, unit suite
and terminal drivers use the standard library at runtime. Strict mypy checks
application, plugin, test, example and tool source, including the launcher.

Run the entire unittest discovery command, not just selected modules or static
interface checks. Retain its test count, failures, errors and skips. Unit tests
exercise APIs and regression invariants; the fourteen offline TUI drivers separately
verify the application's user-facing behavior. Complete verification requires
both, and final results must identify the tested source/ZIP hash.

The following make paid requests through the configured provider. Set credentials
in the provider's documented environment variables, without putting secrets in
commands or reports:

```bash
python3 -B -S -m tools.collective_tui --output /tmp/raychat-live-collective --agents 50 --parallel 8 --live --model accounts/fireworks/models/glm-5p3-flash
python3 -B -S -m tools.self_harness_tui --output /tmp/raychat-live-self-harness --model accounts/fireworks/models/glm-5p3-flash --request-options '{"reasoning_effort":"low"}'
```

## What the terminal scenarios verify

| Driver | Observed behavior |
| --- | --- |
| `bare_tui` | Core starts without feature source/catalog/GEPA, accepts an external provider, exposes no feature tools and recovers from unavailable commands/actions |
| `accept_tui` | Child navigation by keyboard/mouse, exact clipboard bytes, focused double-Escape cancellation, sibling survival, parent return, one/multiple session resume, command-tree cleanup, external package create/check/pack/catalog/search/install/update/live-edit/disable/enable/uninstall/restart |
| `features_tui` | UTF-8 byte paging, directory cursors, hashes and atomic edits, stale-hash rejection, path/symlink confinement, durable memory and paging, skill discovery/loading/session reset, manifest instructions, goal judging/continuation/artifact checks/final suppression |
| `startup_tui` | Compile/import/entrypoint/registration/provider failures name the broken package without a traceback; repairing each source restores commands and chat |
| `profile_upgrade_tui` | Same-version content upgrades, dependency provenance, retry after failed installation, atomic package changes, preserved edits/external packages/links/disabled state/removals; optional read-only copy of an existing installation |
| `persistence_tui` | Checkpointed plugin state across branches, hot settings/code rollback, restart and keyboard/mouse resume; concurrent command checkpoints survive model cancellation and successful local/isolated-child completion; goals and disabled persistence; rejected forks and damaged sessions preserve the active saved conversation |
| `package_download_tui` | Cancellation while headers or response body stall; replacement before server release, no partial activation, successful retry, focused workflow-child cancellation with parent/sibling survival |
| `adversarial_agents_tui` | Separate parent/child queues and histories, focused cancellation with sibling survival, repeated workflows, deferred provider reload, stable picker IDs, bounded Unicode previews, single/double Escape with the picker open, cancellable concurrent session commands and occupied command-lane admission |
| `ui_stress_tui` | Atomic oversized-paste rejection, preserved drafts, discarded newline tails, configured input limits, combining Unicode and exact clipboard content after scrolling, busy commands, saved forks/resume and empty menus |
| `composer_tui` | Editable FIFO queues, per-child draft restoration, command-name completion, automatic mouse-copy feedback, and pushed SDK 4 status expiry/order without transcript pollution |
| `collective_tui` | At least 50 child conversations in bounded workflow batches, independent invoice reconciliation, verified aggregate, per-child context compaction, child follow-up, oversized-input rejection before provider invocation, malformed-response repair and recovery |
| `optimization_tui` | Installed private GEPA engine, stdlib imports, unchanged upstream prompt templates and exact 2,029-byte behavior oracle, offline file-task/incident improvement, parallel score equivalence, canonical saved protocol bytes |
| `plugin_guide_tui` | Complete documented hello/salute packages, session state, model hooks, direct services, live-edit rollback, ZIP/catalog installation and disable/enable/removal |
| `reload_race_tui` | Twenty rapid live edits, concurrent application commands, worker-queued activation, preserved state and visible deferred dependency rejection |
| `self_harness_tui` | Fixed baseline/candidate evaluations, evidence-based proposal, acceptance or rejection, post-selection test comparison with the same provider/settings, retained complete reports |

External package acceptance includes namespace helper modules and live helper
edits. It also verifies that a repository cannot authorize execution through a
forged installation record; catalog/archive metadata must agree; explicit
replacement is required for local edits; disabled package management can be
recovered by explicitly launching its installed path.

Failure injection covers operation errors, cancellation and queued preparation
with throwing rollback callbacks. The TUI must accept another message and a
subsequent plugin update. Failed isolated conversation construction/restoration
records two distinct runtime closures from the worker process, including the
conversation runtime, so provider cleanup cannot mask a leaked session runtime.
All fixtures now load captured packages through the ordinary loader; there is
no source-less module registration path.

The 50-child task reconciles unpaid invoices from 50 distinct files. Every child
must report its own shard and subtotal; the parent writes batch artifacts and
combines them into 24,900 cents. Six large reference reads per child pressure a
10,000-character request budget. The driver measures actual requests rather than
inferring compaction from a final answer. These are harness subagents using the
normal worker conversations and workflow plugin.

Current-task action/result pairs are retained together for as long as the
character budget permits, before spending remaining space on completed history
or dynamic memory. `--keep-recent` limits raw pairs from completed tasks, not
inputs still needed by the active task. Older summaries keep complete factual
groups and never replace the active user prompt.

## Historical evidence and limits

The reports below predate the current file/folder reorganization. They document
their own sources and experiments; they do not establish that the reorganized
tree has passed the new full-unit and fourteen-driver acceptance requirement.

The retained deterministic collective run completed 50 correct children with eight
active at once. All 50 compacted; 305 requests contained compaction, and the
largest child request was 9,993/10,000 characters. It also repaired one deliberately
malformed model response, rejected oversized input before provider invocation and
recovered in the selected child. The aggregate was 24,900 cents.

The aggregation fixture computes exclusively from the current request; it does
not remember evicted report contents between requests. The request that wrote
the aggregate contained all seven complete batch reports at 31,332/32,000
characters. This catches the earlier retention-cap failure that caused a live
parent to keep rereading reports instead of finishing.

The earlier live collective run on the frozen runtime also passed: 50/50 correct
children, eight active at once, 304 compacted child requests and a verified total
of 24,900 cents in 275 seconds. The largest child request was 9,966/10,000
characters; the largest parent request was 31,993/32,000. Child follow-up,
oversized-input rejection and recovery all passed. The complete conditions and
outcome are in [the live report](verification/artifacts/workflow_tui_live_report.json).

An earlier live Self-Harness TUI run accepted one candidate: training 0/6 → 6/6,
validation 2/4 → 4/4 and post-selection tests 2/4 → 4/4, using 83 calls in 83
seconds. The complete paired outcomes and proposal are in
[the report](verification/artifacts/self_harness_tui_report.json).

The subsequent interactive QA reproduced the user's installed-package startup
crash, then verified the upgrade using a read-only copy of that exact old
installation. The default provider also completed a real two-reviewer workflow
and file repair, with approvals entered in the UI and an independent verifier
returning `QA_VERIFY_PASSED total=2440`. Child follow-up, keyboard/mouse switching,
parent return, cancellation, replacement input and denied writes passed. Model
mistakes and corrections are recorded in the
[manual interaction report](verification/artifacts/tui_manual_default_report.json).

The earlier packaged-source acceptance passed the nine terminal drivers that
existed at that time; it did not include the new plugin-guide driver. Its
50 children all compacted and returned correct subtotals, with a 24,900-cent
aggregate, peak concurrency eight, and maximum requests of 9,991/10,000 child
characters and 31,982/32,000 parent characters. Startup errors, existing-package
upgrades, separate child queues, picker cancellation, Unicode copying, oversized
pastes and saved-session navigation are included in the
[QA report](verification/artifacts/tui_qa_report.json), which records the exact
tested archive hash. Full transcripts are retained under
`/tmp/raychat-expanded/qa-release-acceptance-v1/`.

Three further live Self-Harness TUI trials used identical cases and provider
settings, totaling 250 calls. Two improved training from 0/6 to 6/6, validation
from 2/4 to 4/4, and post-selection tests from 2/4 to 4/4. The other improved
training only to 1/6 and stayed at 2/4 on both other splits: it learned the key
rule but often violated the required exact compact JSON formatting. All three
outcomes are retained in the
[Self-Harness QA report](verification/artifacts/self_harness_qa_report.json).
This demonstrates useful but inconsistent improvement on the measured tasks.

Historical evidence is retained under `docs/verification/artifacts/`; the
[index](verification/README.md) separates older API/optimization experiments from
terminal and live-provider evidence. Reports retain their original conditions,
paths and source hashes rather than being relabeled as new results. Full ANSI
transcripts, request logs and failed attempts from this development session are
under `/tmp/raychat-expanded/` and are not durable release attachments.

The final post-reorganization report records the full unit result, pinned static
checks, all thirteen extracted-release drivers, platform/interpreter and exact
source/archive identity. All 50 current-run children compacted and returned correct
subtotals; the aggregate was 24,900 cents. Nine persistence scenarios cover the
checkpoint corrections, including isolated child completion and cancellation.
Eight earlier terminal samples measured 118.67–120.00 full-frame writes per second;
that measurement included repeated identical screens. The current FPS driver
records completed application frames separately from terminal writes, measures
typing latency, and can throttle PTY reads to exercise output backpressure.
Historical live experiments retain their original scope. The separate maximum-strictness audit
is not included in the configured static-checks claim.

The Self-Harness cases deliberately omit an organization-specific deployment
policy. Held-in verifier feedback supplies examples from which the model can
learn that rule. Validation and post-selection tests use different inputs, but
the small public case set has also been used during development, and repeated
trials reuse cases. A passing comparison demonstrates useful instruction learning
on this task family, not broad or statistically established coding improvement.
Failed evaluations, proposals and runs remain evidence; they are not counted as
improvements.

Process cancellation includes isolated plugin commands on the main thread and
in an executor thread. The registered process runner cleaned up commands and
SIGTERM-resistant grandchildren in 0.197–0.249 seconds; cleanup markers and PID
checks were observed before replacement prompts. This proves managed subprocess
cleanup. Arbitrary plugins that detach unmanaged processes are outside this test.

The historical terminal evidence here is macOS/Python 3.12. Portable smoke extracts the exact
archive and runs the same offline TUI drivers on POSIX, retaining an explicit
coverage report with `--smoke-output`. Windows CI checks source compilation and
package/archive integrity, and reports that Windows TUI behavior was not tested.
