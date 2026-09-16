# Verification

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
observations. Retain each run's source identity, complete output, and generated
reports in its chosen output directory or CI artifacts.

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
with `RAYCHAT_AUTH_TOKEN`, `RAYCHAT_MODEL`, and `RAYCHAT_BASE_URL` before launching.
The offline drivers supply their own synthetic environment values:

```bash
python3 -B -S -m tools.collective_tui --output /tmp/raychat-live-collective --agents 50 --parallel 8 --live
python3 -B -S -m tools.self_harness_tui --output /tmp/raychat-live-self-harness --request-options '{"reasoning_effort":"low"}'
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

## Coverage limits

The FPS driver records completed application frames separately from terminal
writes, measures typing latency, and can throttle PTY reads to exercise output
backpressure. A measured application frame rate does not establish the physical
display rate of every terminal emulator.

The Self-Harness cases deliberately omit an organization-specific deployment
policy. Held-in verifier feedback supplies examples from which the model can
learn that rule. Validation and post-selection tests use different inputs, but
repeated trials reuse the small public case set. Passing demonstrates useful
instruction learning on this task family, not general coding improvement.
Failed evaluations and runs are retained as failures rather than improvements.

Portable smoke extracts the exact archive and runs the offline TUI drivers on
POSIX, retaining a coverage report with `--smoke-output`. Windows CI checks source
compilation, provider configuration, HTTP capture/replay, and package/archive
integrity; it reports that Windows TUI behavior was not tested.
