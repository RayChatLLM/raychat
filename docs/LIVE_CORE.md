# Live core updates and recovery

Interactive `python raychat.py` runs a stable supervisor and a replaceable core
process. The supervisor owns native terminal modes and buffers input while cores
are exchanged. `--exec` remains a single noninteractive job.

- `/update /absolute/path/to/source` copies a developer checkout into an isolated
  candidate. `/update` uses the launch source.
- Self-Harness can generate actual Python source changes under `raychat/` and
  `plugins/`, including its own implementation. Configure a fixed outcome evaluator
  with `/self-harness --scores -- VALIDATOR ARG...`, or the explicit exit-code
  mode. Recurring failures and measured improvements determine which proposal is
  submitted to the supervisor. An accepted overlay accompanies that release.
- `/recover previous` selects the previous release. `/recover known-good` selects
  the launch release. Both wait for active work to finish and check restoration.
- **Ctrl+R** opens the supervisor's recovery screen even when the core has crashed
  or stopped responding. `p` selects the previous version; `g` selects the launch
  version. This emergency operation terminates the core, reloads committed journal
  history, and pauses queued work. `/resume-queue` explicitly resumes the queue.
- `/update-log` displays the diagnostic log path. Update status and failures appear
  in the dynamic footer.

The ordinary chat agent can inspect and edit the core through `core_source` and
`core_update`, restore versions through `core_recover`, and read checker diagnostics with
`core_status`. `core_verify` checks exact text in named rendered UI regions. These tools work
without Self-Harness. Ask for a UI change in chat; source edits are validated before
automatic activation. Submission yields the requesting task and displays a **system** notice that
validation is pending. It does not claim activation. Other jobs keep running.
After successful `core_source` inspection, that task can use only core tools and
completion. Attempts to fall back to workspace tools receive a redirect to the
active source tools. Ordinary workspace tools return on the next user task.
After validation and handoff, the supervisor automatically resumes the agent with
`CORE_UPDATE_RESULT` JSON: `request_id`, `session_id`, `status` (`activated`,
`rejected`, `busy`, or `interrupted`), `ok`, original request, active/previous
release IDs, diagnostics, and a fresh screen captured after the replacement owns the
terminal at its actual dimensions. Activation means the new code is running in the
open terminal; no restart is needed. It does not establish that the requested
behavior is correct.

The agent can inspect the result and repair a rejected or incomplete change.
`core_verify` accepts checks such as
`{"region":"system","kind":"wrapped_contains","text":"accounts/vendor/models/name"}`
in a `checks` array. Regions are `system`, `header`, `transcript`, and `composer`;
predicates are `contains`, `absent`, and `wrapped_contains` (text joined across
displayed lines). Missing, inactive, or stale regions fail even an absence check.
SYSTEM evidence excludes the top header and conversation. Its source is
`_sidebar_details` and `_paint_sidebar` in `raychat/ui/controller.py`;
`_paint_header` only paints the title row. `core_source` always reads the active
immutable release, including during ordinary follow-up questions. A developer
checkout can differ from the source currently running.

When the review finishes, the host composes and journals its final report from the
actual update status and listed checks. An unverified model completion cannot
become the visible or saved review report. With no checks, the report explicitly
says the visual outcome is unverified. A passing text check establishes only that
predicate, not layout quality or complete user intent; `task_verified` stays false.
Automatic review permits only the core tools and completion;
workspace commands stay unavailable during that review. The host limits automatic
repairs to three submissions and each review to twenty model turns. Busy or
interrupted results are reported without automatic resubmission. A result for a
different saved session waits until that session is resumed; unrelated queued work
can continue.
If the model reaches the review turn limit, the host commits the actual update
status and available checks with an explicit limit notice, then closes the review.

Results survive handoff and remain pending until the feedback worker durably claims
them immediately before its first model request. A crash before that claim leaves
the result available for delivery. After the claim, recovery retains evidence of
the uncertain provider request and does not replay it. The host clears that evidence
only after the assistant response is committed to the journal.
`core_status` exposes the last structured result, active source, current phase,
phase elapsed time, and captured screen for later questions. ETA remains unknown
(`null`); long-running tasks can postpone activation indefinitely. Deleting
workspace `.raychat` files does not change the UI.

Every launch retains private recovery files outside the workspace, below
`~/.raychat/live/LAUNCH_ID/` (or the configured storage home). Changing the bootstrap
storage location requires one restart of an older supervisor. Source releases are read-only and checked
against their recorded SHA-256 identity before launch. The launch release and
previous release remain available across repeated updates. They are not pruned
while the terminal is open. Keep this directory if recovery may be needed later.
Each successfully activated release also retains its initial compatible state
snapshot independently of later checkpoints.

After closing the terminal, recover before importing the checkout's core code:

```sh
python raychat.py --recover-core ~/.raychat/live/LAUNCH_ID \
  --core-version known-good
# --core-version previous selects the retained previous version instead.
```

A restored core acknowledges readiness before the supervisor grants dispatch
ownership. Failed activation before dispatch restarts the previous immutable
release using the same handoff. After a crash, recovery reads the current committed
journal rather than selecting an earlier history branch. A durable dequeue intent
prevents an uncertain command from being silently returned to the queue. Recovery
does not undo external effects and does not replay interrupted commands.

Isolated workers also watch the exact parent process that launched them. Parent
loss triggers cancellation and resource cleanup, with a bounded exit grace for
blocked workers. This stops orphaned isolated workflows from continuing to dispatch
new steps after core recovery. Already-completed external effects remain recorded.

If the latest state cannot be restored, recovery tries the selected release's
compatible state snapshot. It restores that snapshot's plugin state and drafts,
reads committed history from its journal, and discards stale queued commands and
pending keystrokes. The original latest handoff remains available separately.
If neither state can be restored, emergency recovery opens that version with
plugins and persistence disabled. Original journals
are retained, and the last handoff is kept in `recovery-before-safe.json`. This gives
a usable terminal from which to repair the source or choose another recovery version
without destroying evidence or committed history.

## Validation and activation

The supervisor selects the checkout's `.venv/bin/python` for evaluation when
available, otherwise its launch interpreter. It checks `requirements-dev.txt` pins
before accepting generated code; incompatible global checkers cannot silently
change the gate. Changing this bootstrap selection requires restarting an older
supervisor once.

The supervisor captures its evaluator at launch. Candidate edits cannot replace
`raychat_bootstrap/`, `raychat.py`, tests, build tools, checker configuration, or the
launch configuration. The fixed toolchain normalizes candidate source formatting
before validation; semantic lint/type failures still reject the candidate. Validation
runs imports, strict lint/type/format checks,
the fixed test suite, and portable packaging. The optional Self-Harness outcome
evaluator runs separately and cannot be edited through proposal paths.

Development checkers from `requirements-dev.txt` must be installed in the selected
evaluation interpreter to accept source updates. The running application, transport, supervisor,
and recovery use only the Python standard library. Missing checkers reject an update
and leave the running core available.

After static validation, the core stops dispatching new queued work. Tasks, workflows,
subagents, and background commands finish normally; there is no deadline and no forced
cancellation. Users may type, edit queues, navigate, and use existing cancellation
controls while waiting. After cancellation, activation still waits for every worker
to exit its job and finish cleanup. At idle the supervisor establishes an input barrier, captures
state, and runs a restoration probe without opening the live journal. The old core
then closes its workers, plugin resources, logs, and journal. Only after its process
exits can the replacement acquire the journal's exclusive writer lock. Buffered input
and queued work resume after restoration succeeds and ownership is granted.

The handoff carries semantic conversation history, plugin state and sources, child
conversations and parent relationships, focus, transcript-only notices, drafts and
cursor positions, queue identities and temporary edits, suspended drafts, menus,
scroll/selection state, and partial UTF-8, escape, mouse, or bracketed-paste input.
Render caches and live Python objects are rebuilt in the new process.

## Plugin contract

SDK v4 plugins may register an additional process-handoff contract:

```python
api.on_handoff(export_json, restore_json, idle=background_work_is_idle)
```

`export_json(ctx)` returns finite JSON-compatible data; `restore_json(value, ctx)`
reconstructs resources without starting new work. The optional `idle()` predicate
must remain false while plugin-owned background work is executing. Ordinary
namespaced `ctx.state` is already part of the session snapshot. Plugins with cleanup
or in-process reload resources must also provide a process handoff, otherwise live
activation is rejected with the missing plugin names. Existing `on_reload` handlers
continue to support plugin generation reloads independently of core replacement.

The supervisor is deliberately outside ordinary self-modification. Updating bootstrap
behavior requires a restart. The immutable copies and evaluator boundaries guard the
supported update path; they are not an operating-system sandbox for hostile Python
code running under the same user account.

## Reproducing the terminal checks

With the development checkers installed, run these from the source checkout using
fresh output directories:

```sh
python -m tools.live_core_tui --output build/live-core-check
python -m tools.live_cancel_tui --output build/live-cancel-check
```

Both run the full fixed candidate evaluator and drive actual terminal input.
The cancellation check holds a task and a background command open through validation,
cancels them separately with double Escape, and verifies that activation waits for
the last worker's cleanup. It then checks changed visible behavior, preserved Unicode
input, and continued terminal usability. Each command saves a JSON report and the
terminal transcript.


The live-model check sends ordinary chat requests through the configured provider,
with Self-Harness disabled. It asks for removal and known-good restoration
twice in the same animated terminal, verifies both the label and graphic, and
records source diffs, release identities, keyboard/mouse input, and Unicode drafts.
Each removal runs the fixed candidate evaluator. The provider needs a model capable
of generating valid code in RayChat's JSON action protocol. For the recorded test:

```sh
.venv/bin/python -B -m tools.live_agent_tui --output build/live-agent-check
```

This uses `RAYCHAT_AUTH_TOKEN`, `RAYCHAT_MODEL`, and `RAYCHAT_BASE_URL` and sends
inspected source to that provider. The report and captured screens distinguish successful activations from
rejected or incomplete proposals; a model's completion message alone is not a pass.
