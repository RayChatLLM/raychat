# Live core updates and recovery

Interactive `python raychat.py` runs a stable supervisor and a replaceable core
process. The supervisor owns native terminal modes and buffers input while cores
are exchanged. `--exec` remains a single noninteractive job.

- `/update /absolute/path/to/source` copies a developer checkout into an isolated
  candidate. Live validation checks imports, the fixed test suite and
  deterministic packaging with the application's own interpreter; linters and
  type checkers are development-time tools and do not run in the live gate. `/update` uses the launch source.
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

The ordinary chat agent edits the application through the **source mirror**:
in a trusted workspace the core materializes a writable copy of the active
release's `raychat/` and `plugins/` trees at `.raychat/source` (configurable via
`storage.staging_directory`). The agent reads and edits those files with the
ordinary filesystem tools, exactly like any other project files. At each turn
boundary - once every chat, command worker and runtime is quiescent - the host
fingerprints the mirror and, when its content digest differs from the active
release, submits the whole tree to the supervisor for validation and automatic
live activation. There are no bespoke editing tools and no editing-specific
prompt. A **system** notice reports that validation is pending; other jobs keep
running. Untrusted workspaces get no mirror and submit nothing, the operator can
disable automatic submission with `chat.auto_core_updates`, and probe launches
never submit.

After validation and handoff, the supervisor automatically resumes the agent with
`CORE_UPDATE_RESULT` JSON: `request_id`, `session_id`, `status` (`activated`,
`rejected`, `busy`, or `interrupted`), `ok`, original request, active/previous
release IDs, diagnostics, and a fresh screen captured after the replacement owns
the terminal at its actual dimensions. Activation means the new code is running
in the open terminal; no restart is needed. It does not establish that the
requested behavior is correct - the agent verifies against the result's screen
field or by reading the running state. After an activation the mirror is
refreshed from the activated release, unless the mirror changed again while
validation ran, in which case the newer edits are preserved and resubmitted at
the next boundary.

During the automatic review turn the agent may use the ordinary file and process
tools plus `core_recover` to repair a rejected change; edits resubmit at the next
turn boundary. The host stops automatic resubmission after
`limits.staging_reject_limit` consecutive rejections (default three) and after
`busy` or `interrupted` results; the next user message lifts the suppression. A
tree byte-identical to the last rejected submission is never resubmitted. Each
review is limited to twenty model turns; if the model reaches that limit, the
host commits the actual update status with an explicit limit notice, then closes
the review. A result for a different saved session waits until that session is
resumed; unrelated queued work can continue.

`core_recover` is the one remaining agent tool: it restores the `previous` or
`known-good` release after active work stops, with host approval. Recovery also
resets the source mirror to the restored release, so rolled-back edits cannot
resubmit themselves.

Results survive handoff and remain pending until the feedback worker durably claims
them immediately before its first model request. A crash before that claim leaves
the result available for delivery. After the claim, recovery retains evidence of
the uncertain provider request and does not replay it. The host clears that evidence
only after the assistant response is committed to the journal.
Long-running tasks can postpone activation indefinitely. Deleting
workspace `.raychat` files does not change the UI.

Every launch retains private recovery files outside the workspace, below
`~/.raychat/live/LAUNCH_ID/` (or the configured storage home). Changing the bootstrap
storage location requires one restart of an older supervisor. Source releases are read-only and checked
against their recorded SHA-256 identity before launch. The launch release and
previous release remain available across repeated updates. They are not pruned
while the terminal is open. Keep this directory if recovery may be needed later.
Each successfully activated release also retains its initial compatible state
snapshot independently of later checkpoints.

Recovery writes are serialized within the supervisor. Temporary files are flushed,
synced, and closed before atomic replacement. Windows permission errors during
replacement are retried for up to half a second. If saving still fails, the last
saved file remains intact and the footer reports the failure; the supervisor stays
open. New work is not acknowledged and planned activation is deferred until its
recovery state can be saved. A later successful manifest write clears the warning.

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
