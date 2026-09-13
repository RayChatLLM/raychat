# RayChat

RayChat is a coding chat harness written in pure Python using the standard
library. A small session kernel owns conversation history, approvals, and cancellation.
Independent installed plugins provide tools, providers, skills, memory,
subagents, workflows, goals, and optimization.

Run it with Python 3.10 or newer. No third-party runtime packages are required.
`raychat.py` is the application entrypoint; the configured profile installs its
feature archives on first launch.

```bash
export FIREWORK_API_KEY="your-key"
python3 -B -S raychat.py --workspace ./workspace
```

On Windows, use `py -3` in place of `python3`. The endpoint, model, environment
variable names, plugin selection, limits, storage, and rendering settings live
in [raychat.json](raychat.json). `--config PATH` selects another complete JSON
configuration. CLI flags override settings for that launch. Credentials come
from the configured environment variables.

For a different Chat Completions endpoint:

```bash
python3 -B -S raychat.py --url https://provider.example/v1/chat/completions --model vendor/model
```

Custom endpoints use `LLM_API_KEY` by default. A custom URL requires an explicit
model. Configuration rejects duplicate keys, invalid types and ranges,
non-finite values, unsupported versions, and oversized files.

## Chat controls

- **Enter** submits a message. While working, Enter appends a follow-up to the queue.
- **Escape twice** stops the focused chat's task, HTTP call, command, or plugin
  command. Your draft stays editable; pending messages in that chat are discarded.
- **`/agents`** opens the subagent plugin's session window. Use Up/Down and
  Enter, click a row, or scroll with the mouse wheel. Escape closes the window.
- **`/parent`** returns from a child chat to its parent. The session window also
  has a parent row. You can chat with a child after its delegated task ends.
- **`/system`** toggles the details panel; **`/clear`** starts a fresh conversation.
- **`/quit`** or Ctrl+C closes the application and restores the terminal.

Stopping a child chat does not cancel its parent or siblings. If that child
belongs to a workflow, the parent receives a cancelled child result while the
other children continue. Child chats retain separate histories and drafts during
the application run. Subagent tools remain read-only.

Writes, edits, and process execution require approval unless `--yes` is set.
The approval window shows the complete action and requires reviewing all pages
before typing YES and Enter. Process tools execute argument arrays without a
shell and clean up their process trees on timeout or cancellation. Filesystem
paths stay within the configured workspace.

## Resume a conversation

Interactive conversations save automatically, grouped by workspace.

```bash
python3 -B -S raychat.py --workspace ./workspace --resume
```

With one saved session, it opens immediately. With several, a window lists
sessions newest first with dates, IDs, and prompt previews. Use arrows and Enter
or click a session. The selected chat restores its visible transcript, model
context, and committed plugin state.

```bash
python3 -B -S raychat.py --workspace ./workspace --resume SESSION_ID
```

`--session-dir PATH` changes the storage location; `--no-session` disables
persistence. `/sessions` lists saved
IDs. `/tree` lists completed turns and `/fork ENTRY_ID` selects a conversation
branch. `/resume` opens the only saved session or shows the same picker inside
chat when several exist. `/resume ID` opens a specific saved conversation.
Selecting the current conversation keeps its history and writer open.

Only completed turns become reusable context. Resuming does not replay tools or
goal loops. An active writer lock prevents two processes from modifying the
same saved session. Child conversations in `/agents` belong to the current
application run; the parent's saved transcript includes their workflow reports.

## Run a single job

For automation, pass an explicit prompt or plugin command with `--exec`:

```bash
python3 -B -S raychat.py --exec "Inspect the workspace and summarize it" --workspace ./workspace
python3 -B -S raychat.py --exec /plugins --no-memory
python3 -B -S raychat.py --exec "/optimize verify" --no-memory
```

This prints the final result to stdout and errors to stderr. It never reads
interactive approvals from stdin; mutations are denied unless `--yes` is set.
A successful job exits 0, a failure exits 1, and interruption exits 130.
`--exec` is ephemeral unless resuming a saved conversation. With several saved
sessions, `--exec --resume` requires an explicit ID.

## Plugins

| Plugin | Responsibilities |
| --- | --- |
| `filesystem` | List, read, write, byte edits, confinement, hashes, atomic writes |
| `process` | Argument-array execution, environment policy, bounded output, timeouts, process-tree cleanup |
| `skills` | Discovery, catalog, loading, session-scoped skill content |
| `memory` | Durable memory, paging, actions, context contributions |
| `chat_completions` | Configured HTTP provider, credentials, request/response validation |
| `subagents` | Profiles, routing, isolated child execution, `/agents`, `/parent`, lifecycle events |
| `workflows` | Serial delegation and bounded parallel `delegate_many` batches |
| `goals` | `/goal`, judging, retries, continuation, revisions, final-response filtering |
| `optimization` | GEPA, evaluators, reports, demos, benchmarks, `/optimize`, `/incident` |
| `context` | Instruction assembly and context compaction |
| `plugin_manager` | `/plugins`, discovery, install, live reload and removal |
| `self_harness` | Failure evidence, same-model proposals, isolated validation, gated live promotion |

The default release profile installs independent archives from `plugin_catalog/`
into the user plugin directory. Feature source projects live in `plugins/`;
startup loads installed packages through the package manager. Every package has
a `plugin.json` manifest and registers through `raychat.sdk.PluginAPI`. Adding an
external package requires no changes to the core, CLI, or terminal controller.
The manifest's `instructions` field tells the model how to use that package and
updates with the active plugin generation.

Load a development package with `--plugin PATH`. Disable an installed feature
with `--disable-plugin NAME`; its dependents must also be disabled.
`--no-plugins` creates an empty registry. Workspace plugin discovery requires
explicit trust; see [the plugin guide](docs/PLUGINS.md) and
[the counter example](examples/plugins/counter).

Plugins can be added or edited while the application is running. Trusted source
changes activate at the next idle operation. `/plugins install`, `/plugins link`,
`/plugins reload`, `/plugins disable` and `/plugins uninstall` manage packages.
Use `/plugins new`, `/plugins check` and `/plugins pack` to create and share them,
and `/plugins catalog` and `/plugins search` to discover external packages. A failed
replacement retains the working generation and conversation. Existing child
chats adopt updated profiles at their own next job boundary.

The [Self-Harness plugin](docs/SELF_HARNESS.md) proposes bounded prompt/plugin
changes and validates them before live promotion. For example:

```text
/self-harness --scores -- python3 -B -S evaluator.py
```

Its default gate requires improvement without a held-in or held-out regression;
`--exit-code` supports Nano's simpler command-success validator.

For an embedded session with a supplied model callable:

```python
from raychat.composition import create_session

session = create_session(chat, "workspace", plugins=["filesystem", "context"])
try:
    result = session.run("Read README.md and summarize it")
finally:
    session.close()
```

The kernel itself is `raychat.session.AgentSession`. Features are imported from
their owning plugins; there is no compatibility API or separate linear chat UI.

## Goals and workflows

Use `/goal Implement the change and verify the tests` to continue until a judge
accepts completion. `/goal clear` disables it. Profiles and routing live in the
`plugins.settings.subagents` section of the configuration. The model can request serial
`delegate` actions or bounded parallel `delegate_many` batches.
See [subagents, workflows, and goals](docs/SUBAGENTS.md).

## Optimization

The optimization plugin contains its engine, evaluators, commands, and resources,
including the typed standard-library GEPA port in
`plugins/optimization/gepa/`. Optimization commands run in isolated,
cancellable Python processes. Relative output paths resolve within the workspace.

Enter these commands in chat; automation can pass the same strings to `--exec`:

```text
/optimize verify
/optimize demo --output optimized.txt
/optimize useful-demo
/incident demo --output incident.txt
/optimize --help
```

The active GEPA source is adapted from the pinned upstream launch revision.
Verification reports its current source hash and import audit, upstream
provenance, unchanged prompt templates, and the deterministic transcript oracle.
Read [GEPA provenance](docs/GEPA_PORT.md) for the scope of those checks.

## Repository layout

| Path | Contents |
| --- | --- |
| `raychat.py` | Small application launcher |
| `raychat.json` | Operator configuration and the explicit release allowlist |
| `raychat/configuration.py` | Configuration loading, validation and defaults; replaces the old root configuration module |
| `raychat/` | Session kernel, public SDK, package manager, storage and application composition |
| `raychat/workers.py` | Background conversations, approval handoffs and cancellation, independent of terminal rendering |
| `raychat/ui/` | Terminal input, controller, rendering, selection, state and session pickers |
| `plugins/` | Independent feature projects, including the optimization engine and its resources |
| `plugin_catalog/` | Distributable feature ZIPs, catalog metadata and standard profile; rebuild after plugin edits |
| `tools/` | Acceptance drivers and development/release commands, including `tools/release.py` |
| `tests/` | Executable unit and integration regression tests |
| `docs/verification/` | Indexed historical evidence and final-acceptance status |
| `build/` | Ignored generated ZIP, release folder and local verification output |

There is no root `optimization/` implementation or checked-in `dist/` release
copy. The optimization plugin owns its code; release commands generate
`build/raychat.zip` and `build/raychat/`. Historical reports belong under
`docs/verification/artifacts/`, separate from executable feature resources.

## Development acceptance

Build the distribution packages after changing plugin source. Mypy and Ruff are
development tools; the harness and behavioral drivers use only the standard
library. Run static checks with Python 3.12 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
python3 -B -S -m tools.build_plugin_catalog
.venv/bin/python tools/verify_quality.py
.venv/bin/python tools/check_types.py
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
python3 -B -S -m unittest discover -s tests -v
```

See [strong plugin contracts](docs/TYPING.md) for typed services, validated
settings and events, and the distinction between strict mypy and rejecting all
uses of `Any`.

The complete unit suite is required alongside static checks and actual TUI
acceptance; none substitutes for the others. Record failures, errors and skips
with the tested source revision. Historical packaged QA is retained in
[the verification index](docs/verification/README.md). The current
`verify_quality.py` gate independently inventories source files, applies maximum
mypy and Ruff rules, checks formatting, and fails if sources change during the
run. Its report lists every diagnostic and explicit rule exception.

TUI acceptance drives the actual terminal UI on POSIX. Each output path
must be new; reports, provider requests, and terminal transcripts remain there.

Rendering targets 120 FPS by default. Unchanged frames produce no terminal
output, and the normal chat view skips its hidden ray tracing. The FPS check
records completed application frames, PTY output volume, and typing latency at
four sizes in chat and system views, idle and awaiting an offline provider.
It fails below 90 application FPS or above 100 ms typing latency. These are
application measurements, not measurements of a terminal emulator's display.
Use `--read-bytes-per-second 1000000` to also test output backpressure.

```bash
python3 -B -S -m tools.fps_tui --output ./build/verification-run-1/fps
python3 -B -S -m tools.bare_tui --output ./build/verification-run-1/bare
python3 -B -S -m tools.accept_tui --output ./build/verification-run-1/interaction
python3 -B -S -m tools.features_tui --output ./build/verification-run-1/features
python3 -B -S -m tools.startup_tui --output ./build/verification-run-1/startup
python3 -B -S -m tools.profile_upgrade_tui --output ./build/verification-run-1/profile-upgrade
python3 -B -S -m tools.persistence_tui --output ./build/verification-run-1/persistence
python3 -B -S -m tools.package_download_tui --output ./build/verification-run-1/package-download
python3 -B -S -m tools.adversarial_agents_tui --output ./build/verification-run-1/adversarial-agents
python3 -B -S -m tools.ui_stress_tui --output ./build/verification-run-1/ui-stress
python3 -B -S -m tools.composer_tui --output ./build/verification-run-1/composer
python3 -B -S -m tools.collective_tui --agents 50 --parallel 8 --output ./build/verification-run-1/collective
python3 -B -S -m tools.optimization_tui --output ./build/verification-run-1/optimization
python3 -B -S -m tools.plugin_guide_tui --output ./build/verification-run-1/plugin-guide
python3 -B -S -m tools.reload_race_tui --output ./build/verification-run-1/reload-race
```

These fourteen drivers use deterministic offline model fixtures while exercising
installed plugins, navigation, focused cancellation, commands, file/process operations,
memory, skills, goals, workflow children, context compaction, and the documented
external-plugin authoring/install workflow. The collective run verifies 50 child results and their aggregate, then tests a child follow-up,
context-budget rejection, and recovery. It demonstrates harness behavior under
load; it does not measure a live model's problem-solving ability.

Persistence checks cover plugin checkpoints, failed forks, damaged saved chats
and keyboard/mouse resume. They also preserve concurrent command checkpoints
through cancelled and completed turns, including isolated workflow child chats.
Download checks stall the HTTP server, cancel in the
TUI and require a replacement reply before releasing the server, including from
a workflow child while its parent and sibling remain active.

To measure Self-Harness with the configured live provider:

```bash
python3 -B -S -m tools.self_harness_tui --repetitions 2 --output ./build/verification-run-1/self-harness
```

This spends provider credits. It requires an accepted candidate and improved
results on the experiment's fixed synthetic test cases, and retains evidence if
no improvement occurs. Those results describe that experiment; repeated cases
are not independent samples or evidence of broad coding improvement. The
collective driver also accepts `--live` for a real-provider run.

Build and test the exact release with retained evidence:

```bash
python3 -B -S -m tools.build_portable --smoke --smoke-output ./build/verification-run-1/release
python3 -B -S -m tools.build_portable --check --no-smoke
```

On POSIX, release smoke repeats the offline TUI acceptance against the extracted
ZIP. On Windows, it verifies archive/package integrity and compiles the source;
it does not exercise the Windows TUI. See [release instructions](docs/RELEASE.md)
and [recorded verification results](docs/VERIFICATION.md).

Drag transcript text and release the mouse to copy it automatically.
Confirmation appears below the composer; Escape clears the selection. This works in
child chats too. macOS uses its native clipboard when available; other terminals
receive an OSC 52 clipboard request. `tui.clipboard: "terminal"` always uses OSC 52.
Terminal clipboard permissions may need to be enabled. Resizing the transcript
clears selection because wrapped cell coordinates change.

While a task runs, Enter adds messages to the visible FIFO queue. Shift+Up opens
the newest entry and moves toward older entries; Shift+Down moves toward newer
ones. Click an entry to edit it directly. Edits stay temporary while browsing;
Enter saves all edits and restores your draft, and Escape discards the edits.
Queue dispatch pauses while editing, including in subagent and workflow chats.

Type `/` to browse commands. Up/Down selects; Tab, Enter, or a click fills the
command without executing it. Press Enter again to submit. Plugin command menus
and the dynamic status footer update as plugins are loaded or removed.

## License

RayChat is licensed under the [MIT License](LICENSE). Third-party notices, including the GEPA license, remain with their packages.
