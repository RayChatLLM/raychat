# RayChat

RayChat is a coding chat harness written in pure Python using the standard
library. A small session kernel owns conversation history, approvals, and cancellation.
Independent installed plugins provide tools, providers, skills, memory,
subagents, workflows, goals, and optimization.

Download a release from [GitHub Releases](https://github.com/RayChatLLM/raychat/releases),
extract the entire ZIP, and run `python raychat` in the extracted folder.
Python 3.10+ is required; no installer or pip packages are needed for ordinary use.
See the [release quick start](docs/RELEASE_README.md).

The instructions below apply to a source checkout.

Run it with Python 3.10 or newer. No third-party runtime packages are required.
`raychat.py` is the application entrypoint; the configured profile installs its
feature archives on first launch.

```bash
python raychat.py --workspace ./workspace
```

On first launch, a setup window asks for your API token, model ID, and API base
URL. Paste each value, then choose **Save and continue**. RayChat checks the token
and API base URL by querying `GET /models`. If authentication or the request fails,
the setup window shows an error and keeps your entries so you can correct them
and retry. Settings are saved only after the check succeeds. RayChat saves them in
`~/.raychat/environment/.env` in your user storage and loads them automatically
on later runs. The installation directory can be read-only.

Use `/models` in the TUI to fetch all model IDs from the configured provider's
OpenAI-compatible `GET /models` endpoint. Type to filter; use arrows, Page Up/Down,
Home/End, Enter, or click to inspect an ID. Escape closes the menu. To change the
active model, edit `RAYCHAT_MODEL` in your saved environment file and restart
RayChat. Saved sessions and profile settings cannot override it. Child chats,
task evaluation, and optimization
reflection use the same endpoint, model, and token as the main chat.
The `/system` panel wraps the complete model identifier onto multiple lines,
including its provider path.

Self-Harness is disabled by default. The ordinary chat agent still has the core's
source inspection, live-update, status, and recovery tools. See
[live core updates](docs/LIVE_CORE.md) for activation and recovery controls.

Every launch requires exactly three provider settings. Incomplete settings open
the interactive setup window; `--exec` or launches without a terminal report the
missing values instead. `--help` works without them. `RAYCHAT_BASE_URL` is the API
root; RayChat derives `/chat/completions` and `/models` from it. A complete URL ending
in `/chat/completions` is also normalized to that same root. There is no default
provider address or model, and `--model` and `--url` are removed. Other providers'
credential environment variables are not consulted.

Plugin selection, request options, limits, storage, and rendering settings live
in [raychat.json](raychat.json). `--config PATH` selects another complete JSON
configuration. Provider values belong in the selected environment file;
`storage.home_directory` controls the default location of that file.
Configuration rejects duplicate keys, invalid types and ranges, non-finite values,
unsupported versions, and oversized files.

### Environment files

You can use the setup window or edit the saved file directly. The window has
three editable fields, a masked token, and a **Save and continue** button. Use
Tab/Shift+Tab or click to move between fields; Enter advances or saves. Escape
cancels without saving. Invalid values and write failures stay in the form so
you can correct them. At 80×14 or 80×12, the form uses a compact layout with the
fields, Save button, errors, and keyboard controls visible. Resizing preserves
your entries and focus.

By default, settings live in `~/.raychat/environment/.env` on every platform.
Changing `storage.home_directory` in your selected JSON configuration also moves
this file to that application storage directory. RayChat creates the settings
directory when you save; it does not need write access to its installation.

Use `--env-file PATH` to select another writable settings file for both loading
and saving:

```bash
python raychat.py --env-file ~/raychat-provider.env
```

For an explicit portable setup, use `--portable`. This loads and saves
`environment/.env` beside the launcher instead of the user-storage file. The
installation must be writable for the setup window to save there. These two
options are mutually exclusive; use the same option on subsequent launches.

To fill in a portable file manually, copy your platform's example once:

Linux or macOS:

```bash
cp environment/linux.env environment/.env  # macOS: use environment/macos.env
# Edit environment/.env with your preferred text editor.
python raychat.py --portable --workspace ./workspace
```

Windows (PowerShell):

```powershell
Copy-Item environment/windows.env environment/.env
notepad environment/.env
# Fill in all three values, save, and close Notepad.
python raychat.py --portable --workspace ./workspace
```

All locations use the same format:

```dotenv
RAYCHAT_AUTH_TOKEN=your-api-token
RAYCHAT_MODEL=your-model-id
RAYCHAT_BASE_URL=https://provider.example/v1
```

Quotes around values are optional. Blank lines and full-line `#` comments are
supported; values are literal text, with no shell expansion. Do not source the
file or run an export/PowerShell import command.

Nonblank shell variables override file values; missing or blank variables use
the selected file. Relative `--env-file` paths are resolved from the launch
directory. Portable paths are resolved relative to the launcher, independent of
the current directory or workspace. Other environment files and the platform
examples are not loaded automatically. If you already filled in the installation's
`environment/.env`, launch with `--portable` or move it to your user-storage path.
Restart RayChat after editing settings.

The default file lives outside the installation; the portable `.env` is ignored
by Git. Neither is included in releases or core snapshots. Keep custom settings
files private too. The setup window saves atomically and uses owner-only
permissions on POSIX. There are no default provider values or extra packages.

## Raw HTTP debugging

Enable raw HTTP capture for a launch with `--debug`:

```bash
python raychat.py --debug --debug-dir ./build/http-debug --workspace ./workspace
```

The default directory is `.raychat-http-debug`, resolved relative to the launch
directory. Set `chat.debug` to `true` and `chat.debug_dir` in your configuration
to keep these options enabled. `--debug-dir` selects the destination; it does not
enable capture by itself. `RAYCHAT_HTTP_DEBUG_DIR` also enables capture at the
specified directory and passes it to isolated workers and child agents.

Each connection gets its own subdirectory. `sent.http` contains the bytes
submitted before TLS encryption, including Authorization headers, API keys, and
payloads without redaction. `connection.log` uses readable UTC timestamps for
request progress, response status, and connection failures. `events.jsonl` keeps
the same timestamps alongside precise timing and structured details,
and `send_attempt` / `sent` events so a failed send is distinguishable from a
completed send. `received.http` retains the exact bytes read after TLS
decryption, before HTTP parsing, including duplicate headers and chunk framing.

Each `request-0001`, `request-0002`, etc. subdirectory contains a `curl.txt`
command ready to copy into a POSIX shell and a `curl.ps1` command for PowerShell.
These commands retain the URL, headers, credentials, and payload. Text payloads
up to 4 KiB appear directly in the POSIX command when possible; larger or binary
payloads and the PowerShell command use an adjacent immutable
`request-body-<sha256>.bin` file by absolute path. Previously copied commands keep
their original body when a later capture updates the replay. `request-body.bin`
is a convenience copy of the latest exported body; commands do not reference it.
Retain the trace directory while any saved replay command still needs its files.
Curl recreates HTTP framing itself. Commands for text and byte payloads are also
available when the connection fails, so you can retry the same request manually.
Streamed request bodies get a replay command after sending finishes.

Captures are written incrementally and survive cancellation. A `complete` event
identifies a completed response; `response_closed` with `complete: false`
identifies a partial response, including one stopped at the response-size limit.
Cancelled or interrupted exchanges may have no final event. Existing response
size, timeout, and cancellation limits remain in effect. In debug mode, HTTP
error bodies are also drained into the capture without a separate logging size
cap. Ordinary user-facing errors still redact credentials; the raw files contain
credentials and conversation content.

Capture begins before startup plugin downloads and also covers provider requests
and model discovery. Debugging is disabled by default and creates no capture
files while disabled. `--help` does not enable capture. The separate `--log PATH`
option writes conversation JSONL rather than HTTP traffic.

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
python raychat.py --workspace ./workspace --resume
```

With one saved session, it opens immediately. With several, a window lists
sessions newest first with dates, IDs, and prompt previews. Use arrows and Enter
or click a session. The selected chat restores its visible transcript, model
context, and committed plugin state.

```bash
python raychat.py --workspace ./workspace --resume SESSION_ID
```

`--session-dir PATH` changes the storage location; `--no-session` disables
persistence. `/sessions` lists saved
IDs. `/tree` lists completed turns and `/fork ENTRY_ID` selects a conversation
branch. `/resume` opens the only saved session or shows the same picker inside
chat when several exist. `/resume ID` opens a specific saved conversation.
Selecting the current conversation keeps its history and writer open.

The default application data directory is `.raychat` under your user home.
To use another approved local location, select a complete configuration with
`--config PATH` and set `storage.home_directory` to its absolute path. This moves
user plugin state, trust decisions, default saved sessions, and supervised-core
recovery data. `--session-dir PATH` overrides only the session location. Workspace
plugin state remains under the selected workspace's configured plugin directory;
choose a writable workspace separately with `--workspace PATH`. The installation
directory does not need to hold application data. Directory or security-policy
denials are reported; the application does not clear permissions or require
security-software exclusions.

Only completed turns become reusable context. Resuming does not replay tools or
goal loops. An active writer lock prevents two processes from modifying the
same saved session. Child conversations in `/agents` belong to the current
application run; the parent's saved transcript includes their workflow reports.

## Run a single job

For automation, pass an explicit prompt or plugin command with `--exec`:

```bash
python raychat.py --exec "Inspect the workspace and summarize it" --workspace ./workspace
python raychat.py --exec /plugins --no-memory
python raychat.py --exec "/optimize verify" --no-memory
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
| `self_harness` | Optional, disabled by default: failure evidence, proposals, isolated validation |

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

When explicitly enabled in `raychat.json`, the [Self-Harness plugin](docs/SELF_HARNESS.md)
proposes bounded prompt/plugin changes and validates them before live promotion. For
example:

```text
/self-harness --scores -- python evaluator.py
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

Use `/goal Implement the change and verify the tests` to start work immediately
and continue until a judge accepts completion. No follow-up message is required.
`/goal` shows status; `/goal clear` disables it. Profiles and routing live in the
`plugins.settings.subagents` section of the configuration. The model can request serial
`delegate` actions or bounded parallel `delegate_many` batches.
See [subagents, workflows, and goals](docs/SUBAGENTS.md).

## Optimization

The optimization plugin contains its engine, evaluators, commands, and resources,
including the typed standard-library GEPA port in
`plugins/optimization/gepa/`. Optimization commands run in isolated,
cancellable Python processes. Relative output paths resolve within the workspace.

Protocol exports and JSON reports each publish a complete file. Concurrent exports
to the same destination use last-writer-wins behavior; output and report paths
within one command must be distinct. The two files publish independently. A report
failure may leave a successfully published protocol alongside an older report;
compare the report's optimized-protocol SHA-256 with the protocol before relying on
them together. Export publication failures stop the command without rerunning the
optimization or deleting an existing destination.

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
| `environment/` | Local provider settings and credential-free platform examples |
| `build/` | Ignored generated ZIP, release folder and local verification output |

There is no root `optimization/` implementation or checked-in `dist/` release
copy. The optimization plugin owns its code; release commands generate
`build/raychat.zip` and `build/raychat/`. Local verification output belongs in
`build/`; CI retains its own run artifacts.

## Development acceptance

Build the distribution packages after changing plugin source. Mypy and Ruff are
development tools; the harness and behavioral drivers use only the standard
library. Run static checks with Python 3.12 or newer:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
python -m tools.build_plugin_catalog
.venv/bin/python tools/verify_quality.py
.venv/bin/python tools/check_types.py
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
python -m unittest discover -s tests -v
```

After a catalog rebuild changes artifact names, copy the builder's printed paths
into the sorted `release.source_files` catalog entries in `raychat.json`. See
[catalog publication and release preparation](docs/RELEASE.md#prepare-the-distribution).

See [strong plugin contracts](docs/TYPING.md) for typed services, validated
settings and events, and the distinction between strict mypy and rejecting all
uses of `Any`.

The complete unit suite is required alongside static checks and actual TUI
acceptance; none substitutes for the others. Record failures, errors and skips
with the tested source revision. See [verification instructions](docs/VERIFICATION.md)
for reproducible checks. The current
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
python -m tools.fps_tui --output ./build/verification-run-1/fps
python -m tools.bare_tui --output ./build/verification-run-1/bare
python -m tools.accept_tui --output ./build/verification-run-1/interaction
python -m tools.features_tui --output ./build/verification-run-1/features
python -m tools.startup_tui --output ./build/verification-run-1/startup
python -m tools.profile_upgrade_tui --output ./build/verification-run-1/profile-upgrade
python -m tools.persistence_tui --output ./build/verification-run-1/persistence
python -m tools.package_download_tui --output ./build/verification-run-1/package-download
python -m tools.adversarial_agents_tui --output ./build/verification-run-1/adversarial-agents
python -m tools.ui_stress_tui --output ./build/verification-run-1/ui-stress
python -m tools.composer_tui --output ./build/verification-run-1/composer
python -m tools.collective_tui --agents 50 --parallel 8 --output ./build/verification-run-1/collective
python -m tools.optimization_tui --output ./build/verification-run-1/optimization
python -m tools.plugin_guide_tui --output ./build/verification-run-1/plugin-guide
python -m tools.reload_race_tui --output ./build/verification-run-1/reload-race
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
python -m tools.self_harness_tui --repetitions 2 --output ./build/verification-run-1/self-harness
```

This spends provider credits. It requires an accepted candidate and improved
results on the experiment's fixed synthetic test cases, and retains evidence if
no improvement occurs. Those results describe that experiment; repeated cases
are not independent samples or evidence of broad coding improvement. The
collective driver also accepts `--live` for a real-provider run.

Build and test the exact release with retained evidence:

```bash
python -m tools.build_portable --smoke --smoke-output ./build/verification-run-1/release
python -m tools.build_portable --check --no-smoke
```

On POSIX, release smoke repeats the offline TUI acceptance against the extracted
ZIP. On Windows, it verifies archive/package integrity and compiles the source;
it does not exercise the Windows TUI. See [release instructions](docs/RELEASE.md)
and [recorded verification results](docs/VERIFICATION.md).

Drag transcript text and release the mouse to copy it automatically.
Hold the pointer at the top or bottom edge of the chat to keep scrolling while
extending the selection across screens. Mouse-wheel scrolling also extends a held
selection.
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

Live core updates: `/update SOURCE` validates and activates a new application process
when active work finishes. `/recover previous`, `/recover known-good`, and the
supervisor’s **Ctrl+R** recovery screen retain access to saved releases. Self-Harness
can propose actual core source changes. See [live updates and recovery](docs/LIVE_CORE.md).

## Release builds and CI

`python -m tools.build_user_release` writes the user ZIP and `SHA256SUMS` to `dist/`.
Use `--check` to compare them with a fresh deterministic build. The version comes
from `release-version.txt`; change it deliberately for a new public release.

Routine CI runs Python 3.12 on Linux, macOS, and Windows. Each job runs the full
unit suite once and tests the extracted public launcher. Windows uses a standard
user with the hosted runner’s normal security settings and a CI-only ConPTY
driver. Windows runs the stress module first, then starts the measured slow
modules early to reduce waiting at the end. Each remaining module uses a fresh
standard-library test process, with at most four running concurrently. Every
discovered test runs once.
A separate nightly/manual Windows job runs the same suite with Defender enabled.
For an on-demand check, enable the Defender option when dispatching Release CI.
That job verifies the hosted provisioning daemon’s identity before
exempting that single infrastructure executable from Defender scanning. This
works around a reproduced detection that disconnects the runner even while idle;
RayChat, Python, plugins, and test storage remain scanned. Broader exclusions fail
the acceptance gate, and before/after protection reports are retained.
Strict typing, lint, formatting, and negative type contracts run once on Linux.
The 14 extended Linux terminal scenarios also run nightly or through the manual
extended workflow. Defender and extended terminal jobs do not block routine PR
checks.

After a version change merges into `main`, successful platform jobs publish their
already-tested, matching ZIP as a GitHub Release. Pull requests never publish.
Existing release tags and public assets are never overwritten. CI retains
archive hashes, per-stage timings, and console transcripts. The nightly/manual
Windows job additionally retains before/after Defender evidence.

The previous 12-job run took about 16 minutes elapsed (run `36213496580`).
The consolidated run's measured timings are recorded in the release PR and
its CI artifacts; Python versions other than 3.12 are not exercised by routine CI.
