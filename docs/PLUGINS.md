# Writing and sharing SDK v4 plugins

For typed service keys, validated settings/event schemas, and static contract
checks, see [strong plugin contracts](TYPING.md).

RayChat installs features as independent Python packages. Every
package uses `plugin.json`, the same source capture, dependency resolver,
registration transaction, and live generation switch. Adding a plugin requires
no edits to the session kernel, dispatcher, CLI, or terminal controller. Import
public contracts from `raychat.sdk`; keep implementation helpers in your own
package and use relative imports. Python's standard library is the runtime
dependency environment. Installation does not invoke pip, install wheels, or run
build hooks. Manifest `requires` entries name RayChat plugins, not Python
distributions.

The session kernel owns messages, approvals, cancellation, and durable history.
`raychat.sdk.SessionHost` separates it from the plugin implementation. Package
loading and installation live in the host; features consume the public SDK and
registered services. Plugins execute trusted Python code with the user's access.
Installing a package authorizes its execution; searching a catalog reads metadata
without importing package code.

## A complete first plugin

Start RayChat with the standard feature profile, then enter these commands in its
chat. Paths are relative to the active workspace unless absolute; quote paths
containing spaces. This example uses `/tmp/raychat-plugin-demo` on POSIX; choose a
writable absolute directory on Windows.

Create a scaffold in a new directory:

```text
/plugins new /tmp/raychat-plugin-demo/hello
```

Replace its two files with this complete example; the generated README can stay.
`plugin.json`:

```json
{
  "id": "hello",
  "version": "1.0.0",
  "sdk": 4,
  "entrypoint": "__init__:register",
  "description": "A configurable greeting with a session counter",
  "instructions": "Use {\"action\":\"greet\"} when asked to demonstrate this plugin. It returns the configured greeting and session call count. The operator can use /greet directly.",
  "requires": {},
  "defaults": {"greeting": "Hello"}
}
```

`__init__.py`:

```python
from raychat.event_types import AFTER_TOOL, AfterTool
from raychat.sdk import (
    Action,
    CommandDefinition,
    PluginAPI,
    PluginContext,
    ToolDefinition,
)


def register(api: PluginAPI) -> None:
    def validate_settings(settings: Action) -> None:
        greeting = settings.get("greeting")
        if not isinstance(greeting, str) or not greeting.strip():
            raise ValueError("hello.greeting must be nonempty text")

    api.validate_settings(validate_settings)

    def greeting() -> str:
        return str(api.context.settings["greeting"])

    def validate(action: Action) -> None:
        if action != {"action": "greet"}:
            raise ValueError("greet accepts no arguments")

    def execute(_action: Action, ctx: PluginContext) -> Action:
        ctx.check_cancelled()
        ctx.state["calls"] = ctx.state.get("calls", 0) + 1
        return {
            "ok": True,
            "message": f"{greeting()} (call {ctx.state['calls']})",
        }

    def greet(arguments: str, ctx: PluginContext) -> str:
        if arguments.strip():
            raise ValueError("Usage: /greet")
        result = execute({"action": "greet"}, ctx)
        ctx.checkpoint()
        return str(result["message"])

    def after_tool(event: AfterTool, ctx: PluginContext) -> None:
        if event.action.get("action") == "greet":
            ctx.notify("Greeting tool completed")

    api.register_tool(
        ToolDefinition(
            "greet",
            "Return a greeting and count this call",
            validate,
            execute,
            requires_approval=False,
        )
    )
    api.register_command(CommandDefinition("greet", greet))
    api.register_service("hello_greeting", greeting)
    api.on(AFTER_TOOL, after_tool)
```

Validate and link it, then invoke the command twice:

```text
/plugins check /tmp/raychat-plugin-demo/hello
/plugins link /tmp/raychat-plugin-demo/hello
/greet
/greet
```

A fresh session prints `Hello (call 1)` and `Hello (call 2)` without a model
request. Ask the model, “Use the greet tool once and report its result” to exercise
the model action and the `Greeting tool completed` notification too. Slash
commands are operator entry points; they do not themselves emit tool lifecycle
events. `ToolDefinition.parameters` describes inputs to the model; `validate`
enforces their accepted shape. The argument-free greeting opts out of approval;
tools require approval by default.

`check` imports a captured generation, resolves installed dependencies, validates
registration/default settings, runs configuration hooks and closes its resources.
It does not replace checking actual tool results, approvals and cancellation in
the chat.

### Edit without restarting

While that chat stays open, change the manifest greeting from `"Hello"` to
`"Ahoy"`. The next `/greet` changes the text and continues the counter. Python and
manifest instruction edits use the same live replacement path. `/plugins reload`
requests an explicit refresh. A busy operation keeps its generation until an idle
boundary; `"applied": false` means an update was queued.

A **link** uses the source directory directly. An **install** copies it into
managed storage, so editing the original directory after installing does not
change the installed plugin. `/plugins info hello` shows the active path. Invalid
code/settings leave the previous working generation available; repair the source
and retry. Automatic refresh requires `plugins.auto_reload: true`, the default.

`ctx.state` holds JSON-compatible conversation state, preserved across reload and
saved-session restoration. `ctx.checkpoint()` saves command-driven state changes
when session storage is enabled. An explicit checkpoint also protects your
plugin's state from rollback if a concurrent model turn is cancelled, including
when persistence is disabled. It does not commit that unfinished conversation or
another plugin's uncommitted state. Clearing the session clears its plugin state.
Module globals and closures belong to one generation, not durable conversation
state.

### Pack and install the result

Pack into a new ZIP outside the package directory, then replace the development
link with an installed copy:

```text
/plugins pack /tmp/raychat-plugin-demo/hello /tmp/raychat-plugin-demo/hello-1.0.0.zip
/plugins uninstall hello
/plugins install /tmp/raychat-plugin-demo/hello-1.0.0.zip
/greet
/plugins info hello
```

`pack` returns the archive path and SHA-256, produces deterministic bytes and
refuses to overwrite an existing output. Uninstalling a link preserves the source
directory. Keep credentials and local work outside the package: packaged resources
are shared with its source. Bump the manifest version for a new published release.

Installs default to `WORKSPACE/.raychat/plugins/ID`; add `--user` for
`~/.raychat/plugins/ID`. Use the matching scope for later operations, for example
`/plugins uninstall hello --user`. These home paths follow
`storage.home_directory` when customized. Duplicate IDs across scopes fail rather
than silently shadowing another installation.

Other sources and management commands:

```text
/plugins install https://example.org/hello-1.0.0.zip
/plugins install github:team/hello@v1.0.0
/plugins update hello
/plugins update hello --force
/plugins disable hello
/plugins enable hello
/plugins uninstall hello
```

The URLs illustrate publishing locations; they are not bundled examples. A GitHub
ref resolves to a commit before downloading. Its repository must contain one
plugin at the root. Public repositories need neither Git nor pip; private
repository authentication is not provided.

`update` reuses the recorded install source. Catalog pins resolve to the latest
version available there; a local ZIP or fixed URL is reread at that location.
Install a new source explicitly when its location changes. Edit linked packages
directly instead of running `update`. Local installed edits require `--force` to
replace; an intervening edit while an update is queued rejects it even with force.

Operator-owned records track scope, source, resolved revision/URL, version,
archive/content hashes and enablement. User records live in
`~/.raychat/plugins.lock.json`; workspace records live under the operator's
`~/.raychat/workspaces/WORKSPACE_HASH/`. A repository-supplied lock file cannot
authorize execution. `/plugins` or `/plugins list` opens the plugin picker; use
arrows/Enter or the mouse to toggle a plugin and Escape to close it.

## Publish and discover a catalog

No central marketplace is required. A catalog has `schema: 1` and a `plugins`
array. Each entry contains the **full manifest**, an archive URL and SHA-256.
Generate a working local catalog directly from the ZIP above using stdlib, from
a shell:

```sh
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import zipfile

root = Path("/tmp/raychat-plugin-demo")
archive = root / "hello-1.0.0.zip"
with zipfile.ZipFile(archive) as bundle:
    entry = json.loads(bundle.read("plugin.json"))
entry["url"] = archive.name
entry["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
(root / "catalog.json").write_text(
    json.dumps({"schema": 1, "plugins": [entry]}, indent=2) + "\n",
    encoding="utf-8",
)
PY
```

Register and search it in the chat:

```text
/plugins catalog add demo /tmp/raychat-plugin-demo/catalog.json
/plugins catalog list
/plugins search hello
/plugins install demo/hello@1.0.0
/plugins catalog remove demo
```

Publish the catalog and ZIP together over HTTPS to share them. Readers add the
URL with `/plugins catalog add NAME URL`, search with `/plugins search WORD`, and
install `NAME/ID@VERSION`. Relative archive URLs resolve against the catalog.
`--user` makes a catalog registration user-wide; use that scope when removing it.
Removing a catalog removes a discovery source, not its installed packages.

Missing dependencies can be installed from configured catalogs in the same
transaction. Ambiguous results require `CATALOG/ID@VERSION`. Valid catalogs are
cached; search marks cached responses when a catalog is unavailable. Archive
hashes are checked before importing downloaded code. ZIP extraction rejects
traversal, links, special files, case collisions and excessive sizes/member counts.

## Instructions visible to the model

Every manifest requires a nonempty `instructions` string of at most 8,192
characters. Explain when the capability is useful, show a valid action, and name
operator commands separately from model actions. The host injects active
manifests into each request. These instructions count toward the context limit;
disabling or uninstalling a package removes its text. Installing a newly authored
package or editing its instructions updates the next idle generation without
restarting the chat. Tool schemas and the enabled-action list remain authoritative
for the current session's permissions, including read-only children.

## Settings, CLI metadata and dependencies

A package may include helpers, text/binary resources, documentation and tests.
`entrypoint` is `module:function`; use relative imports within your package.
Manifest defaults are overridden by `plugins.settings.ID` in the selected
`raychat.json`. This is a fragment to merge into the existing configuration:

```json
{"plugins": {"settings": {"hello": {"greeting": "Welcome"}}}}
```

`ctx.settings` is read-only; validate it with `api.validate_settings` during
registration. An override wins over a manifest default, including after reload.
Manifest defaults change live. The application's configuration file and launch
arguments are read at startup, so changing those requires restarting the harness.

Optional declarative `cli` metadata adds launch options without importing plugin
code during parsing. Add this member inside the manifest:

```json
"cli": [{"flags": ["--hello-limit"], "type": "int", "default": 3,
         "environment": "HELLO_LIMIT", "help": "Greeting limit"}]
```

Use `setting` instead of `default` to reference a manifest default key. Types are
`str`, `int`, `float`, `path`; actions are `store`, `append`, `store_true`. `group`
creates a plugin-local mutually exclusive group. `json` serializes an object
default; `invert` negates a boolean default. Duplicate flags fail explicitly.
Values appear through `ctx.options["args"]`; new launch flags require a restart.

### Use another plugin's service

Use services instead of importing another plugin's implementation. The greeting
example publishes `hello_greeting`, a callable returning its configured text.
This complete `salute` consumer declares a direct dependency in `plugin.json`:

```json
{
  "id": "salute",
  "version": "1.0.0",
  "sdk": 4,
  "entrypoint": "__init__:register",
  "description": "Read the greeting supplied by hello",
  "instructions": "The operator can use /salute to read the configured hello greeting. This package adds no model action.",
  "requires": {"hello": "1.0.0"},
  "defaults": {}
}
```

Its `__init__.py`:

```python
from raychat.sdk import CommandDefinition, PluginAPI, PluginContext


def register(api: PluginAPI) -> None:
    greeting = api.require_service("hello_greeting")
    if not callable(greeting):
        raise ValueError("hello_greeting must be callable")

    def salute(arguments: str, _ctx: PluginContext) -> str:
        if arguments.strip():
            raise ValueError("Usage: /salute")
        return str(greeting())

    api.register_command(CommandDefinition("salute", salute))
```

Save those files under `/tmp/raychat-plugin-demo/salute`. With `hello` installed,
use:

```text
/plugins check /tmp/raychat-plugin-demo/salute
/plugins link /tmp/raychat-plugin-demo/salute
/salute
/plugins uninstall salute
```

`/salute` reads the greeting without incrementing hello's counter. Dependencies
register first. `ctx.service(name)` resolves one during execution;
`ctx.optional_service(name, default)` supports optional integrations. Document
service argument/result types as part of your plugin's public contract. Only its
owner may replace a service with `ctx.set_service`.

Required cross-plugin services need a **direct** manifest dependency, even when
another dependency already needs that provider. Versions are exact
`MAJOR.MINOR.PATCH`, without ranges. An install can resolve missing dependencies
from configured catalogs in the same transaction; linking requires dependencies
to be installed already. Cycles and conflicts reject activation. Disable/remove
dependents before their provider, and restore the provider first.

The [counter package](../examples/plugins/counter/__init__.py) provides another
runnable example of session state and a tool.

## Dynamic status without chat noise

Import `StatusItem` from `raychat.sdk` and push updates when state changes:

```python
from raychat.sdk import CommandDefinition, PluginAPI, PluginContext, StatusItem


def register(api: PluginAPI) -> None:
    def refresh(arguments: str, ctx: PluginContext) -> str:
        ctx.set_status("index", StatusItem("Indexing", priority=70))
        # Perform your work here, checking ctx.check_cancelled() in long loops.
        ctx.set_status(
            "index", StatusItem("Index ready", level="success"), ttl_seconds=5
        )
        return "Indexed the workspace."

    api.register_command(
        CommandDefinition(
            "index",
            refresh,
            description="Index this workspace",
            usage="/index",
        )
    )
```

`ctx.set_status(key, item, scope="session", ttl_seconds=None)` replaces that
plugin's keyed item; `None` clears it. Positive finite TTLs expire automatically.
Levels are `info`, `success`, `warning`, and `error`; higher integer priorities
appear first. Temporary feedback precedes persistent items. Long items truncate
at terminal cell boundaries, with an overflow count for additional items.

Session status follows its chat. `scope="application"` is visible across chats,
including while another child is focused. Keys are isolated by plugin and
publishing session. `ctx.notify(text)` sends routine five-second footer feedback.
Neither interface writes chat history, checkpoint state, or model context.
Return substantive command output normally, and raise exceptions for actionable
failures. Use the supplied context in workers so cancellation and stale-job guards
apply. Retained contexts from an unloaded generation cannot publish new status.

Publish from registration/configuration or state transitions, never from a
render callback. The UI reads cached snapshots. Successful reloads replace the
old status contributions; rejected reloads retain the working generation.
SDK 4 removes the former callback registration interface; update the manifest's
`sdk` field and migrate every contributor before reinstalling a package.

## Public registration and execution interfaces

`register(api)` runs once per generation. Registration expires when it returns;
late registration fails. All registrations are attributed to their package.

| Registration | Contract |
| --- | --- |
| `register_tool(ToolDefinition(...))` | Validator and synchronous action callback; JSON result; approval by default |
| `register_command(CommandDefinition(...))` | Argument string and context; returns text; optional `description` and `usage` populate slash completion |
| `register_service(name, value)` | A public dependency interface; consume with `require_service` or `ctx.service` |
| `register_provider(name, factory)` | Factory receives host arguments/environment and returns a callable model |
| `register_worker(name, factory)` | Reconstruct a provider from JSON options in an isolated process |
| `register_instruction(name, callback, priority=50)` | Callback receives session, size limit, context; returns text or `InstructionContribution` |
| `register_menu(name, factory)` | Factory returns SDK `Menu` with choices and a selection callback |
| `register_navigation(name, provider)` | Implement the public SDK `NavigationProvider` interface |
| `register_middleware(name, callback)` | Wrap one message; may return SDK `Continuation(prompt)` |
| `on(event, callback)` | Receive an SDK event and context |
| `configure(callback)` | Initialize after all packages have registered |
| `on_close(callback)` | Release resources on replacement, removal, shutdown or failed activation |
| `on_reload(export, restore)` | Transfer non-JSON resources to a replacement |

`ctx` supplies `workspace`, read-only `settings` and launch `options`, namespaced
JSON `state`, a detached `session.snapshot()`, cancellation checks,
notifications and service access. `ctx.generation` identifies the active
package generation. Required cross-plugin services must have a direct manifest
dependency; optional integrations use `ctx.optional_service`. `ctx.set_service` replaces only a
service owned by the calling plugin. Use services rather than importing another
plugin's implementation or reading runtime internals.

Use `ctx.resource('data/file.bin')` for captured resource bytes. Code and resources
also share an immutable temporary package directory, so relative `__file__`
lookups see the same generation. These paths expire when that generation closes;
keep durable files in an explicit workspace/user storage location.

`ctx.plugin_sources()` returns the serializable package bundle for isolated
workers. A provider descriptor contains `plugin`, `worker`, `source` and JSON
`options`, plus an optional `secrets` tuple when constructing `WorkerDescriptor`.
Those values are redacted from worker errors and never placed on the command
line. Worker names are scoped to the owning plugin. The worker reconstructs the same package generation and calls the
registered worker factory. The HTTP implementation uses this path too.

Raw HTTP debugging is explicitly enabled with `--debug`, `chat.debug`, or
`RAYCHAT_HTTP_DEBUG_DIR`. The resolved directory is inherited by isolated workers,
including child agents. Per-connection `sent.http` and `received.http` files
contain HTTP bytes before TLS encryption and after TLS decryption, including
full headers, payloads, and API keys without redaction; `events.jsonl` records
timing and connection failures. This
capture is separate from conversation logging and from the redacted worker
errors described above. Debugging is off by default and creates no files while
disabled. See [raw HTTP debugging](../README.md#raw-http-debugging) for CLI and
configuration options.

A menu opens with `ctx.emit('ui', {'menu': 'registered-name'})`. A selection
callback can emit `{'session': 'session-id'}` or request a package transaction.
Commands require an idle session at admission by default. Use `while_running=True`
only for callbacks that support starting alongside active work. This is a start
condition, not a lock that excludes future operations; shared resources still
need their own synchronization. A concurrent command that changes durable state
must call `ctx.checkpoint()` before returning. Synchronize competing writes to
your own namespace; a checkpoint commits that namespace's current values.
Use `scope='application'` for
application services and navigation that must work from child chats. Application
commands run in a cancellable worker belonging to the chat where they started.
Only brief UI callbacks, such as opening `/agents`, should set `background=False`;
these run on the UI thread and must not block. `background` defaults to `True`.
Concurrent session commands use the same background command path and retain the
selected session's context. A chat admits one such command at a time; submitting
another command cannot bypass the active command or run its callback on the UI
thread. Normal messages wait for the active work to finish.

An idle session command can request an agent turn by emitting
`ctx.emit("command_task", {"prompt": objective})`. After the command succeeds, its
conversation worker runs that prompt in the same job, with ordinary middleware,
approvals, cancellation, and continuation. The command's returned text becomes
progress feedback; the agent's accepted answer completes the job. Concurrent
commands do not start a second conversation turn. The goals plugin uses this event
for `/goal OBJECTIVE`; status and clear commands do not emit it.

See [the SDK definitions](../raychat/sdk.py) for complete callable signatures.
Registration names share a namespace except package-scoped worker names; use a
plugin prefix for public names to avoid collisions.

SDK 4 lifecycle hooks use shared `EventKey` constants from
`raychat.event_types`: `CONFIGURE`, `SESSION_START`, `SESSION_CLOSE`,
`SESSION_RESET`, `SESSION_RESTORE`, `TURN_START`, `TURN_END`, `TURN_ABORT`,
`CONTEXT`, `BEFORE_TOOL`, `AFTER_TOOL` and `PLUGINS_RELOADED`. Pass the constant
to `api.on`, as the greeting example does. Handlers receive typed payloads:
`AfterTool.action` and `.result`, or `TurnEnded.message`, for example. Observers
return `None`; context hooks return `Context | None`, and tool guards return
`Block | None`. Payload and return types are checked. Required validation or
transform failures abort the affected operation; observer failures emit
diagnostics. `ctx.emit("ui", ...)` is the separate UI notification channel.

## Live updates and state

Automatic reload hashes active source and trusted discovery directories before
idle operations. New manifest packages placed in a trusted workspace's
`.raychat/plugins` become available without restarting. Explicit `--plugin PATH`
loads a package for that invocation. Use `--trust-workspace grant` or `revoke` to
control loose workspace discovery; explicitly installed packages have their own
recorded authorization. `--disable-plugin ID` and `--no-plugins` restrict startup.

Active calls retain their generation. Updates wait until active operations reach
a safe boundary; they do not rewrite Python stack frames or cancel work. Goal
continuations return to the host between messages, allowing a queued change to
activate before the next continuation. A failed/cancelled operation rolls back
its own queued requests without discarding another operation's requests.

Imports, registration, configuration and state restoration complete before the
registry switches. Invalid code, dependencies or hooks retain the old registry.
Installed file replacements and lock-state changes roll back on activation
failure. Rejected unchanged source hashes are not retried on every UI frame.
Disabled plugins do not reappear during discovery.

Register cleanup immediately after acquiring resources, including those created
during registration/configuration. Cleanup must tolerate partial initialization
and failed activation; callbacks take no arguments. Per-operation resources belong
in `try/finally` inside the tool or command.

History, namespaced JSON state, loaded skills, goals and child workers survive
successful replacement. Child histories, drafts and permission allowlists remain
attached to their chat. Non-JSON resource transfer uses
`api.on_reload(export, restore)`. A deliberately transferred resource can register
`on_close(callback, on_reload=False)`; it closes on removal or shutdown. Restore
callbacks must not mutate the active resource before activation succeeds.

`ctx.update_plugins(add=..., remove=..., prepare=..., commit=..., rollback=...)`
is the public transaction API used by package management and Self-Harness.
Preparation and activation run at the next idle boundary. Cancellation or
activation failure invokes `rollback(error)`; successful activation invokes
`commit()`. Transaction hooks must be short and may not start another operation.
The agent `plugins` tool accepts `{"action":"plugins","command":"install ..."}`.
Finish the current turn with `done` before using a newly queued capability.

Self-Harness file promotion holds the package and workspace locks through
validation and records completed stages and undo files before publication.
Startup and cooperating filesystem operations recover interrupted candidates;
conflicting external edits preserve the record and undo files for correction.
A committed decision permits cleanup only, so recovery does not revert later
user edits. This does not provide power-loss durability or coordinate older
writers and external editors. The `.raychat-candidate-` directory prefix is
reserved for private stages and excluded from package source captures.

Core, SDK and terminal-controller source changes still require a restart.

## Standard distribution

| Package | Responsibility |
| --- | --- |
| `filesystem` | List/read/write/edit, confined paths, paging, hashes, atomic replacement |
| `process` | Argument-array commands, environment policy, bounded output, timeouts, process-tree cleanup |
| `skills` | Discovery, catalog, loading, session state and instruction contributions |
| `memory` | Durable storage, paging, actions and instruction contributions |
| `chat_completions` | HTTP, credentials, request/response validation and isolated provider factory |
| `subagents` | Profiles, routing, isolated child execution, lifecycle events and session navigation |
| `workflows` | Serial delegation and bounded parallel batches |
| `goals` | Commands, judging, retries, revisions and continuation requests |
| `optimization` | GEPA, evaluators, demos, benchmarks and reports |
| `context` | Ordered instruction assembly and context compaction |
| `plugin_manager` | Package commands, agent tool and plugin menu |
| `self_harness` | Failure evidence, candidate evaluation and transactional promotion |

### Profile installation

The `raychat/` package contains the host and terminal integration. Feature source
projects live independently in `plugins/ID`; startup never loads that directory.
`python3 -m tools.build_plugin_catalog` builds deterministic archives and a catalog
in `plugin_catalog/`. The configured `plugins.profile` names a local release
profile; its catalog contains full manifests, exact versions and archive hashes.
Generated profiles pin an immutable, content-addressed catalog, whose archives
also have content-addressed names. Rebuilding publishes these completed files
before replacing the profile and retains older artifacts for existing readers.
On first launch, the ordinary package manager installs this profile into the user
plugin directory. Metadata is read before CLI parsing without executing plugins.

A durable profile marker prevents later launches from reinstalling features the
operator removed. Startup compares package hashes, including releases with the
same version label, and updates unchanged packages installed from that profile
through the ordinary installation transaction. Catalog provenance distinguishes
profile packages from external replacements. Linked packages, local edits and
disabled state are preserved; a failed upgrade leaves the previous installation
intact and can be retried. Explicitly upgrade a user package with `/plugins update
ID --user`; local edits still require `--force`. Set `plugins.profile` to `null` for
an installation with no default feature set. A release profile is configuration,
and can name external packages without changing any core Python source.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `/plugins` is unknown | Start with the standard profile and its `plugin_manager` feature enabled, without `--no-plugins`. |
| An edit is not visible | Inspect `/plugins info ID` for the active path, edit the linked source or installed copy, then `/plugins reload`; allow busy work to finish. |
| A manifest default has no effect | Check `plugins.settings.ID` overrides in the selected configuration; changing that file requires restart. |
| Reload is rejected | Read the diagnostic, use `/plugins check PATH`, repair the source/settings/dependency and retry; the prior generation remains active. |
| A required service is missing/refused | Declare its provider directly under `requires`, install the exact version and check the service name. |
| Duplicate ID or registration | Inspect both user/workspace scopes and choose a distinct public registration name. |
| Disable/uninstall fails | Disable/remove active dependents first and use the installation's correct scope. |
| Update reports local edits | Preserve those edits or deliberately replace them with `--force`; update links by editing their source. |
| Catalog install fails | Check the full manifest, exact version, archive location and digest; qualify ambiguous names as `CATALOG/ID@VERSION`. |
| A tool does not stop | Check cancellation during long loops and resource cleanup in `finally`; cancellation targets the focused chat. |

## Cancellation and saved chats

Press Escape twice within `tui.double_escape_seconds` to cancel the focused job.
Its worker remains available for another message. Commands and plugins should
call `ctx.check_cancelled()` during long work and clean resources in `finally`.
Do not swallow cancellation with a broad `BaseException` handler.
HTTP requests, package downloads and optimization work use killable isolated processes; the process plugin
terminates command trees. Arbitrary in-process Python must cooperate with
cancellation.

If an application command is running in the focused chat, double Escape cancels
that command first. It does not stop a parent or sibling workflow. A stalled
`/plugins install`, catalog request or update can be cancelled before its server
responds, and the composer accepts a replacement message immediately.

`/agents` opens the subagent plugin's session menu. Use arrows or the mouse to
switch; `/parent` returns to the parent chat. Cancelling a focused child also
works when that child belongs to a workflow. Sibling jobs continue. One Escape
closes an open picker; a rapid second stops the focused job.

Interactive sessions save under `~/.raychat/sessions` by workspace. `--resume ID`
opens one session; `--resume` opens the only saved session or shows a selector.
`--no-session` disables saving and `--session-dir PATH` changes the location.
`/sessions`, `/resume ID`, `/tree` and `/fork ENTRY_ID` navigate saved history.
Forking restores conversation/plugin state without undoing external file writes.
If a plugin rejects restored state, the current in-memory and saved branches remain
selected; the failed fork does not silently change what the next restart resumes.

## Verify before sharing

Include a README walkthrough that uses an isolated workspace: check/link, invoke
commands and model tools, edit and reload, try an invalid edit, cancel long work,
then pack/install, disable/enable and uninstall. Verify output/files, state
continuity and cleanup. Features with launch flags or providers also need a fresh
startup check. State the supported host/SDK versions and service contracts.

The repository checks below use fresh output directories:

```bash
python3 tools/check_types.py
python3 -m ruff check .
python3 -B -S -m tools.accept_tui --output ./verification-run/plugins
python3 -B -S -m tools.adversarial_agents_tui --output ./verification-run/agents
python3 -B -S -m tools.bare_tui --output ./verification-run/bare
python3 -B -S -m tools.optimization_tui --output ./verification-run/optimization
python3 -B -S -m tools.build_portable --output build/raychat.zip --smoke
```

Behavioral acceptance drives the actual TUI with keyboard/mouse input and checks
rendered results, independent files, and provider requests. The plugin scenario
covers external authoring, ZIPs, catalogs, edits, rollback, cancellation, source
capture, and cleanup. The bare scenario runs an isolated core without feature
packages or a catalog. Existing regression specifications remain available for
static interface checking. The portable bundle includes manifests and acceptance
drivers and excludes installed user packages, credentials, trust files and sessions.
See [the release guide](RELEASE.md) for the remaining feature and stress scenarios.
