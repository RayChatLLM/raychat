# Strong plugin contracts

A strict mypy pass is useful, but it does not prohibit `Any`. Dynamic plugin
loading works with strong types when registration, configuration and events have
explicit contracts. The loader may discover unknown Python objects; consumers
should receive validated values with concrete types.

Host configuration is now a frozen `HostSettings` tree. Use attributes such as
`SETTINGS.tui.target_fps` and `SETTINGS.storage.file_mode`; the string-path host
`setting(...)` accessor has been removed. Loading checks required fields, scalar
types, tuple shapes, finite numbers, and cross-field constraints before exposing
the records. Reading settings during rendering is an ordinary attribute access.

Plugin namespaces and `PluginContext.settings` expose `Mapping[str, object]`.
The old `plugin_setting(...)` accessor has been removed. Each bundled feature
owns a frozen settings class that checks required keys, rejects misspellings,
validates values, and detaches nested containers. `captured_settings(...)` checks
the captured namespace's owner before its plugin schema parses the values.
Negative type fixtures reject incompatible values, record mutation, and use of
unchecked plugin fields. Provider services expose typed timeout and request-option
defaults rather than an untyped settings mapping.

## Services

Use a shared `ServiceKey[T]` instead of assigning the result of a string lookup
to a variable annotated `T`. An annotation alone does not validate the value.

```python
from raychat.sdk import HTTP_PROVIDER, PluginAPI, PluginContext, ProviderService


def register_provider(api: PluginAPI, service: ProviderService) -> None:
    api.register_typed_service(HTTP_PROVIDER, service)


def configured_model(ctx: PluginContext) -> str:
    provider = ctx.require_service(HTTP_PROVIDER)
    return provider.default_model
```

Mypy rejects registering an unrelated implementation, unknown members on the
result, and wrong argument types on its typed methods. The invariant key type
prevents registration from widening `T` to accommodate an incompatible value.
Registration and lookup both validate the outer service class at runtime.
Lookup still enforces manifest dependencies and resolves the current service by
name, so it does not cache an obsolete implementation across replacement.

`ProviderService` is a shared SDK dataclass whose fields include callable and
Protocol contracts. Implementations can supply different provider classes that
satisfy those Protocols. Keep the adapter class/key in a stable shared contract
module, rather than defining separate copies inside independently loaded plugin
generations. `ServiceKey` currently accepts a concrete adapter class; its runtime
check does not inspect method signatures or recursively validate field values.
Mypy checks those when typed implementations construct the adapter.

The `chat_completions`, `subagents`, and `optimization` plugins use this key.
Provider option values now have type `object` and must be validated before use.
`ProviderService.parse_options` accepts JSON text; request option parsing rejects
reserved fields, invalid keys, non-finite values, and oversized documents.
`ProviderSpec` detaches and freezes nested options before sharing them.

`WorkerPayload` declares the complete private worker envelope: plugin, worker,
source, options, and redaction secrets. Provider clients return this contract,
so misspelled envelope fields and incorrect field types fail mypy. The worker
factory validates required option names and concrete values before constructing
its client. HTTP responses are decoded as unknown data and checked before any
field is consumed. The provider package passes the unrestricted mypy and Ruff
checks; legacy string services elsewhere still need concrete contracts.

The HTTP client uses a `RequestOpener` Protocol: it accepts a concrete urllib
`Request` and numeric timeout, and returns an unknown response for validation.
Provider tests use a typed request recorder and bind static imports to the same
captured implementation at runtime. Their registered factory fixture checks the
returned class before exposing it. These tests and fixtures also pass the
unrestricted checks; they contain no `Any` or unchecked implementation casts.
Negative fixtures reject invalid transport arguments and use of an unchecked
response as bytes.

Shared test fixtures preserve those contracts too. `ScriptedChat[T]` returns the
configured reply type, and tool results expose `object` values until field
validators check them. Captured module resolvers validate returned modules;
they no longer cast unknown lookup results to implementation types. Tests that
deliberately simulate an untyped caller have matching negative fixtures proving
the equivalent direct calls fail static checking.

`SUBAGENT_FACTORY` exposes a `SubagentFactoryService` adapter. Its configuration
accepts the frozen `SubagentSetup` record, and child enumeration returns typed
`ChildSessionInfo` records with concrete worker handles. Consumers no longer
construct or mutate another captured plugin's coordinator. Workflow reports
retain failed or cancelled child statuses, their errors, and failed checks.

Captured package transport uses `SourceSnapshot` and `PluginSources` schemas.
Worker reconstruction validates unknown keys, values and base64 source before
compilation. Each generation has an inspectable loader that executes its captured
code objects through the standard import machinery. Relative imports stay within
the generation; source inspection honors Python encoding declarations. Exported
snapshot settings are detached from the live generation.

## Settings and external data

Parse unknown input into a schema once. `MemorySettings.parse(value: object)`
validates each field and returns an immutable dataclass. Memory limits and the
default filename now come from its typed attributes. Invalid booleans, numeric
strings, fractional limits, missing required fields, and out-of-range integers
fail at the boundary. In particular, booleans are not accepted as integer limits.

Package manifests also validate defaults as finite JSON and retain concrete
`CLIArgument` and `ManifestDocument` dictionary shapes. Plugin-specific default
values remain `object` until their owning settings parser validates them.

Every bundled plugin settings validator accepts `object` and rejects malformed
containers through `configuration_fields`. Shared `boolean_field`,
`integer_field`, `number_field`, `text_field`, and `string_list_field` functions
return checked values with concrete types. Numeric overflow is reported as a
`ConfigurationError` with the offending field path. The host configuration's
cross-field comparisons now use these checked values.

The narrow casts inside these shared validators establish checked container
shapes. Every field remains unknown until validated. Do not
replace that validation with `cast(MemorySettings, raw)` or a typed assignment
from `json.loads`: neither checks runtime data.

Apply the same pattern to each plugin's settings, persisted state, tool input,
and provider response. Use a dataclass for internal records or `TypedDict` for a
fixed dictionary shape. Use a recursive JSON value type only at serialization
boundaries; a generic JSON dictionary still cannot check field names or their
relationships.

## Events

SDK 4 uses `EventKey[Payload, Result]` contracts for lifecycle hooks. Import the
shared keys and immutable payload classes from `raychat.event_types`, then
register a handler with `api.on(AFTER_TOOL, handler)`. An `AfterTool` handler reads
`event.action` and `event.result`; it returns `None`. String hook names and the
old dictionary event wrapper are no longer supported.

`CONTEXT` handlers accept `Context` and return `Context | None`; `BEFORE_TOOL`
handlers accept `BeforeTool` and return `Block | None`. The session separately
validates message roles, Unicode and the resulting context budget. Runtime
dispatch validates payloads and results, and key identity prevents a different
contract from impersonating an existing event by reusing its name. Static
consumer fixtures reject incompatible producer values and handler signatures.

All bundled manifests now declare `"sdk": 4`. SDK v2 manifests are rejected
before plugin code executes; there is no compatibility adapter for old handlers.
Custom hooks can define an `EventKey` with a concrete payload class and result
validator. Producers and handlers must import the same shared key declaration.

UI notifications remain a separate callback channel: `ctx.notify(text)` displays
a notice, and `ctx.emit("ui", payload)` requests a registered menu or session.
For communication between plugins, expose a documented service through the SDK.

## Enforcement

Install the pinned development tools with Python 3.12 or newer. Application
runtime compatibility remains Python 3.10+ with only the standard library.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip check
.venv/bin/python tools/verify_quality.py
.venv/bin/python tools/check_types.py
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

`tools/verify_quality.py` is the completion gate. It uses an isolated mypy
configuration with every `Any` restriction and additional optional checks. Ruff
runs with `ALL`, preview rules and `--ignore-noqa`, without the project's
exceptions. The verifier also checks formatting, rejects mypy suppression
directives, and records source hashes before and after checking. It passes every
Python source explicitly to the tools; installed virtual environments,
checker caches, generated release folders and application workspace data are
listed separately. It does not use lint configuration or `.gitignore` to skip
project sources. A changed source inventory makes the run
fail. Full diagnostics and a JSON report default to `build/quality/`; pass another
directory as the sole argument to keep a separate record.

CI runs these checks. `tools/check_types.py` also checks all source files, checks the
same-name launcher separately, and verifies that deliberately invalid consumer
fixtures produce the expected errors. Positive fixtures use `assert_type` to
verify inference, so an accidental regression back to `Any` fails checking.
UI boundary tests invoke malformed calls through unittest's callable assertion
API; matching negative type fixtures prove the direct calls remain invalid.
This retains runtime failure coverage without mypy suppression directives.

Project mypy uses strict mode plus unimported/decorated-Any checks, unreachable
code checks and additional optional error codes. The new service-key, context
payload, shared validators and every plugin settings module reject explicit `Any` and
expressions containing `Any`. Expand that per-module list as boundaries migrate;
do not add broad ignores or skip plugin imports to make CI green.

Ruff selects `ALL` with documented project exceptions in `pyproject.toml`,
including formatter conflicts, documentation policy, unittest conventions, and
existing API/complexity conventions. Function annotation rules remain enabled.
Some legacy dynamic boundaries retain local `ANN401` exceptions with reasons.

For an intentionally stricter audit of the remaining work:

```bash
.venv/bin/python -m mypy --strict --disallow-any-explicit --disallow-any-expr
.venv/bin/python -m ruff check --isolated --preview --select ALL --target-version py310 .
```

These repository-wide audits are not clean yet. Passing configured strict mypy
does not mean the entire repository is free of `Any`. Runtime tests must also
exercise malformed external inputs, dependency enforcement, and replacement:
static checking cannot validate arbitrary values supplied by unchecked Python.

## Additional useful tools

- [Coverage.py](https://coverage.readthedocs.io/en/latest/branch.html): measure
  branch coverage of validation, cancellation and rollback behavior.
- [Hypothesis](https://hypothesis.readthedocs.io/en/latest/): generate adversarial
  payloads, settings and lifecycle sequences, then minimize failing cases.
- [Vulture](https://github.com/jendrikseipp/vulture): identify potentially unused
  functions and variables. Review findings involving dynamic plugin entrypoints
  before deleting them.

These complement the strict completion gate and behavioral suite. Ruff
automatically resolves two mutually exclusive docstring-style rule pairs when
`ALL` is selected; the verifier adds no rule exceptions. See
[Ruff's rule-selection documentation](https://docs.astral.sh/ruff/linter/).

References: [mypy strict and Any flags](https://mypy.readthedocs.io/en/stable/command_line.html),
[Protocol contracts](https://mypy.readthedocs.io/en/stable/protocols.html), and
[TypedDict and tagged unions](https://mypy.readthedocs.io/en/stable/typed_dict.html).
