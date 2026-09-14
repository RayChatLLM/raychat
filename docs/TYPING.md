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

Shared adapters now cover the bundled cross-plugin capabilities in
`raychat.service_contracts`; `HTTP_PROVIDER` remains in the SDK:

| Key | Checked capability |
| --- | --- |
| `CHAT` | Active `Chat` and its fresh-client factory, published together. |
| `MEMORY` | Optional durable `MemoryStoreProtocol` and a concrete context limit. |
| `PROCESS_RUNNER` | Cancellable, bounded process execution returning `CommandResult`, including exit, timeout, truncation and decoding evidence. |
| `ATOMIC_WRITE` | Atomic byte replacement returning the exact byte count and digest. |
| `DELEGATION` | Optional `DelegationExecution`: prepare a typed request, execute its `DelegatedJob`, and receive an `AgentResult`. |
| `OPTIMIZATION` | Lazy loading within the captured generation; each component exposes a checked `OptimizationComponent` with typed entrypoint and binding callbacks. |

These records keep service identity stable across plugin generations. Disabled
memory and delegation are explicit `None` values that consumers must handle.
Delegation consumers use planned jobs instead of reaching into another plugin's
coordinator. `OptimizationBindings` supplies the typed provider, captured source
accessor and protocol to a component before it runs.

Provider option values have type `object` and must be validated before use.
`ProviderService.parse_options` accepts JSON text; request option parsing rejects
reserved fields, invalid keys, non-finite values, and oversized documents.
`ProviderSpec` detaches and freezes nested options before sharing them.

`WorkerPayload` declares the complete private worker envelope: plugin, worker,
source, options, and redaction secrets. Provider clients return this contract,
so misspelled envelope fields and incorrect field types fail mypy. The worker
factory validates required option names and concrete values before constructing
its client. HTTP responses are decoded as unknown data and checked before any
field is consumed. Registry storage and module discovery retain `object` because
entries are heterogeneous; their checked adapters establish the specific
capability before a consumer invokes it. An unknown stored value cannot be used
as a provider, worker or service merely by annotating a variable.

The HTTP client uses a `RequestOpener` Protocol: it accepts a concrete urllib
`Request` and numeric timeout, and returns an unknown response for validation.
Provider tests use a typed request recorder and bind static imports to the same
captured implementation at runtime. Their registered factory fixture checks the
returned class before exposing it. These fixtures contain no `Any` or unchecked
implementation casts.
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

## Pushed status and composer state

SDK 4 status uses immutable `StatusItem` values with concrete text, level and
priority fields. Plugins call `ctx.set_status(key, item, scope=..., ttl_seconds=...)`
to publish or replace a value, and pass `None` to remove it. Scope and level are
literal types; optional expiry is numeric. `Runtime.status_items()` returns a
tuple of `StatusRecord` values. It reads the current status store without
refreshing source, invoking plugin callbacks or waiting for generation activation.
The former `register_status` callback API has been removed.

Queued prompts use `MessageQueue` and immutable `QueuedMessage` records. Editing
preserves the separate composer draft; saving updates queued text while taking
items remains FIFO. Command completion exposes immutable `CommandChoice` records
and returns an explicit acceptance result. Typed fixtures reject wrong queue
indices, edit controls, status payloads and completion choices; positive fixtures
check the exact snapshot and optional queued-text types.

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

The narrow casts inside shared validators establish container shapes only after
checking the container and its keys or elements. They do not prove an arbitrary
nested document matches a domain schema. Every unvalidated field remains
`object`; consumers must narrow it before indexing, arithmetic or method calls.
Do not replace that validation with `cast(MemorySettings, raw)` or a typed
assignment from `json.loads`: neither checks runtime data.

Apply the same pattern to each plugin's settings, persisted state, tool input,
and provider response. Use a dataclass for internal records or `TypedDict` for a
fixed dictionary shape. Use a recursive JSON value type only at serialization
boundaries; a generic JSON dictionary still cannot check field names or their
relationships.

## Events

SDK 4 uses `EventKey[Payload, Result]` contracts for every plugin lifecycle hook.
The general bus's legacy string-handler and dictionary-wrapper compatibility
path has been removed. Import the shared keys and immutable payload classes from
`raychat.event_types`, then register with `api.on(AFTER_TOOL, handler)`.
An `AfterTool` handler reads `event.action` and `event.result` and returns `None`.

Hooks without additional fields use the empty frozen `Lifecycle` record:
`CONFIGURE`, `SESSION_CLOSE`, `SESSION_START`, `SESSION_RESTORE`, `SESSION_RESET`
and `TURN_ABORT`. `TURN_START` and `TURN_END` have their own concrete records.
An unchecked caller cannot substitute the former dictionary payload. Handler
registration preserves each key's invariant payload and result types, while the
heterogeneous registry stores only an adapter that checks both sides of a call.

`CONTEXT` handlers accept `Context` and return `Context | None`; `BEFORE_TOOL`
handlers accept `BeforeTool` and return `Block | None`. The session separately
validates message roles, Unicode and the resulting context budget. Runtime
dispatch validates payloads and results, and key identity prevents a different
contract from impersonating an existing event by reusing its name. Static
consumer fixtures reject incompatible producer values and handler signatures.

All bundled manifests now declare `"sdk": 4`. Older SDK manifests are rejected
before plugin code executes; there is no compatibility adapter for old handlers.
Custom hooks can define an `EventKey` with a concrete payload class and result
validator. Producers and handlers must import the same shared key declaration.

UI notifications use a separate diagnostic callback channel with
`Mapping[str, object]` payloads. `ctx.notify(text)` displays a notice, and
`ctx.emit("ui", payload)` requests a registered menu or session. The UI validates
fields before consuming them; arbitrary dictionary fields do not become typed
application events. Worker event records preserve concrete job identifiers and
checked payloads through cancellation and queue dispatch. For communication
between plugins, expose a documented service through the SDK.

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

The project mypy configuration applies the maximum type policy to every module;
there is no incremental list of specially protected plugins. `strict = true`
enables untyped-call, untyped-definition, generic-`Any` and subclassing-`Any`
checks. Global `disallow_any_expr`, `disallow_any_explicit`,
`disallow_any_unimported` and `disallow_any_decorated` close the remaining `Any`
escape routes. Together these enforce all six `Any` restrictions.

The policy also enables unreachable-code and incomplete-stub warnings, strict
`None` equality, and optional diagnostics for truthiness, unused awaitables,
possibly undefined names, redundant expressions, explicit overrides, mutable
overrides, deprecated APIs, redundant `self`, unimported `reveal_type`, exhaustive
matches and unused or unspecified ignores. Do not add broad ignores, skip plugin
imports or replace a checked contract with an annotation to make a check pass.

`tools/verify_quality.py` is the completion gate. It supplies an isolated maximum
mypy configuration, then runs Ruff with `ALL`, preview rules and
`--ignore-noqa`. `CPY001` is the only global exception: repository policy does
not require per-file copyright headers. An explicit `S311` exception covers
four deterministic optimization files listed below. The gate applies these
exceptions directly and does not inherit other project lint exceptions.

The verifier also checks formatting, rejects mypy suppression directives, and
records source hashes before and after checking. It passes every Python source
explicitly to the tools; installed virtual environments, checker caches,
generated release folders and application workspace data are listed separately.
It does not use lint configuration or `.gitignore` to skip project sources.
A changed source inventory makes the run fail. Full diagnostics and a JSON
report default to `build/quality/`; pass another directory as the sole argument
to keep a separate record.

CI runs these checks. `tools/check_types.py` checks all source files and the
same-name launcher separately, then verifies **115 expected negative contract
diagnostics**. The fixture check compares exact files, lines and error codes;
a missing rejection or an unexpected extra error fails it. Positive fixtures use
`assert_type` to verify inference, so a regression back to `Any` also fails.

Runtime boundary tests deliberately call checked interfaces through an explicit
`object` plus callable-validation seam. Matching negative fixtures prove the
equivalent direct calls remain invalid statically. This preserves malformed-call
coverage without mypy suppression directives. Runtime tests also exercise real
worker processes, terminal restoration, dependency enforcement, hot replacement,
cancellation identity and rollback; static checks cannot validate arbitrary
values supplied by unchecked Python or external JSON.

The four `S311` exceptions are confined to `plugins/optimization/gepa/`:
`optimize_anything.py`, `batch_sampler.py`, `candidate_selector.py` and
`merge.py`. Their `random.Random` calls preserve the original seeded search
behavior for candidates, minibatches and merges. These draws are not used for
credentials or security tokens. `S311` remains enabled everywhere else, and all
other rules remain enabled in these files.

To run the independent Ruff audit directly with the same explicit policy:

```bash
.venv/bin/python -m ruff check --isolated --preview --select ALL \
  --ignore CPY001 --ignore-noqa --target-version py310 \
  --per-file-ignores plugins/optimization/gepa/optimize_anything.py:S311 \
  --per-file-ignores plugins/optimization/gepa/batch_sampler.py:S311 \
  --per-file-ignores plugins/optimization/gepa/candidate_selector.py:S311 \
  --per-file-ignores plugins/optimization/gepa/merge.py:S311 .
```

The verifier records these paths in `ruff_file_rule_exceptions` and explains
why in `ruff_seeded_sampling_reason`. Removing the four file exceptions exposes
the intentional `S311` findings; they are not unresolved typing errors.
The independent maximum audit, formatting check and runtime acceptance remain
separate gates; passing project lint alone does not establish that all pass.

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
`ALL` is selected; the verifier records the global `CPY001` and four-file `S311`
exceptions. See
[Ruff's rule-selection documentation](https://docs.astral.sh/ruff/linter/).

References: [mypy strict and Any flags](https://mypy.readthedocs.io/en/stable/command_line.html),
[Protocol contracts](https://mypy.readthedocs.io/en/stable/protocols.html), and
[TypedDict and tagged unions](https://mypy.readthedocs.io/en/stable/typed_dict.html).
