# Subagents and goal judging

The harness supports isolated read-only subagents, deterministic model routing,
parallel review batches, serialized review workflows, and an optional persistent
goal judge. All orchestration uses the Python standard library.

## Navigate child chats

The subagents plugin owns `/agents`, which opens a window listing the main chat
and its children. Select with Up/Down and Enter, click a row, or scroll with the
mouse wheel. Escape closes the window. `/parent` returns to the parent chat;
the window also includes a parent row.

Each child has its own worker, transcript, draft, and cancellation scope. Press
Escape twice while focused on a child to stop that child's task and submit a
replacement. Sibling chats and the parent remain running. A workflow receives a
cancelled result for that child and still waits for its other children. Follow-up
messages retain the child's completed history and its read-only capabilities.
Child chats remain available in the window for the lifetime of the application.

## Role profiles

The main model is always available as the `primary` profile and is the explicit
default fallback. With no additional profile in the `plugins.settings.subagents` section of
[`raychat.json`](../raychat.json), every delegated task and default goal
judgment uses the same provider endpoint, model, and authentication as the main
chat. These come only from `RAYCHAT_BASE_URL`, `RAYCHAT_MODEL`, and
`RAYCHAT_AUTH_TOKEN`. Setting
`allow_default_fallback` to `false` makes an unsupported delegation fail instead;
the primary profile remains directly available for goal judging.

Add profiles and routes to the same `raychat.json` file used by the rest of the
application. Profiles customize routing, instruction roles, context limits,
timeouts, and request options. They cannot override the provider identity:
`url`, `model`, `key_env`, and literal credential fields are rejected as unknown
settings. For example, merge the relevant profile and route settings into the existing
`plugins.settings.subagents` object; this is a partial excerpt:

```json
{
  "primary_profile": "primary",
  "default_profile": "primary",
  "primary_purposes": ["judge"],
  "allow_default_fallback": true,
  "max_parallel": 4,
  "purpose_routes": {
    "review": "reviewer",
    "judge": "judge-two"
  },
  "profiles": {
    "reviewer": {
      "purposes": ["review"],
      "priority": 20,
      "instruction_role": "developer",
      "context_chars": 64000,
      "keep_recent_turns": 6,
      "api_timeout": 120,
      "request_options": {"temperature": 0}
    },
    "judge-two": {
      "purposes": ["judge"],
      "priority": 20
    }
  },
  "default_purposes": ["*"],
  "default_profile_priority": 0,
  "child_allowed_actions": ["list", "read", "done"],
  "process_poll_seconds": 0.05
}
```

Routing is deterministic. An explicit profile on a delegation wins when it
supports the requested purpose, followed by an exact `purpose_routes` entry,
then the unique highest-priority compatible profile. Ties and missing
capabilities fail closed. Only an explicitly configured default may be used as a
fallback. Profile catalogs given to the model contain names, model IDs,
purposes, and priorities—never endpoints or credentials.

Each newly prepared child snapshots the current primary provider, including its
captured implementation and authentication. A profile's request options override
matching primary request options; other options and the primary timeout are
inherited unless the profile specifies a timeout. Prepared and running children
retain their snapshot while later children use the current primary configuration.

## Delegated workflows

The coordinating model has two text actions:

```json
{"action":"delegate","agent":"reviewer-a","purpose":"review","task":"Review document.txt and return concise feedback."}
```

```json
{"action":"delegate_many","agents":[{"agent":"tests","purpose":"review","task":"Review test coverage."},{"agent":"security","purpose":"security","task":"Review security."}]}
```

`delegate` blocks until one child returns. This makes dependent workflows
unambiguous: reviewer A finishes, the parent receives its structured result,
the parent edits and verifies the workspace, and only then does it delegate to
reviewer B. `delegate_many` is for independent tasks; it uses a bounded thread
pool, but always returns its result array in request order even if agents finish
out of order.

Each configured child owns a fresh, killable Python process, chat client, and
`AgentSession`. The provider key is sent through private stdin, never the process
command line. Children share only the workspace view and are restricted to
`list`, `read`, and `done`; they cannot
write, execute processes, use memory, or recursively delegate. They have no
model-turn cap unless the embedding application explicitly supplies one. One
child failure is reported without discarding successful siblings.

Lifecycle callbacks form a serialized channel with monotonically increasing
event sequence numbers. Start, activity, completion, and failure records include
the batch, agent, purpose, selected profile, and model. The default interfaces
keep these records hidden to preserve a quiet transcript; the parent receives
the complete structured result as `HOST_RESULT`. Configured-process activity is
drained and delivered while the child is still running; its output byte limit is
enforced during that stream, and overflow terminates the child.

## `/goal` mode

Set a goal in chat:

```text
/goal Finish the implementation and verify the full test suite
```

After every apparent completion, a fresh judge receives the objective and the
entire semantic session transcript. It must return a strict `continue` or
`complete` decision. A `continue` decision and its concise feedback are fed back
to the main agent, which resumes work in the same session. There is no automatic
judge-cycle limit. The final response becomes visible only when the judge accepts
completion, the goal is cleared, the user cancels, or an explicit error occurs.

Goal mode does not stop on transient provider failures. Main-agent and judge
timeouts, connection failures, HTTP 408/429, and retryable 5xx responses are
retried without an attempt limit. Backoff starts at 0.5 seconds and is capped at
30 seconds; a provider `Retry-After` value is honored up to that cap. The wait
polls cancellation, and configured child processes preserve retry metadata across
their private transport. A malformed judge decision is also retried because a
fresh response can recover. Authentication failures, invalid configuration,
evidence-limit failures, and other non-retryable errors are still surfaced
instead of causing an infinite retry loop.

With no judge option, the judge uses a fresh client of the main `primary` model.
Select a configured second model by profile name:

```text
/goal --judge judge-two Finish the implementation and verify the full test suite
```

Use `/goal` with no arguments to show the active objective and judge. Use
`/goal clear` or `/goal off` to disable it. `/clear` resets both the conversation
and active goal. Goal lifecycle events are job-scoped and hidden in the clean
default transcript.

## Portability and limits

The coordinator uses a bounded `concurrent.futures.ThreadPoolExecutor` to monitor
independent child processes, plus standard-library locks and process pipes. This
works on supported CPython versions on Windows, macOS, and Linux and lets
cancellation terminate a child whose HTTP call is blocked. Programmatic
`ModelProfile` instances without a process specification remain available to
trusted embedders and deterministic tests; those callables run in-process and
cannot be forcibly cancelled. Parallel fan-out is bounded to eight agents per
batch and configurable to at most 16 concurrent monitors. Subagent reports are
limited to 2,048 characters so serialized feedback remains usable in the parent
context. Judge feedback has the same 2,048-character ceiling. Tasks, errors,
configuration values, names, and complete
judge evidence are also size-validated; over-limit data fails explicitly and is
never silently truncated before the model sees it.


## Collective-task stress test

The optimization plugin provides a reproducible test of the actual workflow,
child-process transport, filesystem and context plugins:

```bash
python3 -B -S raychat.py --no-memory --exec "/benchmark-workflows --agents 50 --output workflow-stress.json"
python3 -B -S raychat.py --no-memory --exec "/benchmark-workflows --agents 50 --live --parallel 4 --padding-pages 2 --output workflow-live.json"
```

Fifty children reconcile separate invoice-ledger shards; the verifier checks
individual results, their combined total and request ordering. The default
local HTTP fixture makes each child read 36,000 bytes under a 16,000-character
request budget, forcing repeated compaction. It also checks follow-up history,
rejecting an oversized prompt before HTTP, recovery and complete worker cleanup.
The `--live` variant uses the configured provider and records its failures; it
requires credentials. Batch size and parallelism remain bounded, so fifty
children do not require fifty simultaneous model calls. Reports are linked in
[Verification](VERIFICATION.md).
