# Release guide

The harness, plugins, acceptance drivers, and release builder use Python 3.10+
standard-library modules. Mypy and Ruff are development-only checkers. Run the
commands below from the repository root; use Python 3.12+ for the static checks.

## Prepare the distribution

Feature source lives in independent `plugins/ID` projects. The core loads their
installed archives through the same package manager used for external plugins.
Install the pinned development tools as described in [the README](../README.md).
Rebuild the catalog after changing a package, then run the complete checks:

```bash
python3 -B -S -m tools.build_plugin_catalog
.venv/bin/python tools/check_types.py
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
python3 -B -S -m unittest discover -s tests -v
```

The explicit `release.source_files` allowlist in `raychat.json` includes the host,
SDK, plugin projects, catalog and archives, acceptance drivers, fixtures,
documentation, and environment templates. New files must be added to this
sorted allowlist. GEPA and its license belong to the optimization package at
`plugins/optimization/gepa/` and `plugins/optimization/GEPA_LICENSE`.

Credentials, user workspaces, memory, logs, bytecode, caches, and previous build
output cannot enter the archive merely because they exist beside the sources.
Every release contains `PORTABLE_MANIFEST.json` with byte counts and SHA-256
hashes. Both the release ZIP and bundled plugin ZIPs store their contents
without compression, so compressor versions cannot change their bytes. Member
order, timestamps and POSIX origin/mode metadata are fixed on every platform.
Generated catalog and profile JSON use UTF-8 with explicit LF endings, matching
the checked-in sources even when rebuilt on Windows.

The complete unit suite must pass in addition to strict typing, lint/format checks
and real terminal acceptance. Retain its full output, including failures, errors
and skips. A static-only run is not full verification. Keep generated evidence
with its source hash in the selected build directory or CI artifacts.

## Drive behavioral acceptance

On POSIX, run the actual TUI drivers. Each `--output` directory must be new;
choose another run directory when repeating the checks.

```bash
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

These fourteen drivers use deterministic offline providers and control the application
through terminal input, mouse events, and visible state. They cover plugin lifecycle and
instructions, chat navigation/resume, transcript copying, focused cancellation,
process cleanup, filesystem operations, skills, durable memory, goals, workflows,
50 child sessions, and context pressure. Optimization acceptance enters five
commands in chat, checks the source/import report and prompt/oracle parity,
measures offline task improvement, and compares sequential/parallel results.
The plugin-guide driver checks the complete external-developer quickstart against
the installed SDK and package commands. Reports and terminal transcripts remain
under the output directories.

The live Self-Harness experiment is separate because it spends provider credits:

```bash
python3 -B -S -m tools.self_harness_tui --repetitions 2 --output ./build/verification-run-1/self-harness
```

It uses the configured provider/model and measures improvement on fixed synthetic
training, validation, and test cases. It exits unsuccessfully when the run does
not demonstrate a benefit and retains the report. This is an experiment on those
cases, not a general coding-performance claim. It is excluded from automatic CI
and offline release smoke. The collective driver can also use a real provider
with `--live`.

## Build and verify the exact ZIP

Use the builder directly to retain release acceptance evidence:

```bash
python3 -B -S -m tools.build_portable --smoke --smoke-output ./build/verification-run-1/release
python3 -B -S -m tools.build_portable --check --no-smoke
```

The first command writes `build/raychat.zip` and
`build/raychat/`, verifies their members, and extracts the ZIP into a
separate temporary directory. Smoke compiles every Python source without
creating bytecode and rebuilds plugin packages to compare them with the bundled
catalog. Required POSIX release smoke runs all fourteen offline TUI drivers against
the extracted application, including `plugin_guide_tui`. Confirm all fourteen appear
in its coverage record before reporting complete acceptance. `smoke.json` records
the platform and exact coverage; driver results and transcripts remain alongside it.

The second command independently rebuilds the archive and compares every byte,
then verifies the materialized release folder. It skips repeating acceptance.
`--output PATH` and `--folder PATH` choose different release destinations.

`tools/release.py` additionally removes recognized disposable caches from the
source tree while preserving repository internals and configured runtime/output trees:

```bash
python3 -B -S -m tools.release
```

It invokes the same archive builder and smoke implementation. Its smoke output is
temporary; use `tools.build_portable --smoke-output` when retaining full evidence
for a handoff. `--no-smoke` creates a development build without behavioral
acceptance and must not be reported as tested.

On Windows PowerShell, replace `python3` with `py -3`. Windows smoke performs
archive/package integrity and source compilation only. The POSIX PTY drivers do
not validate the Windows TUI. CI records this limitation explicitly, preserves
acceptance artifacts, and uses no live provider credentials.

## Recipient quick start

Send `build/raychat.zip`. After extracting it, start the chat from its
root directory:

```bash
cd raychat
export RAYCHAT_AUTH_TOKEN="your-api-token"
export RAYCHAT_MODEL="your-model-id"
export RAYCHAT_BASE_URL="https://provider.example/v1"
python3 -B -S raychat.py --workspace ./workspace
```

The configured release profile installs its feature packages on first launch.
Provider identity comes only from those three environment variables. Platform
templates and loading instructions are included in `environment/` and `README.md`.
Package defaults for role profiles and feature limits live in each `plugin.json`;
operator overrides live in `raychat.json` under `plugins.settings.ID`.
Core defaults remain in `raychat.json`; `--help` lists
launch overrides. Only live model calls need a provider key and external network
access.

Recipients on POSIX can repeat the offline TUI commands above from the extracted
directory, or point a driver at that directory with `--root PATH`. The commands
need no provider credentials or third-party Python packages. Windows users can
extract with `Expand-Archive` and start the application using `py -3`.
