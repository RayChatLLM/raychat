# Plugin migration and verification checklist

Runtime, unit tests and acceptance drivers use only Python's standard library.
Mypy and Ruff are development checks. Behavioral acceptance runs the complete
unit suite and drives the actual terminal UI. The checked milestones below
record the migration work; current acceptance is tracked in QA_TODOS.md.

- [x] Keep the session kernel independent of feature packages; use one registry,
      dependency resolver, source capture, transaction and generation switch.
- [x] Remove legacy worker/runner, direct subagent execution, detached-result
      bypass, duplicate compaction, service proxy and old schema migration paths.
- [x] Remove the remaining source-less test-module loading branch and make its
      fixtures use the same captured package loader as installed plugins.
- [x] Distribute features as external archives installed by the ordinary package
      manager; core startup does not import feature source projects.
- [x] Move feature settings and validation into their owning plugin manifests.
- [x] Put the stdlib GEPA engine and license inside the optimization package;
      remove unused non-stdlib integrations, retain upstream provenance and
      exact prompt/behavior parity checks.
- [x] Publish typed SDK contracts, generate a typed external plugin, support
      dependency discovery, catalog installation and updates without core edits.
- [x] Inject manifest usage instructions into requests and update them on live
      installation, helper edits, activation changes and removal.
- [x] Verify namespace helpers, source cleanup, local-edit protection, complete
      catalog metadata matching, restart behavior and management recovery.
- [x] Verify failed rollback callbacks cannot skip remaining cleanup or strand
      the runtime; failed child construction/restoration closes both runtimes.
- [x] Support transcript selection/copy, subagent keyboard/mouse navigation,
      parent return and one/multiple saved-session resume.
- [x] Verify focused double-Escape cancellation, sibling survival, replacement
      prompts, direct and isolated command-tree cleanup, including executor work.
- [x] Verify filesystem paging/hashes/edits/confinement, durable memory, skill
      discovery/loading/reset, goal judging/continuation/final suppression in TUI.
- [x] Pass full strict mypy for every application/plugin/spec/example/tool source
      plus launcher, and Ruff without blanket ignores or exclusions.
- [x] Run at least 50 actual harness children collectively, verify each result and
      aggregate, bound concurrency, pressure every child's context and recover
      after oversized-input rejection.
- [x] Retain current-task results by available context budget and repeat the live
      50-child aggregation after fixing the completed-history retention cap.
- [x] Demonstrate measured Self-Harness benefit through the TUI with fixed
      paired evaluations and retained rejected as well as accepted outcomes.
- [x] Prove the bare core works with an external provider and no bundled feature
      source, catalog or GEPA.
- [x] Finish the final portable archive/folder byte comparison and preserve the
      exact extracted-release TUI acceptance report.

Evidence, commands, scope and platform limitations are in
[VERIFICATION.md](VERIFICATION.md). The release gate requires all thirteen terminal
scenarios, complete unittest discovery, configured static checks and an independent
archive/folder byte comparison. See [the evidence index](verification/README.md)
for the tested source identity and the separately retained live experiments.
