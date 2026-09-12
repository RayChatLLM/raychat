# QA follow-up and final verification

The starting report was an actual `python3 raychat.py` crash in an existing
installation: the old installed provider imported the removed core constant
`MAX_HTTP_BYTES`. Fresh-home acceptance did not cover that upgrade path.
The later module move also exposed a stale `app_config` import. Literal launcher
checks now accompany packaged acceptance, including existing installed profiles
after SDK and host-configuration changes.

The completed items below describe the earlier interactive QA, before the current
repository reorganization. They are historical evidence, not final acceptance of
the reorganized tree.

- [x] Read the supplied report and reproduce the exact command in a terminal.
- [x] Reconcile existing profile installations by package contents using the
      ordinary transaction path; preserve local edits, external packages,
      disabled packages, removals, and failure retry behavior.
- [x] Launch the user's existing installation successfully and exercise real
      chat, tools, subagents, workflows and navigation interactively.
- [x] Show actionable plugin startup errors and prove repaired packages work.
- [x] Exercise queued child/parent input, focused cancellation, repeated
      workflows and deferred plugin reload through the TUI.
- [x] Disambiguate repeated same-name agent sessions in the picker.
- [x] Preserve combining Unicode characters in rendering and copied text.
- [x] Investigate and fix silent loss of oversized pasted input.
- [x] Exercise persistent-session navigation, error recovery, empty menus,
      resizing, scrolling, and text selection through the TUI.
- [x] Integrate nine terminal regression drivers and verify the portable
      distribution from its extracted archive, including 50 workflow children.
- [x] Restore eager multi-package staging after concurrent lint cleanup changed
      installation to consume only the first item; verify fresh-home startup.

## Final functional acceptance — completed

- [x] Run the complete `python3 -B -S -m unittest discover -s tests -v` suite;
      resolve failures/errors and record the final count and any skips. Run
      without a global configuration override and protect the operator's home.
- [x] Preserve an independently checkpointed plugin command when its concurrent
      model turn is cancelled; preserve complete saved history when that turn
      succeeds. Verify both by restarting the actual terminal application.
- [x] Pass strict mypy, the pinned Ruff checks and Ruff formatting on the final tree.
- [x] Rebuild the plugin catalog from the exact source being packaged.
- [x] Pass all thirteen offline TUI drivers against the extracted release, including
      the documented plugin-authoring/install guide and 50-child workflow test.
- [x] Verify ZIP/folder/manifest consistency and record the tested source and
      archive hashes, platform and interpreter.
- [x] Update the final acceptance report and the [evidence index](verification/README.md)
      without relabeling historical live experiments as current verification.

The user now explicitly requires all tests to pass. Raw-API unit and integration
regressions must run in addition to static checks and actual keyboard/mouse TUI
acceptance. Terminal tests use independent output, file/process and model
request observations. The [final report](verification/artifacts/reorganization_report.json)
records 592 tests (one Windows-only skip), thirteen packaged TUI drivers, the
50-child task and configured static checks. The separate maximum-strictness lint
audit remains outside this pass claim. The original failed Ruff run and its exact
type-fixture import-order correction are retained.
