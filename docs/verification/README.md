# Verification evidence index

**Final functional QA passed on 2026-09-12.** The
[post-reorganization report](artifacts/reorganization_report.json) embeds the
source hashes, terminal results, full-suite metadata and configured static checks.

- 592 tests ran: no failures or errors; one Windows Job Objects test skipped on macOS.
  The suite used the default configuration with OS-enforced protection of the
  operator's RayChat home.
- All thirteen drivers passed against the extracted ZIP, including nine
  persistence scenarios, eight adversarial agent scenarios and the complete
  external-plugin authoring/install walkthrough.
- Fifty actual harness children completed the invoice task with eight active at
  once. All fifty compacted; 305 requests contained compaction. Maximum requests
  were 9,992/10,000 child characters and 31,983/32,000 parent characters. The
  independently verified aggregate was 24,900 cents.
- Current Self-Harness terminal checks verified promotion, held-out regression
  rejection, blocked evaluator cancellation and continued chat. Earlier live
  usefulness experiments remain separately identified below.
- Eight animated terminal samples measured 118.67–120.00 FPS. Literal
  `python3 raychat.py` also opened the existing operator installation, displayed
  plugin/agent menus and goal status, then exited cleanly.

The tested ZIP SHA-256 is
`9c4e33482dec8816d20675e7afaa309e4deeac532297adcfe2dbf583db770e69`.
Its 251 source files and manifest matched the materialized folder byte for byte.
The unit, TUI and FPS runs used the same frozen source on macOS 14.5 ARM64 and
Python 3.12.0. Windows terminal behavior was not exercised.

The initial Ruff run caught two reversed imports in a `TYPE_CHECKING` block in
`tests/typecheck/valid_contracts.py`. The report preserves that failure and proves
the correction changes only those two imports; all runtime and executed test
bytes remain identical. Full configured static checks then passed: mypy 2.3.1,
34 negative type contracts, Ruff 0.16.7 and formatting. Documentation, this
report and corrected Git attributes for the moved GEPA files were added after
behavioral verification, so a newly generated delivery
ZIP has a different archive hash. The separate maximum-strictness audit is still
being developed and is outside this pass claim.

Reproduction commands are in [VERIFICATION.md](../VERIFICATION.md) and
[RELEASE.md](../RELEASE.md). The records below describe their original sources
and conditions; they have not been relabeled as current tests.

## Historical terminal acceptance

| Evidence | Scope |
| --- | --- |
| [Packaged QA](artifacts/tui_qa_report.json) | Earlier extracted ZIP and its recorded archive hash; nine offline drivers, before the plugin-guide driver and current reorganization |
| [Bare core](artifacts/tui_bare_core_report.json), [features](artifacts/tui_features_report.json) | Earlier isolated core and feature-plugin terminal checks |
| [Navigation](artifacts/tui_navigation_report.json), [plugin lifecycle](artifacts/tui_plugin_lifecycle_report.json), [processes](artifacts/tui_process_report.json) | Earlier keyboard/mouse navigation, external package lifecycle and managed-process cancellation |
| [Collective context](artifacts/workflow_tui_context_report.json) | Deterministic 50-child workflow, measured context budgets, current-request aggregation and recovery |
| [Optimization terminal checks](artifacts/optimization_tui_report.json) | Earlier installed optimization commands, import/provenance checks and offline experiments |

These records used actual PTY interaction. The JSON summaries do not contain every
terminal frame or provider request. Paths under `/tmp/raychat-expanded/` refer to
local development evidence and may not exist for a recipient of the repository.

## Live evidence before the move

| Evidence | Scope and limits |
| --- | --- |
| [Default-provider interactive QA](artifacts/tui_manual_default_report.json) | Real-provider two-reviewer workflow, file repair, approvals, child follow-ups and observed model mistakes |
| [Live collective](artifacts/workflow_tui_live_report.json), [later live collective QA](artifacts/workflow_tui_qa_live_report.json) | Real-provider 50-child workflows on their recorded earlier runtimes; do not substitute these for current-tree acceptance |
| [Accepted Self-Harness trial](artifacts/self_harness_tui_report.json) | Paired baseline/candidate evaluation on the fixed synthetic task family |
| [Repeated Self-Harness QA](artifacts/self_harness_qa_report.json) | All three trials, including the flat outcome; repeated cases are not independent evidence of general coding gains |
| [Rejected trial](artifacts/self_harness_tui_rejected_report.json), [provider failure](artifacts/self_harness_tui_provider_failure_report.json) | Negative outcomes retained alongside improvements |

Live experiments use paid provider calls and are separate from offline release
smoke. Their reports preserve model/settings, outcomes and limitations. A useful
Self-Harness result on these small, repeatedly used tasks does not establish broad
or statistically reliable coding improvement.

## Older API, optimization and port records

The remaining families in `artifacts/` predate the terminal acceptance above or
record narrower engine experiments:

- `workflow_stress_report.json`, `workflow_live_report.json` and
  `workflow_glm_report.json`: earlier workflow stress/provider experiments.
- `self_harness_final_report.json`, `self_harness_glm_report.json`,
  `self_harness_live_report.json` and `self_harness_sdk_v2_report.json`: earlier
  harness proposal/evaluation experiments, including unsuccessful runs.
- `port_verification_report.json`: the GEPA source/import audit, prompt provenance
  and behavior oracle for its recorded source hash.
- `offline_optimization_report.json`, `live_optimization_report.json`,
  `useful_optimization_report.json`, `incident_response_offline_report.json` and
  `scalability_report.json`: optimization and parallel-evaluation measurements.
- The accompanying `*protocol.txt` files: exact candidate protocol artifacts for
  those experiments.

Do not rewrite embedded old module paths, `optimization/artifacts` locations or
source hashes to make these reports look current. They explain the earlier
experiment, not the repository's present import or output layout. Current engine
code and feature resources belong to `plugins/optimization/`.

## Evidence scope

The final report embeds all thirteen driver summaries and preserves failed
attempts and negative controls. Full ANSI transcripts and request logs remain
under `/tmp/raychat-expanded/final-thirteen-tui-v5/`; these local paths are not
portable release attachments. Re-run the checked-in drivers to generate a new
complete trace on another machine.

Deterministic providers exercise the real harness workers, package loader,
terminal, storage and tools. Their results establish harness behavior under the
recorded conditions. Live model improvements belong to the separate experiments
above and retain their measured limitations.
