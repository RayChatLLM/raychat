# Filesystem audit and acceptance ledger

Status: in progress. This ledger preserves all requirements from the supplied
checklist (67 checkbox bullets in the supplied text, despite its 69-item label);
implementation and verification are separate gates. Current changes have local
macOS and Linux-container evidence. Native Windows validation uses GitHub Actions;
the latest successful run covers the committed baseline, not the audit revision.
Deployment-software stress remains unverified.

## Shared policy now in use

`raychat/filesystem.py` owns `Path.replace()` publication, unique sibling staging,
OS sidecar locks, bounded regular-file reads, append-only writes, and bounded
owned-resource cleanup. It retries
Win32 5/32/33 for selected publication/deletion operations, plus 145 only when
removing a retired tree. Ordinary POSIX EACCES, missing parents, EXDEV and ENOSPC
propagate immediately. There is no unlink-before-replace or copy-over fallback.
The default deadline is 0.5 seconds with capped jittered backoff; lock acquisition
has a separate budget. An OS call itself can outlast that deadline.

Replacement and permission handling are distinct. Helpers never clear a live
file's read-only attribute or repair ACLs. Mode changes target only the owned
stage. Private snapshots and HTTP replay exports default to 0600; workspace snapshots preserve existing
mode bits and use the configured mode for new files. Ownership, ACLs, timestamps
and extended metadata are not preserved. A read-only stage can remain after a
Windows failure; diagnostics name the leftover. Cleanup never hides a primary
write/publication failure or replays a published business operation.

Runtime scratch directories use `OwnedTemporaryDirectory`: secure creation and
the same bounded retired-tree cleanup, with no implicit permission changes.
Consumers must finish before context exit or explicit cleanup. Failed cleanup
logs and retains the tree. Explicit cleanup detaches its finalizer before trying
deletion, so neither repeated cleanup nor collection revisits a name that another
owner might later reuse. Finalization is a fallback, not restart recovery.

Directory publication reserves scratch trees through `create_scratch_directory`.
Package and portable-release backups move into an unused child of a securely
allocated container; they never delete the reserved name and immediately reuse
it. A failed restoration retains the container and original content. Successful
restoration/commit cleans the retired container. Portable-folder crash recovery
uses the explicit journal and identity checks in `tools/release_folder.py`.

Package installation now uses `raychat/package_transactions.py` and an
operator-owned `plugins.transaction.json` beside its receipt. Under the persistent
scope lock it records original receipts, package identities and exact reserved
containers and their filesystem identities before changing public trees. Every incoming tree is prepared before
the moves. A separate commit record decides restart recovery: pending changes
restore their originals and receipt; committed changes retain their publication
and finish cleanup. Rollback moves new trees into owned retirement slots rather
than recursively deleting a public destination. Recovery is repeatable after
interruption, refuses conflicting external edits, and does not scan unrelated
scratch files. A reused container name or redirected root fails validation.
Manager construction and each subsequent writer recover under the
same lock; receipt changes start from fresh state rather than a stale instance's
cache. This is process-crash recovery, not power-loss durability or atomic
visibility for uncoordinated source readers. Runtime startup, package checks,
live reload capture and watched-source fingerprint scans
now acquire both scope locks in canonical sidecar-path order, refresh receipts,
and hold the locks through dependency discovery and immutable source capture.
They release both locks before importing or registering any plugin code. Explicit
serialized source snapshots need no installed-tree read lock. Each scope has a
one-second acquisition budget; acquisition failure releases previously acquired
locks. Linked development sources and external editors do not participate.
Installation and activation transactions acquire both scopes in the same order
and retain them through validation, commit or rollback. Source reads on that
transaction's thread explicitly borrow its ownership, without reacquisition or
recovery of its pending journal. Other threads still acquire the OS locks. Read
contexts outside a transaction remain nonreentrant. Transactions use the same
one-second acquisition budget per scope; publication retries remain separate.
Plugin code executing during transaction validation cannot wait for a competing
package writer to finish: that writer cannot proceed before this transaction
ends. Ordinary reload releases its read locks before executing plugin code.
Other source readers and orphan allocations made before journal publication
remain pending.
Restart readers make a single attempt at committed cleanup and log contention,
so a leftover retired file does not hide committed packages. A new installation
still requires its predecessor's journal cleanup to finish; persistent permission
failures need deliberate operator correction, never automatic attribute changes.

Private metadata uses `read_regular(..., follow_symlinks=False)`. It rejects
linked/nonregular endpoints, opens with `O_NOFOLLOW` where available, checks the
opened descriptor's filesystem identity against lstat, reads bounded bytes, and
closes before parsing. Identity mismatch closes the descriptor and fails without
reading the substituted file. Package receipts, catalog-cache JSON and transaction
journals use this mode; explicitly operator-selected inputs may follow links.

Catalog builds publish complete archives under `package-<sha256>.zip`, then an
immutable `catalog-<sha256>.json`, and finally replace `profile.json` to pin that
catalog. `catalog.json` is a separately valid convenience snapshot and may name a
different completed build when builders interleave. Existing immutable artifacts
are read and compared; unexpected content fails without replacing it. Catalog
metadata comes from the completed archive's manifest. Source projects must be
quiescent operator-owned inputs. Concurrent builders have last-writer-wins
profile publication; old profiles and their artifacts remain valid because the
builder never deletes them. Newly materialized portable releases include only
the exact current artifact graph in the explicit source allowlist. Distribution
readers use bounded regular-file reads and close before parsing.

Session journals use two persistent sidecars: `.writer.lock` owns the writer
lifetime, and `.lock` protects short IO operations and previews. The lock order
is writer ownership, then the instance's thread mutex, then the IO sidecar;
previews acquire only the IO sidecar and never wait. No lock byte lies inside the
journal itself. Recovery trims only an unterminated tail while holding both
sidecars. New journals use exclusive creation; resume rejects linked/nonregular
files, verifies the opened identity, and never creates a missing journal.
Opening/wrapping errors close owned descriptors and release writer ownership.
Permission errors retain their original type rather than being called contention.
Existing journal bytes and schema are unchanged. This replaces the former
data-file locking protocol, so old-version session writers must be closed before
this version accesses their journals; the two locking protocols do not coordinate.
External editors remain outside the protocol.

HTTP traces are incremental diagnostics in a unique connection-owned directory,
not a multi-file snapshot. `sent.http` includes attempted sends; `sent` events
distinguish completed sends. Raw append failures propagate without replay. A
failed attempt to record an existing transport error is reported through the
logger while the original error propagates. Replay exports use immutable body
names so either independently published command remains valid across a later
export or failure of its companion command. Body endpoint links are rejected
before any target bytes can change. `write_immutable` is shared with catalog
publication; matching content is reused, and unexpected existing content is
retained with an error. There is no automatic trace/body garbage collection.

Snapshot writers flush, fsync and close before publication. This is complete
single-file publication, not a multi-file transaction or a universal power-loss
guarantee. Parent-directory fsync is not provided. Generic helpers reject a linked
destination and intentionally support trusted linked parent directories. The
workspace API separately resolves links within its confinement root; it therefore
intentionally edits an in-root link's target. External editors are outside the
cooperation protocol. Network filesystems and cloud-synchronized data directories
have not been validated and are not covered by the local filesystem guarantee.

Synchronous helpers belong on worker threads. Recovery uses cooperative async
retry sleeps and never awaits after successful replacement. Async snapshot
staging, flush, fsync and close run in a worker. Cancellation joins that worker
cooperatively before the caller may retire its parent or release serialization;
repeated cancellation cannot abandon an open stage. Cancellation before publication
cleans the stage; repeated cancellation during cleanup reports an orphan. Lock sidecars persist and must never be garbage-collected merely because
of their age or contents. Reentry fails; multiple scopes must be locked in canonical
path order. Independent opens contend across processes and threads.

## Subsystem inventory and cooperation contracts

| Subsystem / paths | File class and owner | Contract and remaining gaps |
| --- | --- | --- |
| `tools/build_portable.py`, `tools/release_folder.py`, `tools/release.py` | Materialized release folder, independent ZIP snapshot, private retired trees | Persistent sibling lock covers folder staging, swap, decision and recovery. Journal inventories exact file identities, content and mode bits. Pending recovery conditionally restores the original; committed cleanup preserves later public edits. Partial cleanup accepts only matching remaining inventory entries. Cache cleanup excludes reserved release containers. ZIP and folder are independent publications; arbitrary consumers must stop before replacement or use the folder access context. |
| `raychat/filesystem.py`, `file_lock.py` | Unique owned stages; persistent OS lock sidecars | Central publication/cleanup/locking policy. Unknown orphans are diagnosed and retained. Reclamation requires an exact recorded owner and proof of inactivity; filenames, ages and PID strings do not grant ownership. |
| `plugins/filesystem/operations.py`, `raychat/workspace_files.py` | User workspace snapshots | A workspace sidecar covers reads, append-as-snapshot, anchored edits and hash edits. Streaming edit closes both handles before publication. Explicit multi-file updates retain workspace ownership and lend it only to accesses on their own thread; ordinary accesses remain nonreentrant. Atomic-write service is a last-writer-wins primitive; callers needing a transaction must coordinate. |
| `plugins/memory/store.py` | Shared JSON read–modify–write state | Sidecar covers load and reload–modify–publish. IDs derive from the freshly loaded state. Cached `all`, `page`, `context` represent the instance's last loaded snapshot. |
| `raychat/workspace_trust.py`, `application.py`, `plugin_arguments.py` | Operator-owned workspace trust JSON | Reads and complete grant/revoke updates share persistent `trust.json.lock` in the configured application home. Acquisition is bounded separately from replacement; no-op decisions do not publish. Bounded UTF-8 state rejects malformed, oversized and linked endpoints. CLI metadata honors pending grant/revoke without persisting it. Legacy writers and external editors do not cooperate. |
| `raychat/plugin_manager.py`, `package_transactions.py`, `composition.py`, `plugins.py`, `application.py`, `plugin_arguments.py` | Package receipts, catalog caches, staged downloads, installation trees and reload source captures | Stable scope locks protect a journaled tree/receipt transaction with restart rollback or committed cleanup. Production discovery, metadata, source capture and planning readers use those locks; `paths()` documents its caller-owned read context. Ordinary read locks end before plugin execution and downloads; an active transaction retains its writer locks and lends ownership only to its own thread. Profile plans revalidate both input receipts and publish membership with their packages. Snapshot runtimes defer installed-state access until needed. Exact recorded containers are reclaimed; external changes stop recovery. |
| `raychat/packages.py`, `plugin_sources.py`, `plugins.py` | Package source snapshots and archive extraction into private empty trees | Package traversal rejects links/reparse points before descent, bounds the inventory, reads regular descriptors and compares a second inventory before accepting bytes. Selected roots may resolve through links; descendants may not. Trusted parents and quiescent sources are required. Extraction owns its destination. Generation lifetime review remains separate. |
| `raychat/storage.py` | Append-only session journals | A persistent `.writer.lock` excludes other writers for the journal lifetime. A separate `.lock` serializes append/fsync, recovery, rollback, close and bounded preview reads. Writers acquire the IO lock for at most 0.5 seconds; previews attempt once and fall back to the ID when busy. Only newline-terminated preview records are parsed. Journals are never replaced and appends are never replayed. |
| `raychat_bootstrap/supervisor.py`, `recovery.py`, `releases.py` | Recovery snapshots, immutable release trees, per-child logs | Async publication retains the existing persistence lock and manifest ordering. Children are reaped before streams close/cleanup. Fixed checkpoint names and application crash recovery still need full verification. |
| `raychat/http_debug.py`, `http_replay.py` | Unique connection logs and replay snapshots | One connection owner serializes raw/event appends without retries. Each replay command pins a completed immutable body before publication; older bodies are retained. The fixed body filename is only a convenience snapshot. POSIX and PowerShell commands publish independently and each describes a complete replay. Secondary diagnostic-write failures are logged separately without masking the transport error. |
| `plugins/self_harness/evidence.py`, `candidate.py`, `promotion.py`, `runner.py`, `raychat/workspace_transactions.py` | Shared JSONL evidence, candidate snapshots and isolated evaluation trees | Evidence readers and writers share a persistent sidecar with a 0.5-second acquisition budget. Readers close the bounded suffix before parsing and ignore incomplete records. Under the lock, the next append discards only an unterminated tail, logs its size, then writes once without retries. Candidate promotion now holds both package scopes then the workspace lock through validation. All private stages complete first; conditional rollback checks bytes and file identity, and retires new files before deletion. A core undo journal records pending, committed and rolled-back decisions. Startup source reads and workspace operations recover it before access; conflicting external changes retain the record and undo files. Reserved scratch directories are excluded from package captures. |
| `plugins/optimization/gepa/checkpoint.py`, `state.py`, `optimize_chat_prompt.py`, benchmark scripts | Checkpoints, rollout outputs, reports, protocol snapshots; private fixtures | Shared publication replaces fixed `.tmp` and direct published writes. GEPA owns a persistent run-directory sidecar before cache loading through worker completion and final checkpoint publication. Checkpoints embed required state; cache and inspection files are independent optional snapshots. Generated task components hash typed IDs into bounded ASCII names. Explicit exports are independent last-writer-wins snapshots; protocol reports identify their measured bytes by digest. The CLI rejects aliased protocol/report destinations before work begins. |
| `plugins/skills/store.py`, `configuration.py`, `provider_settings.py`, `distribution.py`, `resources.py`, `presentation.py`, `core_review.py`, `core_tools.py` | Bounded configuration/source/skill reads | Skills deduplicate existing-file identities with samefile and read/close before parsing. Memory and protocol reads use read_regular, which checks the opened descriptor and rejects FIFOs without waiting for a writer. Remaining source/configuration traversal review pending. |
| `plugins/process/*`, `raychat/transport.py`, `workers.py`, `ui/clipboard.py` | Child processes, pipes and private workspaces | Existing cancellation/reaping paths retained; explicit handle-inheritance audit pending. |
| `tools/build_portable.py`, `build_plugin_catalog.py`, `release.py` | Release archives, catalog artifacts and retired build trees | Archive publication centralized. Folder rollback uses bounded replacement and observable cleanup. Catalogs publish content-addressed archives and a pinned catalog before publishing the profile pointer; older artifacts are retained. Cache cleanup is restricted to a quiescent owned checkout with configured and explicit protected paths, link/reparse refusal and one-attempt deletion. Folder crash recovery is now journaled through tools/release_folder.py as described above. |
| Other `tools/*`, `tests/*`, `examples/*` | Isolated acceptance fixtures, intentionally corrupt/live-edited inputs, reports, test subprocesses | Direct writes to exclusively owned fixtures are intentional. Separate published reports from fixtures and audit teardown; do not blanket-convert test mutations. |

### Reviewed private-file and process lifetimes

- Single-case prompt and incident evaluations share `evaluate_case`: each gets a
  fresh owned directory, runs `composition.run_session` with only the synchronous
  filesystem/context plugins, then inspects artifacts after the session closes.
  Those two plugins register no background consumers or resource-retirement
  callbacks. Incident evaluation delegates to this same lifetime. OS errors that
  escape session setup, persistence or shutdown now abort the optimizer unchanged;
  tool-level OS errors remain unsuccessful action results. Candidate execution
  errors still produce diagnostic scores. `run_session` attempts cleanup once and
  preserves an existing run failure or cancellation, logging a secondary cleanup
  error. `AgentSession.close` attempts journal cleanup even after host cleanup
  fails, retaining the first failure and logging a second. Cleanup failure after
  success propagates without rerunning the session. Neither path proves all
  resources retired after a failure. Composition closes a runtime it created if
  session initialization fails, preserving the initialization exception if cleanup
  also fails. Supplied runtimes and journals remain caller-owned until session
  construction succeeds. Runtime construction owns cleanup before manager setup
  and attachment, not only plugin loading. Arbitrary caller/provider background
  lifetimes still require review.
- Offline release cache cleanup requires an operator-owned, quiescent checkout.
  It protects configured top-level exclusions (including build, environment and
  runtime directories), configured workspace/home, default and selected release
  outputs, environment roots identified by `pyvenv.cfg`, and explicit
  `--preserve`/API paths. Protection conservatively ignores case. A preserved
  descendant also protects an enclosing cache tree. Directory symlinks and Windows
  reparse points are pruned before traversal; linked cache entries are refused
  before deletion. Scanning errors propagate instead of silently skipping an
  unreadable subtree. File and directory deletion each get one attempt without
  changing attributes; disappeared cache files are already cleaned up. Initial
  cleanup failure aborts before building. Failure of final cleanup after output
  verification is logged and returned in `cleanup_errors`, without rebuilding or
  rolling back published outputs. Reported removal counts cover completed passes
  only. Other runtime locations must be explicitly preserved or use the
  non-cleaning builder. These checks do not defend against concurrent pathname
  substitution or infer ownership from cache-shaped filenames outside that
  checkout contract. Native Windows junction execution is selected in CI and
  remains unverified for the current changes.
- Optimizer protocol/report exports do not read or merge the existing destination.
  Each call publishes its complete in-memory result with the shared stage/replace
  helper. Last-writer-wins is intentional, including benchmark reports. Protocol
  and report publication is independent; a report failure leaves any successfully
  published protocol in place and never reruns optimization. CLI validation rejects
  equal paths, hard-link aliases and ambiguous case-folded names using the shared
  filesystem identity check; this does not reserve paths against external editors.
  Existing endpoint symlinks are replaced, while selected parent links are followed.
  Reports identify the evaluated protocol by SHA-256, which readers must compare
  before associating independently published artifacts. Private benchmark fixtures
  are written inside fresh owned temporary directories. Experiment report failures
  propagate on their own, but are separately logged when an experiment is already
  failing so the original exception survives. Self-harness benchmark runtime
  ownership begins immediately after construction, covers provider setup and all
  evaluation, and closes the runtime before reading attempt diagnostics. Attempt
  reads close before parsing, reject linked/nonregular files, are bounded to 16 MiB
  and consume only newline-committed records. A diagnostic failure propagates on
  its own and is logged when an experiment is already failing. A runtime shutdown
  failure records `cleanup_error`, retains the scratch directory and preserves an
  existing primary exception. `OwnedTemporaryDirectory.retain` detaches context,
  explicit and finalizer cleanup; automatic age-based reclamation is forbidden.
  Workflow benchmark shutdown separately attempts child enumeration, runtime close
  and liveness verification, even if an earlier step fails. Failure of any step
  retains scratch, records cleanup errors and prevents a successful
  `workers_stopped` result. The primary workflow/cancellation exception survives.
  Its gateway socket is closed even if the serving thread never starts; shutdown,
  socket close and joining are separate non-retried steps. Gateway handlers use
  in-memory state rather than workspace files. Other runtime cleanup scopes still
  need review against this retirement contract.
- GEPA's public optimizer acquires `gepa.lock` before constructing its adapter,
  loading evaluation caches, generating seeds or starting search. A competing run
  fails immediately; the OS releases ownership after a killed process, while the
  sidecar remains. Internal direct engine callers must provide the same ownership.
  All evaluation workers finish before the owning scope exits. Filesystem errors
  propagate even when candidate exceptions are configured to continue: checkpoint
  persistence is never replayed by the search loop. The checkpoint contains its
  own state, including cached evaluations and best outputs. Disk fitness caches
  and rollout inspection files may be ahead of the checkpoint after a crash;
  they are optional independent snapshots, not a multi-file transaction. External
  inspection readers do not acquire the run lock and may see different complete
  versions. Old optimizer processes must stop before sharing a run directory with
  this locking version; older versions do not participate in the protocol.
- The download worker exclusively creates `response` inside its parent's private
  scratch directory. `transport._run_process` stops/reaps the child and drains its
  protocol tasks before `run_child` returns. The parent then reads/closes the
  response and verifies its reported length before exiting the directory scope.
  This file is private IPC output, not a public snapshot; direct exclusive creation
  is intentional. Download cancellation and process-tree tests cover retirement.
- Captured plugin generations write only their own newly allocated directory.
  Source bytes are captured before compilation; retirement unregisters the finder,
  removes generation modules and releases the owned tree only after resource
  cleanup succeeds. Failed registration cleanup, rejected-generation cleanup,
  retired-generation cleanup or final shutdown retains the affected generation's
  modules, finder and files. Retention detaches automatic deletion, logs the owned
  path, and makes later retirement a no-op. It does not turn failed cleanup into
  success or retry callbacks. Generations with deliberately transferred resources
  (`on_reload=False`) stay alive until final runtime shutdown; successful shutdown
  then retires those deferred sources. If final cleanup fails, both active and
  deferred sources remain retained. Successful live publication stays committed
  even when old-generation cleanup fails. Partial registration of a replacement
  also respects transferred-resource ownership: rollback skips those callbacks
  and defers their candidate source files until the active runtime shuts down.
  Initial installation and newly added plugins still own their failed-registration
  cleanup. Coordination while capturing mutable
  installed sources remains a separate outstanding issue.
- Release construction writes its own candidate tree before sealing and exposing
  its identity. Supervisor configuration files are written in a fresh per-run
  directory before launching children. These are private construction writes.
  Read-only mode changes apply to these owned artifacts, not live destinations.
- Supervisor children inherit their explicitly redirected per-child log stream.
  Shutdown attempts bounded termination/reaping before closing the parent's log
  handle; release/recovery directories remain retained if shutdown fails. Release
  validators finish or are killed/reaped before their log context exits.
- Transport, clipboard, release validation and supervisor subprocess calls do not
  opt into additional inherited descriptors. The local Python `Popen` signature
  confirms `close_fds=True`; both Windows command-helper layers explicitly request
  it. No production `pass_fds`, `set_inheritable` or custom handle-list override was
  found. Native Windows handle behavior remains part of the platform acceptance.
- The only runtime `os.chdir` is in the single-request command worker. Its other
  thread is a parent-liveness monitor, which does not use relative filesystem
  paths. The shared UI process does not change its working directory.
- Generated self-harness evaluation scripts now specify UTF-8 for source metadata,
  endpoint configuration and harness text. The generated release dependency check
  likewise reads its requirements as UTF-8. No implicit `read_text()` calls remain
  in runtime/plugin code or these embedded script strings.

## Acceptance evidence

`tests/test_filesystem.py` exercises same-source retries for Win32 5/32/33,
permanent failures, deadline exhaustion, failed fdopen, fsync/cleanup failure,
Unicode and missing parents, thread serialization, reentry, killed-process lock
recovery, and a real child holding a read handle. Ordering uses pipes/barriers.
`tests/test_supervisor_persistence.py` now injects winerror=32 explicitly instead
of treating all PermissionError instances as transient, and checks `.pending`
cleanup. Existing workspace/memory regression tests remain in the suite.

The portable workflow includes filesystem tests on Windows, Linux and macOS with
Python 3.10, 3.12 and 3.14. Configuration is not evidence that those runs passed.
The user-selected Windows environment is GitHub Actions. Its latest verified
results and the exact revision boundary are recorded below.
Separate write/flush/close failures, read-only files, mappings, four-process
increments, symlinks and stale memory-instance updates are also covered locally.
Held-source contention, process termination before/after publication, temporary
and persistent mocked deletion denial, and no deletion of a still-active stage
are now covered. A FIFO input test runs in a bounded child process. Portable-name
integration covers hostile, case-colliding and oversized optimization example IDs;
existing-file identity tests cover hard-linked skills and export paths.
Remaining acceptance includes application restart reclamation, denied directories,
real cross-volume behavior, long paths in the shipped executable, Windows junctions,
and native security-software/deployment tests.

## Original requirement ledger

Native evidence labeled **CI dea3fe4** below refers to the fully successful
[PR run 36158695935](https://github.com/RayChatLLM/raychat/actions/runs/36158695935).
It proves the selected tests on that revision; later changes still require their
own CI run, and hosted Windows success does not prove ordinary-user privileges.

| # | Requirement | Evidence / remaining work |
| --- | --- | --- |
| 1 | Find every place the project touches files. | Pending full audit; see subsystem inventory above. |
| 2 | Classify each file by how it is used. | Pending full audit; see subsystem inventory above. |
| 3 | Create one filesystem utility module. | Implemented: `raychat/filesystem.py`; legacy locking imports delegate to it. |
| 4 | Document ownership and failure behavior. | Helper ownership/budget/failure contracts documented; complete per-caller review pending. |
| 5 | Remove check-then-act assumptions. | Memory load, workspace append and session resume handle operation failures. Package metadata now checks the actual opened descriptor and rejects identity substitution instead of relying on exists/is_file before open. Remaining traversal sites need review. |
| 6 | Use unique, securely created staging files. | Implemented for migrated snapshots: secure mkstemp sibling stages; directory staging remains under review. |
| 7 | Stage beside the destination. | Implemented for migrated snapshots; stage parent is destination parent. |
| 8 | Own the temporary file descriptor explicitly. | fdopen-failure test proves descriptor closure; session wrapping also fixed. Other raw stream owners remain to audit. |
| 9 | Audit `NamedTemporaryFile`; do not ban it outright. | No NamedTemporaryFile use found in runtime/plugin/build sources. |
| 10 | Keep temporary resources alive until consumers finish. | Pending full audit; see subsystem inventory above. |
| 11 | Generate fresh scratch names instead of immediately recycling deleted ones. | Release backup name recycling removed. Release/package staging and backup containers now use secure directory allocation. Remaining name-generation inventory still under review. |
| 12 | Replace direct writes to published snapshots with staged writes. | Migrated workspace, memory, package JSON, recovery, GEPA checkpoints/outputs, optimizer exports and release archives. Catalog/report inventory remains. |
| 13 | Finish all writing layers before publication. | Write/flush/fsync/close injection tests preserve the old destination and prove replacement was not called. |
| 14 | Use `Path.replace()` or `os.replace()` when overwriting is intended. | All runtime file replacements now call the shared Path.replace policy. |
| 15 | Remove weaker fallback paths. | Shared helper has no weaker fallback. Directory rollback protocols remain a separate audit item. |
| 16 | Retry publication of the same completed source. | Test verifies identical source path/content across Win32 5/32/33 retries. |
| 17 | Track whether publication already happened. | Shared publication has no post-success durability step; cleanup errors are reported separately. Release-folder double-failure tests retain original backups; recorded commit decisions prevent rollback after interrupted success. Process-kill recovery also covers partial cleanup. |
| 18 | Keep read handles short-lived. | Pending full audit; see subsystem inventory above. |
| 19 | Distinguish shared reading from shared deletion. | Native child-reader tests passed on Windows, Linux and macOS in CI dea3fe4, distinguishing POSIX old-inode reads from Windows replacement denial. |
| 20 | Audit hidden handle owners. | Pending full audit; see subsystem inventory above. |
| 21 | Give memory mappings their own lifecycle. | Native read-only/mmap lifecycle tests passed on all three operating systems in CI dea3fe4. No runtime mmap consumers were found in the existing inventory. |
| 22 | Control subprocess lifetimes and handle inheritance. | Download/transport, command helpers, clipboard and supervised-core/validator lifetimes reviewed above. Native CI dea3fe4 passed Windows Job Object tests and POSIX termination/inherited-pipe tests; wider owner-lifetime review remains. |
| 23 | Use immutable versions for long-lived readers. | Plugin runtimes retain private captured generations. Generated catalog profiles pin immutable catalogs and archives, retained for old readers. Remaining subsystems still need lifetime review. |
| 24 | Choose the concurrency contract explicitly. | Pending full audit; see subsystem inventory above. |
| 25 | Use a stable sidecar lock. | Memory/workspace/evidence and package writer sidecars are stable. Broader reader coordination remains. |
| 26 | Hide platform locking behind one standard-library wrapper. | One guarded fcntl/msvcrt wrapper with Windows offset zero and byte count one. |
| 27 | Keep OS-backed lock files in place. | FileLock closes descriptors without unlinking; persistent sidecar tested after release. |
| 28 | Use a simple coordinated-read protocol where appropriate. | Pending full audit; see subsystem inventory above. |
| 29 | Bound lock acquisition separately from replacement retries. | Separate configurable acquisition timeout; reentry/thread/process tests pass locally. Package reads and installation/activation transactions acquire both scopes in canonical order; partial-acquisition release and thread-local transaction ownership are tested. Other subsystem lock ordering remains to audit. |
| 30 | Do not treat an exclusive-create sentinel as a complete locking solution. | Locks are OS-backed, not exclusive-create sentinels; killed-holder recovery tested. |
| 31 | Document the cooperation boundary. | Local cooperative boundary and unsupported network/sync semantics documented above. |
| 32 | Classify Windows failures using `exc.winerror`. | Winerror classifier exercised independently from errno; POSIX PermissionError is not retried. |
| 33 | Use a monotonic deadline, capped backoff, and jitter. | Monotonic bounded budget with capped jitter and separate lock deadline implemented. |
| 34 | Do not infer the cause from error 5 alone. | Error 5 receives selected bounded retry only; no inferred cause or permission mutation. |
| 35 | Do not keep retrying permanent conditions. | EACCES, EXDEV, ENOSPC and ENOENT injection verifies one attempt. |
| 36 | Preserve useful diagnostics. | Escaped path, type, errno, winerror, attempts and elapsed logged; stage cleanup logs leftovers. Full integration diagnostics still under review. |
| 37 | Protect responsiveness. | Recovery snapshot staging, fsync and close run off the event loop. Cancellation joins the stage writer before cleanup, while replacement/cleanup retry sleeps remain cooperative; no await follows successful publication. Blocked-fsync, repeated-cancellation and post-publication tests pass. Other bulk I/O/event-loop paths remain under review. |
| 38 | Retire a directory before deleting it. | Pending full audit; see subsystem inventory above. |
| 39 | Delete only resources the operation owns. | Pending full audit; see subsystem inventory above. |
| 40 | Treat missing owned scratch files as already cleaned up. | Missing owned scratch files use unlink(missing_ok=True); tree cleanup verifies root disappearance. |
| 41 | Preserve the primary failure. | Stage failures and owned-directory cleanup retain the primary error. Package rollback preserves publication failures and original backups. Session run/cancellation survives shutdown failure; host cleanup failure survives a second journal cleanup failure. Composition initialization failures survive runtime cleanup failure. Secondary cleanup errors are logged, and cleanup-only failures propagate. Other callers and higher-level paths remain to audit. |
| 42 | Audit `rmtree()` error handling. | Runtime ignore_errors=True sites migrated to bounded, logged cleanup; remaining strict teardown paths need audit. |
| 43 | Design orphan recovery. | Package restart recovery reclaims only exact journal-owned containers after acquiring the scope lock; a live parent can reclaim its recorded child stage after reaping. Unknown stages and pre-journal allocations are deliberately retained. Other subsystem recovery paths still need review. |
| 44 | Use approved, configurable application-data locations. | Absolute storage.home_directory, default sessions, trust state and explicit session-directory overrides are exercised through the shipped launcher on all three operating systems in CI dea3fe4. Remaining application-data locations still need inventory review. |
| 45 | Test with antivirus and relevant index/sync software enabled. | Pending full audit; see subsystem inventory above. |
| 46 | Distinguish security policy from contention. | Pending full audit; see subsystem inventory above. |
| 47 | Keep read-only handling opt-in. | Only the private stage receives chmod; no live permission repair. Access-denied test verifies this. |
| 48 | Specify a metadata policy. | Mode/ACL/ownership/timestamp policy documented above; metadata is not claimed preserved. |
| 49 | Use `Path` and string paths consistently. | Sole runtime chdir is confined to a one-request worker with a filesystem-independent liveness watcher. Remaining path construction inventory still under review. |
| 50 | Generate portable, collision-safe names. | Optimization IDs now use bounded digest components of typed checkpoint keys. Integration verifies hostile, long and case-colliding IDs remain confined and distinct. Other generated-name sites still need review. |
| 51 | Do not use `normcase()` as a universal identity test. | Removed all production normcase calls. Skills and existing export paths use samefile; missing export paths conservatively reject case-folded spelling collisions. No race-free identity claim. |
| 52 | Test long paths in the executable actually shipped. | The actual extracted launcher ran with Unicode installation/configuration/workspace/data/session paths longer than 320 characters on Windows, Linux and macOS, Python 3.10/3.12/3.14, in CI dea3fe4. The launch working directory stays short because Windows CreateProcess has a separate limit; no path-prefix or OS-setting changes are used. |
| 53 | Choose encoding and newlines explicitly. | Runtime/plugin read_text calls, including generated evaluation scripts, specify encoding. Four embedded default-encoding reads were corrected to UTF-8. Broader tooling/newline audit remains. |
| 54 | Define symlink and junction behavior. | Snapshot endpoint rejection and intentional in-root resolution are documented. Real Windows package, bootstrap and cleanup junction fixtures passed in CI dea3fe4. Wider source/configuration traversal and link policy review remains. |
| 55 | State the guarantee precisely. | One-file visibility, serialization and power-loss durability are explicitly distinguished. |
| 56 | Implement durability to the actual requirement. | Snapshot flush/fsync/close is implemented; parent-directory durability is not claimed. |
| 57 | Do not treat several replacements as one transaction. | Packages use an undo journal and commit decision, including profile membership with installed packages. Candidate file batches now have their own undo/commit record and restart recovery before source capture. Catalog profiles pin immutable metadata and archives, published before the profile pointer. Process termination and interleaved catalog builders are tested. Remaining multi-file subsystems still need review. |
| 58 | Keep live databases and logs out of generic replacement helpers. | Session/HTTP/evidence logs retain append protocol. Shared append never retries a partial append. |
| 59 | Validate nonlocal storage separately. | Network/synchronized storage not validated; excluded from the local-filesystem guarantee. |
| 60 | Native platform coverage: | CI dea3fe4 passed all nine Windows/Linux/macOS and Python 3.10/3.12/3.14 combinations, plus 1,074 complete Linux unit tests. Native subprocess tests are included. Prior Linux UID 65534 and macOS ordinary-user evidence is recorded below; ordinary-privilege Windows acceptance remains unproven on hosted runneradmin. |
| 61 | Reader contention: | Child-held destination and staged-source publication tests passed on Windows, Linux and macOS in CI dea3fe4. They use pipe ordering, assert bounded Windows failure with old bytes intact, then successful publication after reader close. |
| 62 | Competing writers: | Thread and four-process increments verified; stale memory instances preserve IDs/updates. Broader snapshot-reader stress pending. |
| 63 | Lock recovery: | Killed native lock holder, thread contention and accidental reentry passed on all three operating systems in CI dea3fe4; recovery retains the persistent sidecar. |
| 64 | Failure and crash boundaries: | Stage fault injection and process termination covered. Package writers are killed after backup, new tree, receipt, commit record and rollback moves; new managers recover from disk and leave unrelated work intact. Candidate writers and recovery workers are also killed at journal, replacement, commit, restore and cleanup boundaries. Portable-folder writers are now killed at journal, backup, publication, decision, rollback and partial-cleanup boundaries; coordinated access recovers exact recorded trees. Other application reclamation remains pending. |
| 65 | Cleanup contention: | Mocked transient/permanent Win32 deletion denial, cleanup-only error 145 and primary-failure preservation tested. Native Windows deletion contention remains pending. |
| 66 | Environment and path edges: | CI dea3fe4 covers Unicode, missing parents, read-only files, mmap, links/junctions, long-path launcher execution, and mocked disk-full/cross-volume failures. Actual denied directories and cross-volume acceptance remain open. |
| 67 | Deployment conditions: | Native deployment/software validation outstanding; no exclusions or permission workarounds introduced. |

## Platform references

- [Microsoft system error codes](https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-)
- [Python nonblocking Windows byte-range locks](https://docs.python.org/3/library/msvcrt.html)
- [Python replacement semantics](https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace)

## Orphan ownership and recovery policy

Never infer inactivity from age, a PID string, `.tmp`/`.pending` suffix, or a
sidecar's existence. A live parent can reclaim a specific stage after receiving
its name from its own child and confirming that child has terminated and its
handles have closed. `test_kill_before_and_after_publication_and_explicit_orphan_cleanup`
exercises that proof and leaves another live writer's stage intact. `remove_owned`
is the final deletion primitive, not a liveness detector.

After a host restart, the current helper cannot prove which `.raychat-*.pending`
files were left by the previous host versus another active instance. They are
retained by design. Automatic reclamation of such unrecorded stages is outside
this recovery policy: implementing it would require durable ownership records and
a cooperative lease held through the closed-stage publication interval. Package
journals already provide the narrower recorded-ownership protocol for their exact
containers. Completed catalog artifacts are retained too, since a reader may hold
an older profile. Persistent scope locks must never be reclaimed as orphan files.
Operator cleanup must first establish ownership and stop/reap all
relevant consumers; there is no glob-delete or stale-PID recovery path.

## Recorded verification checkpoints

The first full local run after the central migration passed 903 tests with one
skip. The portable build with SHA-256
`d3d6cc8497cfe15cf70cf1755957dd8580201e98f80445241b748e0034ee54fa`
passed all fourteen offline POSIX terminal scenarios. Evidence is retained in
`build/filesystem-acceptance`; that archive predates subsequent path-identity,
regular-input, replay-publication and crash-test changes and is not proof of their
acceptance. Those subsequent changes have focused regression coverage; final
verification must be refreshed against the final source/ZIP identity.

The refreshed full suite passed 914 tests with one skip. It predates the subsequent
owned-temporary-directory and offline package rollback changes; those changes have
separate focused tests. The second portable smoke run uses
`build/filesystem-raychat-2.zip` with evidence in `build/filesystem-acceptance-2`;
it passed all fourteen scenarios with archive SHA-256
`6dd289aae3ccef752228ed8229ba2a5f32d7f73c2761505279e7e5af6e040966`.
It predates the subsequent cleanup and reserved-directory changes.

Cleanup verification: 42 filesystem/source-generation/portable tests passed,
then 27 filesystem/package-rollback tests passed after correcting the new test's
lock-release assertion. All other package/download/self-harness integration tests
passed in their 82-test run; that run's sole error was the corrected assertion.
The 37 GEPA/optimizer tests also passed. Strict mypy, launcher typing, Ruff and
format checks passed with zero suppressions in `build/filesystem-quality-3`.

Reserved-directory verification: 57 filesystem, package-transaction/rollback and
portable-build tests passed. Coverage includes allocation failure after moving the
original package, retained originals after failed restoration, and forbidding
delete-and-reuse of release backup names. Strict quality checks passed with zero
suppressions in `build/filesystem-quality-4`.

Package-journal verification: 62 package/recovery/portable tests passed, including
five native child termination boundaries, grouped install/update/removal recovery,
commit interruption, externally edited content, reused container identities and
stale manager receipts. A separate 48-test recovery/orchestration run passed after
adding bounded receipt-read acquisition. Strict quality passed in
`build/filesystem-quality-7`; final typing also rejected all 115 negative contracts.
The first expanded full run exposed startup lock contention. A refreshed complete
run and third portable smoke build are checking the corrected startup budgets;
their results are not yet recorded here.
The final seven recovery tests also verify that committed packages remain readable
while retired-file cleanup fails; strict typing passed for that final adjustment.

The refreshed complete run passed 926 tests with one skip. Subsequent private
metadata read changes passed 95 focused filesystem, package, recovery and download
tests, with strict quality checks in `build/filesystem-quality-8`. Descriptor
substitution and link rejection are tested directly, including descriptor closure
before any bytes from a substituted file are read. The third smoke build predates
this metadata-read change and the final committed-cleanup adjustment.
The third smoke build passed all fourteen terminal scenarios with SHA-256
`62629b13cd6fcc30d8ff927f5a60ab84e2fb9ac4477cfa362e600bde349df73f`;
evidence is in `build/filesystem-acceptance-3`. The subsequent explicit-encoding
changes passed 23 self-harness tests and lint/format checks; generated catalog
archives were refreshed afterward.

Evidence-log verification: 75 evidence, self-harness, HTTP-debug and filesystem
tests passed after putting evidence readers on the writer's persistent sidecar.
Coverage includes a real child reader blocked by the writer, exact suffix
boundaries, recovery across several read chunks, partial writes without replay,
and preservation of all content if recovery cannot read the expected suffix.
Newlines are record commit markers; only an unterminated final record may be
discarded by the next append while holding the lock. Completed records are never
rewritten, and append/recovery errors propagate without retry. This is not a
power-loss durability guarantee. Strict typing, launcher typing, Ruff and format
checks passed with zero suppressions in `build/filesystem-quality-10`.
The evidence tests are included in the native CI matrix; those remote runs remain
unverified here.

Package-capture verification: 63 package, recovery and agent-session tests passed.
The final nine recovery tests passed after extending all five real writer-crash
boundaries to exercise startup with a previously constructed manager: it rejects
capture while the writer holds its scope, then recovers and captures the expected
generation after the child is killed. Separate child probes verify both scope
locks are held during source reads and released before entrypoint imports. Failed
source reads release both locks, and refreshed receipts expose installations made
by another manager. Strict typing, launcher typing, Ruff and formatting passed
with zero suppressions in `build/filesystem-quality-13`. These results cover local
macOS execution; the native CI matrix includes the recovery tests but has not been
run here.

Catalog-publication verification: 72 catalog, portable-build, package and download
tests passed; the final four catalog tests also passed after adding an explicitly
interleaved two-process build. Tests kill builders after archive, immutable
catalog, convenience-catalog and profile publication, verify old readers remain
valid, restart the build, and install its resulting package. Publication failure
keeps the old profile and cleans its owned stage; a corrupted immutable artifact
is rejected and retained. Strict quality passed in `build/filesystem-quality-16`
with zero suppressions.

The refreshed complete Python 3.12 suite passed 940 tests with one skip in
444.849 seconds. The 47 filesystem, package-recovery, evidence-log and catalog
tests also passed natively on macOS Python 3.10.19 and 3.14.3. The isolated 3.10
runtime is under `build/python-runtimes`; the default Python was unchanged.
Docker's local daemon was unavailable, and no native Windows/Linux result is
claimed.

The fourth portable artifact passed all fourteen terminal smoke scenarios and
verified all twelve bundled plugin archives. Its SHA-256 is
`f61c645e83089e4cbd41f3e38f63d449624cf1f8eed7a585042d23e25c265863`,
with evidence in `build/filesystem-acceptance-4`. This archive predates the final
CLI-only typing adjustment and subsequent audit-ledger updates; its catalog
publication and source-capture behavior matches the tested implementation.

Session-journal verification: 70 journal, plugin integration, picker and
filesystem tests passed after the sidecar migration. The 23 durable-session and
concurrent-checkpoint tests passed on each of macOS Python 3.10.19 and 3.14.3.
The final six journal tests passed on 3.10, 3.12 and 3.14, including a real active
writer preview, busy-IO fallback, killed-writer tail recovery, permission failure,
descriptor-wrapping failure, missing-resume behavior and retention of writer
ownership after bounded append/close contention. Strict quality passed with zero
suppressions in `build/filesystem-quality-19`. The terminal persistence driver
passed with evidence in `build/session-journal-persistence`; native Windows
execution of the new preview protocol remains outstanding. Earlier full-suite
and portable smoke checkpoints predate this session-journal change.

HTTP replay-publication verification: 84 HTTP replay/debug/integration, catalog
and filesystem tests passed. The 42 HTTP replay/debug and catalog tests also
passed on macOS Python 3.10.19 and 3.14.3. New cases fail each companion-file
publication and verify both previously exposed commands still reference their
own complete bodies; endpoint symlinks cannot redirect body publication. A
secondary diagnostic disk failure preserves the original transport exception
object and produces a separate error log. Strict quality passed with zero
suppressions in `build/filesystem-quality-20`. Earlier full-suite and portable
smoke artifacts predate these HTTP changes; native Windows remains unverified.

GEPA run-ownership verification: 53 optimizer, catalog and portable-build tests
passed on macOS Python 3.12. The 40 GEPA/optimizer tests passed on each of macOS
Python 3.10.19 and 3.14.3. New cases coordinate a real child through a pipe, reject
a competitor before adapter construction, kill/reap the owner and successfully
restart, propagate a disk-full checkpoint failure after one attempt even with
candidate-error continuation enabled, and resume without cache/inspection files.
The bundled optimization archive and pinned catalog were rebuilt; release source
inventory tests passed. Strict checks passed with zero suppressions using the
pinned toolchain in `build/filesystem-quality-22`. The earlier attempt in
`filesystem-quality-21` used an unpinned interpreter and is not acceptance
evidence. The native CI matrix now includes the GEPA engine tests, but no remote
Windows/Linux execution is claimed. Earlier full-suite and portable smoke
artifacts predate this run-ownership change.

Optimizer export verification: 72 optimizer, self-harness, catalog and portable
build tests passed on macOS Python 3.12. The 35 optimizer/experiment publication
tests passed on Python 3.10.19 and 3.14.3, with the additional protocol/report
publication-boundary test subsequently passing on both. Tests reject colliding
export paths before model work, preserve the old file at each failed publication,
retain an already published protocol when the report fails, avoid rerunning the
experiment, and preserve original experiment/cancellation exceptions when report
publication also fails. The bundled archive and pinned catalog were rebuilt.
Strict checks passed with zero suppressions in `build/filesystem-quality-25`.
Native CI includes the optimizer and experiment publication tests; remote
Windows/Linux execution remains outstanding. Earlier full-suite and portable
smoke artifacts predate these export changes.

Benchmark lifetime verification: 103 self-harness, filesystem, optimizer, catalog
and portable-build tests passed on macOS Python 3.12. The 32 experiment lifecycle
and filesystem tests passed on macOS Python 3.10.19 and 3.14.3. New cases cover
provider setup failure, diagnostic-read failure, cancellation, shutdown failure
alone and alongside a primary error, diagnostic reads after close, incomplete
diagnostic tails, and retained scratch through context exit, explicit cleanup and
finalization. The optimization archive and pinned catalog were rebuilt. Strict
checks passed with zero suppressions in `build/filesystem-quality-27`. Existing
native CI selections include these new tests; remote Windows/Linux execution is
still unverified. Earlier full-suite and portable smoke artifacts predate this
lifecycle change.

Workflow retirement verification: 19 workflow, catalog and portable-build tests
passed on macOS Python 3.12, including the real fifty-child workflow, compaction,
recovery and stopped-worker checks. Three cleanup tests passed on macOS Python
3.10.19 and 3.14.3; the final test-only socket-wrapper typing adjustment also
passed on Python 3.12. Failure cases exercise child enumeration before shutdown,
runtime-close failure, cancellation combined with both failures, a child reporting
alive after close, and gateway-thread startup failure combined with socket cleanup
failure. Scratch is retained whenever worker retirement cannot be verified; no
cleanup step is retried. The optimization archive and pinned catalog were rebuilt.
Strict checks passed with zero suppressions in `build/filesystem-quality-30`.
Native CI now includes workflow cleanup tests; native Windows/Linux results and a
refreshed full-suite/portable smoke run remain outstanding.

### Latest platform evidence (2026-09-23)

This entry supersedes earlier statements that Linux and a refreshed full suite
were outstanding; the earlier entries describe their respective snapshots.

- macOS Python 3.12 completed 961 tests in 353.026 seconds, with one skip
  (`/tmp/raychat-all-tests-6.log`). Subsequent test-only corrections patch
  `Path.replace()` consistently across Python versions; their 55 filesystem-plugin
  and memory tests passed in 16.944 seconds, with one skip
  (`/tmp/raychat-path-replace-tests.log`). No production code changed between them.
- Linux Python 3.10.21 and 3.14.7 each completed 159 selected filesystem,
  publication, journal, optimizer, lifecycle, package recovery, plugin and release
  tests, with one skip, in 58.300 and 67.219 seconds respectively. Logs are
  `/tmp/raychat-linux310-tests-2.log` and `/tmp/raychat-linux314-tests-2.log`.
  These ran on Linux 6.12.54-linuxkit aarch64 as UID 65534, with a private writable
  home, network disabled and no host checkout mount. The allowlisted release ZIP
  was streamed into the container's own temporary filesystem. This does not prove
  Windows sharing semantics, host-mounted storage behavior or deployment stress.
- The Linux input was `build/filesystem-linux-2.zip`, SHA-256
  `608343bd9aa463e8053116bcd1a9e49fcca7a0b05953e3fdae684f1af04691bc`.
  Pinned image digests were
  `python@sha256:31dd4d9529d02d7436659061cb7564cd4733fc90e5e152709a942d53382ec8d0`
  (3.10) and
  `python@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2`
  (3.14). This build used `--no-smoke`; it supplies no refreshed TUI smoke result.
- Strict quality checks passed with zero suppressions and an unchanged source
  inventory in `build/filesystem-quality-31/report.json`.
- GitHub Actions [run 35417643307](https://github.com/RayChatLLM/raychat/actions/runs/35417643307)
  completed successfully for commit
  `1a7d2a32d9b418bd489340a6985939a76095cc31` on 2026-09-19. All ten jobs passed,
  including Windows [Python 3.10](https://github.com/RayChatLLM/raychat/actions/runs/35417643307/job/105829254868),
  [Python 3.12](https://github.com/RayChatLLM/raychat/actions/runs/35417643307/job/105829254780)
  and [Python 3.14](https://github.com/RayChatLLM/raychat/actions/runs/35417643307/job/105829254772).
  This is the current committed HEAD, but the filesystem implementation and its
  new CI selection are uncommitted. These successful baseline jobs therefore do
  not satisfy Windows acceptance for the new changes. Record a successful run
  containing those changes before closing that gate. API evidence is retained in
  `build/filesystem-ci-evidence/latest-run.json` and `baseline-jobs.json`.
  Rechecking the public runs and jobs APIs on 2026-09-23 found the same latest
  successful run and all three successful Windows jobs; no newer revision was
  available for validating the uncommitted work.

The first Linux attempts are excluded from acceptance: their container account
had an unwritable `/nonexistent` home, and four older fault-injection tests patched
`os.replace()` rather than the `Path.replace()` entry point. The writable private
home and corrected test seams resolved those failures without changing production
replacement behavior or destination permissions.

### Evaluator and session failure boundaries

The evaluator now lets session-level `OSError` propagate through GEPA unchanged.
Three new regressions distinguish EACCES/ENOSPC/ENOENT aborts from candidate
execution feedback and tool-level permission failures. The composed-session
regressions cover cancellation plus host cleanup failure, release of the actual
journal writer lock despite host failure, two simultaneous cleanup failures, and
a cleanup failure after a successful run without repeating the run.

On macOS Python 3.12, 58 optimizer, GEPA, catalog and portable-build tests passed
(`/tmp/raychat-optimizer-storage-tests.log`), followed
by 29 journal and session tests (`/tmp/raychat-session-cleanup-tests.log`). The
22 protocol-safety and session-cleanup tests also passed on Python 3.10.19 and
3.14.3 (`/tmp/raychat-failure-lifecycle310.log` and
`/tmp/raychat-failure-lifecycle314.log`). Strict checks passed with zero
suppressions in `build/filesystem-quality-33/report.json`; quality run 32 had a
format-only failure corrected before that passing run. The optimization archive
and catalog were rebuilt, and CI already selects both modified test modules.
These changes postdate the earlier 961-test full suite and Windows baseline.

The refreshed `build/filesystem-linux-3.zip` (SHA-256
`bc253bbb875fdfdb03b7a3adb82b90abc69eada27f489ef0911b6b4ef76a70cb`)
passed all 165 selected tests on Linux Python 3.10.21 and 3.14.7, with one skip
each, in 73.559 and 83.231 seconds respectively. Logs are
`/tmp/raychat-linux310-tests-3.log` and `/tmp/raychat-linux314-tests-3.log`.
These runs used the same pinned images, UID 65534, private home, disabled network
and container-owned filesystem described above. They include both sets of new
regressions and the rebuilt optimization package. This archive was built with
`--no-smoke`; its evidence does not include a refreshed TUI acceptance run.

### Construction and plugin generation ownership

Session construction tests now cover failure after capturing a plugin, journal
restoration failure combined with runtime cleanup failure, preservation of the
caller's journal writer lock, a supplied runtime remaining caller-owned, and
manager attachment failure before plugin loading. Cleanup of a newly created
runtime runs once and cannot replace the initialization exception.

Plugin retirement tests cover final shutdown failure, pending-registration and
completed-registration rollback failures, rejected replacement cleanup, and old
generation cleanup after successful publication. Each uncertain generation keeps
its source files and supports a delayed import even after repeated retirement.
Separate cases prove that deliberately transferred resources retain old or
rejected candidate files until final runtime shutdown, and that failed replacement
registration does not close a resource still owned by the active generation.
These regression fixtures have no background consumers when their test-owned
parent directory is finally removed; production retained generations have no
automatic age-based reclamation policy.

The final macOS Python 3.12 run passed 120 journal, plugin lifecycle, source,
hot-reload, session, worker and optimizer tests in 42.252 seconds
(`/tmp/raychat-construction-retirement-tests-3.log`). All 14 construction/cleanup
and plugin-retirement tests also passed on Python 3.10.19 and 3.14.3
(`/tmp/raychat-retirement310-3.log` and `/tmp/raychat-retirement314-3.log`). Strict
quality checks passed with zero suppressions in
`build/filesystem-quality-36/report.json`. The native CI selection now includes
`tests.test_plugins.PluginRetirementTests` alongside `tests.test_session_journal`.
The earlier 961-test full suite, portable TUI smoke and Windows baseline predate
these ownership changes.

The refreshed `build/filesystem-linux-5.zip` (SHA-256
`62794715a1a462f5133936542a96297329efb28a9f9ee1fccef2b617ea7f6ac3`)
passed 176 selected tests on each of Linux Python 3.10.21 and 3.14.7, with one
skip each, in 59.935 and 67.501 seconds. Logs are
`/tmp/raychat-linux310-tests-5.log` and `/tmp/raychat-linux314-tests-5.log`.
Both runs use the previously recorded pinned images, UID 65534, private writable
home, disabled network and container-owned filesystem. The archive used
`--no-smoke`. The earlier `filesystem-linux-4.zip` passed 175 tests per version
but predates the partial-registration handoff fix; archive 5 supersedes it.

### Release cache cleanup ownership

Release cleanup now has protected-path and environment tests, including a
case-colliding preserved name, literal `~` as a project-relative pathname, and a
protected descendant inside a cache-shaped directory. Tests verify that source
links are skipped, nested cache links and linked cache files are refused, scan
errors propagate, deletion gets one attempt without chmod, and initial cleanup
failure leaves existing outputs and the builder untouched. A real small release
proves that both cleanup passes preserve cache-shaped output names and explicitly
preserved user files. A post-verification cleanup failure is separately reported
without building again or deleting the verified outputs.

The Windows-only fixture creates actual junctions in a bounded child interpreter
using CPython's internal
[`_winapi.CreateJunction` fixture helper](https://github.com/python/cpython/blob/v3.10.19/Modules/_winapi.c#L491).
It checks both ordinary traversal and cache-directory refusal. Creation errors
fail the test on Windows; other platforms explicitly skip it. Runtime code uses
the shared `is_link_or_reparse_point` lstat/attribute check, whose independent
test covers reparse attributes on a directory that is not marked as a symlink.
The CI filesystem step now includes all release cleanup tests. Actual Windows
execution remains pending for the uncommitted revision.

On macOS Python 3.12, 50 release, filesystem and portable-build tests passed
(`/tmp/raychat-release-cleanup-tests-3.log`, one Windows-only skip). All 11 final
release tests subsequently passed on Python 3.10.19, 3.12 and 3.14.3, with the same
skip (`/tmp/raychat-release-final310.log`, `-final312.log`, `-final314.log`).
Strict quality checks passed with zero suppressions in
`build/filesystem-quality-39/report.json`; runs 37 and 38 exposed typing issues
that were corrected before that passing result. The final literal-path fixture
extension also passed Ruff and formatting checks.

`build/filesystem-linux-6.zip`, SHA-256
`f3a0c6f1ef94995cb29f2c44d9f702ff7ec5f9a15511525baedba363d22e9434`,
passed 186 selected tests on each of Linux Python 3.10.21 and 3.14.7, in 53.562
and 61.533 seconds. Logs are `/tmp/raychat-linux310-tests-6.log` and
`/tmp/raychat-linux314-tests-6.log`. Each run skipped the Windows Job Objects and
junction cases. The same pinned images, UID 65534, private home, disabled network
and container-owned filesystem were used. This was a `--no-smoke` build;
full-suite and TUI smoke evidence predates these release-cleanup changes.


### Reload capture before plugin execution

Live reload now captures and compiles every candidate source tree before exporting
handoffs or executing candidate entrypoints. Retirement decisions use the captured
replacement manifests. A later compile failure therefore cannot leave import
side effects from an earlier candidate. An earlier entrypoint cannot change the
bytes subsequently executed for another candidate; final fingerprint validation
still rejects a candidate generation if its installed sources changed during
loading. Existing generation cleanup and retention rules continue to apply.

Two regression tests exercise a later compilation failure and an entrypoint that
mutates another candidate's source. Both preserve the active generation. They
pass on macOS Python 3.10.19, 3.12 and 3.14.3 and are selected in the filesystem CI
matrix. The combined hot-plugin, plugin, package-recovery and package-system
suite passed 91 tests in 26.276 seconds
(`/tmp/raychat-source-capture-tests.log`). Strict quality checks passed with no
suppressions in `build/filesystem-quality-40/report.json`.

That capture-only change was followed by explicit live reload coordination:
`PackageManager.attach` supplies the runtime's source-read context. Capture,
watch scans and final fingerprint validation acquire both scopes. Installation
and activation hold both locks in the same order, with explicit thread-local
transaction ownership permitting their source reads to borrow the existing
locks. Acquisition failures release earlier locks; commit and rollback release
transaction ownership. Both scopes remain held until the last participating
scope transaction ends. The preceding Linux bundle and the GitHub Windows
baseline predate these changes.

Actual child-process lock probes cover startup, package checks, reload capture,
entrypoint execution and transaction completion. Rejected updates release both
locks, another thread cannot borrow ownership from an active transaction, and a
busy second scope releases the first acquired scope while preserving the active
runtime. A later reload succeeds after the competing lock is released.

Remaining installed-source readers were located explicitly: manager inventory and
new-package discovery; install dependency collection and replacement planning;
profile source selection; activation input selection; application startup
selection; and plugin argument metadata discovery. A returned pathname is not a
capture or lock reservation, so wrapping `paths()` alone would not protect a
caller's later manifest or source read. Those callers need contexts encompassing
their complete read, or captured metadata returned from inside such a context.

The focused macOS suite passed 94 tests in 36.759 seconds
(`/tmp/raychat-reload-lock-tests-3.log`). Final package-recovery and capture tests
passed on macOS Python 3.10.19 and 3.14.3: 14 tests in 16.475 and 17.399 seconds
(`/tmp/raychat-reload-lock310.log`, `/tmp/raychat-reload-lock314.log`). Strict
quality passed with zero suppressions in
`build/filesystem-quality-43/report.json`. Run 41 failed fixture lint/formatting;
those issues were corrected before runs 42 and 43 passed.

The refreshed archive `build/filesystem-linux-7.zip`, SHA-256
`dd83a2733c8031ddc3512a19f7627359795e79782c7adea916dc1aa3780eb369`,
passed 191 tests on Linux Python 3.10.21 and 3.14.7 in 83.846 and 99.872 seconds.
Both runs used the pinned images, UID 65534, private writable home, disabled
network and container-owned filesystem described above. The two skips were the
Windows Job Objects and actual junction cases. Logs are
`/tmp/raychat-linux310-tests-7.log` and `/tmp/raychat-linux314-tests-7.log`.
The archive build itself used `--no-smoke`; separate terminal smoke execution
must be assessed from its own results.

A separate child-process probe found a remaining worker-startup issue:
`create_runtime(source=...)` skips installed source capture but still eagerly
constructs a package manager and reads its receipts. Holding the user package
lock while starting a child from a complete serialized source snapshot causes
that child to fail with the bounded active-writer error. Evidence is
`/tmp/raychat-snapshot-worker-lock-probe.log` (child exit 1). Thus snapshot workers
can still depend on a lock held by their validating parent. Snapshot-only startup
must defer installed-state access; commands that actually inspect or mutate
installed packages still need the normal coordination protocol. Existing passing
suites do not cover this startup-under-parent-transaction case.

The complete macOS Python 3.12 standard-library suite passed 993 tests in 414.396
seconds, with the two Windows-only skips (`/tmp/raychat-all-tests-7.log`). This
covers release cleanup, generation retention and coordinated reload before the
snapshot-startup fix below. The separately reproduced snapshot-worker startup
dependency was not covered by that passing suite.

Separate smoke verification of that exact archive succeeded on macOS Python
3.12: 263 Python sources compiled, 12 plugin packages verified, and all 14 offline
POSIX terminal scenarios passed (bare launch, interaction, features, startup,
profile upgrade, persistence, package-download cancellation, plugin guide,
reload race, adversarial agents, UI stress, composer, 50-worker collective, and
optimization). Evidence is in `build/filesystem-portable-smoke-7`, with the final
coverage report in `/tmp/raychat-portable-smoke-7.log`. This refreshes the prior
portable TUI evidence for the current Python changes; it does not provide Windows
TUI evidence or cover the separately reproduced snapshot-startup lock dependency.


### Snapshot startup and package planning

The reproduced snapshot-worker dependency is now fixed. Composition requests
lazy receipt loading when complete captured sources are supplied and preserves
the snapshot's selection instead of consulting installed disabled-state receipts.
The package manager remains available to plugins: its first installed-state use
acquires the normal scope locks and performs recovery and receipt reads. Failed
initialization leaves the cache uninitialized, allowing a later deliberate
operation to succeed after contention ends. Ordinary startup still loads state
eagerly, now under both locks in canonical order.

Inventory refreshes receipts and holds both scopes through manifest and source
inspection. Discovery and activation input selection use the same read context.
Installation captures dependency manifests under those locks, then releases the
locks before staging/downloading. SDK validation for a dependency uses its
captured manifest; unused older-SDK packages remain inspectable. Replacement
planning protects inventory inspection and source fingerprints in one context.
Catalog mutation compares its locked receipt with the original input receipt,
so an intervening cache refresh cannot authorize overwriting a stale plan.

A real child starts from captured sources while both package locks are held,
reports bounded contention when it deliberately requests inventory, and reads
that inventory successfully after the parent releases the locks. Additional
child probes verify coordinated inventory/source reads and that downloads occur
after dependency capture with both locks released. A deterministic intervening
installation and cache refresh rejects a stale catalog mutation without losing
the installed package. The new tests are part of the existing native CI selection.
Profile selection, application startup selection and CLI metadata discovery are
the remaining identified installed-source readers requiring coordination review.

Final focused verification passed 119 package, recovery, hot-plugin,
registration, agent-session and journal tests in 37.275 seconds on macOS Python
3.12 (`/tmp/raychat-snapshot-startup-tests-3.log`). Recovery/capture tests passed
on macOS Python 3.10.19 and 3.14.3: 18 tests in 16.325 and 17.436 seconds
(`/tmp/raychat-snapshot-startup310.log`, `/tmp/raychat-snapshot-startup314.log`).
Strict quality passed with zero suppressions in
`build/filesystem-quality-46/report.json`. Earlier runs 44 and 45 exposed a
fixture coroutine annotation and a private-access lint issue, corrected before
that passing run. The 993-test full suite and 14-scenario smoke report above
predate these snapshot-startup and planning changes.

The refreshed archive `build/filesystem-linux-8.zip`, SHA-256
`75d508fe3ad9d8607973b740ae79c559ced9903236af0698ec425656e4766126`,
passed 195 selected tests on Linux Python 3.10.21 and 3.14.7 in 99.173 and 113.553
seconds. Both used the same pinned images, UID 65534, private home, disabled
network and container-owned filesystem. Both skipped only Windows Job Objects
and actual junction creation. Logs are `/tmp/raychat-linux310-tests-8.log` and
`/tmp/raychat-linux314-tests-8.log`.

A separate bounded macOS terminal run against the extracted archive passed the
50-worker collective scenario (parallelism 8) in 125.558 seconds, using deterministic
offline providers. The report is
`build/filesystem-snapshot-collective-8/result.json`; command completion is logged
in `/tmp/raychat-snapshot-collective-8.log`. This specifically refreshes collective
worker/terminal verification after lazy snapshot startup. It does not refresh
all 14 portable scenarios or the complete 993-test suite, which predate this
change, and it does not supply current Windows CI evidence.


### Profile planning and launch metadata

The remaining identified production callers of package `paths()` now keep both
scope locks through their installed-tree reads. Application startup resolves
selection under the read context before composition captures sources under its
own context. CLI discovery returns detached manifests from the locked read;
argument declarations run after the locks are released. Returned paths remain
non-reserving lookups, and that caller-owned coordination contract is documented
on `paths()`. Linked development sources and external editors remain outside the
cooperating package-writer protocol.

Profile planning captures both receipts and checks installed content while
holding the scope locks. Staging and downloads run after release. Before any
package publication, the transaction revalidates both original receipts, even if
an intervening operation refreshed the manager's cache. Metadata-only profile
completion performs the same check. Profile membership is included in the
package transaction's receipt rather than appended in a later write, so commit
and rollback cover it with the package changes.

New tests interleave another installation in either scope, refresh the planning
manager's cache, and verify rejection without overwriting the operator's package.
They also interleave a write before metadata-only completion and interrupt after
a package commit to verify that membership is already recorded. Child-process
lock probes cover CLI/startup reads and confirm release before declaration or
plugin execution. A failed initial watch scan now closes the newly constructed
runtime exactly once; secondary cleanup failure is logged without masking the
original scan error.

Focused integration verification passed 156 distribution, package, recovery,
plugin, reload, session and feature-integration tests in 41.724 seconds on macOS
Python 3.12 (`/tmp/raychat-package-readers-tests-2.log`). Distribution, recovery
and cleanup tests passed on Python 3.10.19 and 3.14.3: 34 tests in 19.031 and
20.908 seconds (`/tmp/raychat-package-readers310.log`,
`/tmp/raychat-package-readers314.log`). Strict quality passed without suppressions
in `build/filesystem-quality-49/report.json`.

A separate inventory probe identified a shared-state issue in
`application._workspace_trust`: simultaneous grants read the same list and then
publish independent snapshots without serializing the complete update. Two
threads synchronized after their reads both reported successful grants, while
the stored list contained only one workspace. Evidence is
`/tmp/raychat-trust-concurrency-probe.log`. This needs a stable lock around the
trust-list read/modify/publish sequence; replacement retries alone cannot prevent
the lost update. The follow-up below supplies that coordination independently of
the package readers.

The rebuilt archive `build/filesystem-linux-9.zip`, SHA-256
`d9503086f80447a01421176ba2087df7830076458be757192d462aadcdcbe55b`,
passed 206 selected tests on Linux Python 3.10.21 and 3.14.7 in 75.966 and 86.331
seconds. The selection now includes distribution/profile tests. Both runs used
the pinned images, UID 65534, private home, disabled network and container-owned
filesystem, skipping only Windows Job Objects and actual junction creation.
Logs are `/tmp/raychat-linux310-tests-9.log` and
`/tmp/raychat-linux314-tests-9.log`.

Separate bounded macOS terminal runs against the extracted archive passed startup
and profile-upgrade scenarios. Reports are in
`build/filesystem-package-launch-tui-9/startup/result.json` and
`build/filesystem-package-launch-tui-9/profile_upgrade/result.json`, with command
completion in `/tmp/raychat-package-launch-tui-9.log`. These specifically refresh
launch/profile behavior; the complete suite, collective scenario and full
14-scenario smoke results above predate this change. Current Windows CI remains
outstanding, using the user-selected GitHub Actions environment.

## Workspace trust coordination follow-up

`raychat/workspace_trust.py` now holds a persistent sidecar across each bounded
read and complete grant/revoke update. The configured application home and trust
filename remain in use. Lock acquisition defaults to 0.5 seconds independently
of the shared publication retry budget. A changed decision publishes a unique
completed sibling stage through `Path.replace()`; neither the decision nor its
read is replayed. Unchanged decisions do not republish. Missing state means no
grants; malformed JSON, invalid entry types, oversized input/output and linked
endpoints fail without replacing the existing state. Each published snapshot uses
the configured private file mode; existing destination attributes are not changed
to overcome a failure. The one-MiB bound applies to UTF-8 bytes.

Startup and CLI discovery use the same reader. Pending CLI grant/revoke controls
discovery immediately but only application launch persists the decision. In
particular, pending revocation now prevents discovery of previously trusted
workspace metadata. Trust identity retains the existing resolved-path string
contract; the wider case-alias review remains open. Older application versions
and external editors do not acquire this lock and must not write the trust file
concurrently. The lock is released before package-scope locks are acquired.

Nine new tests cover grant/revoke preservation, stable sidecars, no-op writes,
invalid data, Unicode, failed publication and stage cleanup, endpoint links and
pending CLI choices. Pipe-ordered real child writers pause after reading state
and before publication; competing reads and grants fail within a separately
bounded acquisition attempt. After release, independent updates preserve all
decisions. Killing the paused writer leaves the previous snapshot unchanged and
releases OS ownership. Unprivileged Windows symlink creation is not required;
that fixture skips there, while the process tests are selected in Windows CI.

On macOS Python 3.12, 69 trust, plugin-integration, package-recovery and distribution
tests passed in 23.489 seconds (`/tmp/raychat-workspace-trust-tests-2.log`). Trust
and plugin-integration selections each passed 43 tests on macOS Python 3.10.19
and 3.14.3 in 9.979 and 11.380 seconds (`/tmp/raychat-workspace-trust310.log` and
`/tmp/raychat-workspace-trust314.log`). Strict quality passed without suppressions
in `build/filesystem-quality-51/report.json`.

Archive `build/filesystem-linux-10.zip`, SHA-256
`b5a466a12b06756a2e049c49d764ca8a8b214d2b0f5de2c5574e33f687904510`,
contains the implementation and tests. Its focused trust/plugin-integration
selection passed 43 tests each on Linux Python 3.10.21 and 3.14.7 in 10.213 and
12.165 seconds, with no skips. The pinned containers used UID 65534, a private
home, disabled networking and container-owned storage. Logs are
`/tmp/raychat-linux310-trust-10.log` and `/tmp/raychat-linux314-trust-10.log`.
The broader Linux selection and portable TUI evidence above predate this change.
GitHub's runs and jobs APIs were checked again on 2026-09-23: run 35417643307
remains the latest successful Windows evidence for baseline `1a7d2a3`; current
uncommitted changes have no native Windows result yet. No separate Windows test
environment is requested.


## Candidate promotion ownership and conditional rollback

`plugins/self_harness/promotion.py` now owns live candidate publication. Its queued
prepare hook acquires the two package scopes in their existing canonical order,
then the workspace filesystem sidecar. Ownership lasts through live validation
and commit or rollback. Each package acquisition retains its one-second budget;
the workspace acquisition has a separate half-second budget. The package
manager's new explicit `source_update()` context lends its locks to validation on
that same thread. `raychat/workspace_files.py` similarly lends workspace access
only from an explicit update; a second explicit update is rejected, and other
threads/processes still contend. No new filesystem retry loop was added.

The complete original mapping is checked while locked. Aliased paths, existing
hard-link aliases, case-colliding names and the coordination lock are rejected.
Every incoming file is written, synced and closed inside a securely reserved
sibling container before any public target changes. Publication retains
`Path.replace()` and the common bounded policy. The original mode is retained
for existing files; new files use the configured workspace mode. No attribute
change is attempted on a destination to overcome publication failure.

The incoming inode is recorded before publication, so an exception immediately
after a successful replacement still permits conditional rollback. Rollback
checks all recorded targets against both their bytes and filesystem identities
before restoring any of them. A conflicting external edit, including a recreated
file with identical bytes, is preserved and reported through the runtime's
existing chained rollback error. That conflicting set remains on disk for
operator correction; the old runtime generation remains active. Newly created
files are moved into owned retirement slots before cleanup rather than unlinked
at their public names. Cleanup verifies each container's identity, refuses a
reused pathname, reports failures separately, and releases locks on every outcome.

Filesystem tool operations use the same workspace context. Existing self-harness
overlays are read and closed under it, while an absent overlay is an empty
snapshot without creating workspace metadata. Stage cleanup completes before
acceptance logging and notifications. An acceptance-log failure cannot trigger
a second publication through these hooks. External editors and raw atomic-write
service consumers do not participate unless they acquire the shared ownership.

At this stage the owner was **in-memory**, with no undo/commit journal or restart
recovery. The following journal follow-up replaces that limitation. Neither
version promises an atomic view for uncoordinated readers. Earlier full-suite
and portable TUI results predate these changes.

Before the final cleanup-identity guard, 108 self-harness, filesystem, package
recovery and hot-reload tests passed on macOS Python 3.12 in 46.824 seconds
(`/tmp/raychat-candidate-tests-3.log`, one Windows Job Objects skip). The 70-test
self-harness/filesystem selection passed from archive 11 on Linux Python 3.10.21
and 3.14.7 in 33.189 and 36.649 seconds, each with the same skip
(`/tmp/raychat-linux310-candidate-11.log`,
`/tmp/raychat-linux314-candidate-11.log`). Eighteen promotion and existing
self-harness tests also passed on macOS Python 3.10.19 and 3.14.3 in 9.030 and
10.210 seconds (`/tmp/raychat-candidate310.log`,
`/tmp/raychat-candidate314.log`).

The final six promotion tests cover real child lock contention, staged failure,
interruption after publication, conflicting content/identity, portable case
aliases and cleanup of a reused container name. They passed on macOS Python 3.12
in 2.946 seconds (`/tmp/raychat-candidate-tests-5.log`); Python 3.10.19/3.14.3
passed the same six tests in 4.655/5.805 seconds before the final formatting-only
pass (`/tmp/raychat-candidate-final310.log`,
`/tmp/raychat-candidate-final314.log`). Strict quality passed without suppressions
in `build/filesystem-quality-56/report.json`.

The final archive is `build/filesystem-linux-13.zip`, SHA-256
`768e349695f51fc3e1cd504c429c712df52809f3a6ef4bd0ee83028d7dc3a213`.
Its six promotion tests passed on Linux Python 3.10.21 and 3.14.7 in 7.218 and
8.785 seconds with no skips, using the pinned images, UID 65534, private home,
disabled networking and container-owned storage. Logs are
`/tmp/raychat-linux310-candidate-13.log` and
`/tmp/raychat-linux314-candidate-13.log`. The plugin catalog and release source
allowlist were rebuilt; CI now selects the new promotion class and the existing
self-harness regression class. Windows evidence still requires GitHub CI on a
revision containing these changes.


## Candidate transaction journal and restart recovery

`raychat/workspace_transactions.py` now records the complete candidate in
`.raychat/candidate.transaction.json` before the first public replacement. The
record is bounded to 4 MiB and 4,096 changes; each file snapshot is bounded to
64 MiB. All incoming bytes and undo copies are synced, closed and retained in
secure sibling containers. Records bind the workspace path and identity,
confined target/container paths, file hashes, byte counts, identities and mode
bits. The workspace and its metadata parents must be trusted and cooperating;
these records are not an authorization mechanism for hostile writers. Invalid
records, redirected paths, aliases, unexpected container content and replaced
containers are rejected rather than used to overwrite or delete uncertain data.

A pending record restores the original set. Existing files are restored from
completed undo copies through the shared `Path.replace()` policy; newly created
files move into owned retirement slots. Their recorded identities let recovery
recognize a completed restoration if the recovery process is itself killed.
All target states are checked before any restore. A conflict retains the journal
and undo files for explicit operator correction, releases all locks and preserves
the original failure through the runtime's chained error. No published operation
is replayed, and no destination attributes are cleared automatically.

Acceptance publishes a committed decision before cleanup. A rolled-back decision
is likewise published before removing its resources. Recovery of either completed
decision only cleans the exact recorded private resources; it never examines or
reverts subsequent public edits. An interruption after commit publication but
before the in-memory status changes still honors the disk decision. Cleanup
contention is reported separately, retaining the record. Later ordinary access
attempts completed cleanup once; starting another candidate requires that cleanup
to finish. Pending restoration keeps the shared helper's normal bounded budgets.

`workspace_access()` recovers recorded work before new outer access, with its
existing half-second lock budget. Package startup holds both package scopes,
performs that recovery preflight, then captures sources. The scopes exclude a new
candidate until capture finishes; the workspace lock need not span plugin
execution. Filesystem read, list and write operations, overlay loading and
evaluation snapshots also enter the protocol. Preflight leaves a pristine
workspace untouched when no lock or record exists. Explicit update ownership
lends access on its own thread without recovering its active pending record.

Candidate directories use the reserved `.raychat-candidate-` prefix. Package
source enumeration and evaluator copies exclude them, and evaluator copies omit
the workspace lock and transaction record. This prevents private undo bytes from
entering captured plugins or changing a generation's fingerprint when cleanup
finishes. The raw atomic-write service, arbitrary external editors and older
versions remain outside this protocol. Allocations interrupted before journal
publication cannot yet be identified on restart and are retained; no glob/age
cleanup claims ownership of them. This is process-crash recovery on cooperating
local storage, with no universal power-loss durability claim.

Native child writers were killed after the pending record, each public file,
the commit decision, new-file retirement, old-file restoration, the rolled-back
decision and partial cleanup. A recovery child was separately killed during
rollback, then a fresh access finished recovery. Tests also cover a journal
publication failure, an exception immediately after commit publication, retained
external conflicts, later edits after committed cleanup denial, malformed or
redirected records, preserved mode bits and no reclamation of unrelated scratch.
Startup recovery repairs an invalid Python candidate before source capture.
Direct filesystem operations verify restoration before reads, listing or writes;
the service rejects writes to its reserved transaction record.

Before the final source-enumeration exclusion, 152 focused tests passed on macOS
Python 3.12 in 47.103 seconds (`/tmp/raychat-workspace-transactions-2.log`, one
Windows Job Objects skip). A 99-test selection covering transactions, self-harness,
filesystem and package recovery passed from archive 14 on Linux Python 3.10.21
and 3.14.7 in 60.839 and 69.978 seconds, each with the same skip
(`/tmp/raychat-linux310-transactions-14.log`,
`/tmp/raychat-linux314-transactions-14.log`).

After the source-enumeration fix, 64 transaction, package-system, promotion and
source-capture tests passed on macOS Python 3.12 in 15.359 seconds
(`/tmp/raychat-workspace-transactions-4.log`). The final 18 transaction, promotion
and direct-filesystem recovery tests passed on macOS Python 3.10.19 and 3.14.3
in 5.453 and 7.082 seconds (`/tmp/raychat-workspace-transactions-final310.log`,
`/tmp/raychat-workspace-transactions-final314.log`). Strict quality passed without
suppressions in `build/filesystem-quality-60/report.json`.

Archive `build/filesystem-linux-15.zip`, SHA-256
`3af285cd7d330e0b7d0aef9c7a1d7d05a29bffb7310069b66260e791639d679a`,
passed those final 18 tests on Linux Python 3.10.21 and 3.14.7 in 14.004 and
16.979 seconds with no skips. The pinned containers used UID 65534, private home,
disabled networking and container-owned filesystems. Logs are
`/tmp/raychat-linux310-transactions-15.log` and
`/tmp/raychat-linux314-transactions-15.log`. Earlier full-suite and portable TUI
results predate this journal work. GitHub APIs checked on 2026-09-24 still show
run 35417643307 and three successful Windows jobs for baseline `1a7d2a3`; the
uncommitted changes have no matching Windows run.


## Portable-folder journal and rollback retirement

The previous release-folder owner held undo state only in memory and recursively
removed the public tree during rollback. `tools/release_folder.py` now acquires a
stable sibling lock through `access()`, stages all members in a fresh private
`.raychat-release-*` container, verifies bytes, and publishes a bounded inventory
record before moving the original. The builder and release command use this
owner; `--check` and final release verification hold the same access context.
Existing `Path.replace()` retry policy remains centralized in `raychat/filesystem.py`.
There are no new retry loops or automatic attribute changes.

The journal binds the canonical target and parent identity to the exact container
identity and inventories of both trees. Entries retain identity, mode and content
hash; directories are included. Linked/reparse members and special files are
rejected. A pending transaction recognizes staging, backup, publication, retirement
and restoration states. Rollback moves the new public tree into `retired`, then
restores `original`; it never recursively deletes the public pathname. State
conflicts preserve every remaining tree and the record. A committed or rolled-back
decision authorizes only private cleanup, including after later public edits.
Partial deletion is resumable because the remaining inventory must be a matching
subset of the recorded tree. Reused container identities and extra/changed private
members are retained and diagnosed. A failure after commit-record replacement
cannot authorize rollback, even if the caller never received publication success.

The parent must be trusted and consumers must cooperate or stop. The directory
swap has an absent-name interval and is not an atomic multi-file view for arbitrary
readers. Archive and directory outputs publish independently. No universal
power-loss, network-storage or security-software deployment guarantee is made.
The output sidecar remains in place. Lock acquisition is independently bounded to
0.5 seconds; individual replacement/cleanup operations retain the shared policy.
Completed-decision recovery makes one cleanup attempt, allows readers to continue,
and blocks another publication while its unresolved record remains. Unknown
pre-journal allocations are retained. Cache cleanup now prunes reserved release
containers so cache-shaped backup members cannot invalidate recovery.

Validation for this follow-up: nine recovery tests include real child termination
at nine publication/decision/cleanup boundaries (six when the original folder is
absent), interrupted recovery, native sidecar contention, conflicting public
edits, malformed/redirected records, reused containers, staging/journal faults,
and an exception immediately after commit publication. Existing build rollback
and release-cleanup tests remain in the selection. Thirty build/recovery/release
tests pass on macOS Python 3.12 in 6.076 seconds; the 26-test focused selection
passes on macOS 3.10.19 in 5.506 seconds and 3.14.3 in 6.670 seconds. Each skips
only native Windows junction creation. Logs are
`/tmp/raychat-release-folder-2.log`, `/tmp/raychat-release-folder-mac310.log` and
`/tmp/raychat-release-folder-mac314.log`. Quality run 61 passes strict typing,
lint and formatting with zero suppression directives:
`build/filesystem-quality-61/report.json`.

Portable archive `build/filesystem-linux-16.zip` has SHA-256
`49afda4150272f9a6e94e8509a98664ebe099d251289bab91079e84f2f12b207`,
5,174,738 bytes, 336 members and 335 allowlisted source files. The same 26 tests
pass from that extracted archive under Linux Python 3.10.21 in 11.865 seconds and
3.14.7 in 14.919 seconds, with UID 65534, networking disabled, and no checkout
mount. Each skips only the Windows junction test. Logs are
`/tmp/raychat-linux310-release-16.log` and `/tmp/raychat-linux314-release-16.log`.
The full-suite and all-scenario TUI results above predate this change.

GitHub CI remains the selected native Windows environment. The API recheck on
2026-09-24 confirms all three Windows jobs (Python 3.10, 3.12 and 3.14) succeeded
in [run 35417643307](https://github.com/RayChatLLM/raychat/actions/runs/35417643307)
for baseline `1a7d2a32d9b418bd489340a6985939a76095cc31`. The new release recovery
and portable-build tests are included in the workflow's filesystem selection,
but no run covers these uncommitted changes yet. Baseline evidence is retained
in `build/filesystem-ci-evidence/latest-run.json` and `baseline-jobs.json`.


## Async snapshot staging and cancellation ownership

`write_bytes_async()` previously performed stage creation, writes, flush, fsync
and close on the supervisor event loop. These operations now run once in the
loop's executor. Only the completed, closed stage returns to the event loop for
the existing narrowly classified asynchronous replacement retry. This leaves
control events responsive during a slow stage or fsync without changing the
publication policy. No await follows successful replacement.

The pending executor future is shielded. Cancellation joins the same writer
cooperatively, including repeated cancellation requests; the caller retains its
serialization and parent lifetime until the worker closes its handles. The stage
is then cleaned without publication. A staging error after cancellation is chained
to the original cancellation. A staging error without cancellation propagates
normally after owned cleanup. If later cleanup is itself cancelled or denied,
its exact leftover stage is logged separately. No generic workflow is retried.
The deadline still cannot interrupt an operating-system call, so cancellation
completion may wait for a blocked filesystem call even though the event loop
remains responsive.

Five new shared-helper tests hold fsync behind a thread gate and prove the loop
continues, the old snapshot remains visible, repeated cancellation does not return
while the descriptor is open, worker errors preserve the expected failure, and a
scheduled cancellation after replacement cannot change publication success. The
existing supervisor tests continue to verify serialized frozen documents, native
error classification, failed fsync and cancellation during replacement contention.
The ordinary filesystem tools execute in the existing non-daemon conversation
worker (`raychat/workers.py`); their synchronous helper sleeps do not run on the
terminal input thread. The following bootstrap follow-up moves candidate capture, validation cleanup
and seal verification through a joined worker. Wider filesystem inventory and
small synchronous control/log operations remain under review.

Validation: 56 shared-filesystem, supervisor-persistence, recovery and startup
tests pass on macOS Python 3.12 in 3.958 seconds
(`/tmp/raychat-async-stage-2.log`). The 24-test async/persistence/recovery selection
passes on macOS Python 3.10.19 in 2.728 seconds and 3.14.3 in 2.706 seconds
(`/tmp/raychat-async-stage-mac310-2.log`, `/tmp/raychat-async-stage-mac314.log`).
The initial Python 3.10 assertion assumed a direct exception cause; it now checks
through asyncio's additional cancellation context to verify the retained worker
failure. Quality run 63 passes strict typing, lint and formatting with zero
suppressions (`build/filesystem-quality-63/report.json`).

Archive 18's Linux run failed to import `tests.test_supervisor_persistence` because
that existing module was omitted from the release allowlist. The other 47 tests
passed, but that run is not acceptance. The allowlist now includes the missing
module; comparison against all 272 nonignored Python source paths finds no other
omission. Archive 19 is the final artifact for this change:
`build/filesystem-linux-19.zip`, SHA-256
`0c6eb82dcdff7f02a8cb009e7d4fc0199e6970ccff226c2741bea91571b89cd2`,
5,200,431 bytes, 337 members and 336 allowlisted source files.

All 56 tests pass from archive 19 under Linux Python 3.10.21 in 7.086 seconds and
3.14.7 in 8.734 seconds, with no skips. These runs use UID 65534, private container
storage, networking disabled and no mounted checkout. Logs are
`/tmp/raychat-linux310-async-19.log` and `/tmp/raychat-linux314-async-19.log`.
The existing Windows CI filesystem/persistence steps already select these tests.
There is still no GitHub run for the uncommitted work, and the earlier full-suite
and all-scenario TUI results predate this change. The full checklist remains open.


## Bootstrap bulk I/O and joined cancellation

`run_filesystem_task()` in the shared filesystem module runs one caller-owned
operation in an executor and shields its future. Cancellation joins that same
operation before propagating; repeated cancellation cannot release a parent or
stream while a worker still uses it. It has no operation retry or automatic
rollback. A mutating operation can finish before cancellation reaches its caller,
so its outputs need an explicit owner. The async snapshot helper reuses the same
join primitive while retaining its separate cleanup-before-publication policy.

The bootstrap now uses this helper for candidate copying, overlay writing,
release integrity checks, plugin-difference hashing, retained checkpoint loading,
packaging-inventory updates, failure-diagnostic copying, and final validation
cleanup/sealing. Candidate and sealed trees remain owned by the supervisor's
retained release directory; cancellation does not activate or globally delete
them. Verification and plugin hashing finish before a child is spawned, avoiding
an unowned process during cancellation. Diagnostic streams remain open until
their worker finishes. Checker failure stays primary if copying its diagnostics
also fails.

Candidate capture reserves a secure `candidate-*` directory through the shared
allocator. A synchronous capture failure attempts bounded cleanup of only that
new container and preserves the original error. Earlier captures and the fixed
evaluator remain untouched. At validation completion, each direct validator process
is reaped before strict bounded cleanup removes its private build/cache output.
Descendants created by a validator remain subject to that tool's lifetime contract;
this change does not introduce a process-tree supervisor.
Cleanup failure prevents sealing/accepting the candidate; it cannot silently add
leftover validation output to an accepted release digest. No permission repair
is performed by cleanup. Sealing remains a deliberate attribute restriction on
an exclusively owned private candidate.

These are retained local release snapshots in a trusted supervisor directory,
not an atomic snapshot of an arbitrary editor's mutable source tree. Unknown
captures left by process termination are not reclaimed by glob, age or PID.
The worker join cannot interrupt a blocked OS call; the event loop stays usable
while cancellation completion waits for file ownership to end.


Validation for the bootstrap worker follow-up: six new integration tests cover
blocked capture and overlay writers, pre-spawn integrity checks, post-validator
cleanup/sealing, cleanup rejection before seal, failed-capture retirement and
secondary diagnostic failure. The 44-test macOS Python 3.12 run covered bootstrap
ownership, shared async publication, persistence, recovery and complete live-core
handoff tests in 27.944 seconds; it preceded only the added diagnostic-failure
case. All six final bootstrap tests pass separately in 0.060 seconds. Logs are
`/tmp/raychat-bootstrap-fs-1.log` and `/tmp/raychat-bootstrap-fs-2.log`. The final
30-test ownership/async/persistence/recovery selection passes on macOS Python
3.10.19 in 2.883 seconds and 3.14.3 in 2.874 seconds
(`/tmp/raychat-bootstrap-fs-mac310.log`, `/tmp/raychat-bootstrap-fs-mac314.log`).
Quality run 65 passes strict typing, lint and formatting with no suppressions
(`build/filesystem-quality-65/report.json`).

Archive `build/filesystem-linux-20.zip` has SHA-256
`24c7f5ec8e609ae4bb86232cbea60fe6a3bc1ed85142a865a9467dd38133516b`,
5,216,960 bytes, 338 members and 337 allowlisted source files. The final 36-test
selection, adding the release/activation regression class, passes from that
archive on Linux Python 3.10.21 in 4.646 seconds and 3.14.7 in 5.376 seconds, with
no skips. Containers retain UID 65534, private storage, no network and no mounted
checkout. Logs are `/tmp/raychat-linux310-bootstrap-20.log` and
`/tmp/raychat-linux314-bootstrap-20.log`.

The same extracted artifact passed the offline startup terminal driver, including
five plugin failure/repair phases and terminal restoration:
`build/filesystem-bootstrap-tui-20/startup/result.json`. This is startup-driver
evidence, not a new full-suite or fourteen-scenario smoke result. GitHub CI now
selects `tests.test_bootstrap_filesystem` beside the supervisor tests. API results
rechecked on 2026-09-25 still show successful Windows 3.10/3.12/3.14 jobs only for
baseline `1a7d2a3` in run 35417643307; no run covers the uncommitted audit changes.


## Bootstrap source traversal and sealing ownership

Bootstrap capture, release hashing and packaging inventory now inspect entries
with `lstat` and the shared symlink/reparse classifier before traversal. Linked
members (including Windows junctions) and special files are rejected. Optional
absent top-level source components may be omitted, but disappearance of a
previously discovered descendant fails capture. File bytes are read through
`read_regular(..., follow_symlinks=False)`, bounded by the observed size plus one;
identity, size, modification time, mode and link count are revalidated. Directory
metadata is also rechecked after capture/hashing to detect changed inventories.
Selected source roots resolve once; their parents must remain trusted. These
checks are not a hostile-directory substitution defense or an atomic snapshot of
an uncooperative editor's multi-file update.

Copying a hard-linked input is permitted because each private output is created
as an independent file. Hashing/sealing an already materialized release rejects
multiply linked files: changing their attributes could otherwise change another
owner's inode. Sealing first validates the entire inventory and hashes its bytes,
then rechecks each entry immediately before restricting its mode. It never
chmods a linked target merely because directory traversal encountered its name.
Candidate configuration inventory updates require exclusive file ownership.
Source-copy failures retain the primary error and retire only the new capture.

The reserved `.raychat-candidate-*` workspace undo namespace is excluded from
source capture, alongside the existing cache exclusions. This matches package
source capture: transient undo bytes must not become part of an immutable core
release. Ignored scratch is left untouched; it is not garbage-collected. The
native Windows junction fixture is in `tests.test_bootstrap_filesystem`, already
selected by GitHub CI. Symlink fixtures may skip if ordinary creation is unavailable;
junction creation does not require that privilege.

Validation: 46 bootstrap path/ownership, live-core, recovery and persistence tests
pass on macOS Python 3.12 in 16.938 seconds. The 37-test selection using the release
regression class passes on macOS Python 3.10.19 in 4.236 seconds and 3.14.3 in
4.453 seconds. Each skips only the native Windows junction case. Logs are
`/tmp/raychat-bootstrap-paths-2.log`, `/tmp/raychat-bootstrap-paths-mac310.log` and
`/tmp/raychat-bootstrap-paths-mac314.log`. Strict typing, lint and formatting pass
with no suppression directives in `build/filesystem-quality-66/report.json`.

Archive `build/filesystem-linux-21.zip` has SHA-256
`60dc30d11352f838bffcc63591bc3808316032f64f80fa4626917579e13ab3b8`,
5,232,749 bytes, 338 members and 337 allowlisted source files. The same 37 tests
pass from this archive under Linux Python 3.10.21 in 4.766 seconds and 3.14.7 in
5.322 seconds, skipping only the Windows junction case. Containers use UID 65534,
private storage, no network and no mounted checkout. Logs are
`/tmp/raychat-linux310-paths-21.log` and `/tmp/raychat-linux314-paths-21.log`.
The extracted application also passed all five offline startup failure/repair
phases and restored its terminal (`build/filesystem-bootstrap-tui-21/startup/result.json`).
These checks do not replace a final complete suite/all-scenario smoke run or
native Windows results for the uncommitted changes.

### Portable artifact names and core proposal preflight

`filesystem.portable_relative_path` now owns the artifact naming policy used by
package names and proposed core files. It rejects noncanonical relative paths,
Windows device names (including superscript COM/LPT digits), forbidden/control
characters, trailing dots/spaces and components exceeding 255 UTF-8 bytes.
Names are rejected without sanitizing or truncating. The component budget is an
application portability policy, not a universal filesystem or total-path limit.
The Windows naming restrictions follow
[Microsoft's naming documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file);
the application also conservatively reserves CONIN$ and CONOUT$.

`PortablePathIndex` validates implied parent directories as well as explicit
entries. NFC normalization plus case folding is a conservative lexical collision
key, not proof of operating-system identity. Different spellings and file/directory
conflicts are rejected. Exact existing-file replacement requires an explicit
caller option. ZIP validation completes before extraction creates its destination;
package source inventories apply the same naming and collision policy.

Core capture validates the complete proposed batch before allocating scratch or
copying source files. It then checks proposals against the copied tree before
writing any proposed bytes. Exactly spelled existing files may be replaced in
that private candidate. Failures retire only the owned candidate; the source
tree remains unchanged. These checks do not establish hostile-parent protection
or support for every native alias, such as Windows short names.

Validation: 104 filesystem, package, bootstrap and live-core release tests pass
on macOS Python 3.12 (9.475 seconds), 3.10.19 (10.666 seconds) and 3.14.3
(11.413 seconds), with only the native Windows junction fixture skipped.
Strict typing, lint and formatting pass without suppression directives in
`build/filesystem-quality-68/report.json`. Logs are
`/tmp/raychat-portable-paths-22.log` and
`/tmp/raychat-portable-paths-mac{310,314}-22.log`.

Archive `build/filesystem-linux-22.zip` has SHA-256
`c3d594476f782798bfc3b26d4e9726a42d49cd5c0d3e7c0b649bd199b554f3e9`,
5,242,600 bytes, 338 members and 337 allowlisted source files. The same 104 tests
pass from that archive on Linux Python 3.10.21 (11.384 seconds) and 3.14.7
(12.767 seconds), with only the Windows junction fixture skipped. Containers use
UID 65534, private storage, no network and no mounted checkout. Logs are
`/tmp/raychat-portable-paths-linux{310,314}-22.log`. The extracted application
passed all five offline startup failure/repair phases and restored its terminal
(`build/filesystem-bootstrap-tui-22/startup/result.json`).

GitHub CI remains the native Windows evidence source, as requested. Rechecked on
2026-09-25, the latest portable run is
[35417643307](https://github.com/RayChatLLM/raychat/actions/runs/35417643307),
successful for Windows Python 3.10, 3.12 and 3.14 at commit
`1a7d2a32d9b418bd489340a6985939a76095cc31`. This is baseline evidence, not a run
of the uncommitted audit changes. The portable matrix now includes package
transaction tests alongside shared filesystem and bootstrap tests, so future
Windows runs will exercise the new naming checks. Final changed-revision Windows
CI, complete-suite and all-scenario smoke evidence remain outstanding.

### Package source traversal and complete local regression (2026-09-25)

Package capture now uses an explicit bounded inventory instead of recursive glob
traversal. Each discovered member is inspected without following links before
any descent. Symlinks, Windows reparse points and special files fail capture.
The operator-selected root may resolve through a link; this does not authorize
linked descendants. The inventory allows at most 8,192 entries including its
root, in addition to the existing file-count and total-byte limits. Ignored cache
and candidate-undo entries are excluded before descent and remain untouched.

File and standalone manifest reads use `read_regular(..., follow_symlinks=False)`;
the shared reader now rejects a regular-mode entry carrying the Windows reparse
attribute before opening it. Reads close before parsing and compare observed
identity, size, modification time, mode and link count. Hard-linked regular input
is allowed because capture only reads it and returns independent bytes; it never
changes input attributes.

A second bounded inventory checks membership and metadata after all reads.
Linux acceptance exposed why directory metadata alone is insufficient: a newly
added member was initially missed when the directory's reported metadata stayed
unchanged. The regression now also restores the directory's prior modification
time explicitly. Capture rejects late additions, missing files, changed files
and changes to already-read entries. These checks do not create an atomic
snapshot against nonparticipating editors or defend hostile parent substitution;
source roots must remain trusted and quiescent during capture.

The final 111-test selection passes on macOS Python 3.10.19 (28.941 seconds) and
3.14.3 (30.974 seconds), and from the portable archive on Linux Python 3.10.21
(40.304 seconds) and 3.14.7 (48.226 seconds). Each skips only the native Windows
junction fixture. Linux uses UID 65534, private storage, no network and no mounted
checkout. Logs are `/tmp/raychat-package-sources-{mac,linux}{310,314}-24.log`.
`tests.test_package_sources` is included in both the release allowlist and the
GitHub portable matrix.

The complete macOS Python 3.12 suite passes: **1,069 tests in 287.538 seconds**,
with four Windows-only skips (three junction fixtures and Windows Job Objects).
The log is `/tmp/raychat-all-tests-24.log`. Strict typing, lint and formatting pass
without suppression directives in `build/filesystem-quality-72/report.json`.
Earlier archive 23 exposed the Linux inventory regression; its complete-suite and
smoke processes were deliberately interrupted before restarting with the fix.
Those interrupted runs are not acceptance evidence.

Archive `build/filesystem-linux-24.zip` has SHA-256
`119b4b8a9b2f696754d5b8a007cbdc43253fc9c84d284d1bd787bd32c5603e92`,
5,258,679 bytes, 339 members and 338 allowlisted source files. All 14 offline POSIX
portable scenarios pass from it, including startup/upgrade recovery, stalled
package-download cancellation, persistence, plugin lifecycle, adversarial input,
50-child collective execution and optimization. Evidence is
`build/filesystem-portable-smoke-24/smoke.json`; its drivers use deterministic
offline providers. This does not claim Windows TUI or deployment-software coverage.

The user requested a branch and PR, with CI verified on the proposed revision.
Work is on `codex/filesystem-publication-audit`; no commit is made directly to
`main`. Changed-revision Windows results remain required, using GitHub CI.

### Local package archive input (2026-09-25)

`plugin_manager.read_bytes` now uses the shared bounded regular-file reader,
closing its descriptor before ZIP validation. Operator-selected file links remain
supported, unlike linked members within a package tree. Nonregular input becomes
a `PluginError` with the original validation failure chained. Native read errors,
including missing files and permission denial, retain their original exception;
there are no read retries, permission changes or weaker publication paths.
POSIX FIFO input is rejected without waiting for a producer. This does not promise
to interrupt a blocking native OS call on every platform.

A bounded real-child test covers FIFOs as package members, manifests and archive
inputs. Other cases cover the exact byte limit, overflow, descriptor closure,
missing input, selected regular-file links and unchanged permission exceptions.
The package/source/recovery selection passes 73 tests on macOS Python 3.12 in
33.805 seconds (`/tmp/raychat-archive-input-25.log`). The 31-test source/location/
transaction selection passes on macOS Python 3.10.19 in 6.713 seconds and 3.14.3
in 6.797 seconds, and Linux Python 3.10.21 in 4.957 seconds and 3.14.7 in
5.124 seconds. Each skips only the Windows junction fixture. Logs are
`/tmp/raychat-archive-input-{mac,linux}{310,314}-25.log`; the Windows matrix already
selects these modules. Strict typing, lint and formatting pass in
`build/filesystem-quality-74/report.json` without suppression directives.

Linux tests use the isolated unprivileged archive harness and
`build/filesystem-linux-25.zip`, SHA-256
`4dbff1c8dbc445a7bbd2c3455385bd1dbe857a714d207cd4efec41a416738692`,
5,265,104 bytes, 339 members and 338 allowlisted source files. The complete suite
and 14-scenario smoke result above predate this narrowly scoped archive-reader
change; they have not been relabeled as a run of the follow-up revision.

### Release inventory membership revalidation (2026-09-25)

Core source copying and release hashing now enumerate the tree again before
accepting the copy or digest. Revalidation compares exact relative POSIX spellings
and entry versions, including the complete membership set. Checking only the
entries discovered at the start could miss a late addition when directory
metadata stayed unchanged. Copy revalidation applies the same cache/scratch
exclusions as its initial inventory; digest/seal revalidation covers every entry.
The existing digest encoding is unchanged. A detected addition prevents sealing
before any chmod occurs; failed capture retires only its new candidate and leaves
the externally added source bytes untouched.

Package revalidation also compares exact relative string spellings instead of
native Path equality. This prevents Windows case-insensitive path comparison
from hiding a case-only rename. The regression renames an already-read entry and
restores its parent's modification time. These observations still require trusted
parents and quiescent inputs; they do not establish a transaction with external
editors or eliminate the interval after the final observation.

Forty bootstrap, package-source, live-release and persistence tests pass on macOS
Python 3.12 in 4.291 seconds (`/tmp/raychat-inventory-recheck-26.log`). Adding the
two portable ordering regressions produces a 42-test selection, passing on macOS
Python 3.10.19 in 10.980 seconds and 3.14.3 in 12.112 seconds, and Linux Python
3.10.21 in 12.359 seconds and 3.14.7 in 15.522 seconds. Each skips the two Windows
junction fixtures. Logs are `/tmp/raychat-inventory-recheck-{mac,linux}{310,314}-26.log`.
Strict typing, lint and formatting pass without suppressions in
`build/filesystem-quality-75/report.json`.

Linux runs use the unprivileged isolated archive harness with
`build/filesystem-linux-26.zip`, SHA-256
`cbb36b7ecec468df62658bb9fdce6773819a955a6ea3e8043f27a9f1178cc7c9`,
5,269,402 bytes, 339 members and 338 allowlisted source files. That archive passes
all five offline startup failure/repair phases and restores its terminal
(`build/filesystem-bootstrap-tui-26/startup/result.json`). These focused results
supplement the earlier full-suite and full-smoke results; native changed-revision
Windows execution still requires the requested PR and GitHub CI authentication.

### Shipped-launcher long paths and configured storage (2026-09-25)

`PortableBuildTests.test_shipped_launcher_uses_long_configured_data_paths` extracts
the actual portable archive below a Unicode path longer than 320 characters and
runs its `raychat.py` with an isolated interpreter. Application files are made
read-only in the owned fixture. An offline SDK command exercises completed
snapshot replacement and creates sessions through the extracted storage API.
A second launcher invocation resumes the saved custom-directory session using
`--session-dir` and `--resume`. The test verifies the configured data home contains
trust state and default sessions, the custom session directory retains its own
journal, and every application member remains byte-for-byte unchanged.

The fixture deliberately grants trust only to its private workspace. It does not
change the real user's home or use provider credentials. Noninteractive `--exec`
is intentionally in-memory unless resuming, so the fixture explicitly seeds its
saved sessions rather than assuming automatic persistence. POSIX permission bits
exercise a read-only installation; Windows attribute behavior is not presented
as an ACL-denial test. Long paths are used without adding extended-path prefixes
or changing operating-system settings. A Windows failure must be diagnosed from
the actual CI interpreter/environment rather than skipped as a successful test.

All six portable-builder tests pass on macOS Python 3.12 in 3.623 seconds
(`/tmp/raychat-long-launch-27c.log`). Strict typing, lint and formatting pass without
suppressions in `build/filesystem-quality-77/report.json`. The portable CI matrix
already selects this test class, including Windows Python 3.10, 3.12 and 3.14.
The README now documents the existing absolute `storage.home_directory` setting,
the separate session override and the independently selected workspace.

GitHub authentication is now available. The requested PR and changed-revision
CI run are the next validation gate; baseline CI is not substituted for them.

### First PR Windows CI findings (2026-09-25)

PR #1 run 36156050986 exposed three portability defects in revision 42060ca.
Child readiness fixtures emitted platform-native CRLF while their byte protocol
expected LF. They now explicitly configure their owned stdout for LF; production
process capture still preserves native output. The existing-loop process test
expects the native line ending. Candidate plugin activation also compared POSIX
proposal keys to Windows-native manifest paths, silently missing newly added
plugins. Manifest lookup and editable-source keys now use `as_posix()`; native
absolute paths remain the plugin identity keys. Existing activation, rejected
registration, rollback-conflict and lock-lifetime tests cover that change.

The long-path launcher fixture passed a greater-than-320-character working
directory to CreateProcess and failed with Windows error 267. It now launches
from its short private temporary root while retaining long paths for the actual
application, configuration, workspace, trust state and both session locations.
Microsoft documents that a current directory longer than MAX_PATH causes
CreateProcessW to fail:
https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setcurrentdirectory
This test therefore does not claim support for launching subprocesses with a
long working directory. No extended-path prefix or system configuration is
changed. All 74 focused regression tests passed locally before the final helper
argument cleanup (macOS Python 3.12, 36.224 seconds). Updated native CI is required
before recording Windows acceptance.

### Portable checkout line endings (2026-09-25)

Revision 7d7386d passes all three native Windows jobs and the Linux jobs in PR
run 36156895405, including the repaired long-path and candidate-activation tests.
The complete Linux suite passes 1,074 tests (four Windows-only skips). Comparing
build reports exposed a separate packaging difference: Windows checkout converted
`.gitignore` and `pyproject.toml` to CRLF, adding exactly 100 bytes. A local capture
with only those two transformations reproduced the Windows archive SHA-256
`c5226391335e74afb8a0b09b375a86bd2719229e2ffa408260e24bd969dfd22f`
from the Linux archive
`dc266244bd689d5b294a71654a8256513cdf0fe5a99f056291157a006ffe4dda`.
Explicit LF attributes now cover both inputs. The builder continues to preserve
captured source bytes; no platform-dependent rewriting is added to Python code.

### Configuration capture and core diagnostic reads (2026-09-25)

Configuration loading now uses the shared regular-file reader with endpoint links
rejected. The initial byte bound is the observed file size plus one, because the
configuration document supplies its own `limits.max_config_bytes`; that configured
limit remains validated after parsing. A changed length fails capture. The opened
descriptor is checked before reading, including a FIFO substituted after path
inspection, and closes before JSON parsing or plugin/profile expansion. Operator
configuration must be quiescent during startup: this does not make an external
editor participate in a lock protocol or guarantee a snapshot under arbitrary
same-size in-place writes. Read failures are not retried; native OS exceptions
retain the existing ConfigurationError chain, and no file is modified by loading.

Core status uses the same helper to read at most the last 24,000 diagnostic bytes,
seeking and reading on one regular-file descriptor. A missing diagnostic remains
an empty status field; other read failures propagate. UTF-8 decoding replaces a
partial or invalid character at the suffix boundary. The reader closes before
returning status, allowing immediate retirement of the diagnostic file on Windows.
Concurrent append can expose an incomplete diagnostic line, which is acceptable
for this display-only suffix; it is not replayed as an operation or state record.

Core source result paths, suggested read actions and supervisor update keys now
serialize with `Path.as_posix()`. Actual filesystem paths remain native Path
objects. Existing end-to-end source/edit tests now assert every serialized key,
and configuration/core-tool tests are included in the native CI matrix. This
closes the separator mismatch at the core-update wire boundary; broader source
traversal and source-read lifetime review remains open.

Regression coverage includes replacing a configuration from inside its parser,
file growth during capture, a bounded child that replaces the selected file with
a FIFO after inspection, and immediate diagnostic retirement/missing-file reads.
The FIFO test is POSIX-specific. Existing endpoint link tests remain; Windows
symlink creation is skipped only when Windows explicitly denies that privilege.

The 25 focused configuration/core-tool tests pass on macOS Python 3.10.19
(2.369 seconds) and 3.14.3 (2.555 seconds). Strict typing, lint and formatting
pass without suppressions in `build/filesystem-quality-82/report.json`. Updated
native CI is the next gate; the prior CI evidence above is not a substitute.


### Core source inspection and edit-input capture

`core_source` and the input side of `core_update` inspect the retained active
release. Cooperating core updates build separate generations rather than editing
this tree in place. These operations own no public destination, staging file or
cleanup target. A developer checkout must remain quiescent during inspection;
external editors do not participate in the supervisor's generation protocol.

Explicit source names pass the shared portable relative-path validator. Each
component below the approved root is checked without resolving it into a different
object. Linked files and directories, including Windows reparse points, are
rejected even if their targets remain inside the release. The approved root and
its parents may resolve through operator-selected aliases. Search skips linked
entries before descending, traverses only observed ordinary directories, and
selects only regular `.py`/`.json` files. Absent optional `raychat` or `plugins`
roots are empty searches; disappearance after discovery and other I/O errors
propagate. Parent directories must remain trusted during the operation; these
checks are not a sandbox against concurrent hostile ancestor replacement.

Every source read uses `read_regular(..., follow_symlinks=False)`, bounded by the
observed size plus one byte. The helper checks the opened descriptor and closes
it before decoding, hashing, AST parsing or sending a supervisor request. A changed
identity, size, mode, link count, modification timestamp or change timestamp after
capture rejects the result; no read or whole-action retry is introduced. Access
timestamps are deliberately excluded because reading can update them. This does
not promise detection of arbitrary external rewrites that restore all observed
metadata, nor a transaction across multiple independently inspected files.

Search derives the first match's context, AST function boundary and SHA-256 from
the same captured bytes as its matching lines. Replacing the pathname during AST
parsing therefore cannot mix versions in one answer. Edit proposals retain their
existing whole-file digest check and submit changed bytes without writing the live
source. UTF-8 source bytes and original newlines are preserved.

Focused regression evidence: 23 core-tool/review tests pass on local macOS Python
3.10.19 and 3.14.3, each skipping the native Windows junction fixture. Tests replace
the source during AST parsing, inject growth and same-size replacement before a
read for source/search/update actions, and substitute a FIFO in a bounded POSIX
child. The Windows fixture creates a real junction without symlink privileges,
guards against traversal, and checks explicit-read rejection. It is selected in
the existing native CI matrix; this new revision still requires its CI result.
Strict typing, lint and format passed in `build/filesystem-quality-86`. The final
23-test core-tool/review selection also passed on local macOS Python 3.12.
