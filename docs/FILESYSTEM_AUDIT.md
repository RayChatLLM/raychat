# Filesystem ownership and acceptance ledger

This ledger covers all 67 checkbox requirements in the supplied filesystem
checklist (whose download label says 69). It records the supported contract,
implementation and tests together; historical intermediate results are available
in PR #1 and Git history rather than repeated as outstanding work here.

Review status: the repository ownership review and requirement reconciliation are
complete. Acceptance of the consolidated changes is determined by the final
[PR #1 checks](https://github.com/RayChatLLM/raychat/pull/1/checks).

The latest verified native baseline is `40a60b5b76145c0149e301b392995148c2725097`:
[GitHub Actions run 36179526261](https://github.com/RayChatLLM/raychat/actions/runs/36179526261),
all twelve latest job results successful. CI results must match the final PR
head; earlier green results do not validate later changes.

## Shared filesystem policy

`raychat/filesystem.py` owns replacement, securely allocated stages, retired-file
and directory cleanup, regular-file reads, append helpers and OS sidecar locks.
`file_lock.py` delegates to that implementation. Callers do not add filesystem
retry loops.

Publication creates a unique sibling with exclusive creation, writes through its
owned descriptor, flushes, fsyncs and closes it, then calls `Path.replace()`.
Switching to `os.replace()` would not change the underlying platform behavior.
On POSIX an existing reader can retain the old inode; Windows readers may deny
replacement through sharing modes or attributes. Native tests exercise both
behaviors. No failure path deletes the destination first, copies over it, or
changes a live destination's permissions to make replacement succeed.

The configurable default retry budget is 0.5 seconds, measured monotonically
with capped backoff and jitter. Selected replace/delete operations retry Windows
errors 5, 32 and 33; error 145 is additionally eligible only for retired-tree
cleanup. Error 5 is not diagnosed as antivirus, read-only state or an ACL problem
without further evidence. Other errors, including POSIX EACCES, ENOENT, EXDEV and
ENOSPC, propagate immediately. A retry deadline cannot interrupt an OS call that
is already blocked. Lock acquisition has its own budget and is not nested inside
a publication retry.

Temporary names contain 128 random bits and are reserved using exclusive file or
directory creation. Allocation allows at most eight genuine name collisions;
permission denial is not a name collision. This avoids Python 3.10 Windows
`tempfile` allocation repeatedly interpreting an ACL denial as a collision.
Directory swaps reserve a private container and use an unused child inside it;
they do not delete and immediately recycle a reserved name.

Synchronous helpers run on worker threads when called from async bulk-I/O paths.
`run_filesystem_task` and async snapshot staging join the same worker despite
repeated cancellation before the owner releases serialization or removes its
scratch directory. Async replacement uses cooperative sleeps. No await follows
a successful single-file publication, and cleanup trouble cannot replay it.
Diagnostics escape paths and include operation, exception type, errno, winerror,
attempts and elapsed time. Failed cleanup identifies retained owned resources.

`OwnedTemporaryDirectory` deletes only retired, privately owned trees. Its
explicit cleanup detaches automatic finalization before attempting removal, so a
later collection cannot revisit a reused name. Missing owned resources count as
cleaned; other failures are logged and retained. A primary write, execution or
cancellation failure survives a secondary cleanup failure. A cleanup-only failure
is either propagated by the strict operation or reported by its documented
best-effort cleanup wrapper.

`FileLock` keeps persistent sidecars and uses guarded `fcntl.flock`/`msvcrt.locking`
backends. Windows always locks byte zero, length one, on a binary nontruncating
handle. OS ownership, not pathname existence, determines acquisition. The wrapper
coordinates threads and processes; accidental reentry fails. Multiple package
scopes use canonical sidecar order; package scopes precede workspace ownership.
Transaction access may explicitly borrow ownership on the transaction's own
thread. Other readers acquire the normal bounded lock.

## Ownership inventory

The final review covered runtime, bootstrap, every plugin, tools and examples,
including filesystem methods passed to worker threads, embedded child scripts,
archive/stream owners and cleanup paths. The baseline AST inventory contains
1,193 candidate calls/references in 109 files; these include string operations,
network streams and in-memory buffers, not 1,193 filesystem mutations. Test
fixtures were reviewed as intentionally private mutations rather than converted
to production publication APIs.

| Owners and files | Concurrency, publication and retirement contract |
| --- | --- |
| `filesystem.py`, `file_lock.py` | Shared primitives described above; callers supply ownership and concurrency. `write_bytes` is last-writer-wins unless the caller holds a wider lock. |
| `workspace_files.py`, `plugins/filesystem/operations.py` | Workspace sidecar covers reads, append-as-snapshot, anchored edits and hash edits. Streamed edits close source and stage before replacement. Workspace operations recover recorded transactions before access. Arbitrary workspace file sizes are intentional. |
| `plugins/memory/store.py`, `workspace_trust.py` | Sidecars cover bounded reads and reload–modify–publish. Memory IDs derive from current disk state; cached queries describe the instance's last loaded snapshot. Trust uses the configured application home. |
| `storage.py` | Append-only session journal: lifetime `.writer.lock`, then thread mutex, then short-I/O `.lock`. Preview readers take only the I/O lock without waiting. Recovery trims an unterminated tail under ownership; appends are never retried or replaced. |
| `plugin_manager.py`, `package_transactions.py` | Both package scope locks protect journaled tree/receipt updates. Inputs are staged before moves. Commit records decide rollback versus cleanup. Recovery checks identities and content; conflicting external changes retain the record and trees. New writers require previous cleanup to finish. |
| `packages.py`, `plugin_sources.py`, `plugins.py`, composition and discovery callers | Production installed-source readers hold both package locks through discovery and immutable capture, then release before ordinary plugin execution. Captures reject linked descendants and revalidate inventory and opened identities. Roots/parents are trusted and sources quiescent. |
| Captured plugin generations and runtime resources | Generations own unique source directories. Retirement unregisters imports and releases files after callbacks succeed. Failed shutdown retains modules, finder and source tree. Transferred resources keep their source generation until final shutdown. Runtime/session cleanup attempts remaining owners while preserving the primary error. |
| `workspace_transactions.py`, self-harness promotion | Package scopes precede workspace lock. All private stages finish before publication; the undo journal records pending, committed or rolled-back decisions. Rollback is conditional on recorded bytes/identity and retires newly created files before deletion. Restart recovery leaves conflicts intact. |
| `plugins/self_harness/evidence.py` | Persistent sidecar covers append and bounded suffix reads. Only newline-complete records are parsed. A new append repairs an incomplete tail once under lock; failed appends are not replayed. |
| Self-harness evaluation and optimization fixtures | Source copying holds package/workspace ownership and excludes links, secrets and runtime artifacts. Pristine, baseline and candidate trees are private. Fixture writes finish before consumers start; the process service owns evaluator retirement. Custom process services must honor the same lifetime contract. |
| GEPA checkpoints, caches and rollout outputs | Public optimizer owns `gepa.lock` from before adapter/cache construction through worker completion and final checkpoint. Checkpoint is self-contained; optional cache/inspection snapshots may reflect different complete versions. Internal engine callers must supply equivalent ownership. |
| Optimization exports and benchmarks | Protocol/report exports are independent snapshots; aliases are rejected before execution and reports identify measured bytes by digest. Runtime shutdown precedes scratch deletion; unverifiable shutdown retains scratch. Caller-selected stop-marker removal is explicit, single-attempt deletion, not a retryable lock operation. |
| `plugins/skills/store.py`, configuration/resources/core source readers | Bounded regular-file reads close before parsing. Skills intentionally accept configured links and deduplicate existing aliases with `samefile`; discovery requires quiescent roots. Private metadata rejects linked/nonregular endpoints and descriptor substitution. Operator-selected input links follow each API's explicit policy. |
| `http_debug.py`, `http_replay.py`, transcript logs | HTTP appends belong to one connection; failures are not replayed. Replay commands each reference a completed immutable body and publish independently. Transcript appends serialize each write through a sidecar. Mappings accepted as HTTP input remain caller-owned; bytes are captured without assuming ownership. |
| `raychat_bootstrap` | Recovery metadata uses shared async publication and explicit persistence ordering. Release/configuration construction writes private files before exposing them. Normal supervisor/validator shutdown waits for direct children; candidates and logs are retained on failure. Native-creation interruption and repeated-cancellation limits are explicit below. Validator tools own their direct children. |
| `plugins/process`, transport, workers and clipboard | Intended standard streams only; no arbitrary inheritable descriptors. Command service retires process groups on POSIX and Jobs on Windows, including normal leader exit with descendants. Creation, cancellation and stream ownership have dedicated regression tests. Forced owner death and deliberately detached processes are separate limits below. |
| `tools/checker_process.py` | One async workspace owns checker tasks, children, streams and scratch. Concurrent failure cancels and joins siblings; repeated cancellation joins creation/retirement. Failed retirement retains the tree and primary failure. This is a direct-child contract; checkers must not leave independent descendants. |
| `tools/drive_tui.py` | PTY descriptors remain owned through reporting. Graceful EOF failure still kills/reaps the direct child; transcript/mode errors cannot skip closing either descriptor. Actual-child tests cover both failures. |
| Portable folder/release builders | Persistent output lock protects staged folder swap and recovery journal. Exact identities, content and modes govern restoration/cleanup; external changes stop recovery. ZIP and folder are independent complete publications. Strict offline cache cleanup requires a quiescent owned checkout and protects configured/preserved paths and linked trees. |
| Catalog builder and HTTP immutable bodies | Content-addressed artifacts publish before the manifest/profile pointing to them. Existing identical bytes are reused; conflicting existing content fails. Old artifacts are retained for old readers. Catalog builders require quiescent source trees and use last-writer-wins profile publication. |
| Plugin scaffold/pack, tools, examples and acceptance fixtures | Scaffolds reserve a new directory. Pack uses exclusive no-clobber creation; callers consume the export after success, and a failed new export may remain partial. Private inputs/reports complete before their consumers. Diagnostic writes never justify deleting active consumers' files. |

## Permissions, paths and guarantee boundaries

Replacement retries never clear read-only attributes or repair ACLs. Snapshot
mode changes affect the private stage. Private files default to 0600; workspace
snapshots preserve existing mode bits and use configured modes for new files.
Ownership, ACLs, timestamps and extended attributes are not generally preserved.
Explicit transcript opening has a separate POSIX privacy-tightening policy;
failed initialization may leave that deliberate restriction in place. Windows
ACLs and attributes are not changed to repair a failed open or replacement.

Application state uses configured application-home/session locations, not an
assumed writable installation. The extracted launcher is tested with read-only
installation and separate writable state, Unicode and paths longer than 320
characters. Generated artifact components are bounded portable names; typed IDs
use digests to avoid sanitization collisions. `Path` represents native paths and
portable manifests use POSIX separators. Text formats specify UTF-8; byte-exact
formats preserve their specified newlines. No extended Windows path prefix is
blindly prepended. The only runtime `chdir` belongs to a single-request worker;
its liveness watcher performs no relative filesystem operations.

Generic snapshot publication rejects a linked endpoint while permitting trusted
linked parent directories. Workspace APIs intentionally resolve in-root links
and refuse escape. Private metadata, package source descendants and cleanup
roots reject links/reparse points. These checks are not a hostile-parent sandbox:
trusted parents must remain stable during operations. Existing-file identity
checks do not reserve identities against uncooperative mutation.

The guarantee is complete single-file publication and the explicitly documented
cooperative transaction protocols. File flush/fsync precedes replacement, but
parent-directory fsync and universal power-loss durability are not provided.
Several replacements are not one atomic transaction for uncoordinated readers.
No live database is routed through the snapshot helper. Log formats keep their
own append/recovery protocols.

Sidecars coordinate this version's participating local processes. Editors, older
session/GEPA writers, antivirus and other nonparticipants do not acquire them.
Stop old writers before sharing storage with the new lock protocol. SMB/NFS,
cloud-synchronized storage, Controlled Folder Access deployments, and other
security/indexing products are outside the validated deployment scope. Defender
is tested enabled without exclusions; persistent security-policy denial requires
operator-approved configuration, never permission repair or weaker publication.

Automatic orphan recovery only reclaims exact journal-owned resources after
ownership and identity checks, or a parent's recorded stage after child
retirement. Unrecorded/pre-journal stages, unknown scratch trees and immutable old
artifacts are deliberately retained. Age, a PID string or a suffix is not proof
of inactivity; persistent lock sidecars are never garbage-collected. Manual
cleanup must first stop all consumers and establish ownership.

Bootstrap supervisor/validator native creation and repeated cancellation have
not been established as an all-descendants-retired guarantee. Candidate trees
remain retained rather than being removed on validation failure. This boundary
is distinct from the command service's explicitly joined creation owner.

No portable helper can guarantee arbitrary detached descendants retire after
SIGKILL, forced supervisor death or a hostile/custom evaluator's escape from its
ownership group. Validator and checker tools must fulfill their own descendant
contract. Retained candidate/log trees are not automatically deleted on such
failure. These are process/deployment boundaries, not reasons to weaken file
publication or silently retry whole workflows.

## Consolidated final audit changes

The command service retains one shielded native-creation task through cancellation
and joins retirement before propagating the original error. Synchronous calls run
in a context-preserving thread owner; main-thread signal interruption requests
cooperative stop and joins that owner before returning. The synchronous wait
wakes at the existing command polling interval so watcher-thread signals can
reach the main Python thread; the parent-SIGKILL integration test covers this. Four bounded child tests
cover repeated cancellation at handoff, actual POSIX pipe connection with a
pipe-holding descendant, simultaneous creation failure, and SIGTERM during native
creation. The two POSIX-specific fixtures are skipped on Windows; the portable
handoff and exception cases run there. The modified plugin catalog is rebuilt.

The PTY acceptance driver now closes both descriptors even if transcript reporting
fails, and kills/reaps its direct child if graceful EOF input fails. Bootstrap
checker diagnostics read only a 65,536-byte regular-file suffix; diagnostic
validation/read errors do not mask checker failure. Package scope acquisition and
release preserve the original recovery/operation failure if lock closure also
fails. These changes add focused real-child or fault-injection regression tests,
without expanding filesystem retries or changing replacement permissions.

Local process tests pass on Python 3.10 and 3.14; PTY tests pass on 3.10, 3.12 and
3.14. Full-suite/static results and final native counts are recorded on the PR
for the exact consolidated revision, rather than generating another commit just
to insert its own hash here.

## Requirement-by-requirement evidence

Rows identify implementation and regression evidence, not a claim that every
possible filesystem or external application has been validated. Test modules
are under `tests/`; the common native selection is `tools/verify_filesystem.py`.
Native acceptance of the final checkout remains the CI gate described below.

| # | Checklist requirement | Implementation and evidence |
| --- | --- | --- |
| 1 | Inventory all file touches | Runtime/bootstrap/plugin/tool/example call inventory and ownership table above; embedded child scripts and teardown reviewed separately. |
| 2 | Classify file usage | Inventory distinguishes private scratch, snapshots, serialized state, journals/logs and immutable generations; no live database replacements. |
| 3 | One filesystem module | `filesystem.py`; legacy locks delegate; callers use shared stage/replace/read/append/cleanup primitives. |
| 4 | Ownership and failure contracts | Helper docstrings and inventory define overwritten versus no-clobber outputs, owner lifetime, budgets and retained resources. |
| 5 | Remove check-then-act assumptions | Actual open/fstat/identity checks govern reads; exclusive creation reserves outputs. Diagnostic existence/type checks do not suppress later failures. `test_package_sources`, `test_filesystem`, `test_plugin_skills`. |
| 6 | Secure unique staging | Exclusive random sibling allocation, descriptor wrapping and bounded collision handling. Allocation/ACL tests in `test_filesystem` and `test_filesystem_process`. |
| 7 | Stage beside destination | `staged_file` uses destination parent; native cross-volume tests reject EXDEV without fallback. |
| 8 | Descriptor ownership | `owned_stream`, `staged_file`, journal wrapping; failed fdopen/write/close regression tests prove release and primary-error preservation. |
| 9 | NamedTemporaryFile audit | No production/runtime/plugin/build use; no blanket ban added. |
| 10 | Keep temporary consumers alive | Owned generation, checker, benchmark and process-service scopes; `test_checker_process`, `test_process_creation`, plugin retirement and workflow cleanup tests. |
| 11 | Fresh scratch names | Random exclusive directories and unused child slots for backup/retirement; package/release tests reject delete-and-reuse behavior. |
| 12 | Stage shared snapshots | Workspace, memory, trust, recovery, catalog, optimizer exports/checkpoints and reports use publication helpers. Direct construction writes are privately owned. |
| 13 | Finish writing layers first | Write/flush/fsync/close failure injection leaves old destination and forbids replacement; archive wrappers finish before publication. |
| 14 | Keep replacement semantics | Shared helper calls `Path.replace`; directory transactions use the same bounded primitive with explicit rollback. |
| 15 | No weaker fallback | No unlink-destination, copy-over-live or cross-volume move fallback. Failure/rollback tests preserve originals or retain recovery records. |
| 16 | Retry same completed source | Win32 5/32/33 tests assert stable source identity/content; appends and whole workflows are not replayed. |
| 17 | Track publication | No post-success await in async snapshot helper; journal commit decisions distinguish publication from cleanup. Crash-boundary tests cover restart decisions. |
| 18 | Short read handles | Bounded snapshot/source reads close before parsing; workspace streaming edits close before replacement. Skills/configuration/source descriptor tests. |
| 19 | Read sharing vs deletion | Native child-held destination and stage tests distinguish Windows denial from POSIX old-inode access. |
| 20 | Hidden handle owners | Archives, redirected streams, generation imports, session/transcript writers, background workers and stdio captures reviewed; lifetime and failure tests cover owned resources. |
| 21 | Mapping lifecycle | Native mapping fixtures release exported views, mapping and file independently. HTTP accepts caller-owned mappings without claiming retirement. |
| 22 | Child lifetime/inheritance | `close_fds`/intended standard streams; Windows Job and POSIX descendant tests, creation handoff/cancellation tests, checker sibling retirement. Forced/detached-owner limits above. |
| 23 | Immutable long-lived readers | Captured source generations, content-addressed catalogs/packages and replay bodies; old-reader and generation-retention tests. |
| 24 | Explicit concurrency | Inventory names serialized read–modify–write, exclusive owner and last-writer-wins APIs; stale memory and concurrent publication tests. |
| 25 | Stable sidecars | Workspace, trust, memory, evidence, session, package, GEPA and release locks are separate from replaced content. |
| 26 | Portable lock wrapper | Guarded fcntl/msvcrt backends; Windows byte zero/length one; `test_filesystem` and native process tests. |
| 27 | Keep lock files | Descriptor release never unlinks sidecars; killed-holder and persistent-sidecar tests. |
| 28 | Coordinated readers | Workspace/memory/trust/evidence/session previews and installed-source capture acquire their writer's coordination key. |
| 29 | Separate lock budgets/order | Bounded acquisition, thread contention and nonreentry tests; canonical package scope order then workspace; transaction borrowing is thread-local. |
| 30 | No sentinel lock heuristic | OS ownership replaces existence/PID/age inference; killed native lock holder releases ownership automatically. |
| 31 | Cooperation boundary | Local participating versions only; external editors/security software and distributed/sync storage exclusions documented above. |
| 32 | Classify winerror | Tests separate Win32 codes from errno; ordinary POSIX PermissionError is immediate. |
| 33 | Bounded monotonic jitter | `RetryPolicy`/budget helper; exhaustion, configurable delays and selected-code tests. OS calls themselves are not interruptible deadlines. |
| 34 | Do not diagnose error 5 | Selected bounded retry only; no inferred cause or automatic permission mutation. |
| 35 | Permanent errors fail | Missing parent, ENOSPC, EXDEV, invalid input and POSIX denial tests require one attempt; Windows policy denial reaches its fixed budget. |
| 36 | Useful diagnostics | Escaped path/type/errno/winerror/attempts/elapsed and retained-stage/tree messages; exact retained-path tests work on Windows. |
| 37 | Responsive cancellation | Async staging/bulk I/O uses joined workers, cooperative retry sleeps and no await after publication; blocked-fsync/repeated-cancel tests. |
| 38 | Retire before directory deletion | Generations, children, checkers and benchmarks settle owners before scratch cleanup; failed retirement retains owned trees. |
| 39 | Delete only owned resources | Unique scratch and identity-checked journals; recreated-container/external-edit tests preserve unrelated content. Offline cleaner has explicit quiescent ownership. |
| 40 | Missing scratch is clean | Shared owned unlink uses missing_ok; tree cleanup verifies disappearance. Other errors remain visible. |
| 41 | Preserve primary failure | Write/close/cleanup injection, runtime/session shutdown, benchmark rejection, checker retirement and process cancellation tests. |
| 42 | Review rmtree behavior | No runtime ignore_errors=True; shared bounded retired-tree removal and separate strict one-attempt offline cache cleaner. |
| 43 | Orphan policy | Recorded journal recovery only; process-kill tests leave active/unrelated work intact. Unknown/pre-journal stages are retained deliberately. |
| 44 | Configurable writable locations | `storage.home_directory`, session overrides and shipped-launcher tests; verified Windows standard users with read-only installation. |
| 45 | Security software enabled | Standard Windows 3.10/3.14 filesystem suites run again with Defender enabled and no exclusions; other products/sync deployments excluded. |
| 46 | Policy vs contention | Native denied-ACL allocation/publication tests keep bytes and permissions unchanged; diagnostics expose original error instead of claiming a cause. |
| 47 | Opt-in read-only handling | Stage-only modes and separate transcript privacy policy; no destination ACL/attribute repair. Native read-only and descriptor tests. |
| 48 | Metadata policy | Preserve workspace mode bits, private defaults otherwise; no general ownership/ACL/timestamp/xattr preservation or permission-repair promise. |
| 49 | Portable paths | Path-based native paths, POSIX archive names, no filesystem shell commands; sole runtime chdir is isolated in a one-request worker. |
| 50 | Portable unique names | `PortablePathIndex`, portable components and typed-key digests; reserved/case-colliding/hostile/long-name integration tests. |
| 51 | Identity not normcase | No production normcase identity checks; samefile for existing aliases and conservative collision handling for missing export paths. |
| 52 | Actual launcher long paths | Extracted shipped launcher exercised with Unicode paths over 320 characters on all nine OS/Python combinations; no prefix or OS-setting changes. |
| 53 | Encoding/newlines | UTF-8 runtime/tool/embedded-script text; byte-exact fixture/archive formats and LF/CRLF regression tests. |
| 54 | Links and junctions | Endpoint/reparse rejection or explicit configured-link support per owner; native Windows junction and POSIX link fixtures. Trusted-parent boundary stated. |
| 55 | Precise guarantees | Single-file complete publication, cooperative serialization and journaled recovery are distinct from power-loss durability. |
| 56 | Actual durability scope | Flush then fsync then close then replace; no parent-directory fsync or universal crash/power-loss durability claim. |
| 57 | Multiple files need protocol | Package/workspace/release journals and immutable catalog profile graph; independent reports/ZIPs explicitly do not form one transaction. |
| 58 | Logs/databases separate | Sessions, evidence, transcript and HTTP logs have explicit append ownership and no append retry; no live database snapshot replacement. |
| 59 | Nonlocal storage separate | SMB/NFS/cloud sync not supported by this local validation; no claim based on local sidecars. |
| 60 | Native ordinary privileges | Nine Windows/Linux/macOS × Python 3.10/3.12/3.14 jobs, plus verified nonadministrator Windows 3.10/3.14 accounts; final head CI required. |
| 61 | Reader contention | Pipe/event-ordered real child holds destination or stage; bounded denial then success after release, with old bytes preserved. |
| 62 | Competing writers | Thread/four-process increments and repeated distinguishable snapshots with coordinated readers; stale memory-instance regression. |
| 63 | Lock recovery | Killed process, thread contention, reentry and persistent-sidecar tests on native platforms. |
| 64 | Failure/crash boundaries | Creation/write/flush/close/replace injection and killed writers at journal, backup, publication, commit, rollback and partial-cleanup boundaries. |
| 65 | Cleanup contention | Native held scratch file plus transient/persistent mocked denial; useful diagnostics, preserved primary failure, later success and untouched unrelated work. |
| 66 | Environment/path edges | Read-only, native Windows ACL denial, ENOSPC injection, missing/absent paths, real C:/D: cross-volume replacement, mappings, Unicode, long paths, case aliases and links/junctions. |
| 67 | Deployment conditions | Local native matrix and verified enabled Defender; unsupported network/sync/other security products explicitly scoped out. |

## Verification gate and known evidence limits

Baseline run 36179526261 passed 1,136 full-suite tests (seven skips), strict typing,
Ruff and formatting. All nine platform jobs produced identical build/rebuild
SHA-256 `459ce14249242dfab0e9c77d5a62c9db48bb0a7ad6d1fce749533d6b03ea6569`.
All six POSIX jobs passed fourteen terminal scenarios. Both standard Windows
accounts passed 392 filesystem tests twice (thirteen platform skips): ordinary
execution and Defender-enabled execution. Artifacts verify Users membership,
absence of Administrators membership, writable private state, read-only install,
Defender settings before/after and no path/process/extension exclusions.

The first macOS 3.10 attempt failed one terminal process-liveness assertion; one
unchanged-commit rerun passed. Eleven local repetitions also passed. The failed
artifact lacks per-PID state evidence, so the original failure remains
unclassified. The subsequently reproduced async process-creation cancellation
gap has its own regression proof; it is not asserted to explain that CI failure.

The consolidated follow-up is accepted only after local focused regressions,
complete strict checks and all twelve latest native CI jobs pass at its exact PR
head. Record the final run and counts in the PR description without making a
post-validation documentation commit. The PR remains unmerged; main is not used
for audit commits.
