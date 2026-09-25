"""Package installation and catalogs. One service for CLI, UI and agent tools."""

from __future__ import annotations

import base64
import contextvars
import copy
import hashlib
import json
import logging
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import ParseResult, quote, unquote, urljoin, urlparse, urlunparse
from urllib.request import getproxies, proxy_bypass

from raychat.event_types import CONFIGURE, Lifecycle

from .filesystem import (
    FileLock,
    OwnedTemporaryDirectory,
    read_regular,
    write_bytes,
)
from .http_debug import HTTPConnection, HTTPSConnection
from .package_transactions import PackageTransaction
from .packages import (
    MAX_BYTES,
    NAME,
    VERSION,
    Manifest,
    ManifestDocument,
    dependency_order,
    digest,
    discover,
    files,
    pack,
    read_manifest,
    unpack,
)
from .plugin_sources import SourceTree
from .plugins import Runtime
from .sdk import API_VERSION, CancelCheck, PluginContext, PluginError, ServiceKey
from .transport import ProviderProcessError, run_child
from .validation import (
    ConfigurationError,
    array_field,
    boolean_field,
    json_object,
    object_field,
    text_field,
)
from .workspace_files import workspace_access

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from http.client import HTTPConnection as BaseHTTPConnection

    from .distribution import Distribution

_DOWNLOAD_CANCEL: contextvars.ContextVar[CancelCheck | None] = contextvars.ContextVar(
    "package_download_cancel",
    default=None,
)


class InstalledRelease(TypedDict):
    """Record the source and identity of an installed package release."""

    source: str
    version: str
    digest: str
    linked: bool


class PackageRecord(InstalledRelease, total=False):
    """Attach resolved sources and catalog receipts to a staged release."""

    path: str
    resolved: str
    archive_sha256: str
    catalog: str


class InstallationState(TypedDict):
    """Persist installation receipts, operator choices and catalog locations."""

    schema: int
    packages: dict[str, PackageRecord]
    disabled: list[str]
    catalogs: dict[str, str]


class PackageState(InstallationState, total=False):
    """Record optional profile installation history with the package state."""

    profiles: list[str]


_DIGEST_LENGTH = 64
_COMMIT_LENGTH = 40
_MAX_REDIRECTS = 10
_MAX_REPEATS = 4
_REPOSITORY_COMPONENTS = 2


class CatalogRecord(ManifestDocument):
    """Pin a validated package manifest to a location and exact archive bytes."""

    url: str
    sha256: str


class SearchResult(CatalogRecord):
    """Identify a matching catalog release and whether its metadata is cached."""

    catalog: str
    cached: bool


class InventoryItem(ManifestDocument):
    """Describe a checked package's installation and live activation status."""

    path: str
    scope: str
    enabled: bool
    loaded: bool
    modified: bool
    source: str


class InstallResult(TypedDict):
    """Report an immediate or deferred package transaction."""

    applied: bool
    packages: list[str]
    removed: list[str]


class CheckResult(TypedDict):
    """Describe registrations observed from a checked dependency generation."""

    id: str
    version: str
    sdk: int
    tools: list[str]
    commands: list[str]


@dataclass(frozen=True, kw_only=True)
class _ProfilePlan:
    identifier: str
    sources: list[str]
    states: dict[str, PackageState]


@dataclass(frozen=True, kw_only=True)
class _InstallRequest:
    sources: list[str]
    scope: str
    force: bool = False
    linked: bool = False
    ctx: PluginContext | None = None
    preserve_disabled: bool = False
    profile: _ProfilePlan | None = None


@dataclass(kw_only=True)
class _Staging:
    request: _InstallRequest
    temporary: OwnedTemporaryDirectory
    available: dict[str, Manifest]
    staged: dict[str, Path] = field(default_factory=dict)
    records: dict[str, PackageRecord] = field(default_factory=dict)
    manifests: dict[str, Manifest] = field(default_factory=dict)
    visiting: set[str] = field(default_factory=set)
    complete: set[str] = field(default_factory=set)


@dataclass(kw_only=True)
class _PackageChange:
    scope: str
    temporary: OwnedTemporaryDirectory
    ctx: PluginContext | None = None
    staged: Mapping[str, Path] = field(default_factory=dict)
    records: Mapping[str, PackageRecord] = field(default_factory=dict)
    removed: list[str] = field(default_factory=list)
    expected_sources: Mapping[str, str] = field(default_factory=dict)
    profile: _ProfilePlan | None = None


@dataclass(kw_only=True)
class _TransactionState:
    change: _PackageChange
    before: PackageState
    state: PackageState
    transaction: PackageTransaction | None = None
    held: ExitStack = field(default_factory=ExitStack)
    active: bool = False

    def release(self) -> None:
        """Release each acquired installation lock exactly once."""
        self.active = False
        self.held.close()

    def cleanup(self) -> None:
        """Remove staging files before releasing transaction ownership."""
        try:
            self.change.temporary.cleanup()
        finally:
            self.release()


def _apply_enabled(runtime: Runtime, identifier: str, *, enabled: bool) -> None:
    if enabled:
        runtime.disabled.discard(identifier)
    else:
        runtime.disabled.add(identifier)


def _release_key(item: SearchResult) -> tuple[str, tuple[int, ...]]:
    return item["id"], tuple(map(int, item["version"].split(".")))


def _require_version(owner: str, manifest: Manifest, expected: str) -> None:
    if manifest.version != expected:
        message = f"Dependency conflict: {owner} requires {manifest.id}@{expected}."
        raise PluginError(message)


def _identifier(value: object, path: str) -> str:
    result = text_field(value, path)
    if NAME.fullmatch(result) is None:
        raise PluginError("Invalid " + path + ": " + result)
    return result


def _version(value: object) -> str:
    result = text_field(value, "package version")
    if VERSION.fullmatch(result) is None:
        message = "Invalid installed package version."
        raise PluginError(message)
    return result


def _sha256(value: object, path: str) -> str:
    result = text_field(value, path)
    if len(result) != _DIGEST_LENGTH or any(
        char not in "0123456789abcdef" for char in result
    ):
        raise PluginError("Invalid " + path + ".")
    return result


def _schema_one(fields: Mapping[str, object], path: str) -> None:
    schema = fields.get("schema")
    if type(schema) is not int or schema != 1:
        raise PluginError("Invalid " + path + " schema.")


def _package_record(value: object, identifier: str, root: Path) -> PackageRecord:
    fields = object_field(value, "installed package record")
    allowed = {
        "path",
        "linked",
        "source",
        "version",
        "digest",
        "catalog",
        "resolved",
        "archive_sha256",
    }
    if fields.keys() - allowed:
        message = "Invalid installed package record fields."
        raise PluginError(message)
    path = Path(text_field(fields.get("path"), "installed package path"))
    if not path.is_absolute():
        message = "Installed package path must be absolute."
        raise PluginError(message)
    linked = boolean_field(fields.get("linked"), "installed package linked")
    if not linked and path != root / "plugins" / identifier:
        message = "Installed package path is outside its owned directory."
        raise PluginError(message)
    result: PackageRecord = {
        "path": str(path),
        "linked": linked,
        "source": text_field(fields.get("source"), "installed package source"),
        "version": _version(fields.get("version")),
        "digest": _sha256(fields.get("digest"), "installed package digest"),
    }
    if "catalog" in fields:
        result["catalog"] = _identifier(fields["catalog"], "catalog identifier")
    if "resolved" in fields:
        result["resolved"] = text_field(fields["resolved"], "resolved package source")
    if "archive_sha256" in fields:
        result["archive_sha256"] = _sha256(fields["archive_sha256"], "archive digest")
    return result


def _package_state(value: object, root: Path) -> PackageState:
    fields = object_field(value, "plugin installation lock file")
    allowed = {"schema", "packages", "disabled", "catalogs", "profiles"}
    if fields.keys() - allowed:
        message = "Invalid plugin installation lock file fields."
        raise PluginError(message)
    _schema_one(fields, "plugin installation lock file")
    result: PackageState = {
        "schema": 1,
        "packages": {
            _identifier(name, "installed package identifier"): _package_record(
                item,
                name,
                root,
            )
            for name, item in object_field(
                fields.get("packages"),
                "installed packages",
            ).items()
        },
        "disabled": [
            _identifier(item, "disabled plugin")
            for item in array_field(fields.get("disabled"), "disabled plugins")
        ],
        "catalogs": {
            _identifier(name, "catalog identifier"): text_field(url, "catalog URL")
            for name, url in object_field(fields.get("catalogs"), "catalogs").items()
        },
    }
    if "profiles" in fields:
        result["profiles"] = [
            _identifier(item, "profile identifier")
            for item in array_field(fields["profiles"], "profiles")
        ]
    return result


def _catalog_records(value: object) -> list[CatalogRecord]:
    fields = object_field(value, "catalog")
    _schema_one(fields, "catalog")
    results: list[CatalogRecord] = []
    versions: set[tuple[str, str]] = set()
    for entry in array_field(fields.get("plugins"), "catalog plugins"):
        item = object_field(entry, "catalog package")
        location = text_field(item.get("url"), "catalog package URL")
        archive_digest = _sha256(item.get("sha256"), "catalog SHA-256 digest")
        manifest = Manifest.parse({
            key: field for key, field in item.items() if key not in {"url", "sha256"}
        })
        key = (manifest.id, manifest.version)
        if key in versions:
            raise PluginError(
                "Duplicate catalog package version: "
                + manifest.id
                + "@"
                + manifest.version,
            )
        versions.add(key)
        results.append({
            **manifest.document(),
            "url": location,
            "sha256": archive_digest,
        })
    return results


def atomic_json(path: str | Path, value: object) -> None:
    """Persist finite JSON with flush, fsync and atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, indent=2)
    write_bytes(path, encoded.encode("utf-8"))


def read_json(path: str | Path, default: object) -> object:
    """Read bounded regular JSON while retaining unknown field values.

    Returns
    -------
    object
        The validated result described by this operation.

    Raises
    ------
    PluginError
        If the file is linked, non-regular or exceeds the package limit.

    """
    path = Path(path)
    try:
        data = read_regular(path, MAX_BYTES + 1, follow_symlinks=False)
    except FileNotFoundError:
        return default
    except ValueError as error:
        message = "Expected a regular plugin state file: " + str(path)
        raise PluginError(message) from error
    if len(data) > MAX_BYTES:
        error_message = "Plugin state exceeds its size limit."
        raise PluginError(error_message)
    return json_object(data)


def _has_url_scheme(location: str) -> bool:
    # URL parsing treats a Windows drive letter (C:) as a scheme. Keep drive
    # paths local, including drive-relative paths and UNC shares.
    return not PureWindowsPath(location).drive and bool(urlparse(location).scheme)


def _validate_url(url: str) -> ParseResult:
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or bool(parsed.username or parsed.password)
        or (
            parsed.scheme == "http"
            and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        )
    ):
        error_message = "Remote plugins and catalogs require HTTPS."
        raise PluginError(error_message)
    return parsed


@dataclass(frozen=True)
class _HTTPRoute:
    connection: BaseHTTPConnection
    target: str
    headers: dict[str, str]


def _proxy_location(parsed: ParseResult) -> str | None:
    proxy = getproxies().get(parsed.scheme)
    if proxy is None:
        return None
    raw_bypass: object = proxy_bypass(parsed.netloc)
    if type(raw_bypass) is bool:
        bypass = raw_bypass
    elif type(raw_bypass) is int and raw_bypass in {0, 1}:
        bypass = bool(raw_bypass)
    else:
        message = "System proxy bypass did not return a boolean decision."
        raise PluginError(message)
    return None if bypass else proxy


def _request_target(parsed: ParseResult) -> str:
    path = parsed.path or "/"
    if parsed.params:
        path += ";" + parsed.params
    if parsed.query:
        path += "?" + parsed.query
    return path


def _proxy_route(parsed: ParseResult, proxy: str) -> _HTTPRoute:
    address = urlparse(proxy if "://" in proxy else parsed.scheme + "://" + proxy)
    hostname = address.hostname
    if address.scheme not in {"http", "https"} or hostname is None:
        message = "Package proxies require an HTTP or HTTPS endpoint."
        raise PluginError(message)
    headers = {"User-Agent": "RayChat-plugins/2"}
    proxy_headers: dict[str, str] = {}
    if address.username and address.password:
        credentials = unquote(address.username) + ":" + unquote(address.password)
        proxy_headers["Proxy-Authorization"] = "Basic " + base64.b64encode(
            credentials.encode(),
        ).decode("ascii")
    proxy_host = unquote(hostname)
    if parsed.scheme == "https":
        origin = parsed.hostname
        if origin is None:
            message = "Package URL must name a host."
            raise PluginError(message)
        # urllib tunnels HTTPS origins using CONNECT before the target TLS
        # handshake; proxy credentials belong only to that CONNECT request.
        connection = HTTPSConnection(proxy_host, address.port, timeout=30)
        connection.set_tunnel(origin, parsed.port or 443, headers=proxy_headers)
        return _HTTPRoute(connection, _request_target(parsed), headers)
    direct = (
        HTTPSConnection(proxy_host, address.port, timeout=30)
        if address.scheme == "https"
        else HTTPConnection(proxy_host, address.port, timeout=30)
    )
    headers.update(proxy_headers)
    target = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        "",
    ))
    return _HTTPRoute(direct, target, headers)


def _http_route(url: str) -> _HTTPRoute:
    parsed = _validate_url(url)
    host = parsed.hostname
    if host is None:
        message = "Package URL must name a host."
        raise PluginError(message)
    proxy = _proxy_location(parsed)
    if proxy is not None:
        return _proxy_route(parsed, proxy)
    connection = (
        HTTPSConnection(host, parsed.port, timeout=30)
        if parsed.scheme == "https"
        else HTTPConnection(host, parsed.port, timeout=30)
    )
    return _HTTPRoute(
        connection,
        _request_target(parsed),
        {"User-Agent": "RayChat-plugins/2"},
    )


def _http_response(url: str) -> tuple[int, str | None, bytes]:
    route = _http_route(url)
    try:
        route.connection.request("GET", route.target, headers=route.headers)
        response = route.connection.getresponse()
        return (
            response.status,
            response.getheader("Location") or response.getheader("URI"),
            response.read(MAX_BYTES + 1),
        )
    finally:
        route.connection.close()


def _download_http(url: str) -> bytes:
    visits: dict[str, int] = {}
    current = url
    while True:
        status, location, data = _http_response(current)
        if status in {301, 302, 303, 307, 308} and location is not None:
            target = urljoin(current, location)
            parsed = _validate_url(target)
            if urlparse(current).scheme == "https" and parsed.scheme != "https":
                message = "HTTPS downloads cannot redirect to HTTP."
                raise PluginError(message)
            visits[target] = visits.get(target, 0) + 1
            if visits[target] > _MAX_REPEATS or sum(visits.values()) > _MAX_REDIRECTS:
                message = "Package download exceeded its redirect limit."
                raise OSError(message)
            current = target
            continue
        if not HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
            message = f"Package download failed with HTTP status {status}: {current}"
            raise OSError(message)
        if len(data) > MAX_BYTES:
            message = "Download exceeds the package byte limit."
            raise PluginError(message)
        return data


def _download_isolated(url: str, cancel_check: CancelCheck) -> bytes:
    cancel_check()
    with OwnedTemporaryDirectory(prefix="raychat-package-download-") as directory:
        destination = Path(directory) / "response"
        request: dict[str, object] = {
            "mode": "package_download",
            "url": url,
            "destination": str(destination),
        }
        try:
            count = run_child(None, request, cancel_check)
        except ProviderProcessError as error:
            if error.os_error:
                raise OSError(str(error)) from error
            raise PluginError(str(error)) from error
        cancel_check()
        data = read_bytes(destination)
        if count != str(len(data)):
            message = "Downloaded package size does not match its receipt."
            raise PluginError(message)
        return data


def download(url: str) -> bytes:
    """Download bounded bytes through validated HTTP endpoints and redirects.

    Returns
    -------
    bytes
        The complete package or catalog body.

    """
    _validate_url(url)
    cancel_check = _DOWNLOAD_CANCEL.get()
    return (
        _download_http(url)
        if cancel_check is None
        else _download_isolated(url, cancel_check)
    )


def read_bytes(path: str | Path) -> bytes:
    """Read a bounded package file before archive validation.

    Operator-selected file links are followed intentionally. The opened object
    must be regular; POSIX FIFOs are opened nonblocking and rejected. The
    descriptor closes before archive parsing. This read has no retry or
    permission repair and cannot interrupt an already-blocking native OS call.

    Returns
    -------
    bytes
        The validated result described by this operation.

    Raises
    ------
    PluginError
        If the input is nonregular or exceeds the package byte limit.

    """
    try:
        data = read_regular(Path(path), MAX_BYTES + 1)
    except ValueError as error:
        raise PluginError(
            "Package input must be a regular file: " + str(path),
        ) from error
    if len(data) > MAX_BYTES:
        error_message = "Package input exceeds its byte limit."
        raise PluginError(error_message)
    return data


class _SourceOwnership(threading.local):
    """Identify this thread's active package transaction, never another reader."""

    def __init__(self) -> None:
        self.scopes: set[str] = set()
        self.held = ExitStack()


class PackageManager:
    """Manage pinned package releases and atomic plugin generation updates."""

    @staticmethod
    @contextmanager
    def cancellable(check: CancelCheck) -> Iterator[None]:
        """Bind package I/O to this calling operation, independently per thread."""
        token = _DOWNLOAD_CANCEL.set(check)
        try:
            check()
            yield
            check()
        finally:
            _DOWNLOAD_CANCEL.reset(token)

    def __init__(
        self,
        workspace: str | Path,
        home: str | Path,
        *,
        trusted: bool = False,
        defer_state: bool = False,
    ) -> None:
        """Load receipts now, or on first use by a runtime with captured sources."""
        self.workspace = Path(workspace).resolve()
        self.home = Path(home).resolve()
        self.trusted = trusted
        self.runtime: Runtime | None = None
        self._stale_catalogs: set[str] = set()
        self.roots = {"workspace": self.workspace / ".raychat", "user": self.home}
        workspace_id = hashlib.sha256(str(self.workspace).encode("utf-8")).hexdigest()
        # A repository may contain arbitrary files. Installation authorization
        # belongs to the operator, never to a lock file supplied by that repo.
        self.state_roots = {
            "workspace": self.home / "workspaces" / workspace_id,
            "user": self.home,
        }
        self._state_cache: dict[str, PackageState] = {}
        self._source_ownership = _SourceOwnership()
        if not defer_state:
            with self.source_read():
                pass

    @property
    def _states(self) -> dict[str, PackageState]:
        if not self._state_cache:
            with self.source_read():
                pass
        return self._state_cache

    @_states.setter
    def _states(self, value: dict[str, PackageState]) -> None:
        self._state_cache = value

    def _acquire_scope(self, scope: str, *, timeout: float = 0) -> FileLock:
        lock = FileLock(self.state_roots[scope] / "plugins.mutex", timeout=timeout)
        lock.acquire()
        try:
            PackageTransaction.recover(
                self.roots[scope] / "plugins",
                self.state_file(scope),
            )
        except BaseException:
            lock.close()
            raise
        return lock

    @contextmanager
    def _locked_scope(self, scope: str, *, timeout: float = 1.0) -> Iterator[None]:
        lock = self._acquire_scope(scope, timeout=timeout)
        try:
            yield
        finally:
            lock.close()

    def state_file(self, scope: str) -> Path:
        """Locate a scope receipt outside the untrusted workspace.

        Returns
        -------
        Path
            The validated result described by this operation.

        """
        return self.state_roots[scope] / "plugins.lock.json"

    @contextmanager
    def source_read(self) -> Iterator[None]:
        """Hold both scopes through discovery and capture, never plugin execution.

        Acquire in canonical sidecar-path order with a one-second budget per
        scope. Refresh receipts after recovery while both locks are held. This
        context borrows locks only from this thread's explicit package transaction.
        Ordinary read contexts remain nonreentrant. Linked development sources
        and external editors remain outside the cooperating writer protocol.
        """
        if self._source_ownership.scopes:
            yield
            return
        with ExitStack() as held:
            scopes = sorted(
                (str(root), scope) for scope, root in self.state_roots.items()
            )
            for _root, scope in scopes:
                held.enter_context(self._locked_scope(scope))
            # Both scopes exclude new candidates. Recover interrupted file
            # batches before source inspection, then release the workspace lock.
            with workspace_access(self.workspace, existing_only=True):
                pass
            self._states = {scope: self._read(scope) for scope in self.roots}
            yield

    @contextmanager
    def source_update(self) -> Iterator[None]:
        """Own both package scopes through a queued source edit and validation.

        This explicit transaction lends its locks to source capture on the same
        thread. Acquire it before any workspace file lock and retain it until
        commit or rollback finishes. It supplies coordination, not an undo log.
        """
        with self._transaction_sources("source edits"):
            yield

    @contextmanager
    def _transaction_sources(self, scope: str) -> Iterator[None]:
        ownership = self._source_ownership
        if scope in ownership.scopes:
            message = "A package transaction is already active for scope: " + scope
            raise PluginError(message)
        if not ownership.scopes:
            ownership.held.enter_context(self.source_read())
        ownership.scopes.add(scope)
        try:
            yield
        finally:
            ownership.scopes.remove(scope)
            if not ownership.scopes:
                ownership.held.close()

    def _read(self, scope: str) -> PackageState:
        default: PackageState = {
            "schema": 1,
            "packages": {},
            "disabled": [],
            "catalogs": {},
        }
        try:
            return _package_state(
                read_json(self.state_file(scope), default),
                self.roots[scope],
            )
        except ConfigurationError as error:
            raise PluginError(str(error)) from error

    def paths(self, *, include_disabled: bool = False) -> dict[str, Path]:
        """Resolve installed packages while retaining scope and ambiguity checks.

        Hold source_read() through this lookup and subsequent source reads when
        package writers may run. Returned paths do not reserve installed trees.

        Returns
        -------
        dict[str, Path]
            The validated result described by this operation.

        Raises
        ------
        PluginError
            If more than one source claims a plugin identifier.

        """
        disabled = self.disabled
        result: dict[str, Path] = {}
        candidates = []
        for scope, state in self._states.items():
            candidates += [
                (Path(item["path"]), scope) for item in state["packages"].values()
            ]
        if self.trusted:
            candidates += [
                (p, "workspace") for p in discover(self.roots["workspace"] / "plugins")
            ]
        for path, _scope in candidates:
            manifest = read_manifest(path, require_current_sdk=False)
            if manifest.id in disabled and not include_disabled:
                continue
            previous = result.get(manifest.id)
            if previous is not None and previous.resolve() != path.resolve():
                raise PluginError(
                    "Ambiguous plugin ID: "
                    + manifest.id
                    + ". Uninstall or disable one source.",
                )
            result[manifest.id] = path.resolve()
        return result

    @property
    def disabled(self) -> set[str]:
        """Collect the operator-disabled plugin identifiers across scopes.

        Returns
        -------
        set[str]
            The validated result described by this operation.

        """
        return {name for state in self._states.values() for name in state["disabled"]}

    def attach(self, runtime: Runtime, *, inherit_disabled: bool = True) -> None:
        """Bind installation and discovery to a live plugin runtime."""
        self.runtime = runtime
        runtime.source_read = self.source_read
        raw_services: object = runtime.services
        object_field(raw_services, "runtime services")[PLUGIN_MANAGER.name] = self
        if inherit_disabled:
            runtime.disabled.update(self.disabled)

    def new_paths(self) -> list[str]:
        """Find installed package sources absent from the current generation.

        Returns
        -------
        list[str]
            The validated result described by this operation.

        Raises
        ------
        PluginError
            If no runtime has been attached.

        """
        if self.runtime is None:
            error_message = "Plugin discovery requires an attached runtime."
            raise PluginError(error_message)
        known = {
            snapshot["path"] for snapshot in self.runtime.export_sources()["packages"]
        }
        with self.source_read():
            return [
                str(path) for path in self.paths().values() if str(path) not in known
            ]

    def inventory(self) -> list[InventoryItem]:
        """Describe installed metadata and detect source edits without executing it.

        Returns
        -------
        list[InventoryItem]
            The validated result described by this operation.

        """
        with self.source_read():
            return self._inventory()

    def _inventory(self) -> list[InventoryItem]:
        result: list[InventoryItem] = []
        for identifier, path in sorted(self.paths(include_disabled=True).items()):
            manifest = read_manifest(path, require_current_sdk=False)
            scope, record = "workspace", None
            for name, state in self._states.items():
                if identifier in state["packages"]:
                    scope, record = name, state["packages"][identifier]
            result.append(
                {
                    **manifest.document(),
                    "path": str(path),
                    "scope": scope,
                    "enabled": identifier not in self.disabled,
                    "loaded": bool(self.runtime and identifier in self.runtime.plugins),
                    "modified": bool(record and not self._source_matches(path, record)),
                    "source": record.get("source", str(path)) if record else str(path),
                },
            )
        return result

    @staticmethod
    def _source_matches(path: Path, record: PackageRecord) -> bool:
        """Prove installed bytes match their receipt before replacing them.

        Earlier receipts included Finder metadata. Accept that exact recorded
        byte set too; never infer that a changed source file is an operator edit
        we may discard merely because metadata is present.

        Returns
        -------
        bool
            Whether current bytes match the exact stored receipt.

        """
        if digest(files(path, validate_manifest=False)) == record["digest"]:
            return True
        return (
            digest(
                files(
                    path,
                    validate_manifest=False,
                    ignore_finder_metadata=False,
                ),
            )
            == record["digest"]
        )

    def ensure_profile(self, profile: Distribution) -> None:
        """Reconcile unmodified profile releases in one ordinary transaction.

        Archive identities detect changed releases even when versions stay equal.
        User edits, other package sources and removal/disable choices are retained.
        No registration runs during installation or CLI metadata discovery.

        Raises
        ------
        PluginError
            If a runtime is attached or locked installation state changed.

        """
        if self.runtime is not None:
            error_message = "Install a startup profile before attaching a runtime."
            raise PluginError(error_message)
        scope = "user"
        state = self._states[scope]
        catalog_name = profile.id
        previous_catalog = state["catalogs"].get(catalog_name)
        prior_releases = self._prior_releases(previous_catalog)
        origins = copy.deepcopy(state)
        for identifier, origin in origins["packages"].items():
            if "catalog" not in origin and (
                origin["source"]
                == catalog_name + "/" + identifier + "@" + origin["version"]
                or (
                    origin["source"] == identifier + "@" + origin["version"]
                    and (
                        identifier,
                        origin["version"],
                        origin.get("resolved"),
                        origin.get("archive_sha256"),
                    )
                    in prior_releases
                )
            ):
                origin["catalog"] = catalog_name
        if origins != state:
            # Preserve proven origins before refreshing the catalog cache. A
            # failed download/upgrade must not erase the evidence for a retry.
            with self._locked_scope(scope):
                if self._read(scope) != state:
                    error_message = "Installation state changed; retry the profile."
                    raise PluginError(error_message)
                atomic_json(self.state_file(scope), origins)
                self._states[scope] = state = origins
        if self._states[scope]["catalogs"].get(catalog_name) != str(profile.catalog):
            self.catalog("add", catalog_name, str(profile.catalog), scope=scope)
        releases = {
            item["id"] + "@" + item["version"]: item
            for item in self._catalog_records(str(profile.catalog))
        }
        plan = self._profile_plan(profile, releases)
        if plan.sources:
            self._install_sources(
                _InstallRequest(
                    sources=plan.sources,
                    scope=scope,
                    preserve_disabled=True,
                    profile=plan,
                ),
            )
            return
        with self.source_read():
            self._verify_profile(plan)
            state = copy.deepcopy(self._states[scope])
            profiles = state.setdefault("profiles", [])
            if profile.id not in profiles:
                profiles.append(profile.id)
                atomic_json(self.state_file(scope), state)
                self._states[scope] = state

    def _profile_plan(
        self,
        profile: Distribution,
        releases: Mapping[str, CatalogRecord],
    ) -> _ProfilePlan:
        with self.source_read():
            return _ProfilePlan(
                identifier=profile.id,
                sources=self._profile_sources(profile, self._states["user"], releases),
                states=copy.deepcopy(self._states),
            )

    def _verify_profile(self, plan: _ProfilePlan) -> None:
        if self._states != plan.states:
            message = "Installation state changed; retry the profile."
            raise PluginError(message)

    def _profile_sources(
        self,
        profile: Distribution,
        state: PackageState,
        releases: Mapping[str, CatalogRecord],
    ) -> list[str]:
        catalog_name = profile.id
        available = self.paths(include_disabled=True)
        sources = []
        for spec in profile.packages:
            identifier = spec.partition("@")[0]
            if identifier in self.disabled:
                continue
            record = state["packages"].get(identifier)
            if identifier in available:
                if record is None or record["linked"]:
                    continue
                if (
                    record.get("catalog") != catalog_name
                    or not self._source_matches(available[identifier], record)
                    or record.get("archive_sha256") == releases[spec]["sha256"]
                ):
                    continue
            sources.append(catalog_name + "/" + spec)
        return sources

    def catalogs(self) -> dict[str, str]:
        """Collect configured catalog locations in scope precedence order.

        Returns
        -------
        dict[str, str]
            The validated result described by this operation.

        """
        result = {}
        for state in self._states.values():
            result.update(state["catalogs"])
        return result

    def catalog(
        self,
        operation: str,
        name: str = "",
        url: str = "",
        *,
        scope: str = "workspace",
    ) -> dict[str, str]:
        """Validate and atomically persist a catalog addition or removal.

        Returns
        -------
        dict[str, str]
            The validated result described by this operation.

        Raises
        ------
        PluginError
            If the operation, catalog name or locked state is invalid.

        """
        if operation == "list":
            return self.catalogs()
        if not NAME.fullmatch(name):
            error_message = "Catalog name must be an identifier."
            raise PluginError(error_message)
        before = self._states[scope]
        state = copy.deepcopy(before)
        if operation == "add":
            self._catalog_records(url)
            state["catalogs"][name] = url
        elif operation == "remove":
            if name not in state["catalogs"]:
                raise PluginError("Unknown catalog: " + name)
            del state["catalogs"][name]
        else:
            error_message = "Use catalog list, add NAME URL, or remove NAME."
            raise PluginError(error_message)
        with self._locked_scope(scope):
            if self._read(scope) != before:
                error_message = "Catalog state changed; retry."
                raise PluginError(error_message)
            atomic_json(self.state_file(scope), state)
            self._states[scope] = state
        return self.catalogs()

    def _catalog_cache(self, url: str) -> Path:
        return (
            self.home
            / "catalog-cache"
            / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        )

    @staticmethod
    def _catalog_url(catalog: str, location: str) -> str:
        if _has_url_scheme(location):
            return location
        if _has_url_scheme(catalog):
            return urljoin(catalog, location)
        if PureWindowsPath(location).drive:
            return location
        windows = PureWindowsPath(catalog)
        if windows.drive:
            return str(windows.parent / location)
        return str(Path(catalog).parent / location)

    def _prior_releases(self, catalog: str | None) -> set[tuple[str, str, str, str]]:
        if catalog is None:
            return set()
        value = read_json(self._catalog_cache(catalog), {"schema": 1, "plugins": []})
        try:
            records = _catalog_records(value)
        except ConfigurationError as error:
            raise PluginError(
                "Invalid cached profile catalog: " + str(error),
            ) from error
        return {
            (
                item["id"],
                item["version"],
                self._catalog_url(catalog, item["url"]),
                item["sha256"],
            )
            for item in records
        }

    def _catalog_records(self, location: str) -> list[CatalogRecord]:
        cache = self._catalog_cache(location)
        try:
            data = (
                download(location)
                if _has_url_scheme(location)
                else read_bytes(location)
            )
            value = json_object(data)
            self._stale_catalogs.discard(location)
        except OSError:
            self._stale_catalogs.add(location)
            value = read_json(cache, None)
            if value is None:
                raise
        try:
            records = _catalog_records(value)
        except ConfigurationError as error:
            raise PluginError(str(error)) from error
        document: dict[str, object] = {"schema": 1, "plugins": records}
        atomic_json(cache, document)
        return records

    def search(self, query: str = "") -> list[SearchResult]:
        """Search validated catalog metadata in descending release order.

        Returns
        -------
        list[SearchResult]
            The validated result described by this operation.

        """
        results: list[SearchResult] = [
            {
                **item,
                "url": self._catalog_url(url, item["url"]),
                "catalog": name,
                "cached": url in self._stale_catalogs,
            }
            for name, url in self.catalogs().items()
            for item in self._catalog_records(url)
            if query.casefold() in (item["id"] + " " + item["description"]).casefold()
        ]
        return sorted(results, key=_release_key, reverse=True)

    def resolve_source(self, source: str) -> tuple[str, SearchResult | None]:
        """Resolve a local source, immutable GitHub revision or catalog release.

        Returns
        -------
        tuple[str, SearchResult | None]
            The validated result described by this operation.

        Raises
        ------
        PluginError
            If the source cannot be resolved to an unambiguous release.

        """
        path = Path(source).expanduser()
        if path.exists():
            return str(path.resolve()), None
        if source.startswith(("github:", "https://github.com/")):
            spec = (
                source
                .removeprefix("github:")
                .removeprefix("https://github.com/")
                .rstrip("/")
            )
            repo, _separator, ref = spec.partition("@")
            if "/tree/" in repo:
                repo, ref = repo.split("/tree/", 1)
            owner_repo = repo.removesuffix(".git").split("/")
            if len(owner_repo) != _REPOSITORY_COMPONENTS or not all(
                NAME.fullmatch(v) for v in owner_repo
            ):
                error_message = "Use github:OWNER/REPOSITORY@REF."
                raise PluginError(error_message)
            commit_record = json_object(
                download(
                    "https://api.github.com/repos/"
                    + "/".join(owner_repo)
                    + "/commits/"
                    + quote(ref or "HEAD", safe=""),
                ),
            )
            commit = object_field(commit_record, "GitHub commit").get("sha")
            if (
                not isinstance(commit, str)
                or len(commit) != _COMMIT_LENGTH
                or any(c not in "0123456789abcdef" for c in commit)
            ):
                error_message = "GitHub did not resolve an immutable commit."
                raise PluginError(error_message)
            return "https://api.github.com/repos/" + "/".join(
                owner_repo,
            ) + "/zipball/" + commit, None
        if urlparse(source).scheme in {"https", "http"}:
            return source, None
        identifier, _, version = source.partition("@")
        catalog, sep, identifier = identifier.partition("/")
        if not sep:
            identifier, catalog = catalog, ""
        choices = [
            item
            for item in self.search(identifier)
            if item["id"] == identifier
            and (not catalog or item["catalog"] == catalog)
            and (not version or item["version"] == version)
        ]
        if not choices:
            raise PluginError(
                "Plugin not found. Add a catalog or supply a package path/URL: "
                + source,
            )
        if not catalog and len({item["catalog"] for item in choices}) > 1:
            error_message = (
                "Plugin exists in multiple catalogs; use CATALOG/ID@VERSION."
            )
            raise PluginError(
                error_message,
            )
        return choices[0]["url"], choices[0]

    def _stage(self, source: str, target: Path) -> tuple[Manifest, PackageRecord]:
        resolved, expected = self.resolve_source(source)
        path = Path(resolved)
        if not _has_url_scheme(resolved) and path.is_dir():
            members = files(path)
            for name, data in members.items():
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            archive_hash = hashlib.sha256(pack(path)).hexdigest()
        else:
            data = download(resolved) if _has_url_scheme(resolved) else read_bytes(path)
            archive_hash = hashlib.sha256(data).hexdigest()
            if expected and archive_hash != expected["sha256"]:
                error_message = "Catalog package digest does not match the download."
                raise PluginError(error_message)
            unpack(data, target)
        manifest = read_manifest(target)
        if (
            expected
            and manifest.document()
            != Manifest.parse(
                {
                    k: v
                    for k, v in expected.items()
                    if k not in {"url", "sha256", "catalog", "cached"}
                },
            ).document()
        ):
            error_message = "Downloaded manifest does not match the catalog."
            raise PluginError(error_message)
        record: PackageRecord = {
            "source": source,
            "resolved": resolved,
            "archive_sha256": archive_hash,
            "version": manifest.version,
            "digest": digest(files(target)),
            "linked": False,
        }
        if expected is not None:
            record["catalog"] = text_field(expected["catalog"], "catalog")
        return manifest, record

    def install(
        self,
        source: str,
        *,
        scope: str = "workspace",
        force: bool = False,
        linked: bool = False,
        ctx: PluginContext | None = None,
    ) -> InstallResult:
        """Stage a package and its dependencies before queuing one transaction.

        Returns
        -------
        InstallResult
            The validated result described by this operation.

        """
        return self._install_sources(
            _InstallRequest(
                sources=[source],
                scope=scope,
                force=force,
                linked=linked,
                ctx=ctx,
            ),
        )

    def _stage_install(
        self,
        spec: str,
        plan: _Staging,
        expected: tuple[str, str] | None = None,
    ) -> Manifest:
        target = Path(plan.temporary.name) / ("package-" + str(len(plan.staged)))
        manifest, record = self._stage(spec, target)
        if expected is not None and (manifest.id, manifest.version) != expected:
            message = "Dependency resolved to an unexpected version."
            raise PluginError(message)
        if plan.request.preserve_disabled and manifest.id in self.disabled:
            raise PluginError(
                "Profile requires a disabled or removed plugin; "
                "enable or install it explicitly first: " + manifest.id,
            )
        if manifest.id in plan.staged:
            if plan.records[manifest.id]["version"] != manifest.version:
                raise PluginError("Conflicting dependency versions: " + manifest.id)
            return manifest
        plan.staged[manifest.id], plan.records[manifest.id] = target, record
        plan.manifests[manifest.id] = manifest
        return manifest

    def _collect_install(self, identifier: str, plan: _Staging) -> None:
        if identifier in plan.complete:
            return
        if identifier in plan.visiting:
            raise PluginError("Plugin dependency cycle: " + identifier)
        plan.visiting.add(identifier)
        manifest = plan.manifests[identifier]
        for dependency, version in manifest.requires.items():
            if dependency in plan.staged:
                _require_version(identifier, plan.manifests[dependency], version)
                self._collect_install(dependency, plan)
            elif dependency in plan.available:
                _require_version(
                    identifier,
                    Manifest.parse(plan.available[dependency].document()),
                    version,
                )
            else:
                catalog = plan.records[identifier].get("catalog")
                spec = (catalog + "/" if catalog else "") + dependency + "@" + version
                self._stage_install(spec, plan, (dependency, version))
                self._collect_install(dependency, plan)
        plan.visiting.remove(identifier)
        plan.complete.add(identifier)

    def _replacement_sources(self, plan: _Staging) -> dict[str, str]:
        with self.source_read():
            expected_sources: dict[str, str] = {}
            inventory = {item["id"]: item for item in self._inventory()}
            for identifier in plan.staged:
                old = inventory.get(identifier)
                if old is None:
                    continue
                if old["scope"] != plan.request.scope:
                    raise PluginError(
                        "Plugin ID already exists in another scope: " + identifier,
                    )
                expected_sources[old["path"]] = digest(
                    files(old["path"], validate_manifest=False),
                )
                if old["modified"] and not plan.request.force:
                    raise PluginError(
                        "Plugin has local edits; use --force to replace: " + identifier,
                    )
            return expected_sources

    @staticmethod
    def _link_primary(plan: _Staging, primary: str) -> None:
        if not plan.request.linked:
            return
        sources = plan.request.sources
        if (
            len(plan.staged) != 1
            or len(sources) != 1
            or not Path(sources[0]).expanduser().is_dir()
        ):
            message = "link requires a local package with installed dependencies."
            raise PluginError(message)
        plan.records[primary]["linked"] = True
        plan.records[primary]["path"] = str(Path(sources[0]).expanduser().resolve())

    def _stage_roots(self, plan: _Staging) -> _PackageChange:
        primary = self._stage_install(plan.request.sources[0], plan)
        # Stage every requested root before collecting dependencies so an entire
        # profile upgrade is checked as one candidate generation.
        for source in plan.request.sources[1:]:
            self._stage_install(source, plan)
        for identifier in list(plan.staged):
            self._collect_install(identifier, plan)
        expected_sources = self._replacement_sources(plan)
        self._link_primary(plan, primary.id)
        return _PackageChange(
            scope=plan.request.scope,
            staged=plan.staged,
            records=plan.records,
            temporary=plan.temporary,
            ctx=plan.request.ctx,
            expected_sources=expected_sources,
            profile=plan.request.profile,
        )

    def _install_sources(self, request: _InstallRequest) -> InstallResult:
        temporary = OwnedTemporaryDirectory(prefix="raychat-install-")
        try:
            with self.source_read():
                available = {
                    name: read_manifest(path, require_current_sdk=False)
                    for name, path in self.paths(include_disabled=True).items()
                }
            plan = _Staging(
                request=request,
                temporary=temporary,
                available=available,
            )
            return self._transaction(self._stage_roots(plan))
        except BaseException:
            temporary.cleanup()
            raise

    def _transaction(self, change: _PackageChange) -> InstallResult:
        before = self._read(change.scope)
        state = copy.deepcopy(before)
        base = self.roots[change.scope]
        for identifier, record in change.records.items():
            record.setdefault("path", str(base / "plugins" / identifier))
            state["packages"][identifier] = record
            state["disabled"] = [
                value for value in state["disabled"] if value != identifier
            ]
        for identifier in change.removed:
            state["packages"].pop(identifier, None)
            if identifier not in state["disabled"]:
                state["disabled"].append(identifier)
        if change.profile is not None:
            profiles = state.setdefault("profiles", [])
            if change.profile.identifier not in profiles:
                profiles.append(change.profile.identifier)
        plan = _TransactionState(change=change, before=before, state=state)
        return self._apply_transaction(plan)

    def _prepare_transaction(self, plan: _TransactionState) -> None:
        plan.held.enter_context(self._transaction_sources(plan.change.scope))
        plan.active = True
        if self._read(plan.change.scope) != plan.before:
            message = "Installation state changed; retry the operation."
            raise PluginError(message)
        if plan.change.profile is not None:
            self._verify_profile(plan.change.profile)
        for path, expected in plan.change.expected_sources.items():
            if digest(files(path, validate_manifest=False)) != expected:
                raise PluginError(
                    "Installed source changed while the update was queued: " + path,
                )
        sources: dict[str, Path | None] = {
            identifier: source
            for identifier, source in plan.change.staged.items()
            if not plan.change.records[identifier]["linked"]
        }
        for identifier in plan.change.removed:
            old = plan.before["packages"].get(identifier)
            if old is not None and not old["linked"]:
                sources[identifier] = None
        following = json.dumps(plan.state, sort_keys=True, allow_nan=False, indent=2)
        plan.transaction = PackageTransaction.begin(
            self.roots[plan.change.scope] / "plugins",
            self.state_file(plan.change.scope),
            sources,
            following.encode("utf-8"),
        )
        plan.transaction.apply(sources)

    def _rollback_transaction(self, plan: _TransactionState) -> None:
        try:
            self._restore_transaction(plan)
        finally:
            plan.cleanup()

    @staticmethod
    def _restore_transaction(plan: _TransactionState) -> None:
        if plan.active and plan.transaction is not None:
            plan.transaction.rollback()

    def _commit_transaction(self, plan: _TransactionState) -> None:
        try:
            self._finish_transaction(plan)
        finally:
            plan.cleanup()

    def _finish_transaction(self, plan: _TransactionState) -> None:
        if plan.transaction is not None:
            plan.transaction.commit()
        self._states[plan.change.scope] = plan.state
        if self.runtime is not None:
            self.runtime.disabled.update(self.disabled)

    def _validate_transaction(self, plan: _TransactionState) -> None:
        # Validate the complete installed dependency graph before commit.
        paths = self.paths(include_disabled=True)
        paths.update({
            name: Path(item["path"]) for name, item in plan.change.records.items()
        })
        for name in plan.change.removed:
            paths.pop(name, None)
        dependency_order({
            name: read_manifest(path, require_current_sdk=False)
            for name, path in paths.items()
        })

    def _apply_transaction(self, plan: _TransactionState) -> InstallResult:
        change = plan.change
        if self.runtime is None:
            try:
                self._prepare_transaction(plan)
                self._validate_transaction(plan)
                self._commit_transaction(plan)
            except BaseException:
                try:
                    self._rollback_transaction(plan)
                except (OSError, PluginError):
                    logging.getLogger(__name__).exception(
                        "Package rollback failed; journal retained for scope=%r",
                        change.scope,
                    )
                raise
            return {
                "applied": True,
                "packages": list(change.records),
                "removed": change.removed,
            }
        removed_active = [
            identifier
            for identifier in [*change.removed, *change.records]
            if identifier in self.runtime.plugins
        ]
        context = change.ctx or self.runtime.context("plugin_manager")

        def prepare() -> None:
            self._prepare_transaction(plan)

        def commit() -> None:
            self._commit_transaction(plan)

        def rollback(_error: BaseException) -> None:
            self._rollback_transaction(plan)

        applied = context.update_plugins(
            add=[record["path"] for record in change.records.values()],
            remove=removed_active,
            prepare=prepare,
            commit=commit,
            rollback=rollback,
        )
        return {
            "applied": applied,
            "packages": list(change.records),
            "removed": change.removed,
        }

    def set_enabled(
        self,
        identifier: str,
        *,
        enabled: bool,
        scope: str = "workspace",
        ctx: PluginContext | None = None,
    ) -> bool:
        """Queue a dependency-checked activation change with a locked receipt.

        Returns
        -------
        bool
            The validated result described by this operation.

        """
        runtime, paths = self._activation_inputs(
            identifier,
            enabled=enabled,
            scope=scope,
        )
        before = self._read(scope)
        state = copy.deepcopy(before)
        state["disabled"] = [v for v in state["disabled"] if v != identifier]
        if not enabled:
            state["disabled"].append(identifier)
        held = ExitStack()
        prepared = False

        def prepare() -> None:
            nonlocal prepared
            held.enter_context(self._transaction_sources(scope))
            prepared = True
            if self._read(scope) != before:
                error_message = "Plugin state changed; retry."
                raise PluginError(error_message)
            atomic_json(self.state_file(scope), state)

        def commit() -> None:
            self._states[scope] = state
            _apply_enabled(runtime, identifier, enabled=enabled)
            held.close()

        def rollback(_error: BaseException) -> None:
            if not prepared:
                return
            try:
                if self._read(scope) == state:
                    atomic_json(self.state_file(scope), before)
            finally:
                held.close()

        context = ctx or runtime.context("plugin_manager")
        return context.update_plugins(
            add=[paths[identifier]]
            if enabled and identifier not in runtime.plugins
            else [],
            remove=[identifier]
            if not enabled and identifier in runtime.plugins
            else [],
            prepare=prepare,
            commit=commit,
            rollback=rollback,
        )

    def _activation_inputs(
        self,
        identifier: str,
        *,
        enabled: bool,
        scope: str,
    ) -> tuple[Runtime, dict[str, Path]]:
        runtime = self.runtime
        if runtime is None:
            error_message = "Plugin activation requires an attached runtime."
            raise PluginError(error_message)
        with self.source_read():
            paths = self.paths(include_disabled=True)
            if identifier not in paths:
                raise PluginError("Unknown plugin: " + identifier)
            if enabled and any(
                identifier in v["disabled"]
                for k, v in self._states.items()
                if k != scope
            ):
                error_message = (
                    "Plugin is disabled in another scope; enable it there first."
                )
                raise PluginError(
                    error_message,
                )
            return runtime, paths

    def uninstall(
        self,
        identifier: str,
        *,
        scope: str = "workspace",
        ctx: PluginContext | None = None,
    ) -> InstallResult:
        """Queue package removal while preserving linked source directories.

        Returns
        -------
        InstallResult
            The validated result described by this operation.

        Raises
        ------
        PluginError
            If this scope does not own the installed package.

        """
        if identifier not in self._states[scope]["packages"]:
            raise PluginError("Plugin is not installed in this scope: " + identifier)
        temporary = OwnedTemporaryDirectory(prefix="raychat-uninstall-")
        return self._transaction(
            _PackageChange(
                scope=scope,
                removed=[identifier],
                temporary=temporary,
                ctx=ctx,
            ),
        )

    def update(
        self,
        identifier: str,
        *,
        scope: str = "workspace",
        force: bool = False,
        ctx: PluginContext | None = None,
    ) -> InstallResult:
        """Resolve an installed release source again and queue its replacement.

        Returns
        -------
        InstallResult
            The validated result described by this operation.

        Raises
        ------
        PluginError
            If the package is missing or linked to a development source.

        """
        record = self._states[scope]["packages"].get(identifier)
        if record is None or record.get("linked"):
            error_message = "Only installed release packages can be updated."
            raise PluginError(error_message)
        source = record["source"]
        if not urlparse(source).scheme and "@" in source and not Path(source).exists():
            source = source.rsplit("@", 1)[0]
        return self.install(source, scope=scope, force=force, ctx=ctx)

    def check(self, path: str | Path) -> CheckResult:
        """Load a captured dependency generation and report its registrations.

        Returns
        -------
        CheckResult
            The validated result described by this operation.

        """
        runtime = Runtime(self.workspace)
        trees: list[SourceTree] = []
        try:
            with self.source_read():
                available = self.paths(include_disabled=True)
                manifest = read_manifest(path)
                available[manifest.id] = Path(path).resolve()
                ordered = dependency_order(
                    {
                        name: read_manifest(candidate, require_current_sdk=False)
                        for name, candidate in available.items()
                    },
                    [manifest.id],
                )
                trees.extend(SourceTree(available[name]) for name in ordered)
            runtime.load([tree.entrypoint() for tree in trees])
            runtime.emit(CONFIGURE, Lifecycle(), strict=True)
            return {
                "id": manifest.id,
                "version": manifest.version,
                "sdk": manifest.sdk,
                "tools": list(runtime.tools),
                "commands": list(runtime.commands),
            }
        finally:
            try:
                runtime.close()
            finally:
                # Includes captures whose entrypoint failed before runtime.load().
                for tree in trees:
                    tree.retire()


PLUGIN_MANAGER = ServiceKey("plugin_manager", PackageManager)


def scaffold(path: str | Path) -> Path:
    """Create a shareable SDK package with checked action and settings fields.

    Returns
    -------
    Path
        The validated result described by this operation.

    Raises
    ------
    PluginError
        If the destination exists or its name is not a valid plugin ID.

    """
    path = Path(path).expanduser().resolve()
    if path.exists():
        error_message = "Scaffold destination already exists."
        raise PluginError(error_message)
    if not NAME.fullmatch(path.name):
        error_message = "Destination name must be a valid plugin ID."
        raise PluginError(error_message)
    path.mkdir(parents=True)
    manifest = Manifest(
        path.name,
        "1.0.0",
        API_VERSION,
        "__init__:register",
        "A shareable RayChat plugin",
        {},
        {"greeting": "Hello"},
        instructions=(
            'Use {"action":"greet"} to get a greeting. '
            "The operator can use /greet. Calls are counted in session state."
        ),
    )
    atomic_json(path / "plugin.json", manifest.document())
    (path / "__init__.py").write_text(
        """from raychat.sdk import (
    CommandDefinition, PluginAPI, PluginContext, ToolDefinition,
)
from raychat.validation import integer_field, object_field, text_field


def register(api: PluginAPI) -> None:
    def validate(action: dict[str, object]) -> None:
        if set(action) != {'action'}:
            raise ValueError('No arguments are accepted.')

    def execute(_action: dict[str, object], ctx: PluginContext) -> dict[str, object]:
        raw_state: object = ctx.state
        state = object_field(raw_state, 'greeting state')
        calls = integer_field(state.get('calls', 0), 'calls', minimum=0) + 1
        state['calls'] = calls
        return {
            'ok': True,
            'message': text_field(ctx.settings['greeting'], 'greeting'),
            'calls': calls,
        }

    def greet(_arguments: str, ctx: PluginContext) -> str:
        return text_field(execute({'action': 'greet'}, ctx)['message'], 'greeting')

    api.register_tool(ToolDefinition(
        'greet', 'Return a greeting', validate, execute, requires_approval=False,
    ))
    api.register_command(CommandDefinition(
        'greet', greet, description='Greet the user', usage='/greet [name]',
    ))
""",
        encoding="utf-8",
    )
    (path / "README.md").write_text(
        f"# {path.name}\n\n"
        "Use `/greet` or the `greet` tool. Settings live under "
        f"`plugins.settings.{path.name}`.\n\n"
        "Check with `/plugins check PATH`, share with `/plugins pack PATH`, "
        "install with `/plugins install ZIP`. In the chat, run `/greet`, "
        "then ask the model to use the greeting tool. Change plugin.json "
        "instructions and the greeting implementation, then repeat without "
        "restarting.\n",
        encoding="utf-8",
    )
    return path
