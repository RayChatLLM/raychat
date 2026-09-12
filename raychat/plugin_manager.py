"""Package installation and catalogs. One service for CLI, UI and agent tools."""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Any, TypedDict, cast
from urllib.parse import ParseResult, quote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from raychat.event_types import CONFIGURE, Lifecycle
from raychat.type_support import override

from .distribution import Distribution
from .file_lock import FileLock
from .packages import (
    MAX_BYTES,
    NAME,
    VERSION,
    Manifest,
    dependency_order,
    digest,
    discover,
    files,
    pack,
    read_manifest,
    unpack,
)
from .plugins import Runtime
from .sdk import API_VERSION, CancelCheck, PluginContext, PluginError
from .validation import json_object, text_field

_DOWNLOAD_CANCEL: contextvars.ContextVar[CancelCheck | None] = contextvars.ContextVar(
    "package_download_cancel", default=None
)


class InstalledRelease(TypedDict):
    source: str
    version: str
    digest: str
    linked: bool


class PackageRecord(InstalledRelease, total=False):
    path: str
    resolved: str
    archive_sha256: str
    catalog: str


class InstallationState(TypedDict):
    schema: int
    packages: dict[str, PackageRecord]
    disabled: list[str]
    catalogs: dict[str, str]


class PackageState(InstallationState, total=False):
    profiles: list[str]


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".plugins-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path: str | Path, default: object) -> Any:  # noqa: ANN401 - JSON file schemas are validated by the consuming operation
    path = Path(path)
    if not path.exists():
        return default
    if path.is_symlink() or not path.is_file():
        raise PluginError("Expected a regular plugin state file: " + str(path))
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        error_message = "Plugin state exceeds its size limit."
        raise PluginError(error_message)
    return json_object(data)


def _validate_url(url: str) -> ParseResult:
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or (
            parsed.scheme == "http"
            and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        )
    ):
        error_message = "Remote plugins and catalogs require HTTPS."
        raise PluginError(error_message)
    return parsed


class PackageRedirects(HTTPRedirectHandler):
    @override
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        parsed = _validate_url(newurl)
        if urlparse(req.full_url).scheme == "https" and parsed.scheme != "https":
            error_message = "HTTPS downloads cannot redirect to HTTP."
            raise PluginError(error_message)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str) -> bytes:
    _validate_url(url)
    cancel_check = _DOWNLOAD_CANCEL.get()
    if cancel_check is not None:
        from .transport import ProviderProcessError, run_child

        cancel_check()
        with tempfile.TemporaryDirectory(
            prefix="raychat-package-download-"
        ) as directory:
            destination = Path(directory) / "response"
            try:
                count = run_child(
                    None,
                    {
                        "mode": "package_download",
                        "url": url,
                        "destination": str(destination),
                    },
                    cancel_check,
                )
            except ProviderProcessError as exc:
                if exc.os_error:
                    raise OSError(str(exc)) from exc
                raise PluginError(str(exc)) from exc
            cancel_check()
            data = read_bytes(destination)
            if count != str(len(data)):
                raise PluginError("Downloaded package size does not match its receipt.")
            return data
    with build_opener(PackageRedirects()).open(
        Request(url, headers={"User-Agent": "RayChat-plugins/2"}),  # noqa: S310 - HTTP(S) endpoint validated before opening the request
        timeout=30,
    ) as response:
        data = response.read(MAX_BYTES + 1)
    if not isinstance(data, bytes):
        error_message = "Download did not return bytes."
        raise PluginError(error_message)
    if len(data) > MAX_BYTES:
        error_message = "Download exceeds the package byte limit."
        raise PluginError(error_message)
    return data


def read_bytes(path: str | Path) -> bytes:
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        error_message = "Package input exceeds its byte limit."
        raise PluginError(error_message)
    return data


class PackageManager:
    @contextmanager
    def cancellable(self, check: CancelCheck) -> Iterator[None]:
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
    ) -> None:
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
        self._states = {scope: self._read(scope) for scope in self.roots}

    def _scope_lock(self, scope: str) -> FileLock:
        return FileLock(self.state_roots[scope] / "plugins.mutex")

    def state_file(self, scope: str) -> Path:
        return self.state_roots[scope] / "plugins.lock.json"

    def _read(self, scope: str) -> PackageState:
        value = read_json(
            self.state_file(scope),
            {"schema": 1, "packages": {}, "disabled": [], "catalogs": {}},
        )
        if (
            not isinstance(value, dict)
            or value.get("schema") != 1
            or not isinstance(value.get("packages"), dict)
            or not isinstance(value.get("catalogs"), dict)
            or not isinstance(value.get("disabled"), list)
        ):
            error_message = "Invalid plugin installation lock file."
            raise PluginError(error_message)
        for identifier, item in value["packages"].items():
            if (
                not isinstance(identifier, str)
                or not NAME.fullmatch(identifier)
                or not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not Path(item["path"]).is_absolute()
                or type(item.get("linked")) is not bool
                or not isinstance(item.get("source"), str)
                or not isinstance(item.get("version"), str)
                or not VERSION.fullmatch(item["version"])
                or not isinstance(item.get("digest"), str)
                or len(item["digest"]) != 64
                or (
                    "catalog" in item
                    and (
                        not isinstance(item["catalog"], str)
                        or not NAME.fullmatch(item["catalog"])
                    )
                )
            ):
                error_message = "Invalid installed package record."
                raise PluginError(error_message)
            if (
                not item["linked"]
                and Path(item["path"]) != self.roots[scope] / "plugins" / identifier
            ):
                error_message = "Installed package path is outside its owned directory."
                raise PluginError(
                    error_message,
                )
        if not all(
            isinstance(v, str) and NAME.fullmatch(v) for v in value["disabled"]
        ) or not all(
            isinstance(k, str) and NAME.fullmatch(k) and isinstance(v, str)
            for k, v in value["catalogs"].items()
        ):
            error_message = "Invalid disabled plugin or catalog entry."
            raise PluginError(error_message)
        return cast("PackageState", value)

    def paths(self, *, include_disabled: bool = False) -> dict[str, Path]:
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
        return {name for state in self._states.values() for name in state["disabled"]}

    def attach(self, runtime: Runtime) -> None:
        self.runtime = runtime
        runtime.services["plugin_manager"] = self
        runtime.disabled.update(self.disabled)

    def new_paths(self) -> list[str]:
        if self.runtime is None:
            error_message = "Plugin discovery requires an attached runtime."
            raise PluginError(error_message)
        known = {
            snapshot["path"] for snapshot in self.runtime.export_sources()["packages"]
        }
        return [str(path) for path in self.paths().values() if str(path) not in known]

    def inventory(self) -> list[dict[str, Any]]:
        result = []
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
        """
        if self.runtime is not None:
            error_message = "Install a startup profile before attaching a runtime."
            raise PluginError(error_message)
        scope = "user"
        state = self._states[scope]
        catalog_name = profile.id
        previous_catalog = state["catalogs"].get(catalog_name)
        previous = (
            read_json(self._catalog_cache(previous_catalog), {})
            if previous_catalog is not None
            else {}
        )
        previous_records = previous.get("plugins", [])
        if not isinstance(previous_records, list):
            error_message = "Invalid cached profile catalog."
            raise PluginError(error_message)
        # Earlier dependency receipts used unqualified names. Only an exact
        # archived catalog receipt establishes their origin; IDs alone do not.
        prior_releases = {
            (
                item.get("id"),
                item.get("version"),
                self._catalog_url(previous_catalog, item["url"]),
                item.get("sha256"),
            )
            for item in previous_records
            if previous_catalog is not None
            and isinstance(item, dict)
            and isinstance(item.get("url"), str)
        }
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
            with self._scope_lock(scope):
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
        if sources:
            self._install_sources(sources, scope=scope, preserve_disabled=True)
        with self._scope_lock(scope):
            state = self._read(scope)
            profiles = state.setdefault("profiles", [])
            if profile.id not in profiles:
                profiles.append(profile.id)
                atomic_json(self.state_file(scope), state)
                self._states[scope] = state

    def catalogs(self) -> dict[str, str]:
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
        if operation == "list":
            return self.catalogs()
        if not NAME.fullmatch(name):
            error_message = "Catalog name must be an identifier."
            raise PluginError(error_message)
        state = copy.deepcopy(self._states[scope])
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
        with self._scope_lock(scope):
            if self._read(scope) != self._states[scope]:
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
        if urlparse(location).scheme:
            return location
        return (
            urljoin(catalog, location)
            if urlparse(catalog).scheme
            else str(Path(catalog).parent / location)
        )

    def _catalog_records(self, url: str) -> list[dict[str, Any]]:
        cache = self._catalog_cache(url)
        try:
            data = download(url) if urlparse(url).scheme else read_bytes(url)
            value = json_object(data)
            self._stale_catalogs.discard(url)
        except OSError:
            self._stale_catalogs.add(url)
            value = read_json(cache, None)
            if value is None:
                raise
        if (
            not isinstance(value, dict)
            or value.get("schema") != 1
            or not isinstance(value.get("plugins"), list)
        ):
            error_message = "Catalog must contain schema=1 and a plugins array."
            raise PluginError(error_message)
        versions = set()
        for item in value["plugins"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("url"), str)
                or not isinstance(item.get("sha256"), str)
            ):
                error_message = "Catalog package requires a URL and SHA-256 digest."
                raise PluginError(error_message)
            if len(item["sha256"]) != 64 or any(
                c not in "0123456789abcdef" for c in item["sha256"]
            ):
                error_message = "Invalid catalog SHA-256 digest."
                raise PluginError(error_message)
            manifest = Manifest.parse(
                {k: v for k, v in item.items() if k not in {"url", "sha256"}},
            )
            key = (manifest.id, manifest.version)
            if key in versions:
                raise PluginError(
                    "Duplicate catalog package version: "
                    + manifest.id
                    + "@"
                    + manifest.version,
                )
            versions.add(key)
        atomic_json(cache, value)
        return cast("list[dict[str, Any]]", value["plugins"])

    def search(self, query: str = "") -> list[dict[str, Any]]:
        results = []
        for name, url in self.catalogs().items():
            for item in self._catalog_records(url):
                if (
                    query.casefold()
                    in (item["id"] + " " + item["description"]).casefold()
                ):
                    resolved = dict(item)
                    resolved["url"] = self._catalog_url(url, item["url"])
                    results.append(
                        {
                            **resolved,
                            "catalog": name,
                            "cached": url in self._stale_catalogs,
                        },
                    )
        return sorted(
            results,
            key=lambda item: (item["id"], tuple(map(int, item["version"].split(".")))),
            reverse=True,
        )

    def resolve_source(self, source: str) -> tuple[str, dict[str, object] | None]:
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
            if len(owner_repo) != 2 or not all(NAME.fullmatch(v) for v in owner_repo):
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
            commit = (
                commit_record.get("sha") if isinstance(commit_record, dict) else None
            )
            if (
                not isinstance(commit, str)
                or len(commit) != 40
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
        if not urlparse(resolved).scheme and path.is_dir():
            members = files(path)
            for name, data in members.items():
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            archive_hash = hashlib.sha256(pack(path)).hexdigest()
        else:
            data = download(resolved) if urlparse(resolved).scheme else read_bytes(path)
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
    ) -> dict[str, Any]:
        return self._install_sources(
            [source],
            scope=scope,
            force=force,
            linked=linked,
            ctx=ctx,
        )

    def _install_sources(
        self,
        sources: list[str],
        *,
        scope: str,
        force: bool = False,
        linked: bool = False,
        ctx: PluginContext | None = None,
        preserve_disabled: bool = False,
    ) -> dict[str, Any]:
        temporary = tempfile.TemporaryDirectory(prefix="raychat-install-")
        staging = Path(temporary.name)
        staged: dict[str, Path] = {}
        records: dict[str, PackageRecord] = {}
        manifests: dict[str, Manifest] = {}
        available = self.paths(include_disabled=True)
        visiting: set[str] = set()
        complete: set[str] = set()
        try:

            def stage(spec: str, expected: tuple[str, str] | None = None) -> Manifest:
                target = staging / ("package-" + str(len(staged)))
                manifest, record = self._stage(spec, target)
                if expected is not None and (manifest.id, manifest.version) != expected:
                    error_message = "Dependency resolved to an unexpected version."
                    raise PluginError(error_message)
                if preserve_disabled and manifest.id in self.disabled:
                    raise PluginError(
                        "Profile requires a disabled or removed plugin; enable or "
                        "install it explicitly first: " + manifest.id,
                    )
                if manifest.id in staged:
                    if records[manifest.id]["version"] != manifest.version:
                        raise PluginError(
                            "Conflicting dependency versions: " + manifest.id,
                        )
                    return manifest
                staged[manifest.id], records[manifest.id] = target, record
                manifests[manifest.id] = manifest
                return manifest

            def collect(identifier: str) -> None:
                if identifier in complete:
                    return
                if identifier in visiting:
                    raise PluginError("Plugin dependency cycle: " + identifier)
                visiting.add(identifier)
                manifest = manifests[identifier]
                for dependency, version in manifest.requires.items():
                    if dependency in staged:
                        if manifests[dependency].version != version:
                            error_message = f"Dependency conflict: {identifier} requires {dependency}@{version}."
                            raise PluginError(
                                error_message,
                            )
                        collect(dependency)
                    elif dependency in available:
                        if read_manifest(available[dependency]).version != version:
                            error_message = f"Dependency conflict: {manifest.id} requires {dependency}@{version}."
                            raise PluginError(
                                error_message,
                            )
                    else:
                        catalog = records[identifier].get("catalog")
                        spec = (
                            (catalog + "/" if catalog else "")
                            + dependency
                            + "@"
                            + version
                        )
                        stage(spec, (dependency, version))
                        collect(dependency)
                visiting.remove(identifier)
                complete.add(identifier)

            primary = stage(sources[0])
            # Stage every requested root before resolving dependencies so the
            # transaction sees the complete profile, including version upgrades.
            for source in sources[1:]:
                stage(source)
            for identifier in list(staged):
                collect(identifier)
            expected_sources = {}
            for identifier in staged:
                old = next(
                    (item for item in self.inventory() if item["id"] == identifier),
                    None,
                )
                if old and old["scope"] != scope:
                    raise PluginError(
                        "Plugin ID already exists in another scope: " + identifier,
                    )
                if old:
                    expected_sources[old["path"]] = digest(
                        files(old["path"], validate_manifest=False),
                    )
                if old and old["modified"] and not force:
                    raise PluginError(
                        "Plugin has local edits; use --force to replace: " + identifier,
                    )
            if linked:
                if (
                    len(staged) != 1
                    or len(sources) != 1
                    or not Path(sources[0]).expanduser().is_dir()
                ):
                    error_message = (
                        "link requires a local package with installed dependencies."
                    )
                    raise PluginError(
                        error_message,
                    )
                records[primary.id]["linked"] = True
                records[primary.id]["path"] = str(
                    Path(sources[0]).expanduser().resolve(),
                )
            return self._transaction(
                scope,
                staged,
                records,
                [],
                temporary,
                ctx,
                expected_sources,
            )
        except BaseException:
            temporary.cleanup()
            raise

    def _transaction(
        self,
        scope: str,
        staged: Mapping[str, Path],
        records: Mapping[str, PackageRecord],
        removed: list[str],
        temporary: tempfile.TemporaryDirectory[str],
        ctx: PluginContext | None,
        expected_sources: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        state = copy.deepcopy(self._states[scope])
        before = self._read(scope)
        backups: dict[Path, Path] = {}
        written: list[Path] = []
        base = self.roots[scope]
        for identifier, record in records.items():
            record.setdefault("path", str(base / "plugins" / identifier))
            state["packages"][identifier] = record
            state["disabled"] = [v for v in state["disabled"] if v != identifier]
        for identifier in removed:
            state["packages"].pop(identifier, None)
            if identifier not in state["disabled"]:
                state["disabled"].append(identifier)

        held: list[FileLock] = []

        def release() -> None:
            while held:
                held.pop().close()

        def prepare() -> None:
            lock = self._scope_lock(scope)
            lock.acquire()
            held.append(lock)
            if self._read(scope) != before:
                error_message = "Installation state changed; retry the operation."
                raise PluginError(error_message)
            for path, expected in (expected_sources or {}).items():
                if digest(files(path, validate_manifest=False)) != expected:
                    raise PluginError(
                        "Installed source changed while the update was queued: " + path,
                    )
            for identifier, source in staged.items():
                record = records[identifier]
                if record["linked"]:
                    continue
                target = Path(record["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    backup = target.parent / (
                        ".backup-" + identifier + "-" + os.urandom(8).hex()
                    )
                    backups[target] = backup
                    os.replace(target, backup)
                written.append(target)
                incoming = target.parent / (
                    ".incoming-" + identifier + "-" + os.urandom(8).hex()
                )
                try:
                    shutil.copytree(source, incoming)
                    os.replace(incoming, target)
                finally:
                    shutil.rmtree(incoming, ignore_errors=True)
            atomic_json(self.state_file(scope), state)

        def rollback(error: BaseException) -> None:
            try:
                if not held:
                    return
                for target in reversed(written):
                    shutil.rmtree(target, ignore_errors=True)
                    if target in backups:
                        os.replace(backups[target], target)
                if self._read(scope) == state:
                    atomic_json(self.state_file(scope), before)
            finally:
                try:
                    temporary.cleanup()
                finally:
                    release()

        def commit() -> None:
            try:
                self._states[scope] = state
                for backup in backups.values():
                    shutil.rmtree(backup, ignore_errors=True)
                if self.runtime is not None:
                    self.runtime.disabled.update(self.disabled)
                for identifier in removed:
                    old = before["packages"].get(identifier)
                    if old and not old.get("linked"):
                        path = Path(old["path"])
                        if path.parent == base / "plugins":
                            shutil.rmtree(path, ignore_errors=True)
            finally:
                try:
                    temporary.cleanup()
                finally:
                    release()

        added = [record["path"] for record in records.values()]
        if self.runtime is None:
            try:
                prepare()
                # Validate the complete installed dependency graph before commit.
                paths = self.paths(include_disabled=True)
                paths.update(
                    {name: Path(item["path"]) for name, item in records.items()},
                )
                for name in removed:
                    paths.pop(name, None)
                dependency_order(
                    {
                        name: read_manifest(path, require_current_sdk=False)
                        for name, path in paths.items()
                    },
                )
                commit()
            except BaseException as exc:
                rollback(exc)
                raise
            return {"applied": True, "packages": list(records), "removed": removed}
        removed_active = [
            identifier
            for identifier in [*removed, *records]
            if identifier in self.runtime.plugins
        ]
        if ctx is None:
            ctx = self.runtime.context("plugin_manager")
        applied = ctx.update_plugins(
            add=added,
            remove=removed_active,
            prepare=prepare,
            commit=commit,
            rollback=rollback,
        )
        return {"applied": applied, "packages": list(records), "removed": removed}

    def set_enabled(
        self,
        identifier: str,
        enabled: bool,
        *,
        scope: str = "workspace",
        ctx: PluginContext | None = None,
    ) -> bool:
        runtime = self.runtime
        if runtime is None:
            error_message = "Plugin activation requires an attached runtime."
            raise PluginError(error_message)
        paths = self.paths(include_disabled=True)
        if identifier not in paths:
            raise PluginError("Unknown plugin: " + identifier)
        if enabled and any(
            identifier in v["disabled"] for k, v in self._states.items() if k != scope
        ):
            error_message = (
                "Plugin is disabled in another scope; enable it there first."
            )
            raise PluginError(
                error_message,
            )
        before = self._read(scope)
        state = copy.deepcopy(before)
        state["disabled"] = [v for v in state["disabled"] if v != identifier]
        if not enabled:
            state["disabled"].append(identifier)
        held: list[FileLock] = []

        def release() -> None:
            while held:
                held.pop().close()

        def prepare() -> None:
            lock = self._scope_lock(scope)
            lock.acquire()
            held.append(lock)
            if self._read(scope) != before:
                error_message = "Plugin state changed; retry."
                raise PluginError(error_message)
            atomic_json(self.state_file(scope), state)

        def commit() -> None:
            self._states[scope] = state
            if enabled:
                runtime.disabled.discard(identifier)
            else:
                runtime.disabled.add(identifier)
            release()

        def rollback(error: BaseException) -> None:
            if not held:
                return
            try:
                if self._read(scope) == state:
                    atomic_json(self.state_file(scope), before)
            finally:
                release()

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

    def uninstall(
        self,
        identifier: str,
        *,
        scope: str = "workspace",
        ctx: PluginContext | None = None,
    ) -> dict[str, Any]:
        if identifier not in self._states[scope]["packages"]:
            raise PluginError("Plugin is not installed in this scope: " + identifier)
        temporary = tempfile.TemporaryDirectory(prefix="raychat-uninstall-")
        return self._transaction(scope, {}, {}, [identifier], temporary, ctx)

    def update(
        self,
        identifier: str,
        *,
        scope: str = "workspace",
        force: bool = False,
        ctx: PluginContext | None = None,
    ) -> dict[str, Any]:
        record = self._states[scope]["packages"].get(identifier)
        if record is None or record.get("linked"):
            error_message = "Only installed release packages can be updated."
            raise PluginError(error_message)
        source = record["source"]
        if not urlparse(source).scheme and "@" in source and not Path(source).exists():
            source = source.rsplit("@", 1)[0]
        return self.install(source, scope=scope, force=force, ctx=ctx)

    def check(self, path: str | Path) -> dict[str, Any]:
        from .plugin_sources import SourceTree

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
        runtime = Runtime(self.workspace)
        trees: list[SourceTree] = []
        try:
            modules = []
            for name in ordered:
                tree = SourceTree(available[name])
                trees.append(tree)
                modules.append(tree.entrypoint())
            runtime.load(modules)
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


def scaffold(path: str | Path) -> Path:
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
        instructions='Use {"action":"greet"} to get a greeting. The operator can use /greet. Calls are counted in session state.',
    )
    atomic_json(path / "plugin.json", manifest.document())
    (path / "__init__.py").write_text(
        """from raychat.sdk import Action, CommandDefinition, PluginAPI, PluginContext, ToolDefinition


def register(api: PluginAPI) -> None:
    def validate(action: Action) -> None:
        if set(action) != {'action'}:
            raise ValueError('No arguments are accepted.')

    def execute(action: Action, ctx: PluginContext) -> Action:
        ctx.state['calls'] = ctx.state.get('calls', 0) + 1
        return {'ok': True, 'message': ctx.settings['greeting'], 'calls': ctx.state['calls']}

    def greet(arguments: str, ctx: PluginContext) -> str:
        return str(execute({'action': 'greet'}, ctx)['message'])

    api.register_tool(ToolDefinition('greet', 'Return a greeting', validate, execute, False))
    api.register_command(CommandDefinition('greet', greet, description='Greet the user', usage='/greet [name]'))
""",
        encoding="utf-8",
    )
    (path / "README.md").write_text(
        f"# {path.name}\n\nUse `/greet` or the `greet` tool. Settings live under `plugins.settings.{path.name}`.\n\nCheck with `/plugins check PATH`, share with `/plugins pack PATH`, install with `/plugins install ZIP`. In the chat, run `/greet`, then ask the model to use the greeting tool. Change plugin.json instructions and the greeting implementation, then repeat without restarting.\n",
        encoding="utf-8",
    )
    return path
