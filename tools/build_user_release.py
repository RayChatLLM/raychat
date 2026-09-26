"""Build the small public download around the verified source distribution."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from raychat.filesystem import write_bytes
from tools import build_portable


class _Arguments(argparse.Namespace):
    output: Path
    check: bool


def release_version(root: Path) -> str:
    """Read the explicit stable release version.

    Returns
    -------
    str
        A three-component, non-prefixed semantic version.

    Raises
    ------
    ValueError
        The version file is not an ordinary stable version.

    """
    version = (root / "release-version.txt").read_text(encoding="utf-8").strip()
    if (
        re.fullmatch(
            r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)",
            version,
        )
        is None
    ):
        message = "release-version.txt must contain a stable version such as 0.1.0."
        raise ValueError(message)
    return version


def build(root: Path) -> tuple[str, bytes]:
    """Wrap the complete allowlisted source without removing recovery support.

    Returns
    -------
    tuple[str, bytes]
        Version and deterministic, platform-independent ZIP bytes.

    """
    version = release_version(root)
    _, sources = build_portable.build_archive(root)
    members = {"_raychat/" + name: data for name, data in sources.items()}
    members["raychat"] = sources["tools/release_launcher.py"]
    members["README.md"] = sources["docs/RELEASE_README.md"]
    members.update({
        "plugins/" + name.removeprefix("plugin_catalog/"): data
        for name, data in sources.items()
        if name.startswith("plugin_catalog/")
    })
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(
                f"raychat-v{version}/{name}",
                build_portable.FIXED_TIMESTAMP,
            )
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return version, buffer.getvalue()


def main() -> int:
    """Write or independently reproduce the user ZIP and its checksum.

    Returns
    -------
    int
        Zero after the complete archive has been built or matched.

    Raises
    ------
    ValueError
        An existing archive does not match a fresh deterministic build.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(namespace=_Arguments())
    version, raw = build(Path(__file__).resolve().parents[1])
    name = f"raychat-v{version}.zip"
    destination = args.output / name
    digest = hashlib.sha256(raw).hexdigest()
    checksums = f"{digest}  {name}\n".encode()
    if args.check:
        if (
            destination.read_bytes() != raw
            or (args.output / "SHA256SUMS").read_bytes() != checksums
        ):
            message = "The user release differs from an independent build."
            raise ValueError(message)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        write_bytes(destination, raw, mode=0o644)
        write_bytes(args.output / "SHA256SUMS", checksums, mode=0o644)
    report: dict[str, str] = {
        "version": version,
        "sha256": digest,
        "archive": str(destination),
    }
    sys.stdout.write(
        json.dumps(report) + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
