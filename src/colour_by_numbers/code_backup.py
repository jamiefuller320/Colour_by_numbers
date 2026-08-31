"""Weekly off-repo code backups (tarball + optional S3 upload).

Uses the same credential names as the value_investor data backup:
``BACKUP_S3_URI``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
``AWS_DEFAULT_REGION``. Objects are isolated under
``s3://<bucket>/colour-by-numbers/code/`` unless ``BACKUP_S3_URI`` already
contains ``colour-by-numbers``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = Path("output/backups")
DEFAULT_S3_OBJECT_PREFIX = "colour-by-numbers/code"
ARCHIVE_PREFIX = "colour-by-numbers-code"
BUNDLE_ARCNAME = ".code-backup/repo.bundle"

EXCLUDE_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".eggs",
        ".tox",
        "output",
        "node_modules",
        ".cursor",
    }
)
EXCLUDE_FILE_SUFFIXES = (".pyc", ".pyo", ".egg")
EXCLUDE_FILE_NAMES = frozenset({".DS_Store"})


@dataclass
class BackupManifest:
    created_at: str
    kind: str
    archive_name: str
    commit: str
    branch: str
    paths: list[str]
    file_count: int
    bytes: int
    sha256: str
    has_git_bundle: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "kind": self.kind,
            "archive_name": self.archive_name,
            "commit": self.commit,
            "branch": self.branch,
            "paths": self.paths,
            "file_count": self.file_count,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "has_git_bundle": self.has_git_bundle,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackupManifest:
        return cls(
            created_at=str(data["created_at"]),
            kind=str(data.get("kind") or "code"),
            archive_name=str(data["archive_name"]),
            commit=str(data.get("commit") or ""),
            branch=str(data.get("branch") or ""),
            paths=list(data.get("paths") or []),
            file_count=int(data.get("file_count") or 0),
            bytes=int(data.get("bytes") or 0),
            sha256=str(data.get("sha256") or ""),
            has_git_bundle=bool(data.get("has_git_bundle")),
        )


@dataclass
class BackupSnapshot:
    archive_path: Path
    manifest_path: Path
    manifest: BackupManifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_path": str(self.archive_path),
            "manifest_path": str(self.manifest_path),
            "manifest": self.manifest.to_dict(),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_stamp(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _is_git_repo(repo_root: Path) -> bool:
    return (repo_root / ".git").exists()


def _git(repo_root: Path, *args: str) -> str | None:
    git = shutil.which("git")
    if git is None or not _is_git_repo(repo_root):
        return None
    try:
        result = subprocess.run(
            [git, "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_head(repo_root: Path) -> tuple[str, str]:
    commit = _git(repo_root, "rev-parse", "HEAD") or ""
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") or ""
    return commit, branch


def _tracked_files(repo_root: Path) -> list[Path] | None:
    listing = _git(repo_root, "ls-files", "-z")
    if listing is None:
        return None
    files: list[Path] = []
    for raw in listing.split("\0"):
        if not raw:
            continue
        path = repo_root / raw
        if path.is_file():
            files.append(path)
    return files


def _should_skip_dir(name: str) -> bool:
    return name in EXCLUDE_DIR_NAMES or name.endswith(".egg-info")


def _walk_source_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [name for name in dirnames if not _should_skip_dir(name)]
        for name in filenames:
            if name in EXCLUDE_FILE_NAMES or name.endswith(EXCLUDE_FILE_SUFFIXES):
                continue
            path = Path(dirpath) / name
            if path.is_file():
                files.append(path)
    return files


def collect_backup_files(repo_root: Path) -> list[Path]:
    """Tracked files when git is available; otherwise a filtered tree walk."""
    tracked = _tracked_files(repo_root)
    if tracked is not None:
        return sorted(tracked)
    return sorted(_walk_source_files(repo_root))


def _create_git_bundle(repo_root: Path, dest: Path) -> Path | None:
    git = shutil.which("git")
    if git is None or not _is_git_repo(repo_root):
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [git, "-C", str(repo_root), "bundle", "create", str(dest), "--all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("git bundle failed: %s", exc)
        return None
    return dest if dest.exists() and dest.stat().st_size > 0 else None


def create_code_backup_snapshot(
    *,
    repo_root: Path | None = None,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    now: datetime | None = None,
    include_git_bundle: bool = True,
) -> BackupSnapshot:
    """Create a gzip tarball of the repository source plus an optional git bundle."""
    repo_root = Path(repo_root or Path.cwd())
    sources = collect_backup_files(repo_root)
    if not sources:
        raise FileNotFoundError(f"No source files found to back up under {repo_root}")

    stamp = _archive_stamp(now)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    archive_path = backup_dir / f"{ARCHIVE_PREFIX}-{stamp}.tar.gz"
    manifest_path = backup_dir / f"{ARCHIVE_PREFIX}-{stamp}.manifest.json"

    bundle_path: Path | None = None
    if include_git_bundle:
        bundle_path = _create_git_bundle(
            repo_root, backup_dir / f"{ARCHIVE_PREFIX}-{stamp}.bundle"
        )

    file_count = 0
    with tarfile.open(archive_path, "w:gz") as tar:
        for source in sources:
            rel = source.relative_to(repo_root).as_posix()
            tar.add(source, arcname=rel, recursive=False)
            file_count += 1
        if bundle_path is not None:
            tar.add(bundle_path, arcname=BUNDLE_ARCNAME, recursive=False)
            file_count += 1

    commit, branch = _git_head(repo_root)
    digest = _sha256_file(archive_path)
    created = now or datetime.now(UTC)
    manifest = BackupManifest(
        created_at=created.isoformat(),
        kind="code",
        archive_name=archive_path.name,
        commit=commit,
        branch=branch,
        paths=_top_level_paths(sources, repo_root),
        file_count=file_count,
        bytes=archive_path.stat().st_size,
        sha256=digest,
        has_git_bundle=bundle_path is not None,
    )
    _write_json(manifest_path, manifest.to_dict())
    if bundle_path is not None:
        bundle_path.unlink(missing_ok=True)
    return BackupSnapshot(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _top_level_paths(sources: Iterable[Path], repo_root: Path) -> list[str]:
    tops: set[str] = set()
    for source in sources:
        rel = source.relative_to(repo_root).as_posix()
        tops.add(rel.split("/", 1)[0])
    return sorted(tops)


def verify_backup_snapshot(
    archive_path: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    archive_path = Path(archive_path)
    manifest_path = Path(
        manifest_path
        or archive_path.with_name(
            archive_path.name.replace(".tar.gz", ".manifest.json")
        )
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found for {archive_path}")
    manifest = BackupManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    actual = _sha256_file(archive_path)
    return {
        "ok": actual == manifest.sha256,
        "expected_sha256": manifest.sha256,
        "actual_sha256": actual,
        "manifest": manifest.to_dict(),
    }


def restore_backup_snapshot(
    archive_path: Path,
    *,
    dest_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Extract a code backup archive into ``dest_dir`` (default: cwd)."""
    dest_dir = Path(dest_dir or Path.cwd())
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    restored: list[str] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() and not member.isdir():
                continue
            if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                continue
            restored.append(member.name)
            if dry_run:
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(extracted.read())

    return {
        "archive": str(archive_path),
        "dest_dir": str(dest_dir),
        "dry_run": dry_run,
        "restored_paths": len(restored),
        "members": restored[:50],
    }


def list_local_snapshots(backup_dir: Path = DEFAULT_BACKUP_DIR) -> list[dict[str, Any]]:
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(
        backup_dir.glob(f"{ARCHIVE_PREFIX}-*.manifest.json"), reverse=True
    ):
        if "monthly" in manifest_path.name:
            continue
        try:
            manifest = BackupManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, TypeError, KeyError):
            continue
        archive_path = backup_dir / manifest.archive_name
        rows.append(
            {
                "manifest_path": str(manifest_path),
                "archive_path": str(archive_path),
                "archive_exists": archive_path.exists(),
                "created_at": manifest.created_at,
                "commit": manifest.commit,
                "bytes": manifest.bytes,
                "file_count": manifest.file_count,
            }
        )
    return rows


def resolve_backup_s3_base(s3_uri: str | None = None) -> str | None:
    """Return the S3 prefix for this repo, isolated from value_investor data."""
    raw = (s3_uri or os.environ.get("BACKUP_S3_URI") or "").strip()
    if not raw:
        return None
    raw = raw.rstrip("/")
    if "colour-by-numbers" in raw:
        return raw
    if raw.startswith("s3://"):
        rest = raw[5:]
        bucket = rest.split("/", 1)[0]
        if not bucket:
            return None
        return f"s3://{bucket}/{DEFAULT_S3_OBJECT_PREFIX}"
    return f"{raw}/{DEFAULT_S3_OBJECT_PREFIX}"


def _month_key_from_snapshot(snapshot: BackupSnapshot) -> str:
    raw = snapshot.manifest.created_at
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    created = datetime.fromisoformat(raw)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return created.strftime("%Y-%m")


def monthly_backup_dest_names(snapshot: BackupSnapshot) -> tuple[str, str]:
    month_key = _month_key_from_snapshot(snapshot)
    return (
        f"{ARCHIVE_PREFIX}-monthly-{month_key}.tar.gz",
        f"{ARCHIVE_PREFIX}-monthly-{month_key}.manifest.json",
    )


def _aws_cp(source: Path, dest: str) -> None:
    if not shutil.which("aws"):
        raise RuntimeError("aws CLI not found — install AWS CLI or upload artifact manually")
    subprocess.run(["aws", "s3", "cp", str(source), dest], check=True)


def upload_backup_snapshot(
    snapshot: BackupSnapshot,
    *,
    s3_uri: str | None = None,
) -> dict[str, Any]:
    """Upload archive + manifest to S3 using value_investor credential names."""
    base = resolve_backup_s3_base(s3_uri)
    if not base:
        return {"uploaded": False, "reason": "BACKUP_S3_URI not configured"}
    archive_dest = f"{base}/{snapshot.archive_path.name}"
    manifest_dest = f"{base}/{snapshot.manifest_path.name}"
    _aws_cp(snapshot.archive_path, archive_dest)
    _aws_cp(snapshot.manifest_path, manifest_dest)
    return {
        "uploaded": True,
        "archive_dest": archive_dest,
        "manifest_dest": manifest_dest,
    }


def upload_monthly_backup_pin(
    snapshot: BackupSnapshot,
    *,
    s3_uri: str | None = None,
) -> dict[str, Any]:
    """Overwrite the calendar-month pin under ``monthly/`` on S3."""
    base = resolve_backup_s3_base(s3_uri)
    if not base:
        return {"uploaded": False, "reason": "BACKUP_S3_URI not configured"}
    archive_name, manifest_name = monthly_backup_dest_names(snapshot)
    monthly_base = f"{base}/monthly"
    archive_dest = f"{monthly_base}/{archive_name}"
    manifest_dest = f"{monthly_base}/{manifest_name}"
    _aws_cp(snapshot.archive_path, archive_dest)
    _aws_cp(snapshot.manifest_path, manifest_dest)
    return {
        "uploaded": True,
        "month_key": _month_key_from_snapshot(snapshot),
        "archive_dest": archive_dest,
        "manifest_dest": manifest_dest,
    }


def snapshot_from_payload(data: dict[str, Any]) -> BackupSnapshot:
    manifest = BackupManifest.from_dict(data.get("manifest") or {})
    archive_path = Path(data.get("archive_path") or "")
    manifest_path = Path(data.get("manifest_path") or "")
    if not archive_path or not manifest_path:
        raise ValueError("payload missing archive_path or manifest_path")
    return BackupSnapshot(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def try_upload_backup_snapshot(
    snapshot: BackupSnapshot,
    *,
    s3_uri: str | None = None,
) -> dict[str, Any]:
    try:
        return upload_backup_snapshot(snapshot, s3_uri=s3_uri)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        logger.warning("Code backup S3 upload failed: %s", exc)
        return {
            "uploaded": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def try_upload_monthly_backup_pin(
    snapshot: BackupSnapshot,
    *,
    s3_uri: str | None = None,
) -> dict[str, Any]:
    try:
        return upload_monthly_backup_pin(snapshot, s3_uri=s3_uri)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        logger.warning("Monthly code backup S3 pin failed: %s", exc)
        return {
            "uploaded": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
