"""Tests for weekly code backup snapshots and S3 delivery."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "colour_by_numbers"


def _load(name: str, filename: str):
    path = _SRC / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_backup = _load("cbn_code_backup_under_test", "code_backup.py")
_cli = _load("cbn_code_backup_cli_under_test", "code_backup_cli.py")

ARCHIVE_PREFIX = _backup.ARCHIVE_PREFIX
BUNDLE_ARCNAME = _backup.BUNDLE_ARCNAME
collect_backup_files = _backup.collect_backup_files
create_code_backup_snapshot = _backup.create_code_backup_snapshot
list_local_snapshots = _backup.list_local_snapshots
monthly_backup_dest_names = _backup.monthly_backup_dest_names
resolve_backup_s3_base = _backup.resolve_backup_s3_base
restore_backup_snapshot = _backup.restore_backup_snapshot
snapshot_from_payload = _backup.snapshot_from_payload
try_upload_backup_snapshot = _backup.try_upload_backup_snapshot
try_upload_monthly_backup_pin = _backup.try_upload_monthly_backup_pin
upload_backup_snapshot = _backup.upload_backup_snapshot
upload_monthly_backup_pin = _backup.upload_monthly_backup_pin
verify_backup_snapshot = _backup.verify_backup_snapshot
main = _cli.main


def _seed_repo(root: Path) -> Path:
    (root / "src" / "colour_by_numbers").mkdir(parents=True)
    (root / "src" / "colour_by_numbers" / "code_backup.py").write_text(
        "print('backup')\n", encoding="utf-8"
    )
    (root / "README.md").write_text("# Colour by Numbers\n", encoding="utf-8")
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "junk.py").write_text("secret\n", encoding="utf-8")
    (root / "output" / "tmp").mkdir(parents=True)
    (root / "output" / "tmp" / "plate.png").write_bytes(b"png")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"pyc")
    return root


def test_walk_excludes_venv_output_and_caches(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path / "repo")
    files = {path.relative_to(repo).as_posix() for path in collect_backup_files(repo)}
    assert "src/colour_by_numbers/code_backup.py" in files
    assert "README.md" in files
    assert ".venv/lib/junk.py" not in files
    assert "output/tmp/plate.png" not in files
    assert "__pycache__/mod.cpython-312.pyc" not in files


def test_snapshot_verify_and_restore_roundtrip(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path / "repo")
    backup_dir = tmp_path / "backups"
    snapshot = create_code_backup_snapshot(
        repo_root=repo,
        backup_dir=backup_dir,
        include_git_bundle=False,
        now=datetime(2026, 8, 31, 13, 15, tzinfo=UTC),
    )
    assert snapshot.archive_path.exists()
    assert snapshot.manifest_path.exists()
    assert snapshot.archive_path.name.startswith(ARCHIVE_PREFIX)
    assert snapshot.manifest.kind == "code"
    assert snapshot.manifest.file_count >= 2
    assert "src" in snapshot.manifest.paths
    assert "README.md" in snapshot.manifest.paths

    verify = verify_backup_snapshot(snapshot.archive_path)
    assert verify["ok"] is True

    dest = tmp_path / "restored"
    result = restore_backup_snapshot(snapshot.archive_path, dest_dir=dest)
    assert result["restored_paths"] >= 2
    assert (dest / "src/colour_by_numbers/code_backup.py").read_text(
        encoding="utf-8"
    ) == "print('backup')\n"
    assert (dest / "README.md").exists()
    assert not (dest / ".venv").exists()
    assert not (dest / "output/tmp/plate.png").exists()


def test_restore_rejects_path_traversal(tmp_path: Path) -> None:
    import tarfile

    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="../escape.txt")
        payload = b"nope"
        info.size = len(payload)
        import io

        tar.addfile(info, io.BytesIO(payload))

    dest = tmp_path / "safe"
    dest.mkdir()
    result = restore_backup_snapshot(archive, dest_dir=dest)
    assert result["restored_paths"] == 0
    assert not (tmp_path / "escape.txt").exists()


def test_resolve_s3_base_uses_same_bucket_as_value_investor() -> None:
    assert resolve_backup_s3_base(None) is None
    assert (
        resolve_backup_s3_base("s3://my-bucket/ftse-value-investor/backups/")
        == "s3://my-bucket/colour-by-numbers/code"
    )
    assert (
        resolve_backup_s3_base("s3://my-bucket/colour-by-numbers/code/")
        == "s3://my-bucket/colour-by-numbers/code"
    )


def test_monthly_backup_dest_names_use_manifest_month(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path / "repo")
    snapshot = create_code_backup_snapshot(
        repo_root=repo,
        backup_dir=tmp_path / "backups",
        include_git_bundle=False,
        now=datetime(2026, 8, 31, 13, 15, tzinfo=UTC),
    )
    archive_name, manifest_name = monthly_backup_dest_names(snapshot)
    assert archive_name == f"{ARCHIVE_PREFIX}-monthly-2026-08.tar.gz"
    assert manifest_name == f"{ARCHIVE_PREFIX}-monthly-2026-08.manifest.json"


def test_upload_targets_isolated_prefix(monkeypatch, tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path / "repo")
    snapshot = create_code_backup_snapshot(
        repo_root=repo,
        backup_dir=tmp_path / "backups",
        include_git_bundle=False,
        now=datetime(2026, 8, 31, 13, 15, tzinfo=UTC),
    )
    calls: list[list[str]] = []

    monkeypatch.setenv("BACKUP_S3_URI", "s3://shared-bucket/ftse-value-investor/backups/")
    monkeypatch.setattr(
        _backup.shutil,
        "which",
        lambda name: "/usr/bin/aws" if name == "aws" else None,
    )
    monkeypatch.setattr(
        _backup.subprocess,
        "run",
        lambda cmd, check=True: calls.append(cmd),
    )

    result = upload_backup_snapshot(snapshot)
    assert result["uploaded"] is True
    assert result["archive_dest"].startswith("s3://shared-bucket/colour-by-numbers/code/")
    assert snapshot.archive_path.name in result["archive_dest"]
    assert len(calls) == 2
    assert calls[0][0:3] == ["aws", "s3", "cp"]

    monthly = upload_monthly_backup_pin(snapshot)
    assert monthly["uploaded"] is True
    assert monthly["month_key"] == "2026-08"
    assert monthly["archive_dest"] == (
        "s3://shared-bucket/colour-by-numbers/code/monthly/"
        f"{ARCHIVE_PREFIX}-monthly-2026-08.tar.gz"
    )


def test_try_upload_without_s3_uri_is_soft_skip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("BACKUP_S3_URI", raising=False)
    repo = _seed_repo(tmp_path / "repo")
    snapshot = create_code_backup_snapshot(
        repo_root=repo,
        backup_dir=tmp_path / "backups",
        include_git_bundle=False,
    )
    result = try_upload_backup_snapshot(snapshot)
    assert result["uploaded"] is False
    assert result.get("reason") == "BACKUP_S3_URI not configured"
    monthly = try_upload_monthly_backup_pin(snapshot)
    assert monthly["uploaded"] is False


def test_try_upload_records_aws_errors(monkeypatch, tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path / "repo")
    snapshot = create_code_backup_snapshot(
        repo_root=repo,
        backup_dir=tmp_path / "backups",
        include_git_bundle=False,
    )
    monkeypatch.setenv("BACKUP_S3_URI", "s3://bucket/prefix")
    monkeypatch.setattr(_backup.shutil, "which", lambda name: None)
    result = try_upload_backup_snapshot(snapshot)
    assert result["uploaded"] is False
    assert "aws CLI not found" in result["error"]


def test_snapshot_from_payload_and_list(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path / "repo")
    snapshot = create_code_backup_snapshot(
        repo_root=repo,
        backup_dir=tmp_path / "backups",
        include_git_bundle=False,
    )
    restored = snapshot_from_payload(snapshot.to_dict())
    assert restored.archive_path == snapshot.archive_path
    assert restored.manifest.sha256 == snapshot.manifest.sha256
    rows = list_local_snapshots(tmp_path / "backups")
    assert len(rows) == 1
    assert rows[0]["archive_exists"] is True


def test_cli_snapshot_and_deliver(monkeypatch, tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path / "repo")
    backup_dir = tmp_path / "backups"
    rc = main(
        [
            "snapshot",
            "--json",
            "--no-git-bundle",
            "--repo-root",
            str(repo),
            "--backup-dir",
            str(backup_dir),
        ]
    )
    assert rc == 0
    snapshots = list(backup_dir.glob("*.tar.gz"))
    assert len(snapshots) == 1

    payload_path = tmp_path / "backup.json"
    payload_path.write_text(
        json.dumps(
            {
                "archive_path": str(snapshots[0]),
                "manifest_path": str(snapshots[0].with_name(
                    snapshots[0].name.replace(".tar.gz", ".manifest.json")
                )),
                "manifest": json.loads(
                    snapshots[0]
                    .with_name(snapshots[0].name.replace(".tar.gz", ".manifest.json"))
                    .read_text(encoding="utf-8")
                ),
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []
    monkeypatch.setenv("BACKUP_S3_URI", "s3://bucket/ftse/")
    monkeypatch.setattr(
        _cli._backup.shutil,
        "which",
        lambda name: "/usr/bin/aws" if name == "aws" else None,
    )
    monkeypatch.setattr(
        _cli._backup.subprocess,
        "run",
        lambda cmd, check=True: calls.append(cmd),
    )
    rc = main(
        ["deliver", "--from-json", str(payload_path), "--upload", "--upload-monthly", "--json"]
    )
    assert rc == 0
    saved = json.loads(payload_path.read_text(encoding="utf-8"))
    assert saved["upload"]["uploaded"] is True
    assert saved["upload_monthly"]["uploaded"] is True
    assert "colour-by-numbers/code" in saved["upload"]["archive_dest"]
    assert len(calls) == 4


def test_cli_verify_restore_and_list(tmp_path: Path, capsys) -> None:
    repo = _seed_repo(tmp_path / "repo")
    backup_dir = tmp_path / "backups"
    snapshot = create_code_backup_snapshot(
        repo_root=repo,
        backup_dir=backup_dir,
        include_git_bundle=False,
    )
    assert main(["verify", str(snapshot.archive_path)]) == 0
    dest = tmp_path / "out"
    assert main(["restore", str(snapshot.archive_path), "--dest", str(dest)]) == 0
    assert (dest / "README.md").exists()
    assert main(["list", "--backup-dir", str(backup_dir)]) == 0
    assert "bytes" in capsys.readouterr().out


def test_git_bundle_embedded_when_repo_available(tmp_path: Path) -> None:
    import subprocess
    import shutil

    if shutil.which("git") is None:
        return
    repo = _seed_repo(tmp_path / "repo")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "README.md", "src"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=repo, check=True, capture_output=True)

    snapshot = create_code_backup_snapshot(
        repo_root=repo,
        backup_dir=tmp_path / "backups",
        include_git_bundle=True,
    )
    assert snapshot.manifest.has_git_bundle is True
    assert snapshot.manifest.commit
    dest = tmp_path / "from-bundle"
    restore_backup_snapshot(snapshot.archive_path, dest_dir=dest)
    assert (dest / BUNDLE_ARCNAME).exists()
    assert (dest / BUNDLE_ARCNAME).stat().st_size > 0
