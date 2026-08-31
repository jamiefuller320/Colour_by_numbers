"""CLI for weekly code backups and S3 delivery.

Loads ``code_backup.py`` as a sibling file so CI can run this script without
installing the full colour-by-numbers stack (Pillow, rembg, …).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_backup_module():
    path = Path(__file__).resolve().parent / "code_backup.py"
    spec = importlib.util.spec_from_file_location("cbn_code_backup_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cbn_code_backup_impl"] = module
    spec.loader.exec_module(module)
    return module


_backup = _load_backup_module()
DEFAULT_BACKUP_DIR = _backup.DEFAULT_BACKUP_DIR
create_code_backup_snapshot = _backup.create_code_backup_snapshot
list_local_snapshots = _backup.list_local_snapshots
restore_backup_snapshot = _backup.restore_backup_snapshot
snapshot_from_payload = _backup.snapshot_from_payload
try_upload_backup_snapshot = _backup.try_upload_backup_snapshot
try_upload_monthly_backup_pin = _backup.try_upload_monthly_backup_pin
upload_backup_snapshot = _backup.upload_backup_snapshot
upload_monthly_backup_pin = _backup.upload_monthly_backup_pin
verify_backup_snapshot = _backup.verify_backup_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot this repository and upload to S3 using the same "
            "BACKUP_S3_URI / AWS_* credentials as value_investor"
        ),
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true")
    common.add_argument("--repo-root", type=Path, default=Path.cwd())
    common.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser(
        "snapshot",
        parents=[common],
        help="Create a code tarball + manifest",
    )
    snap.add_argument("--upload", action="store_true", help="Upload when BACKUP_S3_URI is set")
    snap.add_argument(
        "--upload-monthly",
        action="store_true",
        help="Pin snapshot to monthly/ key on S3 (overwrite same calendar month)",
    )
    snap.add_argument(
        "--no-git-bundle",
        action="store_true",
        help="Skip embedding a git bundle --all in the archive",
    )
    snap.add_argument(
        "--strict-upload",
        action="store_true",
        help="Exit non-zero when --upload is set but upload fails",
    )
    snap.set_defaults(func=_cmd_snapshot)

    deliver = sub.add_parser(
        "deliver",
        parents=[common],
        help="Upload an existing snapshot described by snapshot --json output",
    )
    deliver.add_argument(
        "--from-json",
        type=Path,
        required=True,
        help="Path to snapshot JSON (updated in place when --upload runs)",
    )
    deliver.add_argument("--upload", action="store_true")
    deliver.add_argument("--upload-monthly", action="store_true")
    deliver.add_argument("--strict-upload", action="store_true")
    deliver.set_defaults(func=_cmd_deliver)

    sub.add_parser(
        "list",
        parents=[common],
        help="List local snapshots under output/backups",
    ).set_defaults(func=_cmd_list)

    verify = sub.add_parser(
        "verify",
        parents=[common],
        help="Verify archive checksum against manifest",
    )
    verify.add_argument("archive", type=Path)
    verify.add_argument("--manifest", type=Path, default=None)
    verify.set_defaults(func=_cmd_verify)

    restore = sub.add_parser(
        "restore",
        parents=[common],
        help="Extract archive into a directory",
    )
    restore.add_argument("archive", type=Path)
    restore.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Directory to extract into (default: --repo-root)",
    )
    restore.add_argument("--dry-run", action="store_true")
    restore.set_defaults(func=_cmd_restore)

    args = parser.parse_args(argv)
    return int(args.func(args))


def _upload_snapshot(snapshot, *, strict: bool):
    if strict:
        try:
            result = upload_backup_snapshot(snapshot)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return None, 1
    else:
        result = try_upload_backup_snapshot(snapshot)
    if strict and not result.get("uploaded") and result.get("error"):
        print(str(result.get("error")), file=sys.stderr)
        return result, 1
    return result, 0


def _upload_monthly(snapshot, *, strict: bool):
    if strict:
        try:
            result = upload_monthly_backup_pin(snapshot)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return None, 1
    else:
        result = try_upload_monthly_backup_pin(snapshot)
    if strict and not result.get("uploaded") and result.get("error"):
        print(str(result.get("error")), file=sys.stderr)
        return result, 1
    return result, 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    snapshot = create_code_backup_snapshot(
        repo_root=args.repo_root,
        backup_dir=args.backup_dir,
        include_git_bundle=not args.no_git_bundle,
    )
    upload_result = None
    if args.upload:
        upload_result, code = _upload_snapshot(snapshot, strict=args.strict_upload)
        if code != 0:
            return code
    monthly_result = None
    if args.upload_monthly:
        monthly_result, code = _upload_monthly(snapshot, strict=args.strict_upload)
        if code != 0:
            return code
    payload = snapshot.to_dict()
    if upload_result is not None:
        payload["upload"] = upload_result
    if monthly_result is not None:
        payload["upload_monthly"] = monthly_result
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Snapshot: {snapshot.archive_path}")
        print(f"  files: {snapshot.manifest.file_count}")
        print(f"  bytes: {snapshot.manifest.bytes}")
        print(f"  sha256: {snapshot.manifest.sha256}")
        if snapshot.manifest.commit:
            print(f"  commit: {snapshot.manifest.commit}")
        if upload_result:
            print(f"  upload: {upload_result}")
        if monthly_result:
            print(f"  upload_monthly: {monthly_result}")
    return 0


def _cmd_deliver(args: argparse.Namespace) -> int:
    if not args.upload and not args.upload_monthly:
        print("deliver requires --upload and/or --upload-monthly", file=sys.stderr)
        return 2
    payload = json.loads(args.from_json.read_text(encoding="utf-8"))
    snapshot = snapshot_from_payload(payload)
    if args.upload:
        upload_result, code = _upload_snapshot(snapshot, strict=args.strict_upload)
        if code != 0:
            return code
        payload["upload"] = upload_result
    if args.upload_monthly:
        monthly_result, code = _upload_monthly(snapshot, strict=args.strict_upload)
        if code != 0:
            return code
        payload["upload_monthly"] = monthly_result
    args.from_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    rows = list_local_snapshots(args.backup_dir)
    if args.json:
        print(json.dumps(rows, indent=2))
    elif not rows:
        print("No local snapshots")
    else:
        for row in rows:
            print(
                f"{row['created_at']}  {row['bytes']} bytes  "
                f"{'ok' if row['archive_exists'] else 'missing archive'}  "
                f"{row['archive_path']}"
            )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify_backup_snapshot(args.archive, manifest_path=args.manifest)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Verify: {'ok' if result['ok'] else 'FAIL'}")
    return 0 if result["ok"] else 1


def _cmd_restore(args: argparse.Namespace) -> int:
    dest = args.dest if args.dest is not None else args.repo_root
    result = restore_backup_snapshot(
        args.archive,
        dest_dir=dest,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        label = "Would restore" if args.dry_run else "Restored"
        print(f"{label} {result['restored_paths']} member(s) from {result['archive']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
