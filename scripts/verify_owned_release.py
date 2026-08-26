#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path


REQUIRED_PATHS = (
    "app/backend/fedrbtvis/__init__.py",
    "app/backend/tests/__init__.py",
    "app/frontend/package-lock.json",
    "app/frontend/src",
    "evidence/legacy/observations.csv",
    "evidence/legacy/manifest.json",
    "evidence/legacy/README.md",
    "docs/architecture.md",
    "docs/provenance.md",
    "THIRD_PARTY.md",
    "README.md",
    ".gitignore",
)
FORBIDDEN_WORKTREE_PATHS = (
    "Backend",
    "HetVis-main",
    ".claude/settings.local.json",
    "项目完整总结.md",
    "面经.md",
    "report.pdf",
)
FORBIDDEN_SOURCE_TERMS = (
    "FedWXR",
    "FedNRID",
    "FedCorr",
    "HetVis",
    "Django",
    "Flower",
)
PYTHON_SCAN_DIRS = ("app/backend/fedrbtvis", "app/backend/tests")
FRONTEND_SCAN_DIR = "app/frontend/src"
TRACKED_FORBIDDEN_SUFFIXES = (".pt", ".pth")
LARGE_FILE_LIMIT = 20 * 1024 * 1024

_PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))"
)
_FRONTEND_IMPORT_RE = re.compile(
    r"""(?:from\s*["']([^"']+)["']|import\s*\(?["']([^"']+)["']|require\(["']([^"']+)["']\))"""
)
_STRING_PATH_RE = re.compile(
    r"""["']([^"']*(?:HetVis-main|FedWXR|FedNRID|FedCorr|django|flower)[^"']*)["']""",
    re.IGNORECASE,
)


def _tracked_files(root: Path) -> list[Path] | None:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    text = result.stdout.decode("utf-8", errors="replace")
    return [Path(item) for item in text.split("\0") if item]


def _python_forbidden_import(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = _PY_IMPORT_RE.match(line)
        if match is None:
            continue
        spec = match.group(1) or match.group(2) or ""
        for term in FORBIDDEN_SOURCE_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", spec, re.IGNORECASE):
                return term
        for string_match in _STRING_PATH_RE.finditer(line):
            value = string_match.group(1)
            for term in FORBIDDEN_SOURCE_TERMS:
                if re.search(rf"{re.escape(term)}", value, re.IGNORECASE):
                    return term
    return None


def _frontend_forbidden_import(path: Path) -> tuple[str, str] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in _FRONTEND_IMPORT_RE.finditer(text):
        imported = next(
            (group for group in match.groups() if group is not None),
            "",
        )
        normalized = imported.replace("\\", "/")
        for term in FORBIDDEN_SOURCE_TERMS:
            if re.search(rf"{re.escape(term)}", normalized, re.IGNORECASE):
                return term, normalized
    return None


def audit_repository(root: Path) -> tuple[list[str], list[str]]:
    root = Path(root).resolve()
    issues: list[str] = []

    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            issues.append(f"missing required path: {relative}")

    for relative in FORBIDDEN_WORKTREE_PATHS:
        if (root / relative).exists():
            issues.append(f"forbidden worktree path exists: {relative}")

    for relative in PYTHON_SCAN_DIRS:
        base = root / relative
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except Exception as error:
                issues.append(f"python AST failure: {path.relative_to(root)}: {error}")
                continue
            forbidden = _python_forbidden_import(path)
            if forbidden is not None:
                issues.append(
                    f"forbidden source reference {forbidden!r}: "
                    f"{path.relative_to(root)}"
                )

    frontend_base = root / FRONTEND_SCAN_DIR
    if frontend_base.is_dir():
        for path in frontend_base.rglob("*"):
            if not path.is_file() or path.suffix not in {".ts", ".tsx", ".js", ".jsx"}:
                continue
            forbidden = _frontend_forbidden_import(path)
            if forbidden is not None:
                term, imported = forbidden
                issues.append(
                    f"forbidden frontend import {imported!r}: "
                    f"{path.relative_to(root)}"
                )

    sys.path.insert(0, str(root / "app" / "backend"))
    try:
        from fedrbtvis.legacy import LegacyRepository

        LegacyRepository.from_directory(root / "evidence" / "legacy")
    except Exception as error:
        issues.append(f"legacy evidence invalid: {error}")

    tracked = _tracked_files(root)
    if tracked is not None:
        for relative in tracked:
            parts = relative.parts
            if (
                "node_modules" in parts
                or "dist" in parts
                or "__pycache__" in parts
                or (parts and parts[0] in {"runs", "data"})
            ):
                issues.append(f"tracked forbidden artifact: {relative.as_posix()}")
            if relative.suffix.lower() in TRACKED_FORBIDDEN_SUFFIXES:
                issues.append(f"tracked forbidden artifact: {relative.as_posix()}")

        large = [
            relative
            for relative in tracked
            if (root / relative).is_file()
            and (root / relative).stat().st_size > LARGE_FILE_LIMIT
        ]
        for relative in large:
            issues.append(f"oversized tracked file: {relative.as_posix()}")

    return issues, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the owned FedRBTVis release")
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    args = parser.parse_args()
    issues, warnings = audit_repository(args.root)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for issue in issues:
        print(f"ISSUE: {issue}")
    if issues:
        print(f"OWNED_RELEASE_AUDIT=FAIL ({len(issues)} issues)")
        return 1
    print("OWNED_RELEASE_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
