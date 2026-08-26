import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "verify_owned_release.py"
EVIDENCE = REPO_ROOT / "evidence" / "legacy"
FORBIDDEN_PUBLIC_DOC_PATH = re.compile(
    r"[A-Za-z]:[/\\](Users|personal|desktop)|/Users/|/home/"
)


def public_documentation_paths() -> list[Path]:
    paths = [REPO_ROOT / "README.md", REPO_ROOT / "THIRD_PARTY.md"]
    for directory in (
        REPO_ROOT / "docs",
        REPO_ROOT / "app" / "frontend" / "docs",
        REPO_ROOT / "evidence" / "legacy",
    ):
        paths.extend(path for path in directory.rglob("*.md") if path.is_file())
    return paths


def write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_minimal_root(root: Path) -> None:
    required = (
        "app/backend/fedrbtvis/__init__.py",
        "app/backend/tests/__init__.py",
        "app/frontend/package-lock.json",
        "app/frontend/src/main.ts",
        "evidence/legacy/observations.csv",
        "evidence/legacy/manifest.json",
        "evidence/legacy/README.md",
        "docs/architecture.md",
        "docs/provenance.md",
        "THIRD_PARTY.md",
        "README.md",
        ".gitignore",
    )
    for relative in required:
        write(root / relative)
    shutil.copytree(
        REPO_ROOT / "app/backend/fedrbtvis",
        root / "app/backend/fedrbtvis",
        dirs_exist_ok=True,
    )
    csv = (EVIDENCE / "observations.csv").read_bytes().replace(b"\r\n", b"\n")
    (root / "evidence/legacy/observations.csv").write_bytes(csv)
    shutil.copy2(EVIDENCE / "manifest.json", root / "evidence/legacy/manifest.json")
    shutil.copy2(EVIDENCE / "README.md", root / "evidence/legacy/README.md")


class ReleaseAuditTest(unittest.TestCase):
    def run_audit(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(root)],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_rejects_forbidden_worktree_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Backend").mkdir(parents=True)
            result = self.run_audit(root)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Backend", result.stdout + result.stderr)

    def test_rejects_forbidden_frontend_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_minimal_root(root)
            write(
                root / "app/frontend/src/bad.ts",
                'import legacy from "../../HetVis-main/foo";\n',
            )
            result = self.run_audit(root)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HetVis-main", result.stdout + result.stderr)

    def test_passes_minimal_allowed_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_minimal_root(root)
            result = self.run_audit(root)

        self.assertEqual(result.returncode, 0)
        self.assertNotIn("PUBLIC_REVIEW_REQUIRED", result.stdout)
        self.assertNotIn("G5_BLOCKED", result.stdout)

    def test_rejects_report_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_minimal_root(root)
            write(root / "report.pdf", "forbidden public artifact")
            result = self.run_audit(root)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("report.pdf", result.stdout + result.stderr)

    def test_legacy_csv_requires_lf_line_endings(self) -> None:
        attributes_path = REPO_ROOT / ".gitattributes"
        attributes = (
            attributes_path.read_text(encoding="utf-8").splitlines()
            if attributes_path.is_file()
            else []
        )

        self.assertIn("evidence/legacy/*.csv text eol=lf", attributes)

    def test_public_documentation_has_no_machine_specific_paths(self) -> None:
        findings = []
        for path in public_documentation_paths():
            text = path.read_text(encoding="utf-8")
            if FORBIDDEN_PUBLIC_DOC_PATH.search(text):
                findings.append(str(path.relative_to(REPO_ROOT)))

        self.assertEqual(findings, [], "machine-specific paths: " + ", ".join(findings))

    def test_provenance_distinguishes_private_history_from_public_candidate(self) -> None:
        text = (REPO_ROOT / "docs/provenance.md").read_text(encoding="utf-8")

        self.assertIn("private local evidence repository retains legacy Git history", text)
        self.assertIn("sanitized public candidate is a clean one-commit export", text)
        self.assertNotIn(
            "The current repository is therefore not suitable for direct public export.",
            text,
        )


if __name__ == "__main__":
    unittest.main()
