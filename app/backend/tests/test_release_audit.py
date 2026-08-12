import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "verify_owned_release.py"
EVIDENCE = REPO_ROOT / "evidence" / "legacy"


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
        "report.pdf",
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
    shutil.copy2(EVIDENCE / "observations.csv", root / "evidence/legacy/observations.csv")
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
        self.assertIn("PUBLIC_REVIEW_REQUIRED", result.stdout)


if __name__ == "__main__":
    unittest.main()
