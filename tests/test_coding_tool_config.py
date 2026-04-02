import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metaharness.integrations.coding_tool.config import load_coding_tool_project
from metaharness.integrations.coding_tool.runtime import _resolve_command_shell, make_backend


class CodingToolConfigTests(unittest.TestCase):
    def test_make_backend_applies_codex_backend_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "baseline").mkdir()
            (root / "tasks.json").write_text("[]", encoding="utf-8")
            (root / "metaharness.json").write_text(
                """
                {
                  "objective": "demo",
                  "constraints": [],
                  "required_files": [],
                  "backends": {
                    "codex": {
                      "use_oss": true,
                      "local_provider": "ollama",
                      "model": "gpt-oss:20b",
                      "approval_policy": "never",
                      "sandbox_mode": "workspace-write"
                    }
                  }
                }
                """.strip(),
                encoding="utf-8",
            )

            project = load_coding_tool_project(root)
            backend = make_backend("codex", project)

            self.assertTrue(backend.use_oss)
            self.assertEqual("ollama", backend.local_provider)
            self.assertEqual("gpt-oss:20b", backend.model)
            self.assertIsNone(backend.timeout_seconds)
            self.assertEqual([], project.allowed_write_paths)

    def test_make_backend_can_override_local_codex_config_to_hosted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "baseline").mkdir()
            (root / "tasks.json").write_text("[]", encoding="utf-8")
            (root / "metaharness.json").write_text(
                """
                {
                  "objective": "demo",
                  "constraints": [],
                  "required_files": [],
                  "backends": {
                    "codex": {
                      "use_oss": true,
                      "local_provider": "ollama",
                      "model": "gpt-oss:20b",
                      "approval_policy": "never",
                      "sandbox_mode": "workspace-write"
                    }
                  }
                }
                """.strip(),
                encoding="utf-8",
            )

            project = load_coding_tool_project(root)
            backend = make_backend(
                "codex",
                project,
                overrides={"use_oss": False, "local_provider": "", "model": ""},
            )

            self.assertFalse(backend.use_oss)
            self.assertIsNone(backend.local_provider)
            self.assertIsNone(backend.model)

    def test_load_project_reads_allowed_write_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "baseline").mkdir()
            (root / "tasks.json").write_text("[]", encoding="utf-8")
            (root / "metaharness.json").write_text(
                """
                {
                  "objective": "demo",
                  "constraints": [],
                  "required_files": [],
                  "allowed_write_paths": ["AGENTS.md", "scripts"],
                  "backends": {}
                }
                """.strip(),
                encoding="utf-8",
            )

            project = load_coding_tool_project(root)
            self.assertEqual(["AGENTS.md", "scripts"], project.allowed_write_paths)

    def test_resolve_command_shell_falls_back_when_zsh_is_unavailable(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which") as which:
                which.side_effect = lambda name: {
                    "bash": "/usr/bin/bash",
                    "zsh": None,
                    "sh": "/usr/bin/sh",
                }.get(name)
                self.assertEqual("/usr/bin/bash", _resolve_command_shell())


if __name__ == "__main__":
    unittest.main()
