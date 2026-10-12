"""Deleting one project of the projects root, and the administrator's remove-all clean-up."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from chem_agent_bo.lab import service as lab_service
from chem_agent_bo.lab.api import ADMIN_REMOVE_ALL_CONFIRM, create_app
from chem_agent_bo.lab.service import LabBOService, ProjectBusyError

from test_lab_mode import _create_small_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Projects(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "root"
        self.root.mkdir()
        self.service = LabBOService(projects_root=self.root)
        _create_small_project(self.service, "native_a")
        _create_small_project(self.service, "native_b")
        self.outside = self.base / "workspace" / "ext"
        _create_small_project(self.service, str(self.outside))
        self.service.register_project(self.outside, handle="registered_x")
        (self.root / "notes.txt").write_text("not a project", encoding="utf-8")

    def handles(self):
        return sorted(item["handle"] for item in self.service.list_projects())


class DeleteProjectTests(_Projects):
    def test_native_project_moves_to_backups(self):
        result = self.service.delete_project("native_a")
        backup = Path(result["backup_dir"])
        self.assertFalse((self.root / "native_a").exists())
        self.assertTrue((backup / "project.yaml").exists())
        self.assertEqual(backup.parent, self.root / "_backups")
        self.assertTrue(backup.name.startswith("deleted_native_a_"))
        self.assertEqual(self.handles(), ["native_b", "registered_x"])

    def test_registered_project_points_to_registration(self):
        with self.assertRaises(ValueError) as caught:
            self.service.delete_project("registered_x")
        self.assertIn("DELETE /api/projects/registered_x/registration", str(caught.exception))
        self.assertTrue((self.outside / "project.yaml").exists())

    def test_busy_project_is_left_alone(self):
        lock = lab_service._project_ask_lock(self.root / "native_a")
        lock.acquire()
        try:
            with self.assertRaises(ProjectBusyError):
                self.service.delete_project("native_a")
        finally:
            lock.release()
        self.assertTrue((self.root / "native_a" / "project.yaml").exists())

    def test_api(self):
        client = TestClient(create_app(self.root))
        self.assertEqual(client.delete("/api/projects/native_a").status_code, 200)
        self.assertEqual(client.delete("/api/projects/native_a").status_code, 404)
        self.assertEqual(client.delete("/api/projects/registered_x").status_code, 400)


class RemoveAllTests(_Projects):
    def test_dry_run_changes_nothing(self):
        result = self.service.remove_all_projects()
        self.assertTrue(result["dry_run"])
        self.assertEqual(sorted(item["handle"] for item in result["removed"]), ["native_a", "native_b"])
        self.assertEqual([item["handle"] for item in result["unregistered"]], ["registered_x"])
        self.assertEqual(result["skipped"], ["notes.txt"])
        self.assertEqual(self.handles(), ["native_a", "native_b", "registered_x"])

    def test_real_run_moves_unregisters_and_keeps(self):
        result = self.service.remove_all_projects(keep=["native_b"], dry_run=False)
        backup = Path(result["backup_dir"])
        self.assertEqual(self.handles(), ["native_b"])
        self.assertTrue((backup / "native_a" / "project.yaml").exists())
        self.assertTrue((self.outside / "project.yaml").exists(), "a registered project's files stay")
        self.assertTrue((self.root / "notes.txt").exists())
        self.assertEqual(result["kept"], ["native_b"])

    def test_purge_deletes(self):
        result = self.service.remove_all_projects(dry_run=False, purge=True)
        self.assertEqual(result["backup_dir"], "")
        self.assertEqual(self.handles(), [])
        self.assertFalse((self.root / "native_a").exists())
        self.assertFalse((self.root / "_backups").exists() and any((self.root / "_backups").glob("remove_all_*")))

    def test_busy_project_blocks_everything(self):
        lock = lab_service._project_ask_lock(self.outside)
        lock.acquire()
        try:
            with self.assertRaises(ProjectBusyError):
                self.service.remove_all_projects(dry_run=False)
        finally:
            lock.release()
        self.assertEqual(self.handles(), ["native_a", "native_b", "registered_x"])

    def test_api_token_and_confirmation(self):
        client = TestClient(create_app(self.root))
        url = "/api/admin/projects/remove-all"
        with mock.patch.dict(os.environ, {"TRACE_ADMIN_TOKEN": ""}):
            self.assertEqual(client.post(url, json={}).status_code, 404)
        with mock.patch.dict(os.environ, {"TRACE_ADMIN_TOKEN": "s3cret"}):
            self.assertEqual(client.post(url, json={}).status_code, 403)
            self.assertEqual(client.post(url, json={}, headers={"X-Trace-Admin-Token": "wrong"}).status_code, 403)
            ok = {"X-Trace-Admin-Token": "s3cret"}
            listed = client.post(url, json={}, headers=ok)
            self.assertEqual(listed.status_code, 200)
            self.assertTrue(listed.json()["dry_run"])
            self.assertEqual(client.post(url, json={"dry_run": False}, headers=ok).status_code, 400)
            self.assertEqual(self.handles(), ["native_a", "native_b", "registered_x"])
            done = client.post(url, json={"dry_run": False, "confirm": ADMIN_REMOVE_ALL_CONFIRM,
                                          "keep": ["native_a"]}, headers=ok)
            self.assertEqual(done.status_code, 200, done.text)
            self.assertEqual(self.handles(), ["native_a"])

    def test_cli_lists_then_removes(self):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(PROJECT_ROOT), str(PROJECT_ROOT / "third_party/atlas/src"), str(PROJECT_ROOT / "third_party/olympus/src")])}
        command = [sys.executable, str(PROJECT_ROOT / "experiments/run_lab_bo.py"), "remove-all-projects",
                   "--projects-root", str(self.root)]
        listed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, cwd=PROJECT_ROOT)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("Nothing was changed", listed.stdout)
        self.assertEqual(self.handles(), ["native_a", "native_b", "registered_x"])
        removed = subprocess.run(command + ["--yes", "--keep", "native_b"], capture_output=True, text=True,
                                 encoding="utf-8", env=env, cwd=PROJECT_ROOT)
        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.assertEqual(self.handles(), ["native_b"])


if __name__ == "__main__":
    unittest.main()
