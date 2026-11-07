"""Portability and archive-safety tests for the procedural release exporter."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts import export_multistep_release as release


class ReleaseTests(unittest.TestCase):
    def test_normalization_changes_only_top_level_asset_root(self):
        original = [{"asset_root": "/private/host", "family_id": "f",
                     "evidence": [{"path": "banks/seed-71000/assets/reference.png"}],
                     "certificate": {"label": "Supported", "pixels": [[1, 2, 3]]}}]
        snapshot = deepcopy(original)
        portable = release.portable_cases(original)
        self.assertEqual(original, snapshot)
        expected = deepcopy(original)
        del expected[0]["asset_root"]
        self.assertEqual(portable, expected)

    def test_other_absolute_paths_and_traversal_are_rejected(self):
        for path in ("/private/image.png", "../image.png", "a/../image.png",
                     "C:/private/image.png", "a\\b.png"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                release.portable_cases([{"evidence": [{"path": path}]}])
        with self.assertRaises(ValueError):
            release.portable_cases([{"evidence": [], "nested": {"path": "/secret"}}])
        with self.assertRaises(ValueError):
            release.public_tree({"/private/key": "otherwise relative"})

    def test_archive_is_reproducible_and_extracts_to_a_new_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "source.png"
            original.write_bytes(b"synthetic RGB fixture")
            before = (original.read_bytes(), original.stat().st_mtime_ns)
            a, b = root / "one.tar.gz", root / "two.tar.gz"
            generated = {"cases.json": b"[]\n"}
            assets = {"banks/fixture.png": original}
            release.write_archive(a, generated, assets, source_root=root)
            release.write_archive(b, generated, assets, source_root=root)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            relocated = release.extract_checked(a, root / "new-location")
            self.assertEqual((relocated / "banks/fixture.png").read_bytes(), before[0])
            self.assertEqual((relocated / "cases.json").read_bytes(), b"[]\n")
            self.assertEqual((original.read_bytes(), original.stat().st_mtime_ns), before)
            with tarfile.open(a) as archive:
                self.assertTrue(all(m.uid == m.gid == m.mtime == 0 for m in archive))

    def test_symlinks_and_malicious_tar_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.write_bytes(b"fixture")
            link = root / "link"
            link.symlink_to(source)
            with self.assertRaises(ValueError):
                release.write_archive(root / "bad-source.tar.gz", {}, {"asset": link}, source_root=root)
            allowed, external = root / "allowed", root / "external"
            allowed.mkdir()
            external.mkdir()
            (external / "image.png").write_bytes(b"external")
            (allowed / "bank").symlink_to(external, target_is_directory=True)
            with self.assertRaises(ValueError):
                release.write_archive(root / "parent-link.tar.gz", {},
                                      {"asset": allowed / "bank/image.png"},
                                      source_root=allowed)
            archive = root / "traversal.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo(release.PREFIX + "/../outside")
                info.size = 1
                handle.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                release.extract_checked(archive, root / "extract")
            self.assertFalse((root / "outside").exists())

    def test_checksum_verification_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"release fixture"
            (root / "cases.json").write_bytes(payload)
            record = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            manifest = release.encoded({"files": {"cases.json": record}})
            (root / "release_manifest.json").write_bytes(manifest)
            checks = record["sha256"] + "  cases.json\n"
            checks += hashlib.sha256(manifest).hexdigest() + "  release_manifest.json\n"
            (root / "SHA256SUMS").write_text(checks)
            release.verify_files(root)
            (root / "cases.json").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                release.verify_files(root)


if __name__ == "__main__":
    unittest.main()
