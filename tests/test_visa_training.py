"""Keep official held-out VisA images outside auxiliary SFT and constrain extraction."""
import csv
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_visa_training import download, select


class VisaTrainingTests(unittest.TestCase):
    def test_official_test_rows_are_never_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["object", "split", "label", "image", "mask"])
                writer.writeheader()
                for category in ("a", "b", "c", "d"):
                    for split in ("train", "test"):
                        for label, count in (("normal", 3), ("anomaly", 2)):
                            for index in range(count):
                                writer.writerow(dict(object=category, split=split, label=label,
                                    image=f"{category}/{split}/{label}/{index}.png", mask=""))
            selected, references = select(path)
            self.assertEqual(len(selected), 16)
            self.assertTrue(all(row["split"] == "train" for row in selected))
            self.assertTrue(all("/train/" in ref for ref in references.values()))
            self.assertFalse(set(references.values()) & {row["image"] for row in selected})
            partitions = [{row["object"] for row in selected if row["partition"] == split}
                          for split in ("train", "validation")]
            self.assertFalse(partitions[0] & partitions[1])

    def test_only_whitelisted_regular_members_are_written(self):
        content = io.BytesIO()
        wanted = "candle/Data/Images/Normal/a.JPG"
        with tarfile.open(fileobj=content, mode="w") as archive:
            for name in ("../" + wanted, "archive/" + wanted, "archive/unwanted.txt"):
                data = name.encode()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        content.seek(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.prepare_visa_training.urllib.request.urlopen", return_value=content):
                content.headers = {}
                download("https://example.invalid/unused", {wanted}, root)
            self.assertEqual((root / wanted).read_bytes(), ("archive/" + wanted).encode())
            self.assertEqual(len(list(root.rglob("*.*"))), 1)
            # A completed cache needs no second network pass.
            with patch("scripts.prepare_visa_training.urllib.request.urlopen") as network:
                download("https://example.invalid/unused", {wanted}, root)
                network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
