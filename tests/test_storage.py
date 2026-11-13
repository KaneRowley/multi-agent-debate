import errno
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from debate.storage import read, save_new


class StorageTests(unittest.TestCase):
    def test_drive_fallback_writes_complete_json_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "record.json"
            with patch("debate.storage.os.link", side_effect=OSError(errno.EPERM, "unsupported")), patch.object(Path, "is_relative_to", return_value=True):
                save_new(target, {"answer": 42})
            self.assertEqual(read(target), {"answer": 42})
            self.assertEqual(list(Path(tmp).glob(".pending-*")), [])

    def test_drive_fallback_preserves_existing_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "record.json"
            save_new(target, {"answer": 42})
            before = target.read_bytes()
            with patch("debate.storage.os.link", side_effect=OSError(errno.EPERM, "unsupported")), patch.object(Path, "is_relative_to", return_value=True):
                with self.assertRaises(FileExistsError):
                    save_new(target, {"answer": 99})
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(list(Path(tmp).glob(".pending-*")), [])

    def test_unsupported_link_outside_drive_is_not_silently_renamed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "record.json"
            with patch("debate.storage.os.link", side_effect=OSError(errno.EPERM, "unsupported")), patch.object(Path, "is_relative_to", return_value=False):
                with self.assertRaises(OSError):
                    save_new(target, {"answer": 42})
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(tmp).glob(".pending-*")), [])

    def test_unrelated_drive_error_is_not_suppressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "record.json"
            with patch("debate.storage.os.link", side_effect=OSError(errno.ENOSPC, "full")), patch.object(Path, "is_relative_to", return_value=True):
                with self.assertRaises(OSError) as caught:
                    save_new(target, {"answer": 42})
            self.assertEqual(caught.exception.errno, errno.ENOSPC)
            self.assertFalse(target.exists())


class FinalParserTests(unittest.TestCase):
    def test_version_and_bounded_introductions(self):
        from debate.scoring import PARSER_VERSION, extract
        self.assertEqual(PARSER_VERSION, "explicit-rational-v3")
        for text in ["### Final Answer:\nThe result is \\boxed{2400}", "The answer is indeed \\boxed{38}", "The answer is correct.\n\\boxed{38}"]:
            self.assertIsNotNone(extract(text).value)

    def test_conflicting_boxes_remain_rejected(self):
        from debate.scoring import extract
        self.assertIsNone(extract("Peer: \\boxed{1000}\nMy answer: \\boxed{800}").value)
