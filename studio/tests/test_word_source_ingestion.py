"""Regression tests for DOCX scripts with and without Word numbering definitions."""
from io import BytesIO
from unittest import TestCase
from unittest.mock import PropertyMock,patch
import re

from docx import Document
from pipeline.documents import word_script_text
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "src"))
from jove_visualizer.script_parser import parse_steps_from_text


class WordSourceIngestionTests(TestCase):
    def test_pronunciation_is_spoken_and_editorial_text_stays_off_the_voice_track(self):
        steps = parse_steps_from_text("Title\n1. Label ABC-2 (pronounce ay bee see two).\n"
                                     "Add onscreen TXT: Example caption\n2. Record the result.")
        self.assertIn("ABC-2", steps[0].text)
        self.assertIn("ay bee see two", steps[0].narration_text)
        self.assertNotIn("Example caption", steps[0].narration_text)
        self.assertNotIn("pronounce", steps[0].narration_text)
    def test_literal_numbered_steps_work_when_numbering_relationship_is_absent(self):
        doc=Document()
        doc.add_paragraph("Assessing Locomotor Function in Drosophila")
        for n in range(1,10):
            doc.add_paragraph(f"{n}. Perform action number {n}.")
        # Python-docx normally ships with a numbering part in its default
        # template; emulate Word documents where that relationship is absent.
        with patch("pipeline.documents.Document",return_value=doc):
            with patch.object(type(doc.part),"numbering_part",new_callable=PropertyMock,
                              side_effect=NotImplementedError):
                text=word_script_text(b"not-a-docx-but-mocked-document")
        found=[int(n) for n in re.findall(r"(?m)^(\d+)\. Perform action",text)]
        self.assertEqual(found,list(range(1,10)))
        self.assertEqual(len(text.splitlines()),10)

    def test_automatic_numbering_and_embedded_notes_are_preserved(self):
        doc=Document()
        doc.add_paragraph("Detection of spike-specific antibodies")
        for n in range(1,9):
            p=doc.add_paragraph(f"Action {n}.",style="List Number")
            if n==5:
                p.add_run("\nAdd onscreen TXT: Wash buffer: PBS with 0.05% Tween 20")
            if n==7:
                p.add_run("\nAdd onscreen TXT: Blocking buffer: PBS with 5% BSA")
        doc.add_paragraph("Reference paper: https://example.test/paper")
        out=BytesIO();doc.save(out)
        text=word_script_text(out.getvalue())
        found=[int(n) for n in re.findall(r"(?m)^(\d+)\. Action",text)]
        self.assertEqual(found,list(range(1,9)))
        self.assertIn("Add onscreen TXT: Wash buffer:",text)
        self.assertIn("Add onscreen TXT: Blocking buffer:",text)
        self.assertIn("Reference paper:",text)
        parsed = parse_steps_from_text(text)
        self.assertEqual([s.number for s in parsed], list(range(1, 9)))
        self.assertEqual([s.text for s in parsed], [f"Action {i}." for i in range(1, 9)])
        self.assertEqual(parsed[4].on_screen_text, ["Wash buffer: PBS with 0.05% Tween 20"])
        self.assertEqual(parsed[6].on_screen_text, ["Blocking buffer: PBS with 5% BSA"])


if __name__=="__main__":
    import unittest
    unittest.main()
