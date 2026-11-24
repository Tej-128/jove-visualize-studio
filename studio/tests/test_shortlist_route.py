import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from pipeline.automatic_edit import AutomaticEditor


class ShortlistRouteTests(unittest.TestCase):
    def test_retry_discards_broader_selection_and_only_opens_listed_videos(self):
        editor = Mock()
        editor.steps = [SimpleNamespace(number=1, text='Observe the sample')]
        editor.library.videos = {'1': {}, '2': {}}
        editor.plans = {1: {'shots': [{'video_id': '2'}]}}
        editor.backups = dict(editor.plans)
        editor.deadline = float('inf')
        editor.action_catalog = None
        editor.catalog.return_value = []
        plans = AutomaticEditor.run_shortlist(editor, [{'step': 1, 'video_id': '1'}])
        # Every step is finalized (kept footage or a labelled gap); the broader selection is discarded.
        self.assertIs(plans, editor.finalize.return_value)
        self.assertFalse(editor.plans)
        editor.catalog.assert_called_once_with('1')
        self.assertFalse(editor.backups)
