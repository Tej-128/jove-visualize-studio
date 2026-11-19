from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
import copy
import unittest
from pipeline.scientific_audit import apply_audit,schema
from pipeline.automatic_edit import AutomaticEditor


def plan():
    return {'step':1,'text':'Freeze flies.','shots':[{'video_id':'123','start':1.,'end':3.,'evidence':'demonstrated','covered_actions':['Freeze flies']}],
            'missing_actions':[],'unobservable_details':[],'dense_checked':True,'needs_review':False}


def answer(**changes):
    return {'status':'compatible','reason':'Visible action is compatible.','conflicting_source_quotes':[],
        'shots':[{'shot_id':'0','medium':'experimental_footage','boundary_suitable':True,
                  'actions':[{'quote':'Freeze flies','support':'directly_visible','observation':'Physical handling is visible.'}]}],**changes}


class ScientificAuditTests(unittest.TestCase):
    def test_default_editor_requires_independent_auditing(self):
        with TemporaryDirectory() as t:
            editor=AutomaticEditor(None,SimpleNamespace(videos={}),Mock(),t,[],{})
            self.assertTrue(editor.audit_required)

    def test_explicit_source_contradiction_overrides_selector_approval(self):
        audit=answer(status='contradiction',reason='Source identifies sedation, not freezing.',conflicting_source_quotes=['Ethanol sedation assay'])
        self.assertIsNone(apply_audit(plan(),audit,SimpleNamespace(text='Freeze flies.'),'Ethanol sedation assay'))

    def test_immobility_inference_cannot_be_used_as_visible_freezing(self):
        audit=answer();audit['shots'][0]['actions'][0].update(support='inferred_only',observation='Flies are still; freezing is inferred.')
        self.assertIsNone(apply_audit(plan(),audit,SimpleNamespace(text='Freeze flies.'),''))

    def test_animation_is_never_upgraded_to_demonstrated_experimental_footage(self):
        audit=answer();audit['shots'][0]['medium']='animation'
        result=apply_audit(plan(),audit,SimpleNamespace(text='Freeze flies.'),'')
        self.assertIsNone(result)

    def test_unrelated_section_title_or_mixed_scene_is_rejected(self):
        for change in [{'boundary_suitable':False},{'medium':'mixed'},{'medium':'title_card'}]:
            audit=answer();audit['shots'][0].update(change)
            self.assertIsNone(apply_audit(plan(),audit,SimpleNamespace(text='Freeze flies.'),''))

    def test_all_actions_must_be_covered_and_quote_invention_is_rejected(self):
        step=SimpleNamespace(text='Freeze flies and count them.')
        audit=answer();result=apply_audit(plan(),audit,step,'')
        self.assertIn('count them',result['missing_actions'])
        audit['shots'][0]['actions'][0]['quote']='A different action'
        self.assertIsNone(apply_audit(plan(),audit,step,''))

    def test_correct_live_action_can_pass_without_modifying_selector_result(self):
        original=plan();result=apply_audit(original,answer(),SimpleNamespace(text='Freeze flies.'),'')
        self.assertTrue(result['scientific_audit_checked']);self.assertFalse(result['needs_review'])
        self.assertNotIn('scientific_audit',original)


if __name__=='__main__':unittest.main()
