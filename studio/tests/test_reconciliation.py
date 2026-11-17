from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
import json
import unittest
from pipeline.reconciliation import checked_interval,reconcile_step,surrounding_interval
from pipeline.runner import run_timing,write_json,Jobs


def assessment(**changes):
    return dict(suitable=True,complete_action=True,continuous_context=True,scientifically_compatible=True,
        start=1.,end=11.,action_start=3.,action_end=8.,confidence=.95,reason='Complete action and continuous context.',
        scientific_conflicts=[],critical_uncertainties=[])|changes


class DurationReviewTests(unittest.TestCase):
    def test_only_numbers_required_by_the_script_can_create_a_numeric_blocker(self):
        unspecified=assessment(critical_uncertainties=[{'requirement_quote':'defined height','constraint_type':'explicit_quantity','issue':'Exact numeric height not visible.'}])
        self.assertIsNotNone(checked_interval(unspecified,0,12,10,script_text='Mark a scoring region at a defined height.'))
        self.assertTrue(unspecified['notes'])
        specified=assessment(critical_uncertainties=[{'requirement_quote':'20 individuals','constraint_type':'explicit_quantity','issue':'Count not visible.'}])
        self.assertIsNone(checked_interval(specified,0,12,10,script_text='Group 20 individuals.'))
        missing_action=assessment(critical_uncertainties=[{'requirement_quote':'mark a region','constraint_type':'visible_action','issue':'Marking is absent.'}])
        self.assertIsNone(checked_interval(missing_action,0,12,10,script_text='Position the tube and mark a region.'))
        invented=assessment(critical_uncertainties=[{'requirement_quote':'wait exactly 60 seconds','constraint_type':'explicit_quantity','issue':'Wait not timed.'}])
        self.assertIsNone(checked_interval(invented,0,12,10,script_text='Allow brief acclimation.'))

    def test_context_proposal_preserves_complete_action_and_never_stretches_video(self):
        self.assertEqual(surrounding_interval(3,8,0,15,10),(.5,10.5))
        self.assertEqual(surrounding_interval(1,6,0,11,10),(0,10))
        self.assertIsNone(surrounding_interval(1,26,0,30,10))
        self.assertIsNone(surrounding_interval(1,6,0,7,10))
    def test_complete_action_and_verified_context_fit_natural_narration(self):
        result=checked_interval(assessment(),0,12,10)
        self.assertIsNotNone(result)
        self.assertEqual(result[-1].tempo,1)

    def test_never_crops_action_or_accepts_unverified_science_or_forced_speed(self):
        for change in [dict(action_start=.5),dict(action_end=11.5),dict(start=-1),dict(end=20),
                       dict(complete_action=False),dict(continuous_context=False),dict(confidence=.8),
                       dict(scientific_conflicts=['Wrong treatment']),dict(critical_uncertainties=['Unverified count']),
                       dict(end=6,action_end=5),dict(start=float('nan')),dict(scientific_conflicts=None)]:
            self.assertIsNone(checked_interval(assessment(**change),0,25,10),change)

    def test_duration_is_sent_to_both_visual_reviews_and_saved_audio_is_retained(self):
        shot={'step':1,'text':'Transfer flies.','video_id':'1','start':3.,'end':8.,'tier':'suggested_timestamp',
              'audio_path':'saved.wav','audio_duration':10.,'tempo':2.,'needs_timing_review':True}
        gateway=Mock();gateway.ask.return_value=assessment(start=.5,end=10.5)
        library=SimpleNamespace(video=lambda vid:Path('source.mp4'))
        with TemporaryDirectory() as t,patch('pipeline.reconciliation.duration',return_value=100),\
             patch('pipeline.reconciliation.sample_frames',return_value=[]):
            result,attempts=reconcile_step(SimpleNamespace(number=1,text='Transfer flies.'),shot,[],10,library,gateway,t)
        self.assertEqual(result['audio_path'],'saved.wav')
        self.assertEqual(result['tempo'],1)
        self.assertFalse(result['needs_timing_review'])
        self.assertEqual(gateway.ask.call_count,2)
        for args in gateway.ask.call_args_list:self.assertEqual(args.args[1]['target_selected_seconds'],10)

    def test_unresolved_timing_cannot_render_and_can_render_only_after_verified_fit(self):
        for compatible in [False,True]:
            with self.subTest(compatible=compatible),TemporaryDirectory() as t:
                folder=Path(t)/('a'*32);folder.mkdir()
                state={'id':'a'*32,'owner':'a@example.test','output_folder_id':'private-output','status':'running'}
                write_json(folder/'state.json',state);(folder/'input.txt').write_text('1. Transfer flies.')
                shot={'step':1,'text':'Transfer flies.','video_id':'1','start':3.,'end':8.,'tier':'suggested_timestamp',
                      'audio_duration':10.,'tempo':2.,'needs_timing_review':True}
                write_json(folder/'report.json',{'selected':[shot],'issues':[{'step':1,'reason':'Narration adjustment is outside the pilot automatic range.'}]})
                chosen={**shot,'start':1.,'end':11.,'tempo':1.,'needs_timing_review':False} if compatible else None
                gateway=SimpleNamespace(calls=[],prior_calls=[])
                with patch('pipeline.runner.DriveStore'),patch('pipeline.runner.Library') as lib,\
                     patch('pipeline.runner.ModelGateway',return_value=gateway),patch('pipeline.reconciliation.reconcile_step',return_value=(chosen,[])),\
                     patch('pipeline.runner.render',return_value={'frames':300}) as render:
                    lib.from_folder.return_value=SimpleNamespace(video=lambda vid:Path('source.mp4'))
                    run_timing(folder,'a@example.test',{'google_service_account':{},'LIBRARY_FOLDER_ID':'lib','OPENAI_API_KEY':'test'})
                final=json.loads((folder/'state.json').read_text())
                self.assertEqual(final['status'],'complete' if compatible else 'needs_review')
                self.assertEqual(render.call_count,int(compatible))
                self.assertTrue(final['timing_attempted'])

    def test_other_owner_cannot_launch_timing_pass(self):
        with TemporaryDirectory() as t:
            jobs=Jobs(t);ident='a'*32;(Path(t)/ident).mkdir()
            write_json(Path(t)/ident/'state.json',{'id':ident,'owner':'a@example.test','status':'needs_review','step_count':9,'completed_visual_steps':9})
            with self.assertRaises(PermissionError):jobs.resolve_timing(ident,'b@example.test',{})
            jobs.executor.shutdown()

if __name__=='__main__':unittest.main()
