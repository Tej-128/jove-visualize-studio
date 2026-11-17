from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import unittest
import wave
import zipfile
import xml.etree.ElementTree as ET
import imageio_ffmpeg
from pipeline.automatic_edit import AutomaticEditor,validated_plan,plan_schema
from pipeline.action_render import cut_schedule,render_actions
from pipeline.media import execute,sample_window
from pipeline.providers import ModelGateway
from pipeline.runner import run,write_json

def response(window='w0',**extra):
    return dict(shots=[dict(window_id=window,start=1.,end=3.,evidence='demonstrated',
        covered_actions=['Transfer flies'],reason='Transfer is visible.',scientific_conflicts=[])],
        missing_actions=[],unobservable_details=[])|extra

def windows():return [dict(window_id='w0',video_id='101',start=.5,end=4.,tier='suggested_timestamp')]

class AutomaticSelectionTests(unittest.TestCase):
    def test_model_contract_prevents_the_observed_live_format_errors(self):
        step=SimpleNamespace(number=1,text='Transfer flies and count them.')
        schema=plan_schema(step,windows())
        shot=schema['properties']['shots']['items']['properties']
        self.assertEqual(shot['evidence']['enum'],['demonstrated','illustrative','uncertain'])
        self.assertEqual(shot['covered_actions']['type'],'array')
        self.assertEqual(shot['window_id']['enum'],['w0'])
        self.assertIn('Transfer flies',shot['covered_actions']['items']['enum'])
        self.assertNotIn('Unrelated source caption',shot['covered_actions']['items']['enum'])

    def test_gateway_sends_strict_schema_and_does_not_reuse_an_unconstrained_response(self):
        client=Mock();client.chat.completions.create.return_value=SimpleNamespace(id='test',usage=SimpleNamespace(total_tokens=1),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"shots":[],"missing_actions":[],"unobservable_details":[]}'))])
        with TemporaryDirectory() as t,patch('pipeline.providers.OpenAI',return_value=client):
            gateway=ModelGateway('test','test',3,Path(t)/'cache')
            gateway.ask('rules',{'step':'Transfer flies.'})
            schema=plan_schema(SimpleNamespace(text='Transfer flies.'),windows())
            gateway.ask('rules',{'step':'Transfer flies.'},schema=schema)
            self.assertEqual(client.chat.completions.create.call_count,2)
            sent=client.chat.completions.create.call_args.kwargs['response_format']
            self.assertEqual(sent['type'],'json_schema');self.assertTrue(sent['json_schema']['strict'])
            self.assertEqual(sent['json_schema']['schema'],schema)

    def test_unobservable_metadata_does_not_delete_useful_footage(self):
        step=SimpleNamespace(number=1,text='Transfer flies into the tube.')
        plan=validated_plan(response(unobservable_details=['Specimen sex cannot be established visually.']),windows(),step)
        self.assertEqual(len(plan['shots']),1);self.assertTrue(plan['needs_review'])

    def test_multiple_actions_can_use_multiple_shots_in_one_step(self):
        step=SimpleNamespace(number=1,text='Transfer flies and count them.')
        answer=response()
        answer['shots'].append(dict(window_id='w0',start=3.,end=4.,evidence='illustrative',
            covered_actions=['count them'],reason='Counting action.',scientific_conflicts=[]))
        plan=validated_plan(answer,windows(),step)
        self.assertEqual(len(plan['shots']),2);self.assertTrue(plan['needs_review'])

    def test_hallucinated_sources_out_of_range_times_and_conflicts_cannot_be_accepted(self):
        step=SimpleNamespace(number=1,text='Transfer flies into the tube.')
        for change in [dict(window_id='invented'),dict(end=40.),dict(scientific_conflicts=['Shows the opposite operation']),
                       dict(covered_actions=['Freeze the flies']),dict(start=float('nan'))]:
            answer=response();answer['shots'][0].update(change)
            self.assertIsNone(validated_plan(answer,windows(),step),change)

    def test_plain_script_and_fresh_matches_use_suggested_windows_without_full_scan(self):
        step=SimpleNamespace(number=1,text='Transfer flies into the tube.')
        matches=[dict(step_number=1,best_match=dict(journal_id='101',start_timestamp='00:01',end_timestamp='00:04',caption_passage='Transfer flies.'))]
        with TemporaryDirectory() as t:
            library=SimpleNamespace(videos={'101':{'sha256':'a'*64,'filename':'101.mp4'}},video=lambda _:Path('source.mp4'),cache=Path(t))
            library.restore_overviews=Mock()
            gateway=Mock();gateway.ask.side_effect=[response(),response('d0')]
            with patch('pipeline.automatic_edit.duration',return_value=20),patch('pipeline.automatic_edit.sample_window',return_value=[]):
                editor=AutomaticEditor(None,library,gateway,t,[step],{1:{'duration':7}},workers=1)
                result=editor.run(matches)
            self.assertEqual(result[0]['shots'][0]['video_id'],'101')
            self.assertTrue(result[0]['dense_checked'])
            self.assertEqual(gateway.ask.call_count,2)
            dense_window=gateway.ask.call_args_list[1].args[1]['windows'][0]
            self.assertGreaterEqual(dense_window['end']-dense_window['start'],3.5)
            library.restore_overviews.assert_not_called()
            self.assertNotIn('reference',json.dumps(gateway.ask.call_args_list[0].args[1]).lower())

    def test_missing_caption_video_is_checked_before_caption_fallback_exhausts_the_budget(self):
        step=SimpleNamespace(number=1,text='Transfer flies into the tube.')
        index=Mock();index.connection.execute.return_value.fetchall.return_value=[('101',)]
        with TemporaryDirectory() as t:
            library=SimpleNamespace(videos={'101':{'sha256':'a'*64,'filename':'101.mp4'},
                '202':{'sha256':'b'*64,'filename':'202_Fly_Climbing.mp4'}},cache=Path(t),restore_overviews=Mock())
            editor=AutomaticEditor(index,library,Mock(),t,[step],{1:{'duration':7}},workers=1)
            editor.catalog=Mock(return_value=[{'step':1,'video_id':'202','start':1,'end':4}])
            tiers=[]
            def phase(steps,pools,tier):
                tiers.append(tier)
                if tier=='wider_library_visual' and pools.get(1):
                    editor.plans[1]={'step':1,'text':step.text,'shots':[{'video_id':'202'}],
                        'missing_actions':[],'unobservable_details':[],'needs_review':False,'dense_checked':True}
            editor.inspect_phase=phase
            with patch('pipeline.automatic_edit.Selector.wider_candidates') as caption_fallback:
                result=editor.run([{'step_number':1,'best_match':None}])
            self.assertEqual(result[0]['shots'][0]['video_id'],'202')
            editor.catalog.assert_called_once_with('202')
            caption_fallback.assert_not_called()
            self.assertLess(tiers.index('suggested_timestamp'),tiers.index('wider_library_visual'))

    def test_provider_failure_preserves_an_explicitly_unverified_retrieved_candidate(self):
        step=SimpleNamespace(number=1,text='Transfer flies into the tube.')
        matches=[dict(step_number=1,best_match=dict(journal_id='101',start_timestamp='00:01',end_timestamp='00:04',caption_passage='Transfer flies.'))]
        with TemporaryDirectory() as t:
            library=SimpleNamespace(videos={'101':{'sha256':'a'*64,'filename':'101.mp4'}},video=lambda _:Path('source.mp4'),cache=Path(t),
                                    restore_overviews=Mock(),has_prepared_overview=lambda _:True)
            gateway=Mock();gateway.ask.side_effect=TimeoutError('provider unavailable')
            with patch('pipeline.automatic_edit.duration',return_value=20),patch('pipeline.automatic_edit.sample_window',return_value=[]),\
                 patch('pipeline.automatic_edit.screen_video',side_effect=TimeoutError),\
                 patch('pipeline.automatic_edit.Selector.wider_candidates',return_value=[]):
                result=AutomaticEditor(None,library,gateway,t,[step],{1:{'duration':7}},workers=1).run(matches)
            self.assertEqual(len(result),1)
            self.assertTrue(result[0]['needs_review'])
            self.assertEqual(result[0]['shots'][0]['evidence'],'uncertain')
            self.assertEqual(result[0]['shots'][0]['tier'],'unverified_retrieval')

    def test_explicitly_rejected_window_is_not_relabelled_as_verified_fallback(self):
        step=SimpleNamespace(number=1,text='Transfer flies into the tube.')
        matches=[dict(step_number=1,best_match=dict(journal_id='101',start_timestamp='00:01',end_timestamp='00:04',caption_passage='Transfer flies.'))]
        with TemporaryDirectory() as t:
            library=SimpleNamespace(videos={'101':{'sha256':'a'*64,'filename':'101.mp4'}},video=lambda _:Path('source.mp4'),cache=Path(t),
                                    restore_overviews=Mock(),has_prepared_overview=lambda _:True)
            gateway=Mock();gateway.ask.return_value={'shots':[],'missing_actions':['Transfer absent'],'unobservable_details':[]}
            with patch('pipeline.automatic_edit.duration',return_value=20),patch('pipeline.automatic_edit.sample_window',return_value=[]),\
                 patch('pipeline.automatic_edit.screen_video',side_effect=TimeoutError),\
                 patch('pipeline.automatic_edit.Selector.wider_candidates',return_value=[]):
                result=AutomaticEditor(None,library,gateway,t,[step],{1:{'duration':7}},workers=1).run(matches)
            self.assertEqual(result,[])

    def test_default_runner_routes_plain_script_to_new_editor_without_reference_input(self):
        with TemporaryDirectory() as t:
            folder=Path(t)/('b'*32);folder.mkdir()
            write_json(folder/'state.json',{'id':'b'*32,'owner':'a@example.test','output_folder_id':'private-output'})
            write_json(folder/'matches.json',[{'step_number':1,'script_step':'Transfer flies.','best_match':None}])
            (folder/'narration-001.wav').write_bytes(b'existing voice')
            write_json(folder/'narration-001.json',{'spoken':'Transfer flies.','voice_id':'voice','generation_speed':.88,'duration':2})
            step=SimpleNamespace(number=1,text='Transfer flies.',previous_text='',next_text='');index=Mock()
            config={'google_service_account':{},'OUTPUT_FOLDER_ID':'output','LIBRARY_FOLDER_ID':'library',
                    'OPENAI_API_KEY':'test','ELEVENLABS_API_KEY':'test','voice':{'id':'voice','speed':.88}}
            with patch('pipeline.runner.DriveStore'),patch('pipeline.runner.Library') as lib,\
                 patch('pipeline.runner.CorpusIndex',return_value=index),patch('pipeline.runner.parse_steps_from_text',return_value=[step]),\
                 patch('pipeline.runner.run_automatic_edit') as auto,patch('pipeline.runner.match_steps') as match,\
                 patch('pipeline.runner.generate_step_audio') as tts,patch('pipeline.runner.run_full_subset') as full:
                lib.from_folder.return_value=SimpleNamespace(index=lambda:'.',videos={'101':{}})
                run(folder,'a@example.test','1. Transfer flies.',config)
                auto.assert_called_once();full.assert_not_called();match.assert_not_called();tts.assert_not_called()
            self.assertFalse((folder/'reference.xlsx').exists())

    def test_dense_counterevidence_overrides_a_positive_coarse_assessment(self):
        step=SimpleNamespace(number=1,text='Transfer flies into the tube.')
        with TemporaryDirectory() as t:
            library=SimpleNamespace(videos={'101':{'sha256':'a'*64,'filename':'101.mp4'}},video=lambda _:Path('source.mp4'),cache=Path(t))
            gateway=Mock();gateway.ask.side_effect=[response(),{'shots':[],'missing_actions':['Transfer absent'],'unobservable_details':[]}]
            with patch('pipeline.automatic_edit.duration',return_value=20),patch('pipeline.automatic_edit.sample_window',return_value=[]):
                editor=AutomaticEditor(None,library,gateway,t,[step],{1:{'duration':7}},workers=1)
                result=editor.inspect(step,[{'video_id':'101','start':.5,'end':4}], 'suggested_timestamp')
            self.assertIsNone(result)
            self.assertEqual(editor.attempts[1][-1]['status'],'rejected')

    def test_parallel_model_requests_keep_their_own_ledger_records(self):
        client=Mock()
        client.chat.completions.create.return_value=SimpleNamespace(id='test',usage=SimpleNamespace(total_tokens=1),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"shots":[]}'))])
        with TemporaryDirectory() as t,patch('pipeline.providers.OpenAI',return_value=client):
            gateway=ModelGateway('test','test',3,Path(t)/'cache')
            with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda i:gateway.ask('rules',{'step':i}),range(3)))
            saved=json.loads((Path(t)/'model-requests.json').read_text())
            self.assertEqual(len(saved),3);self.assertTrue(all(x['status']=='complete' for x in saved))
            self.assertTrue(all('elapsed_seconds' in x for x in saved))

class MultiShotMediaTests(unittest.TestCase):
    def test_long_action_cannot_create_extreme_video_or_voice_playback_speed(self):
        plans=[{'step':1,'text':'Transfer flies.','needs_review':False,'missing_actions':[],'unobservable_details':[],
                'shots':[dict(video_id='101',start=1.,end=61.,reason='A long complete action')]}]
        shots,adjustments,total=cut_schedule(plans,{1:{'duration':7}})
        self.assertLessEqual(shots[0]['video_speed'],1.151)
        self.assertGreaterEqual(adjustments[0]['tempo'],.95)
        self.assertGreater(adjustments[0]['additional_viewing_seconds'],30)
        self.assertTrue(shots[0]['review'])
    def test_multi_shot_schedule_keeps_speech_natural_and_reports_extreme_video_speed(self):
        plans=[{'step':1,'text':'Transfer flies.','needs_review':False,'missing_actions':[],'unobservable_details':[],
                'shots':[dict(video_id='101',start=1.,end=3.,reason='First action'),dict(video_id='101',start=4.,end=6.,reason='Second action')]}]
        shots,adjustments,total=cut_schedule(plans,{1:{'duration':7}})
        self.assertEqual(sum(s['frames'] for s in shots),total)
        self.assertEqual(shots[0]['timeline_end_frame'],shots[1]['timeline_start_frame'])
        self.assertTrue(all(s['review'] for s in shots))
        self.assertAlmostEqual(adjustments[0]['tempo'],1,places=2)

    def test_real_multi_shot_edit_has_no_cards_complete_audio_and_portable_timeline(self):
        with TemporaryDirectory() as t:
            root=Path(t);source=root/'source.mp4';voice=root/'voice.wav';out=root/'out';out.mkdir()
            execute(['-f','lavfi','-i','testsrc2=size=320x180:rate=30:duration=9','-c:v','libx264','-threads','1','-preset','ultrafast',source],root/'make-video.log')
            execute(['-f','lavfi','-i','sine=frequency=440:sample_rate=44100:duration=4','-c:a','pcm_s16le',voice],root/'make-voice.log')
            samples=sample_window(source,1,3,root/'samples',8)
            self.assertGreaterEqual(len(samples),6)
            self.assertTrue(all(1<=s['time']<=3 for s in samples))
            self.assertEqual(len(list((root/'samples').glob('*.log'))),1)
            plans=[dict(step=1,text='Transfer and count.',needs_review=True,missing_actions=[],unobservable_details=['Count uncertain'],
                        shots=[dict(video_id='101',start=1.,end=3.,reason='Transfer.',evidence='demonstrated',tier='suggested_timestamp'),
                               dict(video_id='101',start=4.,end=6.,reason='Counting.',evidence='illustrative',tier='suggested_timestamp')])]
            result=render_actions(plans,{1:{'path':str(voice),'duration':4}},SimpleNamespace(video=lambda _:source),out,width=320,height=180)
            count,_=imageio_ffmpeg.count_frames_and_secs(str(out/'preview.mp4'))
            self.assertEqual(count,result['frames']);self.assertEqual(result['shots'],2)
            self.assertEqual(result['placeholder_steps'],[]);self.assertTrue(result['review_only'])
            with wave.open(str(out/'narration.wav')) as audio:self.assertAlmostEqual(audio.getnframes()/audio.getframerate(),4,places=1)
            with zipfile.ZipFile(out/'editable-package.zip') as archive:
                self.assertIn('media/shot-002.mp4',archive.namelist());self.assertIn('narration.wav',archive.namelist())
            mlt=ET.parse(out/'editable.mlt').getroot()
            resources=[x.text for x in mlt.findall('.//property[@name="resource"]')]
            self.assertNotIn(str(source),resources);self.assertIn('media/shot-001.mp4',resources)

class CheckpointTests(unittest.TestCase):
    def test_changed_owner_or_script_cannot_restore_a_signed_checkpoint(self):
        from pipeline.checkpoints import signature,valid
        config={'google_service_account':{'private_key':'unit-test-only-not-a-real-key'}}
        state={'id':'a'*32,'owner':'invited@example.test','input_sha256':'a'*64}
        state['checkpoint_signature']=signature(state,config)
        self.assertTrue(valid(state,config))
        self.assertFalse(valid({**state,'owner':'other@example.test'},config))
        self.assertFalse(valid({**state,'input_sha256':'b'*64},config))
        self.assertFalse(valid({**state,'checkpoint_signature':None},config))

    def test_signed_restart_restores_saved_narration_and_marks_interrupted_job_for_rebuild(self):
        import hashlib
        from pipeline.checkpoints import signature
        from pipeline.runner import Jobs
        owner='invited@example.test';ident='c'*32;script=b'1. Transfer flies.'
        config={'google_service_account':{'private_key':'unit-test-only-not-a-real-key'},'OUTPUT_FOLDER_ID':'output'}
        state={'id':ident,'owner':owner,'status':'running','input_sha256':hashlib.sha256(script).hexdigest()}
        state['checkpoint_signature']=signature(state,config)
        store=Mock();store.children.side_effect=[[
            {'id':'job-folder','name':'job-'+ident,'createdTime':'2026-01-01'}],[
            {'id':'state','name':'state.json'},{'id':'input','name':'input.txt'},
            {'id':'voice','name':'narration-001.wav'},{'id':'notes','name':'selections.json'}]]
        store.request.return_value.json.return_value=state
        def download(file_id,dest):
            Path(dest).write_bytes(script if file_id=='input' else b'{}' if file_id=='notes' else b'saved narration')
        store.download.side_effect=download
        with TemporaryDirectory() as t,patch('pipeline.runner.DriveStore',return_value=store):
            jobs=Jobs(t)
            restored=jobs.restore_latest(owner,config)
            self.assertEqual(restored,ident)
            self.assertEqual(jobs.state(ident,owner)['status'],'needs_review')
            self.assertEqual((Path(t)/ident/'narration-001.wav').read_bytes(),b'saved narration')
            self.assertTrue((Path(t)/ident/'automatic'/'selections.json').exists())
            with self.assertRaises(PermissionError):jobs.state(ident,'other@example.test')
            jobs.executor.shutdown()

    def test_new_edit_audio_report_is_recovered_from_its_actual_frame_plan(self):
        from pipeline.runner import Jobs
        with TemporaryDirectory() as t:
            jobs=Jobs(t);ident='d'*32;folder=Path(t)/ident;folder.mkdir()
            write_json(folder/'state.json',{'id':ident,'owner':'a@example.test'})
            write_json(folder/'edit_plan.json',{'audio_adjustments':[{'step':1,'tempo':1.,'review':False}]})
            path=jobs.file(ident,'a@example.test','audio-adjustments.json')
            self.assertEqual(json.loads(path.read_text())[0]['tempo'],1.)
            jobs.executor.shutdown()

if __name__=='__main__':unittest.main()
