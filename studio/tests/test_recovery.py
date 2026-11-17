from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
import json
import unittest
from pipeline.providers import ModelGateway,VisionBudgetExceeded
from pipeline.runner import run,write_json,Jobs


class RecoveryTests(unittest.TestCase):
    def test_completed_assessment_is_reused_without_another_paid_request(self):
        client=Mock()
        client.chat.completions.create.return_value=SimpleNamespace(
            id='test-id',usage=SimpleNamespace(total_tokens=1),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"suitable": false}'))])
        with TemporaryDirectory() as t,patch('pipeline.providers.OpenAI',return_value=client):
            first=ModelGateway('test','test-model',1,Path(t)/'cache')
            result=first.ask('rules',{'step':'Tap tube.'})
            second=ModelGateway('test','test-model',1,Path(t)/'cache')
            self.assertEqual(second.ask('rules',{'step':'Tap tube.'}),result)
            self.assertEqual(client.chat.completions.create.call_count,1)
            self.assertEqual(second.calls,[])
            self.assertEqual(second.cache_hits,1)
            with self.assertRaises(VisionBudgetExceeded):first.ask('rules',{'step':'Different action.'})
            self.assertEqual(client.chat.completions.create.call_count,1)

    def test_resume_reuses_matches_and_narration_and_pauses_at_budget(self):
        with TemporaryDirectory() as t:
            folder=Path(t)/('a'*32);folder.mkdir()
            write_json(folder/'state.json',{'id':'a'*32,'owner':'a@example.test','output_folder_id':'existing-output'})
            write_json(folder/'matches.json',[{'step_number':1,'script_step':'Tap tube.','best_match':None}])
            (folder/'narration-001.wav').write_bytes(b'saved audio')
            write_json(folder/'narration-001.json',{'spoken':'Tap tube.','voice_id':'voice','generation_speed':.88,'duration':2})
            step=SimpleNamespace(number=1,text='Tap tube.',previous_text='',next_text='')
            index=Mock();store=Mock();gateway=SimpleNamespace(calls=[],prior_calls=[])
            cfg={'google_service_account':{},'OUTPUT_FOLDER_ID':'output','LIBRARY_FOLDER_ID':'library',
                 'OPENAI_API_KEY':'test','ELEVENLABS_API_KEY':'test','voice':{'id':'voice','speed':.88}}
            with patch('pipeline.runner.DriveStore',return_value=store),patch('pipeline.runner.Library') as library,\
                 patch('pipeline.runner.CorpusIndex',return_value=index),patch('pipeline.runner.parse_steps_from_text',return_value=[step]),\
                 patch('pipeline.runner.match_steps') as match,patch('pipeline.runner.generate_step_audio') as tts,\
                 patch('pipeline.runner.ModelGateway',return_value=gateway),patch('pipeline.runner.Selector') as selector:
                library.from_folder.return_value=SimpleNamespace(index=lambda:'.',videos={'1':{}})
                selector.return_value.select.side_effect=VisionBudgetExceeded('limit')
                selector.return_value.attempts=[]
                run(folder,'a@example.test','Tap tube.',cfg)
                match.assert_not_called();tts.assert_not_called();store.folder.assert_not_called()
            state=json.loads((folder/'state.json').read_text())
            self.assertEqual(state['status'],'needs_review')
            self.assertIn('request limit',state['detail'])
            self.assertTrue((folder/'report.json').exists())
            index.connection.close.assert_called_once()

    def test_other_user_cannot_resume_a_saved_project(self):
        with TemporaryDirectory() as t:
            jobs=Jobs(t);ident='a'*32;folder=Path(t)/ident;folder.mkdir()
            write_json(folder/'state.json',{'id':ident,'owner':'a@example.test','status':'failed'})
            with self.assertRaises(PermissionError):jobs.resume(ident,'b@example.test',{})
            self.assertEqual(jobs.state(ident,'a@example.test')['status'],'failed')
            jobs.executor.shutdown()

if __name__=='__main__':unittest.main()
