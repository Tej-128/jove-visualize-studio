from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch,Mock
import unittest
import hashlib
import zipfile
from pipeline.library import Library
from pipeline.runner import Jobs
from jove_visualizer.config import Settings
from jove_visualizer.models import ScriptStep,Candidate
from jove_visualizer.pipeline import match_steps
from jove_visualizer.rerank import ScientificReranker

class MatchingRuntimeTests(unittest.TestCase):
    def test_reports_activity_before_retrieval_and_after_reranking(self):
        with TemporaryDirectory() as t:
            events=[]
            index=SimpleNamespace(index_dir=Path(t),retrieve=lambda step:events.append('retrieve') or [])
            result=match_steps(index,[ScriptStep(1,'Tap the tube.')],Settings(openai_enabled=False),progress=lambda done,total,msg:events.append(msg))
            self.assertTrue(events[0].startswith('Retrieving caption'))
            self.assertEqual(events[1],'retrieve')
            self.assertTrue(any('reranking completed' in msg for msg in events))
            self.assertEqual(len(result),1)

    def test_sdk_does_not_multiply_application_attempts(self):
        settings=Settings(openai_max_retries=2,openai_timeout_seconds=60)
        client=Mock();client.responses.create.side_effect=TimeoutError('test timeout')
        candidate=Candidate(1,'1','Example',[],0,1000,'Tap the tube.','example.vtt')
        with patch('openai.OpenAI',return_value=client) as constructor,patch('jove_visualizer.rerank.time.sleep'):
            outcome=ScientificReranker(settings,api_key='test').rerank(ScriptStep(1,'Tap the tube.'),[candidate])
        self.assertEqual(constructor.call_args.kwargs['max_retries'],0)
        self.assertEqual(client.responses.create.call_count,2)
        self.assertFalse(outcome.ai_used)

    def test_title_cache_restores_and_rejects_wrong_index_or_contents(self):
        with TemporaryDirectory() as t:
            root=Path(t);archive=root/'source.zip'
            names=['article_cache_meta.json','article_ids.json','article_title_embeddings.npy','article_titles.json']
            with zipfile.ZipFile(archive,'w') as z:
                for name in names:z.writestr('v2_cache/'+name,b'prepared test bytes')
            class Store:
                def download(self,ident,dest):Path(dest).write_bytes(archive.read_bytes())
            lib=object.__new__(Library);lib.cache=root;lib.store=Store()
            checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
            lib.manifest={'title_cache':{'file_id':'cache','index_sha256':'index-a','sha256':checksum}}
            folder=root/'index';folder.mkdir()
            lib.restore_title_cache(folder,'index-a')
            self.assertEqual((folder/'v2_cache'/names[0]).read_bytes(),b'prepared test bytes')
            with self.assertRaisesRegex(ValueError,'does not match'):lib.restore_title_cache(folder,'index-b')
            with zipfile.ZipFile(archive,'w') as z:z.writestr('corpus.sqlite',b'not permitted')
            lib.manifest['title_cache']['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'contents'):lib.restore_title_cache(folder,'index-a')

if __name__=='__main__':unittest.main()
