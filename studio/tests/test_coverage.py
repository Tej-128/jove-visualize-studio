from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
from pipeline.coverage import storyboards,screen_video
from pipeline.media import execute
from pipeline.runner import Jobs,run_full_subset,write_json
from pipeline.providers import VisionBudgetExceeded
from pipeline.coverage import valid_prepared_manifest
from pipeline.coverage import suggested_candidates
from pipeline.library import Library
import json
import hashlib
import zipfile
import sqlite3


class CoverageTests(unittest.TestCase):
    def test_original_timestamps_survive_missing_storyboard_discovery_but_unavailable_sources_do_not(self):
        row={'step_number':1,'best_match':{'journal_id':'1','start_timestamp':'02:26','end_timestamp':'02:35','caption_passage':'Transfer flies.','confidence_score':.9},
             'alternatives':[{'journal_id':'absent','start_timestamp':'00:01','end_timestamp':'00:05'}]}
        candidates=suggested_candidates([row],{'1'})
        self.assertEqual(len(candidates),1)
        self.assertEqual((candidates[0]['start'],candidates[0]['end']),(146,155))
        self.assertEqual(candidates[0]['tier'],'suggested_timestamp')

    def test_source_captions_ground_the_actual_interval_and_missing_captions_do_not_error(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            with sqlite3.connect(root/'corpus.sqlite') as db:
                db.execute('CREATE TABLE passages(article_id TEXT,start_ms INTEGER,end_ms INTEGER,text TEXT)')
                db.executemany('INSERT INTO passages VALUES(?,?,?,?)',[('1',10000,20000,'Draw the line on the tube.'),('1',100000,120000,'Analyze software distances.'),('2',10000,20000,'Unrelated specimen.')])
            db.close()
            lib=object.__new__(Library);lib.index=lambda:root
            self.assertEqual(lib.caption_context('1',12,18)[0]['text'],'Draw the line on the tube.')
            self.assertEqual(lib.caption_context('missing',12,18),[])
            self.assertEqual(lib.caption_context("1' OR 1=1 --",12,18),[])

    def test_prepared_overviews_are_bound_to_source_hashes_and_cannot_reference_other_files(self):
        with TemporaryDirectory() as t:
            root=Path(t);archive=root/'source.zip';source_sha='a'*64
            manifest={'version':1,'interval':3.,'duration':10,'frame_count':4,'sheets':[{'file':'sheet-0000.jpg','start':0,'end':10}]}
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('1/'+source_sha[:16]+'/storyboards.json',json.dumps(manifest))
                z.writestr('1/'+source_sha[:16]+'/sheet-0000.jpg',b'image bytes')
            lib=object.__new__(Library);lib.cache=root;lib.videos={'1':{'sha256':source_sha}}
            lib.manifest={'overview_cache':{'version':1,'interval':3.,'sources':{'1':source_sha},'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'file_id':'test'}}
            lib.store=SimpleNamespace(download=lambda ident,dest:Path(dest).write_bytes(archive.read_bytes()))
            lib.restore_overviews();self.assertTrue(lib.has_prepared_overview('1'))
            prepared=root/'full-overviews'/'1'/source_sha[:16]
            self.assertEqual(storyboards(None,prepared),manifest)
            manifest['sheets'][0]['file']='../../secrets.toml'
            self.assertFalse(valid_prepared_manifest(manifest,prepared))
            lib.videos['1']['sha256']='b'*64
            self.assertFalse(lib.has_prepared_overview('1'))

    def make_project(self,root):
        folder=Path(root)/('a'*32);folder.mkdir()
        write_json(folder/'state.json',{'id':'a'*32,'owner':'a@example.test','output_folder_id':'output','status':'needs_review'})
        (folder/'input.txt').write_text('1. Transfer sample.')
        (folder/'narration-001.wav').write_bytes(b'saved narration')
        write_json(folder/'narration-001.json',{'spoken':'Transfer sample.','duration':2})
        return folder

    def run_survey(self,folder,screen):
        videos={str(i):{'sha256':str(i).zfill(64)} for i in range(1,31)}
        lib=SimpleNamespace(videos=videos,video=lambda ident:Path('source-'+ident+'.mp4'),restore_overviews=lambda:None,has_prepared_overview=lambda ident:False)
        config={'google_service_account':{},'LIBRARY_FOLDER_ID':'library','OPENAI_API_KEY':'test'}
        def review(verified,steps,audios,destination):
            write_json(Path(destination)/'edit_plan.json',{'fps':'30','shots':[{'step':1,'timeline_start_frame':0,'timeline_end_frame':60,'audio_duration':2.,'start':.2,'end':2.2,'video_id':'review-placeholder','kind':'footage_not_verified'}]})
            return {'placeholder_steps':[1],'review_only':True}
        with patch('pipeline.runner.DriveStore'),patch('pipeline.runner.Library') as factory,\
             patch('pipeline.runner.ModelGateway',return_value=SimpleNamespace(calls=[],prior_calls=[])),\
             patch('pipeline.runner.CorpusIndex') as index,patch('pipeline.runner.generate_step_audio') as tts,\
             patch('pipeline.coverage.screen_video',side_effect=screen),patch('pipeline.runner.render') as render,\
             patch('pipeline.render.render_review_draft',side_effect=review):
            factory.from_folder.return_value=lib
            run_full_subset(folder,'a@example.test',config)
            index.assert_not_called();tts.assert_not_called();render.assert_not_called()

    def test_all_thirty_videos_are_screened_without_caption_or_embedding_dependencies(self):
        calls=[]
        def screen(vid,path,sha,*args,**kwargs):
            calls.append(vid)
            return {'video_id':vid,'source_sha256':sha,'status':'screened_by_storyboards','candidates':[]}
        with TemporaryDirectory() as t:
            folder=self.make_project(t);self.run_survey(folder,screen)
            coverage=json.loads((folder/'coverage.json').read_text())
            self.assertEqual(set(calls),{str(i) for i in range(1,31)})
            self.assertEqual(len(calls),30);self.assertTrue(coverage['complete'])
            state=json.loads((folder/'state.json').read_text())
            self.assertEqual(state['status'],'needs_review')
            self.assertTrue(state['full_subset_finished'])

    def test_request_limit_resume_does_not_skip_or_repeat_completed_videos(self):
        first=[];second=[]
        def screen(vid,path,sha,*args,**kwargs):
            first.append(vid)
            if len(first)==8:raise VisionBudgetExceeded('limit')
            return {'video_id':vid,'source_sha256':sha,'status':'screened_by_storyboards','candidates':[]}
        def resumed(vid,path,sha,*args,**kwargs):
            second.append(vid)
            return {'video_id':vid,'source_sha256':sha,'status':'screened_by_storyboards','candidates':[]}
        with TemporaryDirectory() as t:
            folder=self.make_project(t);self.run_survey(folder,screen)
            coverage=json.loads((folder/'coverage.json').read_text())
            self.assertFalse(coverage['complete']);self.assertEqual(len(coverage['videos']),7)
            self.run_survey(folder,resumed)
            self.assertEqual(len(second),23)
            self.assertEqual(set(first[:7])&set(second),set())
            self.assertTrue(json.loads((folder/'coverage.json').read_text())['complete'])

    def test_other_owner_cannot_launch_full_subset_search(self):
        with TemporaryDirectory() as t:
            folder=self.make_project(t);jobs=Jobs(t)
            with self.assertRaises(PermissionError):jobs.full_subset(folder.name,'b@example.test',{})
            jobs.executor.shutdown()

    def test_decodes_whole_timeline_and_records_real_sample_times(self):
        with TemporaryDirectory() as t:
            root=Path(t);source=root/'source.mp4'
            execute(['-f','lavfi','-i','testsrc2=size=320x180:rate=30:duration=7.2','-an','-c:v','libx264','-threads','1','-preset','ultrafast',source],root/'make.log')
            result=storyboards(source,root/'evidence')
            self.assertEqual(result['frame_count'],3)
            self.assertEqual(result['first_sample_seconds'],0)
            self.assertAlmostEqual(result['last_sample_seconds'],6)
            self.assertLessEqual(result['maximum_sample_gap_seconds'],3.1)
            with patch('pipeline.coverage.execute',side_effect=AssertionError('must reuse prepared evidence')):
                self.assertEqual(storyboards(source,root/'evidence'),result)

    def test_no_caption_dependency_and_every_storyboard_batch_is_examined(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            dest=root/'arbitrary-video'/('a'*16);dest.mkdir(parents=True)
            sheets=[]
            for i in range(17):
                name=f'sheet-{i}.jpg';(dest/name).write_bytes(b'test image bytes')
                sheets.append({'file':name,'start':i*48,'end':(i+1)*48})
            manifest={'sheets':sheets,'duration':816,'interval':3,'frame_count':272}
            gateway=Mock();gateway.ask.return_value={'summary':'No action seen.','candidates':[]}
            with patch('pipeline.coverage.storyboards',return_value=manifest):
                result=screen_video('arbitrary-video','source.mp4','a'*64,[SimpleNamespace(number=1,text='Transfer sample.')],gateway,root)
            self.assertEqual(gateway.ask.call_count,3)
            self.assertEqual(result['status'],'screened_by_storyboards')
            self.assertEqual(result['summaries'][-1]['end'],816)
            self.assertEqual(gateway.ask.call_args_list[0].args[2][0]['detail'],'high')

    def test_out_of_bounds_and_invented_step_candidates_are_rejected(self):
        with TemporaryDirectory() as t:
            root=Path(t);dest=root/'video'/('a'*16);dest.mkdir(parents=True);(dest/'sheet.jpg').write_bytes(b'image')
            manifest={'sheets':[{'file':'sheet.jpg','start':0,'end':30}],'duration':30,'interval':3,'frame_count':10}
            candidates=[{'step':1,'start':1,'end':5,'confidence':.9}, {'step':99,'start':1,'end':5,'confidence':.9},
                        {'step':1,'start':-1,'end':5,'confidence':.9},{'step':1,'start':1,'end':float('inf'),'confidence':.9}]
            gateway=Mock();gateway.ask.return_value={'candidates':candidates}
            with patch('pipeline.coverage.storyboards',return_value=manifest):
                result=screen_video('video','source.mp4','a'*64,[SimpleNamespace(number=1,text='Action.')],gateway,root)
            self.assertEqual(len(result['candidates']),1)
            self.assertEqual(result['candidates'][0]['status'],'discovery_only')

if __name__=='__main__':unittest.main()
