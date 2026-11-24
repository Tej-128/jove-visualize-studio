from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
import time
import unittest
import numpy as np
from pipeline.action_catalog import ActionCatalog,validate_observations
from pipeline.job_queue import JobQueue
from pipeline.library_retrieval import LibraryCorpusIndex
from pipeline.editorial import action_units,quality_report
from pipeline.automatic_edit import validated_plan
from pipeline.action_render import cut_schedule,TimingCoverageError
from evaluate_edit import evaluate


class EditorialGateTests(unittest.TestCase):
    def test_action_contract_does_not_split_apparatus_but_keeps_separate_actions(self):
        self.assertEqual(action_units('Transfer flies and count them.'),['Transfer flies','count them'])
        self.assertEqual(action_units('Place tubes and vials on the bench.'),['Place tubes and vials on the bench'])
        self.assertEqual(action_units('Add 0.5 mL of reagent and incubate at 37.5 degrees.'),
                         ['Add 0.5 mL of reagent','incubate at 37.5 degrees'])
        self.assertEqual(action_units('Group flies by sex and exposure day.'),['Group flies by sex and exposure day'])
        self.assertEqual(action_units('Freeze and count flies, then assign a score.'),['Freeze','count flies','assign a score'])

    def test_model_cannot_silently_drop_the_second_script_action(self):
        answer={'shots':[{'window_id':'w0','start':1.,'end':3.,'evidence':'demonstrated',
            'covered_actions':['Transfer flies'],'reason':'Visible transfer','scientific_conflicts':[]}],
            'missing_actions':[],'unobservable_details':[]}
        plan=validated_plan(answer,[{'window_id':'w0','video_id':'12','start':0.,'end':10.,'tier':'suggested_timestamp'}],
            SimpleNamespace(number=1,text='Transfer flies and count them.'))
        self.assertIn('count them',plan['missing_actions']);self.assertTrue(plan['needs_review'])

    def test_subsecond_action_is_held_and_flagged_instead_of_stretched_across_long_vo(self):
        plan={'step':1,'shots':[{'start':1.,'end':1.7}]}
        shots,adjustments,_=cut_schedule([plan],{1:{'duration':8.58}})
        self.assertEqual(shots[0]['video_speed'],1.)
        self.assertGreater(shots[0]['hold_seconds'],7);self.assertGreater(adjustments[0]['hold_seconds'],7)
        self.assertTrue(shots[0]['review'])

    def test_output_with_uncertain_footage_never_passes_quality_gate(self):
        step=SimpleNamespace(number=1)
        plan={'step':1,'needs_review':True,'dense_checked':False,'missing_actions':[],
            'shots':[{'evidence':'uncertain'}]}
        report=quality_report([step],[plan],[{'video_speed':1,'tempo':1}],250)
        self.assertFalse(report['automated_checks_passed']);self.assertFalse(report['publication_approved'])
        self.assertTrue(report['processing_within_ten_minutes'])

    def test_reference_evaluation_does_not_double_count_overlapping_shots(self):
        plan={'shots':[{'step':1,'video_id':'12','start':1.,'end':4.},{'step':1,'video_id':'12','start':2.,'end':5.}]}
        out=evaluate(plan,[{'step':1,'video_id':'12','start':1.,'end':5.}])
        self.assertEqual(out['steps'][0]['reference_overlap_seconds'],4)


class QueueTests(unittest.TestCase):
    def test_update_transfers_unclaimed_jobs_to_new_build_and_disables_old_worker(self):
        with TemporaryDirectory() as t:
            queue=JobQueue(Path(t)/'jobs.sqlite');queue.activate_build('old')
            queue.enqueue('a','owner','script',{'ENGINE_BUILD':'old'})
            queue.activate_build('new')
            self.assertIsNone(queue.claim('old-worker','old'))
            with self.assertRaises(ValueError):queue.enqueue('b','owner','another script',{'ENGINE_BUILD':'old'})
            self.assertEqual(queue.claim('new-worker','new')['id'],'a')

    def test_configured_worker_executes_a_plain_script_without_credentials_in_queue(self):
        from pipeline.runner import Jobs,write_json
        import threading
        with TemporaryDirectory() as t:
            done=threading.Event()
            def fake_run(folder,owner,text,config):
                self.assertEqual(text,'1. Transfer cells.')
                self.assertEqual(config['OPENAI_API_KEY'],'private-test-marker')
                state=json.loads((folder/'state.json').read_text());state['status']='complete';write_json(folder/'state.json',state)
                done.set()
            config={'OPENAI_API_KEY':'private-test-marker','ENGINE_BUILD':'new','voice':{'id':'approved','speed':.88}}
            with patch('pipeline.runner.run',side_effect=fake_run):
                jobs=Jobs(t,worker_config=config)
                try:
                    ident=jobs.submit('owner','1. Transfer cells.',config)
                    self.assertTrue(done.wait(5))
                finally:
                    jobs.stop_event.set();jobs.worker_thread.join(timeout=5);jobs.executor.shutdown()
                self.assertEqual(jobs.queue.get(ident,'owner')['status'],'complete')

    def test_duplicate_requests_share_one_job_without_persisting_credentials(self):
        with TemporaryDirectory() as t:
            queue=JobQueue(Path(t)/'jobs.sqlite')
            cfg={'OPENAI_API_KEY':'secret-marker','ELEVENLABS_API_KEY':'secret-marker','google_service_account':{'private_key':'secret-marker'},
                 'voice':{'id':'approved','speed':.88},'ENGINE_BUILD':'build','LIBRARY_FOLDER_ID':'library'}
            self.assertEqual(queue.enqueue('a','one@example.test','1. Transfer cells.',cfg),'a')
            self.assertEqual(queue.enqueue('b','one@example.test','1. Transfer cells.',cfg),'a')
            with queue.connection() as db:stored=db.execute('SELECT config FROM jobs').fetchone()[0]
            self.assertNotIn('secret-marker',stored)
            with self.assertRaises(PermissionError):queue.get('a','other@example.test')
            self.assertEqual(queue.enqueue('c','one@example.test','1. Transfer cells.',{**cfg,'LIBRARY_FOLDER_ID':'different'}),'c')

    def test_two_workers_cannot_claim_same_job_and_expired_lease_is_recovered(self):
        with TemporaryDirectory() as t:
            queue=JobQueue(Path(t)/'jobs.sqlite');queue.enqueue('a','owner','script',{'ENGINE_BUILD':'build'})
            with ThreadPoolExecutor(max_workers=2) as pool:
                claimed=list(pool.map(lambda worker:queue.claim(worker,'build'),['one','two']))
            self.assertEqual(sum(x is not None for x in claimed),1)
            record=next(x for x in claimed if x)
            with queue.connection() as db:db.execute('UPDATE jobs SET lease_until=0')
            recovered=JobQueue(queue.path).claim('replacement','new-build')
            self.assertEqual(recovered['id'],record['id'])
            self.assertFalse(queue.heartbeat('a','one' if queue.get('a','owner')['worker']=='two' else 'two'))


class CatalogTests(unittest.TestCase):
    def test_catalog_is_reused_across_different_scripts_and_rejects_changed_sources(self):
        videos={'12':{'sha256':'a'*64}}
        row={'start':1,'end':8,'action':'Transfer cells with a pipette','apparatus':'pipette and tube',
             'specimen':'cells','context':'transfer into a tube','limitations':'Motion not verified'}
        with TemporaryDirectory() as t:
            catalog=ActionCatalog(t,videos)
            catalog.data['videos']['12']={'source_sha256':'a'*64,'duration':30,'complete':True,
                'observations':validate_observations({'observations':[row]},'12','a'*64,0,30)}
            catalog.save();restored=ActionCatalog(t,videos)
            self.assertTrue(restored.ready)
            self.assertEqual(restored.search('Transfer cells')[0]['video_id'],'12')
            self.assertEqual(restored.search('Pipette into the tube')[0]['video_id'],'12')
            self.assertEqual(restored.search('Transfer cells',video_ids=['99']),[])
            self.assertEqual(restored.search('Count zebrafish'),[])
            changed=ActionCatalog(t,{'12':{'sha256':'b'*64}})
            with self.assertRaises(ValueError):changed.load(catalog.path)

    def test_catalog_rejects_invalid_intervals_and_remains_discovery_only(self):
        row={'start':1,'end':8,'action':'Transfer cells','apparatus':'tube','specimen':'cells','context':'handling','limitations':'sparse frames'}
        self.assertEqual(validate_observations({'observations':[{**row,'end':50}]},'12','a'*64,0,30),[])
        self.assertEqual(validate_observations({'observations':[{**row,'start':float('nan')}]},'12','a'*64,0,30),[])
        result=validate_observations({'observations':[row]},'12','a'*64,0,30)
        self.assertEqual(result[0]['status'],'discovery_only')

    def test_human_rejected_footage_cannot_return_as_an_approved_plan(self):
        from pipeline.automatic_edit import AutomaticEditor
        with TemporaryDirectory() as t:
            root=Path(t);(root/'review-exclusions.json').write_text(json.dumps([{'step':1,'video_id':'12','start':1,'end':3}]))
            library=SimpleNamespace(videos={'12':{'sha256':'a'*64}})
            step=SimpleNamespace(number=1,text='Transfer cells.')
            editor=AutomaticEditor(None,library,Mock(),root/'automatic',[step],{1:{'duration':2}})
            answer={'shots':[{'window_id':'w0','start':1,'end':3,'evidence':'demonstrated','covered_actions':['Transfer cells'],
                    'reason':'Model proposes previously rejected shot','scientific_conflicts':[]}],'missing_actions':[],'unobservable_details':[]}
            self.assertIsNone(editor.validate(answer,[{'window_id':'w0','video_id':'12','start':0,'end':5,'tier':'suggested_timestamp'}],step))


class AvailableMediaTests(unittest.TestCase):
    def test_unavailable_best_semantic_and_lexical_hits_cannot_crowd_out_available_media(self):
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
        db.execute('CREATE TABLE passages(passage_id INTEGER PRIMARY KEY,article_id TEXT,normalized_text TEXT)')
        db.executemany('INSERT INTO passages VALUES(?,?,?)',[(1,'unavailable','transfer cells'),(2,'available','transfer cells')])
        db.execute("CREATE VIRTUAL TABLE passage_fts USING fts5(normalized_text,content='passages',content_rowid='passage_id')")
        db.execute("INSERT INTO passage_fts(passage_fts) VALUES('rebuild')")
        embedder=SimpleNamespace(encode_queries=lambda *args,**kw:np.array([[1.,0.]]))
        base=SimpleNamespace(connection=db,embeddings=np.array([[1.,0.],[.9,.1]]),embedder=embedder)
        index=LibraryCorpusIndex(base,['available'])
        semantic,_=index._semantic_top_k(['Transfer cells'],1)
        lexical=index._lexical_top_k('Transfer cells',1)
        self.assertEqual(semantic[0][0][0],2);self.assertEqual(lexical[0][0],2)
        db.close()


class WordInputTests(unittest.TestCase):
    def test_plain_word_automatic_numbering_and_table_order_survive_extraction(self):
        from docx import Document
        from io import BytesIO
        from pipeline.documents import word_script_text
        document=Document();document.add_heading('A new assay',0)
        document.add_paragraph('Transfer cells into a tube.',style='List Number')
        table=document.add_table(rows=1,cols=1);table.cell(0,0).text='Note: keep the tube upright.'
        document.add_paragraph('Incubate the tube.',style='List Number')
        data=BytesIO();document.save(data)
        extracted=word_script_text(data.getvalue())
        self.assertIn('1. Transfer cells',extracted);self.assertIn('2. Incubate',extracted)
        self.assertLess(extracted.index('Note:'),extracted.index('2. Incubate'))


if __name__=='__main__':unittest.main()
