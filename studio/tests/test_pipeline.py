from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from io import BytesIO
import unittest
import zipfile
import math
import wave
from pipeline.timing import fit_audio,atempo_filters,allocate_frames,pacing_outliers
from pipeline.storage import safe_extract
from pipeline.sheets import read_candidates
from pipeline.selection import Selector

class TimingTests(unittest.TestCase):
    def test_seven_seconds_into_ten_and_review(self):
        result=fit_audio(7,10)
        self.assertAlmostEqual(result.tempo,.7)
        self.assertAlmostEqual(result.duration_multiplier,10/7)
        self.assertTrue(result.needs_review)

    def test_natural_small_adjustment(self):
        self.assertFalse(fit_audio(9,10).needs_review)

    def test_invalid_durations(self):
        for value in [0,-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):fit_audio(value,10)

    def test_pacing_changes_are_visible(self):
        self.assertEqual(pacing_outliers([.7,1,.95]),[1])

    def test_rounding_does_not_accumulate_gaps(self):
        rows=allocate_frames([.101]*300)
        self.assertEqual(rows[-1][1],round(30.3*30000/1001))
        self.assertTrue(all(a[1]==b[0] for a,b in zip(rows,rows[1:])))

    def test_chained_tempo_preserves_ratio(self):
        product=math.prod(float(part.split('=')[1]) for part in atempo_filters(.17).split(','))
        self.assertAlmostEqual(product,.17)

class InputTests(unittest.TestCase):
    def test_changed_video_checksum_does_not_reuse_old_cached_media(self):
        from pipeline.library import Library
        import hashlib
        with TemporaryDirectory() as temp:
            data={'old':b'old media','new':b'new approved media'}
            calls=[]
            class Store:
                def download(self,file_id,dest):
                    calls.append(file_id);Path(dest).write_bytes(data[file_id])
            lib=object.__new__(Library);lib.cache=Path(temp);lib.store=Store()
            def asset(key):return {'filename':'1.mp4','file_id':key,'sha256':hashlib.sha256(data[key]).hexdigest()}
            lib.videos={'1':asset('old')};old=lib.video('1')
            lib.videos={'1':asset('new')};new=lib.video('1')
            self.assertNotEqual(old,new)
            self.assertEqual(new.read_bytes(),data['new'])
            self.assertEqual(lib.video('1'),new)
            self.assertEqual(calls,['old','new'])

    def test_personal_drive_output_is_rejected_before_file_creation(self):
        from pipeline.storage import DriveStore
        store=object.__new__(DriveStore);calls=[]
        store.info=lambda ident:{'id':ident}
        store.request=lambda *args,**kwargs:calls.append(args)
        with self.assertRaisesRegex(ValueError,'Shared Drive'):store.folder('personal-folder','job')
        self.assertEqual(calls,[])

    def test_transfer_parts_reassemble_original_bytes_and_reject_corruption(self):
        from pipeline.library import Library
        import hashlib
        with TemporaryDirectory() as temp:
            root=Path(temp)
            data={'a':b'original first bytes','b':b' and remaining bytes'}
            class Store:
                def download(self,file_id,dest):
                    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data[file_id])
            lib=object.__new__(Library);lib.cache=root;lib.store=Store()
            parts=[{'file_id':k,'sha256':hashlib.sha256(v).hexdigest()} for k,v in data.items()]
            dest=root/'whole'
            lib.download_asset({'parts':parts},dest)
            self.assertEqual(dest.read_bytes(),data['a']+data['b'])
            (root/'parts'/parts[0]['sha256']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):lib.download_asset({'parts':parts},dest)

    def test_zip_traversal_is_rejected(self):
        with TemporaryDirectory() as t:
            p=Path(t)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','bad')
            with self.assertRaises(ValueError):safe_extract(p,Path(t)/'out')
            self.assertFalse((Path(t)/'escape').exists())

    def test_exact_times_and_human_feedback_preserved(self):
        from openpyxl import Workbook
        b=Workbook();s=b.active;s.append(['Step','Script Step','Journal ID','Start','End','Exact time stamps','Feedback Notes'])
        s.append([1,'Action',123,'01:00','02:00','01:12 - 01:19','Accurate'])
        data=BytesIO();b.save(data)
        row=read_candidates(data.getvalue())[0]
        self.assertEqual((row['start'],row['end']),(72,79));self.assertEqual(row['feedback'],'Accurate')
        self.assertEqual(row['original_start'],'01:00')

class SearchOrderTests(unittest.TestCase):
    def selector(self,success_tier=None):
        class TestSelector(Selector):
            def wider_candidates(self,step,excluded):return self.index.retrieve(step)
            def inspect(self,step,candidate,tier,audio):
                self.attempts.append((tier,candidate['video_id']))
                if tier==success_tier:return {'needs_timing_review':False,'confidence':.9,'tempo':1,'step':step.number}
            def full_video_windows(self,step,vid):return [{'video_id':vid,'start':1,'end':4}]
        index=SimpleNamespace(retrieve=lambda s:[SimpleNamespace(article_id='4',start_ms=0,end_ms=4000,text='action')])
        return TestSelector(index,SimpleNamespace(videos={'1':1,'2':2,'3':3,'4':4}),None,'.')

    def test_no_widening_until_all_timestamp_options_checked(self):
        s=self.selector('suggested_timestamp')
        s.select(SimpleNamespace(number=1),[{'video_id':str(i)} for i in (1,2,3)],3)
        self.assertEqual(s.attempts,[('suggested_timestamp',str(i)) for i in (1,2,3)])

    def test_full_suggested_videos_before_library(self):
        s=self.selector('full_library_retrieval')
        s.select(SimpleNamespace(number=1),[{'video_id':str(i)} for i in (1,2,3)],3)
        self.assertEqual([x[0] for x in s.attempts],['suggested_timestamp']*3+['full_suggested_video']*3+['full_library_retrieval'])

    def test_missing_match_is_not_invented(self):
        s=self.selector();self.assertIsNone(s.select(SimpleNamespace(number=1),[{'video_id':'1'}],3))

class AccessTests(unittest.TestCase):
    def test_verified_email_and_explicit_invitation_are_both_required(self):
        from pipeline.access import invited
        self.assertTrue(invited(' A@EXAMPLE.TEST ',True,['a@example.test']))
        self.assertFalse(invited('a@example.test',False,['a@example.test']))
        self.assertFalse(invited('a@example.test','true',['a@example.test']))
        self.assertFalse(invited('b@example.test',True,['a@example.test']))
        self.assertFalse(invited('',True,['']))

    def test_other_owner_cannot_read_job_or_path(self):
        from pipeline.runner import Jobs,write_json
        with TemporaryDirectory() as t:
            j=Jobs(t);ident='a'*32;folder=Path(t)/ident;folder.mkdir()
            write_json(folder/'state.json',{'owner':'a@example.test','id':ident})
            with self.assertRaises(PermissionError):j.state(ident,'b@example.test')
            with self.assertRaises(ValueError):j.state('../escape','a@example.test')
            j.executor.shutdown()

if __name__=='__main__':unittest.main()

