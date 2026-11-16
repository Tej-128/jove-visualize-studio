from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave
import numpy as np
import imageio_ffmpeg
from fractions import Fraction
from pipeline.media import execute,retime_audio
from pipeline.render import render

class RealMediaTests(unittest.TestCase):
    def test_seven_to_ten_seconds_preserves_tone_pitch(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source.wav';dest=root/'slower.wav'
            execute(['-f','lavfi','-i','sine=frequency=1000:sample_rate=48000:duration=7','-c:a','pcm_s16le',source],root/'make.log')
            retime_audio(source,dest,.7,300,Fraction(30))
            with wave.open(str(dest)) as w:
                self.assertEqual(w.getnframes(),480000)
                signal=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(float)
            middle=signal[48000:96000]
            spectrum=np.abs(np.fft.rfft(middle*np.hanning(len(middle))))
            peak=np.argmax(spectrum)*48000/len(middle)
            self.assertLess(abs(peak-1000),3)

    def test_complete_edit_has_expected_frame_count_and_portable_assets(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source.mp4';audio=root/'source.wav'
            execute(['-f','lavfi','-i','testsrc2=size=320x180:rate=30000/1001:duration=12','-an','-c:v','libx264','-threads','1','-preset','ultrafast',source],root/'video.log')
            execute(['-f','lavfi','-i','sine=frequency=440:sample_rate=44100:duration=2.8','-c:a','pcm_s16le',audio],root/'audio.log')
            shots=[{'step':i+1,'text':f'Action {i+1}','path':str(source),'video_id':'synthetic','start':a,'end':b,
                'audio_path':str(audio),'audio_duration':2.8,'tempo':2.8/(b-a),'confidence':1,'tier':'synthetic_test_only'}
                for i,(a,b) in enumerate([(1.,4.),(6.,9.)])]
            out=root/'output';out.mkdir()
            try:result=render(shots,out,width=320,height=180)
            except Exception:
                print((out/'render.log').read_text(errors='replace')[-4000:])
                raise
            count,_=imageio_ffmpeg.count_frames_and_secs(str(out/'preview.mp4'))
            self.assertEqual(count,result['frames'])
            self.assertEqual(result['dissolves'],1)
            self.assertTrue((out/'editable-package.zip').is_file())
            import xml.etree.ElementTree as E
            mlt=E.parse(out/'editable.mlt').getroot()
            resources=[x.text for x in mlt.findall('.//property[@name="resource"]')]
            self.assertIn('media/shot-001.mp4',resources)
            self.assertNotIn(str(source),resources)

if __name__=='__main__':unittest.main()

