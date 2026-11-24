"""Exercise the actual MP4 at an internal cut and across two protocol steps."""
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
from pipeline.action_render import render_actions


class AllShotDissolves(unittest.TestCase):
    def test_every_join_in_export(self):
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("black", "white", "gray"):
                subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                                f"color={name}:size=160x90:rate=25:duration=4.4",
                                str(folder / (name + ".mp4"))], check=True)
            plans, audios = [], {}
            for number, colors in enumerate((("black", "white"), ("gray",)), 1):
                duration = 4 * len(colors)
                wav = folder / f"{number}.wav"
                with wave.open(str(wav), "wb") as file:
                    file.setparams((1, 2, 48000, 0, "NONE", "uncompressed"))
                    file.writeframes(bytes(duration*96000))
                audios[number] = {"duration": duration, "path": wav}
                shots = [dict(start=0, end=4, video_id=color, reason="transition test") for color in colors]
                plans.append(dict(step=number, text="test", shots=shots, needs_review=False,
                                  missing_actions=[], unobservable_details=[]))
            library = SimpleNamespace(video=lambda color: folder / (color + ".mp4"))
            output = folder / "output"
            render_actions(plans, audios, library, output, fps="25", width=160, height=90)
            frames = subprocess.check_output([ffmpeg, "-v", "error", "-i", str(output / "preview.mp4"),
                                              "-an", "-vf", "scale=1:1,format=gray", "-f", "rawvideo", "-"])
            self.assertEqual(len(frames), 300)
            for cut in (100, 200):
                left, right = frames[cut-5], frames[cut+5]
                for i in range(9):
                    self.assertAlmostEqual(frames[cut-4+i], left+(right-left)*i/8, delta=4)
