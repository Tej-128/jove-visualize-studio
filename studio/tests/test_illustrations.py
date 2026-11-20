"""Regression tests for JoVE-only default and explicit synthetic-illustration opt-in."""
import base64
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
import json
import unittest

from PIL import Image
from pipeline.illustrations import permitted,create_illustration_plan
from pipeline.job_queue import JobQueue
from pipeline.action_render import render_actions
from pipeline.editorial import quality_report


class SourcePolicyTests(unittest.TestCase):
    def test_default_does_not_enable_illustration(self):
        self.assertFalse(permitted({}))
        self.assertFalse(permitted({"ALLOW_AI_ILLUSTRATIONS": False}))
        self.assertFalse(permitted({"ALLOW_AI_ILLUSTRATIONS": "true"}))

    def test_no_image_provider_call_when_not_opted_in(self):
        client=Mock()
        with TemporaryDirectory() as t:
            with self.assertRaises(PermissionError):
                create_illustration_plan(SimpleNamespace(number=1,text="Wash the cells"),
                                         {"duration":4},t,{},client=client)
            client.images.generate.assert_not_called()
            self.assertFalse((Path(t)/"illustrations").exists())

    def test_queue_preserves_separate_policy_and_does_not_store_keys(self):
        with TemporaryDirectory() as t:
            q=JobQueue(Path(t)/"queue.sqlite")
            cfg={"ENGINE_BUILD":"test","ALLOW_AI_ILLUSTRATIONS":True,
                 "IMAGE_MODEL":"gpt-image-1.5","OPENAI_API_KEY":"secret-value"}
            a=q.enqueue("a","owner","1. Wash cells.",cfg)
            b=q.enqueue("b","owner","1. Wash cells.",{**cfg,"ALLOW_AI_ILLUSTRATIONS":False})
            self.assertNotEqual(a,b)
            claimed=q.claim("worker","test")
            self.assertIs(claimed["config"]["ALLOW_AI_ILLUSTRATIONS"],True)
            self.assertEqual(claimed["config"]["IMAGE_MODEL"],"gpt-image-1.5")
            with q.connection() as db:
                records=[row[0] for row in db.execute("SELECT config FROM jobs")]
            self.assertNotIn("secret-value","".join(records))

    def test_opt_in_produces_labelled_review_plan(self):
        buf=BytesIO()
        Image.new("RGB",(512,512),"white").save(buf,"PNG")
        response=Mock()
        response.data=[SimpleNamespace(b64_json=base64.b64encode(buf.getvalue()).decode())]
        client=Mock()
        client.images.generate.return_value=response
        def fake_ffmpeg(args,log):
            Path(args[-1]).write_bytes(b"test encoded fixture")
        with TemporaryDirectory() as t:
            with patch("pipeline.illustrations.execute",side_effect=fake_ffmpeg):
                result=create_illustration_plan(
                    SimpleNamespace(number=2,text="Transfer the sample to a tube."),
                    {"duration":6.5},t,
                    {"ALLOW_AI_ILLUSTRATIONS":True},client=client)
            self.assertEqual(result["step"],2)
            self.assertEqual(result["shots"][0]["medium"],"generated_illustration")
            self.assertEqual(result["shots"][0]["evidence"],"illustrative")
            self.assertTrue(result["needs_review"])
            self.assertFalse(result["scientific_audit_checked"])
            self.assertTrue((Path(t)/"illustrations"/"step-002.png").is_file())
            self.assertTrue((Path(t)/"illustrations"/"step-002.mp4").is_file())
            client.images.generate.assert_called_once()
            with patch("pipeline.illustrations.execute",side_effect=AssertionError("re-encoded")):
                create_illustration_plan(SimpleNamespace(number=2,text="Transfer the sample to a tube."),
                                         {"duration":6.5},t,{"ALLOW_AI_ILLUSTRATIONS":True},client=client)
            client.images.generate.assert_called_once()

    def test_synthetic_illustration_never_passes_scientific_release_gate(self):
        step=SimpleNamespace(number=1)
        plan={"step":1,"needs_review":True,"dense_checked":False,"scientific_audit_checked":False,
              "missing_actions":["Transfer cells"],"shots":[{"medium":"generated_illustration","evidence":"illustrative"}]}
        report=quality_report([step],[plan],[{"video_speed":1.0,"tempo":1.0}],2)
        self.assertFalse(report["automated_checks_passed"])
        self.assertFalse(report["publication_approved"])

    def test_renderer_rejects_external_synthetic_file_path(self):
        with TemporaryDirectory() as t:
            folder=Path(t)/"job"
            folder.mkdir()
            outside=Path(t)/"untrusted.mp4"
            outside.write_bytes(b"incorrect")
            shots=[{"step":1,"video_id":"ai-illustration-step-1","path":str(outside),
                    "medium":"generated_illustration","frames":120,"start":0.0,"end":4.0}]
            with patch("pipeline.action_render.cut_schedule",return_value=(shots,[],120)):
                with self.assertRaisesRegex(ValueError,"Untrusted illustration"):
                    render_actions([{"step":1}],{},Mock(),folder)


if __name__=="__main__":
    unittest.main()
