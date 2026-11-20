"""Optional AI-created scientific illustrations. Never classify synthetic media as experiment footage.

Only called after JoVE footage selection, for explicitly opted-in jobs. Outputs are
review-only illustrations with an unmistakable provenance watermark.
"""
from pathlib import Path
from io import BytesIO
import base64
import math
import re
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .media import execute
from .editorial import action_units

IMAGE_MODEL_DEFAULT = "gpt-image-1.5"
SOURCE_POLICY_ILLUSTRATION = "jove_first_optional_illustrations"


def permitted(config):
    return config.get("ALLOW_AI_ILLUSTRATIONS") is True


def create_illustration_plan(step, audio, folder, config, client=None):
    """Create one labelled review-only diagram, never synthetic experimental evidence."""
    if not permitted(config):
        raise PermissionError("Illustration creation requires explicit per-job opt-in.")
    if not config.get("OPENAI_API_KEY") and client is None:
        raise ValueError("Illustration provider has no configured API key.")
    seconds = float(audio["duration"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Narration duration must be positive.")
    seconds = max(seconds, .5)
    target = Path(folder) / "illustrations"
    target.mkdir(parents=True, exist_ok=True)
    filename = f"step-{int(step.number):03}"
    png = target / (filename + ".png")
    video = target / (filename + ".mp4")
    if not video.is_file() or video.stat().st_size == 0:
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=config["OPENAI_API_KEY"], timeout=90, max_retries=0)
        prompt = (
            "Create a rigorously labelled, educational SCIENTIFIC SCHEMATIC ILLUSTRATION "
            "(not a photograph, not purported experimental footage). Render in a clean "
            "flat vector-like textbook visual style. Do not add any textual claims, "
            "measurements or outcomes unsupported by the input. Avoid photorealism, "
            "invented equipment configurations or specific specimen attributes. "
            "Where the specific action cannot be faithfully illustrated from this "
            "instruction alone, depict only neutral relevant apparatus. "
            "Scientific script action: " + step.text[:2000]
        )
        result = client.images.generate(
            model=config.get("IMAGE_MODEL", IMAGE_MODEL_DEFAULT),
            prompt=prompt, size="1536x1024", quality="medium"
        )
        payload = result.data[0].b64_json
        if not payload:
            raise ValueError("Illustration provider returned no image bytes.")
        image = Image.open(BytesIO(base64.b64decode(payload, validate=True))).convert("RGB")
        canvas = ImageOps.pad(image, (1280, 720), color="#f3f7fa")
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rectangle((0, 0, 1280, 65), fill=(8, 24, 44, 240))
        draw.rectangle((0, 655, 1280, 720), fill=(8, 24, 44, 240))
        draw.text((24, 20), "AI-GENERATED SCIENTIFIC ILLUSTRATION", font=ImageFont.load_default(size=27), fill="white")
        draw.text((24, 674), "NOT ORIGINAL EXPERIMENTAL FOOTAGE - REVIEW ONLY", font=ImageFont.load_default(size=23), fill="#ffe29b")
        canvas.save(png)
        # Extra decoder frames are intentional; the authoritative timeline is the narration clock.
        execute(["-loop", "1", "-framerate", "30000/1001", "-i", png,
                 "-t", f"{seconds+1:.3f}", "-r", "30000/1001",
                 "-an", "-c:v", "libx264", "-threads", "1",
                 "-preset", "veryfast", "-pix_fmt", "yuv420p", video],
                target / (filename + ".log"))
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError("Generated illustration could not be converted to video.")
    return {
        "step": step.number, "text": step.text, "shots": [{
            "video_id": "ai-illustration-step-" + str(step.number),
            "start": 0.0, "end": seconds, "path": str(video),
            "tier": "explicit_opt_in_illustration",
            "medium": "generated_illustration", "evidence": "illustrative",
            "covered_actions": [], "reason": "Synthetic educational illustration: not verified experimental footage."
        }],
        "missing_actions": action_units(step.text),
        "unobservable_details": ["Synthetic illustration does not demonstrate an actual experimental procedure."],
        "needs_review": True, "dense_checked": False, "scientific_audit_checked": False,
        "source_provenance": "AI-generated diagram, opt-in, not JoVE experimental footage",
    }
