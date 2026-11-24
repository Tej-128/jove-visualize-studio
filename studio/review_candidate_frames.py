"""Render indexed review frames for every local candidate of a requested step."""
import argparse
import io
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
for name in ("candidates", "media", "output"):
    parser.add_argument("--"+name, type=Path, required=True)
parser.add_argument("--step", type=int, required=True)
args = parser.parse_args()
media = {m["name"].split("_")[0].split(".")[0]: m for m in json.loads(args.media.read_bytes())["media"] if m["role"]=="Video"}
for index, action in enumerate(json.loads(args.candidates.read_bytes())["actions"]):
    if action["step"] != args.step or not action["local"]:
        continue
    sheet = Image.new("RGB", (1100, 75*len(action["local"])))
    draw = ImageDraw.Draw(sheet)
    for row, candidate in enumerate(action["local"]):
        vid = candidate["id"]
        start, end = candidate["visual_review_ms"]
        end = min(end, media[vid]["duration"]*1000)
        for col in range(10):
            sec = (start+(col+.5)*(end-start)/10)/1000
            data = subprocess.check_output(["ffmpeg", "-v","error","-ss",f"{sec:.3f}","-i",media[vid]["fingerprint"]["path"],"-frames:v","1","-vf","scale=110:56:force_original_aspect_ratio=decrease","-c:v","mjpeg","-f","image2pipe","-"])
            if data:
                sheet.paste(Image.open(io.BytesIO(data)),(col*110,row*75+14))
            draw.text((col*110,row*75),f"{vid} {sec:.1f}",fill="white")
    sheet.save(args.output/f"step-{args.step}-action-{index}.jpg",quality=89)
