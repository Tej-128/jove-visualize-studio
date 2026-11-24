"""Prepare a reviewed shot list and its word-aligned audit for the Premiere plugin."""
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FPS = 25
HANDLE = 10
SIZE = (960, 540)
DARK = "#152d36"
ACCENT = "#89dec8"


def font(size):
    return ImageFont.truetype("segoeui.ttf", size)


def text(draw, xy, value, size, fill):
    if draw.textlength(value, font=font(size)) > SIZE[0] - xy[0] - 18:
        raise ValueError(f"Text exceeds safe width: {value}")
    draw.text(xy, value, font=font(size), fill=fill)


def graphic(shot):
    card = shot.get("diagram")
    layer = Image.new("RGBA", SIZE, DARK if card else (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if card:
        text(draw, (50, 42), "PROTOCOL SCHEMATIC", 20, ACCENT)
        text(draw, (50, 102), card["title"], 40, "white")
        text(draw, (50, 178), card["description"], 25, "white")
        labels = card["nodes"]
        spacing = min(148, 806 / max(1, len(labels)-1))
        start = 480 - spacing * (len(labels)-1)/2
        for i, label in enumerate(labels):
            x = int(start + i*spacing)
            if i and card.get("connect"):
                draw.line((x-spacing+33,311,x-39,311),fill=ACCENT,width=3)
                draw.polygon([(x-34,311),(x-43,305),(x-43,317)],fill=ACCENT)
            draw.rounded_rectangle((x-30,278,x+30,345),radius=12,outline=ACCENT,width=3)
            draw.text((x,309),label,font=font(31),fill="white",anchor="mm")
        text(draw, (50, 407), card["note"], 22, ACCENT)
    elif shot.get("caption"):
        draw.rectangle((0,458,960,540),fill=(10,27,34,242))
        text(draw,(24,461),shot["caption"][0],24,ACCENT)
        text(draw,(24,497),shot["caption"][1],21,"white")
    return layer


def prepare(plan, manifest, transcript, project, *, allow_diagrams=False):
    if any(shot.get("diagram") for shot in plan) and not allow_diagrams:
        raise ValueError("Restore the reviewed clip or explicitly enable diagrams.")
    media = {v["id"]: v for v in manifest["media"]}
    words = [w for seg in transcript["segments"] for w in seg["words"]]
    voice = media[transcript["media_id"]]
    if voice["role"] != "Vo":
        raise ValueError("Word timings must belong to the original voiceover")
    work = project.parent
    videos = project / "Videos"
    videos.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(voice["fingerprint"]["path"],project/"Full original voiceover.wav")
    cursor = 0
    report = []
    for i, shot in enumerate(plan,1):
        start, end = [round(shot[k]*FPS) for k in ("start","end")]
        if start != cursor or end <= start:
            raise ValueError("Gap or overlap in narration timeline")
        cursor = end
        spoken = "".join(w["word"] for w in words if start/FPS <= (w["start"]+w["end"])/2 < end/FPS).strip()
        if not shot["observation"].strip():
            raise ValueError("Every shot needs a visual review")
        name = f"{i:02}-step-{shot['step']:02}.mp4"
        png = work / f"graphic-{i:02}.png"
        graphic(shot).save(png)
        cmd = ["ffmpeg","-v","error","-y","-filter_complex_threads","1"]
        if shot.get("diagram"):
            cmd += ["-loop","1","-framerate",str(FPS),"-i",str(png)]
        else:
            source = media[shot["media_id"]]
            at = shot["source_in"] - HANDLE/FPS
            if at < 0 or at + (end-start+2*HANDLE)/FPS > source["duration"]+.001:
                raise ValueError(f"Missing handles in {name}")
            cmd += ["-ss",str(round(at,4)),"-i",source["fingerprint"]["path"],"-i",str(png)]
            filt = "[0:v]setpts=PTS-STARTPTS,fps=25,scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2,setsar=1"
            if "detail" in shot:
                x,y,w,h,on,off=shot["detail"]
                filt += (f"[base];[base]split[wide][region];[region]crop={w}:{h}:{x}:{y},"
                         "scale=320:180,drawbox=x=0:y=0:w=iw:h=ih:color=white:t=3[detail];"
                         f"[wide][detail]overlay=12:12:enable='between(t,{on+.4},{off+.4})'")
            cmd += ["-filter_complex",filt+"[clip];[clip][1:v]overlay[out]","-map","[out]"]
        subprocess.run(cmd+["-frames:v",str(end-start+2*HANDLE),"-an","-c:v","libx264","-threads","2","-pix_fmt","yuv420p","-crf","17",str(videos/name)],check=True)
        report.append(dict(shot, file=name, spoken=spoken))
    if abs(cursor/FPS-voice["duration"])>1/FPS:
        raise ValueError("Timeline must include the complete voiceover")
    (work/"shot-review.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan","manifest","transcript","project"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--allow-diagrams", action="store_true", help="Enable explicitly requested schematic shots")
    args = parser.parse_args()
    prepare(*(json.loads(p.read_bytes()) for p in (args.plan,args.manifest,args.transcript)),args.project,allow_diagrams=args.allow_diagrams)

