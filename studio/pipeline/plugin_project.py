"""Prepare the inputs consumed by the original desktop editing workflow."""
from pathlib import Path
import wave
import zipfile
from docx import Document
from .sheets import read_candidates


def package_project(folder, steps, audios, library):
    workbook = folder / 'matches.xlsx'
    ids = sorted({row['video_id'] for row in read_candidates(workbook.read_bytes())})
    if not ids or any(i not in library.videos or not i.isdigit() for i in ids):
        raise ValueError('The shortlist is empty or a suggested source is unavailable.')
    script = Document()
    for step in steps:
        script.add_paragraph(step.text)
    script.save(folder / 'Script.docx')
    with wave.open(str(folder / 'Voiceover.wav'), 'wb') as dst:
        for index, audio in enumerate(audios):
            with wave.open(str(audio['path']), 'rb') as src:
                spec = src.getparams()[:3]
                if index == 0:
                    dst.setparams(src.getparams())
                elif spec != dst.getparams()[:3]:
                    raise ValueError('Narration segments have incompatible audio formats.')
                dst.writeframes(src.readframes(src.getnframes()))
    target = folder / 'Premiere-project.zip'
    temporary = target.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w') as archive:
        for path in (workbook, folder / 'Script.docx', folder / 'Voiceover.wav'):
            archive.write(path, path.name)
        for ident in ids:
            path = Path(library.video(ident))
            archive.write(path, 'Videos/' + ident + Path(library.videos[ident]['filename']).suffix)
    temporary.replace(target)
    return target
