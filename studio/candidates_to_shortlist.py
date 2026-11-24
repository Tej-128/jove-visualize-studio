"""Turn a saved candidate search into the shortlist workbook the editor reads.

Input is either a candidates.json ({"actions": [{"step", "action", "local": [{"id", "start_ms", "end_ms",
"caption", "title"}]}]}) or an offline-discovery job folder (step-NN.json files with caption transcripts
and storyboard observations). Works for any numbered script; nothing is tuned to one protocol.

Each step's candidates are interleaved across its actions, so every action gets its best leads first.
Overlapping ranges from the same video are dropped and each step keeps at most --per-step rows. The
output has the Step / Script Step / Journal ID / Start / End / Matching CC Text columns that
pipeline.sheets.read_candidates expects. Ranks are search order only, never a visual judgement.

    python candidates_to_shortlist.py --input candidates.json --output matches.xlsx
    python candidates_to_shortlist.py --input phase2-results/drosophila/<job> --output matches.xlsx
"""
import argparse
import json
from pathlib import Path

from openpyxl import Workbook

HEADER = ['Step', 'Script Step', 'Journal ID', 'Start', 'End', 'Matching CC Text', 'Source title', 'Search rank']


def _row(video_id, start, end, caption='', title=''):
    return {'video_id': str(video_id).removesuffix('.0'), 'start': float(start), 'end': float(end),
            'caption': str(caption or '')[:2000], 'title': str(title or '')[:300]}


def from_candidates_json(data):
    steps, texts = {}, {}
    for action in data.get('actions', []):
        number = int(action['step'])
        texts.setdefault(number, []).append(str(action.get('action', '')).strip())
        steps.setdefault(number, []).append([_row(c['id'], c['start_ms'] / 1000, c['end_ms'] / 1000, c.get('caption'), c.get('title'))
                                             for c in action.get('local') or []])
    return steps, {k: '; '.join(v) for k, v in texts.items()}


def from_discovery_folder(folder):
    steps, texts = {}, {}
    for path in sorted(Path(folder).glob('step-*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        number = int(data['step']);texts[number] = data.get('text', '')
        for action in data.get('actions', []):
            rows = [_row(c['video_id'], c['start'], c['end'], c.get('caption'), c.get('title')) for c in action.get('transcripts', [])]
            rows += [_row(o['video_id'], o['start'], o['end'], o.get('caption') or o.get('action')) for o in action.get('observations', [])]
            steps.setdefault(number, []).append(rows)
    return steps, texts


def interleave(lists, per_step):
    chosen = []
    for rank in range(max((len(rows) for rows in lists), default=0)):
        for rows in lists:
            if rank >= len(rows):
                continue
            c = rows[rank]
            if c['end'] <= c['start'] or any(o['video_id'] == c['video_id'] and c['start'] < o['end'] and c['end'] > o['start'] for o in chosen):
                continue
            chosen.append({**c, 'rank': rank + 1})
            if len(chosen) >= per_step:
                return chosen
    return chosen


def write_shortlist(source, output, per_step=6):
    source = Path(source)
    steps, texts = (from_discovery_folder(source) if source.is_dir()
                    else from_candidates_json(json.loads(source.read_text(encoding='utf-8'))))
    book = Workbook();sheet = book.active;sheet.title = 'Shortlist';sheet.append(HEADER)
    counts = {}
    for number in sorted(steps):
        rows = interleave(steps[number], per_step);counts[number] = len(rows)
        for c in rows:
            sheet.append([number, texts.get(number, ''), c['video_id'], round(c['start'], 3), round(c['end'], 3),
                          c['caption'], c['title'], c['rank']])
    book.save(output)
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-step', type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(write_shortlist(args.input, args.output, args.per_step)))
