"""Offline reference evaluation. This module is NEVER imported by the editor.

References stay private and are supplied after generation. Overlap is a retrieval
diagnostic, not a substitute for an editor reviewing scientific action coverage.
"""
from pathlib import Path
import argparse
import json


def evaluate(plan,references):
    shots=plan.get('shots',[]);rows=[]
    for ref in references:
        selected=[s for s in shots if s['step']==ref['step']]
        overlap=[]
        for s in selected:
            if str(s['video_id'])!=str(ref['video_id']):continue
            a=max(s['start'],ref['start']);b=min(s['end'],ref['end'])
            if b>a:overlap.append((a,b))
        merged=[]
        for a,b in sorted(overlap):
            if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(merged[-1][1],b))
            else:merged.append((a,b))
        seconds=sum(b-a for a,b in merged)
        rows.append({'step':ref['step'],'has_footage':bool(selected),
            'reference_video_retrieved':any(str(s['video_id'])==str(ref['video_id']) for s in selected),
            'reference_overlap_seconds':round(seconds,4),
            'reference_coverage':round(seconds/(ref['end']-ref['start']),4),
            'review_flag':any(s.get('review',True) for s in selected)})
    return {'schema_version':1,'steps':rows,'reference_overlap_steps':sum(r['reference_overlap_seconds']>0 for r in rows),
            'required_steps':len(rows),'scientific_acceptance':'Not measured. Requires independent playback review.',
            'limitation':'A different source can be equally valid. Matching a reference does not prove all narrated details.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(evaluate(json.loads(a.plan.read_text()),json.loads(a.references.read_text())),indent=2),encoding='utf-8')
