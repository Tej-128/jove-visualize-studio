"""Script-derived action contracts and release checks, independent of model opinion."""
import math
import re


def action_units(text):
    """Keep exact script quotes. Split coordinated actions, not lists of apparatus."""
    verbs=r'(?:add|adjust|allow|assign|aspirate|centrifuge|collect|continue|count|cut|dissect|expose|fill|freeze|incubate|insert|invert|label|load|mark|measure|mix|mount|observe|pipette|place|position|pour|prepare|record|remove|repeat|resuspend|rinse|seal|select|shake|stain|start|stop|tap|transfer|turn|vortex|wash)'
    pieces=re.split(r';\s*|(?<!\d)\.(?=\s+[A-Z]|$)\s*|\s+(?:and\s+then|then|and|while)\s+(?='+verbs+r'\b)',text,flags=re.I)
    return list(dict.fromkeys(p.strip(' ,;.') for p in pieces if len(p.strip(' ,;.'))>=3)) or [text.strip()]


def coverage_gaps(step,shots):
    covered=[q.lower() for s in shots for q in s.get('covered_actions',[])]
    return [unit for unit in action_units(step.text) if not any(unit.lower()==q for q in covered)]


def metadata_only_review(plan):
    """Permit an annotated cut only when every action passed both visual checks."""
    shots = plan.get('shots', [])
    return bool(shots and plan.get('dense_checked') and plan.get('scientific_audit_checked')
                and plan.get('scientific_audit', {}).get('status') == 'compatible'
                and not plan.get('missing_actions') and not plan.get('needs_timing_review')
                and all(s.get('medium') == 'experimental_footage' and s.get('evidence') == 'demonstrated'
                        for s in shots))


def quality_report(steps,plans,adjustments,processing_seconds=None):
    """A rendered file never by itself passes the editorial release gate."""
    expected={s.number for s in steps};found={p['step'] for p in plans}
    omitted=sorted(expected-found);reasons=[]
    if omitted:reasons.append('Required steps have no usable footage.')
    if any(p.get('missing_actions') for p in plans):reasons.append('Some required physical actions are not covered.')
    if any(not p.get('dense_checked') or any(s.get('evidence')=='uncertain' for s in p['shots']) for p in plans):
        reasons.append('Some shots have not passed close visual checking.')
    if any(not p.get('scientific_audit_checked') for p in plans):reasons.append('Some selections have not passed the independent scientific compatibility audit.')
    if any(p.get('needs_review') for p in plans):reasons.append('Scientific illustration or unseen details require editorial review.')
    if any(not math.isfinite(a['video_speed']) or not .85<=a['video_speed']<=1.151 for a in adjustments):
        reasons.append('Footage playback is outside the approved pilot range.')
    if any(not .95<=a['tempo']<=1.05 for a in adjustments):reasons.append('Speech pacing is outside the approved range.')
    if any(a.get('additional_viewing_seconds',0)>.3 for a in adjustments):reasons.append('Additional viewing pauses require review.')
    duration_ok=processing_seconds is not None and processing_seconds<=600
    return {'schema_version':1,'automated_checks_passed':not reasons,'publication_approved':False,
            'human_scientific_review_required':True,'reasons':reasons,'omitted_steps':omitted,
            'step_coverage':{'required':len(expected),'retained':len(found&expected)},
            'processing_within_ten_minutes':duration_ok if processing_seconds is not None else None,
            'timing_scope':'This invocation only; saved stages are identified separately in report.json.'}
