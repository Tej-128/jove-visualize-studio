"""Script-derived action contracts and release checks, independent of model opinion."""
import math
import re


def action_units(text):
    """Keep exact script quotes. Split coordinated actions, not lists of apparatus."""
    verbs=(r'(?:add|adjust|allow|aliquot|anaesthetize|anesthetize|apply|assign|aspirate|attach|autoclave|blot|centrifuge|clean|collect|connect|continue|count|cut|decant|detach|dilute|discard|disinfect|dissect|dissolve|drain|euthanize|excise|expose|fill|flush|freeze|harvest|heat|homogenize|immerse|incubate|inject|inoculate|insert|invert|label|load|mark|measure|mix|mount|observe|open|perfuse|pipette|place|position|pour|prepare|press|record|remove|repeat|restrain|resuspend|rinse|rotate|scrape|seal|select|shake|shave|sonicate|stain|start|stir|stop|streak|suture|swirl|tap|thaw|transfect|transfer|trypsinize|turn|vortex|wash|weigh|wipe)')
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
    """A rendered review draft is never a scientific approval."""
    expected={s.number for s in steps};found={p['step'] for p in plans}
    missing_steps=sorted(expected-found);gaps=sorted(p['step'] for p in plans if not p.get('shots'));reasons=[]
    if missing_steps:reasons.append('Some script steps are missing from the timeline.')
    if gaps:reasons.append('Some steps have no suitable footage and use labelled cards.')
    if any(p.get('missing_actions') for p in plans if p.get('shots')):reasons.append('Some required actions are not covered by footage.')
    if any(p.get('status')=='representative_only' for p in plans):
        reasons.append('Representative footage illustrates general actions; protocol details remain unverified.')
    if any(not p.get('dense_checked') for p in plans if p.get('shots')):reasons.append('Some shots have not passed close visual checking.')
    if any(not math.isfinite(a['video_speed']) or not .85<=a['video_speed']<=1.151 for a in adjustments):
        reasons.append('Footage playback is outside the approved pilot range.')
    if any(not .95<=a['tempo']<=1.05 for a in adjustments):reasons.append('Speech pacing is outside the approved range.')
    if any(a.get('hold_seconds',0)>.3 for a in adjustments):reasons.append('Some footage is shorter than its narration; last frames are held.')
    if any(a.get('additional_viewing_seconds',0)>.3 for a in adjustments):reasons.append('Additional viewing pauses require review.')
    duration_ok=processing_seconds is not None and processing_seconds<=600
    statuses={k:sum(p.get('status')==k for p in plans) for k in ('selected','representative_only','no_suitable_footage')}
    return {'schema_version':2,'automated_checks_passed':not reasons,'publication_approved':False,
            'human_scientific_review_required':True,'reasons':reasons,'omitted_steps':missing_steps,'gap_steps':gaps,
            'step_status':statuses,'narration_retained':not missing_steps,
            'step_coverage':{'required':len(expected),'retained':len(found&expected),'with_footage':len(found&expected)-len(gaps)},
            'processing_within_ten_minutes':duration_ok if processing_seconds is not None else None,
            'timing_scope':'This invocation only; saved stages are identified separately in report.json.'}
