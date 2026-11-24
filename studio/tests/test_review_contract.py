"""Review-draft contract tests. Steps come from several fields on purpose: nothing may be tuned to one script."""
import json
import unittest
import wave
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import imageio_ffmpeg

from pipeline.action_render import render_actions
from pipeline.automatic_edit import AutomaticEditor
from pipeline.editorial import quality_report
from pipeline.media import execute
from pipeline.review_contract import (CONFLICT_TYPES, DIFFERENCE_TO_CONFLICT, DIFFERENCES, STATUSES, contract_record,
                                      decompose_step, review_markdown, review_plan, review_schema, rule_decomposition)

STEPS = {
    'plate': 'Add 100 microliters of the diluted antigen to each well of a 96-well plate.',
    'seal': 'Seal the plate and incubate it overnight at 4 degrees Celsius.',
    'animal': 'Restrain the mouse and inject 200 µL of the compound intraperitoneally.',
    'microscopy': 'Place the slide on the stage and focus on the cells at 40x magnification.',
    'software': 'Open the image in the analysis software and measure the mean fluorescence intensity.',
}


def windows():
    return [dict(window_id='w0', video_id='70511', start=10., end=30., tier='suggested_timestamp'),
            dict(window_id='w1', video_id='55403', start=380., end=420., tier='suggested_timestamp')]


def shot(window='w0', start=12., end=16., level='representative', illustrates=None, **extra):
    return dict(window_id=window, start=start, end=end, action_start=start, action_end=end, evidence_level=level,
                illustrates=illustrates or [], visible_observations=['Gloved hand pipettes into a multiwell plate.'],
                unverified_details=['Reagent identity', 'Volume per well'], editor_note='Representative pipetting.', **extra)


def answer(shots=(), exclusions=(), missing=(), flags=()):
    return dict(status='representative_only', shots=list(shots), exclusion_notes=list(exclusions),
                missing_actions=list(missing), review_flags=list(flags))


class DecompositionTests(unittest.TestCase):
    def test_rules_split_actions_and_collect_unverifiable_quantities_across_fields(self):
        seal = rule_decomposition(STEPS['seal'])
        self.assertEqual([a['quote'] for a in seal['actions']], ['Seal the plate', 'incubate it overnight at 4 degrees Celsius'])
        self.assertEqual([a['kind'] for a in seal['actions']], ['physical', 'waiting'])
        self.assertIn('4 degrees Celsius', seal['details'])
        animal = rule_decomposition(STEPS['animal'])
        self.assertEqual(len(animal['actions']), 2)
        self.assertIn('200 µL', animal['details'])
        plate = rule_decomposition(STEPS['plate'])
        self.assertIn('100 microliters', plate['details']);self.assertIn('96-well', plate['details'])

    def test_model_actions_must_be_quoted_from_the_step(self):
        gateway = Mock()
        gateway.ask.return_value = {'actions': [
            {'quote': 'open the image in the analysis software', 'visible_action': 'software on screen', 'kind': 'screen'},
            {'quote': 'calibrate the microscope', 'visible_action': 'invented', 'kind': 'physical'}],
            'details': ['mean fluorescence intensity']}
        result = decompose_step(STEPS['software'], gateway)
        self.assertEqual(result['method'], 'model')
        self.assertEqual([a['quote'] for a in result['actions']], ['Open the image in the analysis software'])
        self.assertEqual(result['actions'][0]['kind'], 'screen')

    def test_provider_failure_falls_back_to_rules(self):
        gateway = Mock();gateway.ask.side_effect = TimeoutError
        result = decompose_step(STEPS['microscopy'], gateway)
        self.assertEqual(result['method'], 'rules');self.assertEqual(result['model_error'], 'TimeoutError')
        self.assertGreaterEqual(len(result['actions']), 1)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.decomposition = rule_decomposition(STEPS['plate'])
        self.quote = self.decomposition['actions'][0]['quote']

    def plan(self, model_answer, text=STEPS['plate'], decomposition=None):
        return review_plan(model_answer, windows(), 2, text, decomposition or self.decomposition)

    def test_representative_footage_from_a_different_experiment_is_kept_and_labelled(self):
        plan = self.plan(answer([shot(illustrates=[self.quote])]))
        self.assertEqual(plan['status'], 'representative_only')
        self.assertEqual(len(plan['shots']), 1);self.assertEqual(plan['shots'][0]['video_id'], '70511')
        self.assertIn('Reagent identity', plan['shots'][0]['unverified_details'])
        self.assertIn('100 microliters', plan['unobservable_details'])
        self.assertEqual(plan['missing_actions'], []);self.assertTrue(plan['needs_review'])

    def test_directly_visible_actions_are_selected_even_with_unverified_details(self):
        plan = self.plan(answer([shot(level='directly_visible', illustrates=[self.quote])]))
        self.assertEqual(plan['status'], 'selected')
        self.assertTrue(plan['shots'][0]['unverified_details'])

    def test_typed_exclusion_removes_only_its_window_and_untyped_exclusion_is_ignored(self):
        model = answer([shot(illustrates=[self.quote]), shot('w1', 385., 390., illustrates=[self.quote])],
                       exclusions=[{'window_id': 'w1', 'conflict_type': 'incompatible_apparatus',
                                    'reason': 'A robotic medium exchanger, not a pipette.', 'evidence': 'frame 386 s'},
                                   {'window_id': 'w0', 'conflict_type': 'different_experiment',
                                    'reason': 'The source studies organoids.', 'evidence': ''}])
        plan = self.plan(model)
        self.assertEqual([s['video_id'] for s in plan['shots']], ['70511'])
        self.assertEqual(plan['exclusion_notes'][0]['conflict_type'], 'incompatible_apparatus')
        self.assertTrue(any('Untyped exclusion ignored' in f for f in plan['review_flags']))

    def test_invented_windows_times_and_actions_are_dropped(self):
        for change in [dict(window='w9'), dict(end=99.), dict(start=float('nan')), dict(illustrates=['Calibrate the microscope'])]:
            kwargs = dict(illustrates=[self.quote]);kwargs.update(change)
            window = kwargs.pop('window', 'w0')
            plan = self.plan(answer([shot(window, **kwargs)]))
            self.assertEqual(plan['shots'], [], change);self.assertEqual(plan['status'], 'no_suitable_footage')

    def test_multi_action_coverage_and_missing_actions_are_computed(self):
        decomposition = rule_decomposition(STEPS['seal'])
        seal, incubate = [a['quote'] for a in decomposition['actions']]
        plan = self.plan(answer([shot(illustrates=[seal])]), STEPS['seal'], decomposition)
        self.assertEqual(plan['missing_actions'], [incubate]);self.assertEqual(plan['status'], 'representative_only')

    def test_schema_is_strict_and_closed(self):
        schema = review_schema(self.decomposition['actions'], windows())
        def walk(node):
            if node.get('type') == 'object':
                self.assertFalse(node['additionalProperties'])
                self.assertEqual(set(node['required']), set(node['properties']))
                for child in node['properties'].values():walk(child)
            if node.get('type') == 'array':walk(node['items'])
        walk(schema)
        verdict = schema['properties']['windows']['items']['properties']
        self.assertEqual(verdict['window_id']['enum'], ['w0', 'w1'])
        self.assertEqual(verdict['illustrates']['items']['enum'], [self.quote])
        self.assertEqual(verdict['difference']['enum'], list(DIFFERENCES))
        self.assertIn('invisible_details_only', verdict['difference']['enum'])
        self.assertEqual(set(DIFFERENCE_TO_CONFLICT.values()), set(CONFLICT_TYPES))


    def test_record_has_exactly_the_editorial_contract_shape(self):
        plan = self.plan(answer([shot(illustrates=[self.quote])],
                                exclusions=[{'window_id': 'w1', 'conflict_type': 'unsuitable_medium', 'reason': 'Title card.', 'evidence': '400 s'}]))
        record = contract_record(plan, {'hold_seconds': 2., 'source_video_seconds': 4., 'original_audio_seconds': 6.})
        self.assertEqual(set(record), {'step', 'status', 'shots', 'missing_actions', 'exclusion_notes', 'review_flags'})
        self.assertEqual(set(record['shots'][0]), {'video_id', 'start', 'end', 'evidence_level', 'illustrates',
                                                    'visible_observations', 'unverified_details', 'editor_note'})
        self.assertEqual(record['exclusion_notes'], [{'video_id': '55403', 'reason': 'unsuitable_medium: Title card.'}])
        self.assertTrue(any('held for 2.0 s' in f for f in record['review_flags']))
        self.assertIn(record['status'], STATUSES)
        self.assertIn('| 2 | representative only |', review_markdown([record], [SimpleNamespace(number=2, text=STEPS['plate'])]))


def verdict(window='w0', difference='invisible_details_only', start=12., end=16., illustrates=(), **extra):
    return dict(window_id=window, visible_operation=extra.pop('seen', 'Gloved hand works at a multiwell plate.'),
                apparatus='multiwell plate, pipette', difference=difference, illustrates=list(illustrates),
                start=start, end=end, action_start=start, action_end=end, visible_observations=[],
                unverified_details=['Liquid identity'], reason=extra.pop('reason', ''), evidence=extra.pop('evidence', ''))


class VerdictTests(unittest.TestCase):
    """The live ELISA run showed the model calling same-operation clips 'different operation'. Code now decides."""
    def setUp(self):
        self.decomposition = rule_decomposition(STEPS['seal'])
        self.seal, self.incubate = [a['quote'] for a in self.decomposition['actions']]

    def plan(self, verdicts, narration=None, keep_all=False):
        return review_plan({'windows': verdicts, 'review_flags': []}, windows(), 3, STEPS['seal'], self.decomposition,
                           narration_seconds=narration, keep_all=keep_all)

    def test_same_operation_from_another_experiment_is_kept_as_representative(self):
        plan = self.plan([verdict(illustrates=[self.seal], seen='Covers a cell-culture plate with adhesive film.',
                                  reason='Same sealing operation; the plate contents cannot be confirmed.')])
        self.assertEqual(len(plan['shots']), 1);self.assertEqual(plan['shots'][0]['evidence_level'], 'representative')
        self.assertIn('Covers a cell-culture plate with adhesive film.', plan['shots'][0]['visible_observations'])
        self.assertEqual(plan['exclusion_notes'], []);self.assertEqual(plan['missing_actions'], [self.incubate])

    def test_only_visible_differences_exclude_and_only_their_window(self):
        plan = self.plan([verdict(difference='none', illustrates=[self.seal]),
                          verdict('w1', 'apparatus', 380., 420., reason='Robotic medium exchanger, no plate sealing.', evidence='386 s')])
        self.assertEqual([s['video_id'] for s in plan['shots']], ['70511']);self.assertEqual(plan['shots'][0]['evidence_level'], 'directly_visible')
        self.assertEqual(plan['exclusion_notes'][0]['conflict_type'], 'incompatible_apparatus')
        self.assertEqual(plan['exclusion_notes'][0]['video_id'], '55403')

    def test_code_covers_each_action_once_then_fills_short_steps(self):
        verdicts = [verdict('w0', 'invisible_details_only', 12., 14., illustrates=[self.seal]),
                    verdict('w0', 'none', 20., 23., illustrates=[self.seal]),
                    verdict('w1', 'invisible_details_only', 385., 395., illustrates=[self.incubate])]
        short = self.plan(verdicts, narration=5)
        self.assertEqual([(s['start'], s['evidence_level']) for s in short['shots']], [(20., 'directly_visible'), (385., 'representative')])
        self.assertEqual(short['missing_actions'], [])
        long = self.plan(verdicts, narration=30)
        self.assertEqual(len(long['shots']), 3)
        self.assertEqual(len(self.plan(verdicts, narration=5, keep_all=True)['shots']), 3)

    def test_a_window_kept_and_excluded_keeps_the_footage_and_flags_it(self):
        plan = self.plan([verdict(illustrates=[self.seal]), verdict('w0', 'operation', 10., 30., reason='Pipetting.')])
        self.assertEqual(len(plan['shots']), 1);self.assertEqual(plan['exclusion_notes'], [])
        self.assertTrue(any('both kept and excluded' in f for f in plan['review_flags']))


class EditorFinalizeTests(unittest.TestCase):
    def test_every_step_leaves_the_editor_with_footage_or_a_labelled_gap(self):
        steps = [SimpleNamespace(number=1, text=STEPS['plate']), SimpleNamespace(number=2, text=STEPS['software'])]
        quote = rule_decomposition(STEPS['plate'])['actions'][0]['quote']
        rows = [{'step': 1, 'video_id': '70511', 'start': 10., 'end': 30., 'caption': ''},
                {'step': 2, 'video_id': '55403', 'start': 380., 'end': 420., 'caption': ''}]
        def ask(rules, payload, images=(), schema=None):
            if payload['windows'][0]['video_id'] == '70511':
                window = payload['windows'][0]
                start = window['start'] + .2
                return answer([shot(window['window_id'], start, min(window['end'] - .2, start + 3), illustrates=[quote])])
            return answer([], exclusions=[{'window_id': payload['windows'][0]['window_id'], 'conflict_type': 'unsuitable_medium',
                                           'reason': 'Software dialog unrelated to the measurement.', 'evidence': '400 s'}],
                          missing=['Open the image in the analysis software'])
        gateway = Mock();gateway.ask.side_effect = ask
        with TemporaryDirectory() as t:
            library = SimpleNamespace(videos={'70511': {'sha256': 'a' * 64, 'filename': '70511.mp4'},
                                              '55403': {'sha256': 'b' * 64, 'filename': '55403.mp4'}},
                                      video=lambda _: Path('source.mp4'), cache=Path(t), has_prepared_overview=lambda _: True)
            editor = AutomaticEditor(None, library, gateway, t, steps, {1: {'duration': 3}, 2: {'duration': 4}},
                                     workers=1, audit_required=False)
            with patch('pipeline.automatic_edit.duration', return_value=600), patch('pipeline.automatic_edit.sample_window', return_value=[]),\
                 patch('pipeline.automatic_edit.screen_video', return_value={'candidates': []}):
                plans = editor.run_shortlist(rows)
        self.assertEqual([p['step'] for p in plans], [1, 2])
        self.assertEqual(plans[0]['status'], 'representative_only');self.assertTrue(plans[0]['dense_checked'])
        self.assertEqual(plans[1]['status'], 'no_suitable_footage');self.assertEqual(plans[1]['shots'], [])
        self.assertEqual(plans[1]['exclusion_notes'][0]['conflict_type'], 'unsuitable_medium')
        sent = gateway.ask.call_args_list[0].args[1]
        self.assertEqual(sent['step_actions'][0]['quote'], quote);self.assertIn('100 microliters', sent['unverifiable_details'])


class ReviewDraftRenderTests(unittest.TestCase):
    def test_gap_steps_and_short_footage_keep_the_complete_narration(self):
        with TemporaryDirectory() as t:
            root = Path(t);source = root / 'source.mp4';out = root / 'out';out.mkdir()
            execute(['-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=30:duration=6', '-c:v', 'libx264', '-threads', '1',
                     '-preset', 'ultrafast', source], root / 'make-video.log')
            audios = {}
            for number, seconds in [(1, 3), (2, 2.5), (3, 2)]:
                voice = root / f'voice-{number}.wav'
                execute(['-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=44100:duration={seconds}', '-c:a', 'pcm_s16le', voice],
                        root / f'make-voice-{number}.log')
                audios[number] = {'path': str(voice), 'duration': seconds}
            base = dict(missing_actions=[], unobservable_details=['Reagent identity'], needs_review=True, on_screen_text=[])
            plans = [dict(base, step=1, text=STEPS['plate'], status='representative_only',
                          shots=[dict(video_id='101', start=1., end=4., reason='Pipetting.', evidence='illustrative',
                                      evidence_level='representative', unverified_details=['Volume per well'])]),
                     dict(base, step=2, text=STEPS['software'], status='no_suitable_footage', shots=[],
                          missing_actions=['Open the image in the analysis software']),
                     dict(base, step=3, text=STEPS['seal'], status='representative_only',
                          shots=[dict(video_id='101', start=2., end=2.8, reason='Brief sealing.', evidence='illustrative')])]
            result = render_actions(plans, audios, SimpleNamespace(video=lambda _: source), out, width=320, height=180)
            count, _ = imageio_ffmpeg.count_frames_and_secs(str(out / 'preview.mp4'))
            self.assertEqual(count, result['frames']);self.assertEqual(result['frames'], round(7.5 * 30000 / 1001))
            self.assertEqual(result['placeholder_steps'], [2]);self.assertFalse(result['publication_approved'])
            with wave.open(str(out / 'narration.wav')) as audio:
                self.assertAlmostEqual(audio.getnframes() / audio.getframerate(), 7.5, places=1)
            adjustments = json.loads((out / 'audio-adjustments.json').read_text())
            self.assertTrue(adjustments[1]['slate']);self.assertGreater(adjustments[2]['hold_seconds'], 1)
            captions = [x.text for x in ET.parse(out / 'editable.mlt').getroot().findall('.//property[@name="shotcut:caption"]')]
            self.assertTrue(any('[no suitable footage]' in c for c in captions))
            quality = quality_report([SimpleNamespace(number=n) for n in (1, 2, 3)], plans, adjustments, 100)
            self.assertFalse(quality['publication_approved']);self.assertTrue(quality['narration_retained'])
            self.assertEqual(quality['gap_steps'], [2]);self.assertEqual(quality['step_status']['no_suitable_footage'], 1)


class FakeGateway:
    """Stands in for the model only. Step 1 gets representative footage; step 2 gets a typed exclusion."""
    def __init__(self):
        self.calls, self.prior_calls, self.deadline = [], [], None

    def ask(self, instructions, payload, images=(), schema=None):
        self.calls.append({'status': 'complete'})
        if 'actions' in (schema or {}).get('properties', {}):
            raise TimeoutError('decomposition unavailable; rules are used')
        window = payload['windows'][0]
        if window['video_id'] != '101' or window['start'] > 15:
            return {'windows': [verdict(window['window_id'], 'medium', window['start'], window['end'],
                                        reason='Title card only.', evidence=f"{window['start']:.0f} s")], 'review_flags': []}
        start, end = window['start'], window['end']
        if 'enum' not in schema['properties']['windows']['items']['properties']['start']:
            start, end = start + .2, min(end - .2, start + 3.2)
        return {'windows': [verdict(window['window_id'], 'invisible_details_only', start, end,
                                    illustrates=[payload['step_actions'][0]['quote']])], 'review_flags': []}


class HostedJobPathTests(unittest.TestCase):
    def test_hosted_job_always_renders_review_files_for_every_step(self):
        from pipeline.runner import run_automatic_edit
        with TemporaryDirectory() as t:
            root = Path(t);folder = root / 'job';folder.mkdir();source = root / '101.mp4'
            execute(['-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=30:duration=40', '-c:v', 'libx264', '-threads', '1',
                     '-preset', 'ultrafast', source], root / 'make-video.log')
            steps = [SimpleNamespace(number=1, text=STEPS['plate'], on_screen_text=[]),
                     SimpleNamespace(number=2, text=STEPS['software'], on_screen_text=[])]
            audios = []
            for step, seconds in zip(steps, (3, 2)):
                voice = root / f'voice-{step.number}.wav'
                execute(['-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=44100:duration={seconds}', '-c:a', 'pcm_s16le', voice],
                        root / f'make-voice-{step.number}.log')
                audios.append({'path': str(voice), 'duration': seconds})
            (folder / 'matches.json').write_text(json.dumps([
                {'step_number': 1, 'best_match': {'journal_id': '101', 'start_timestamp': '00:02', 'end_timestamp': '00:12',
                                                  'caption_passage': 'Pipette into the wells.'}},
                {'step_number': 2, 'best_match': {'journal_id': '101', 'start_timestamp': '00:20', 'end_timestamp': '00:30',
                                                  'caption_passage': 'Title card.'}}]), encoding='utf-8')
            library = SimpleNamespace(videos={'101': {'sha256': 'a' * 64, 'filename': '101.mp4'}}, video=lambda _: source,
                                      cache=root / 'cache', has_prepared_overview=lambda _: True, restore_overviews=Mock())
            checkpoint, store = Mock(), Mock()
            config = {'OPENAI_API_KEY': 'unused', 'OUTPUT_FOLDER_ID': 'out', 'HOSTED_SHORTLIST': True}
            with patch('pipeline.runner.ModelGateway', return_value=FakeGateway()), \
                 patch('pipeline.action_catalog.open_catalog', side_effect=RuntimeError('no catalogue')), \
                 patch('pipeline.automatic_edit.screen_video', return_value={'candidates': []}):
                run_automatic_edit(folder, steps, audios, SimpleNamespace(embedder=None), library, config, checkpoint, store, 'remote')
            review = json.loads((folder / 'review.json').read_text(encoding='utf-8'))
            self.assertFalse(review['publication_approved'])
            self.assertEqual([r['step'] for r in review['steps']], [1, 2])
            self.assertEqual(review['steps'][0]['status'], 'representative_only')
            self.assertEqual(review['steps'][1]['status'], 'no_suitable_footage')
            self.assertEqual(review['steps'][1]['exclusion_notes'][0]['reason'], 'unsuitable_medium: Title card only.')
            self.assertIn('| 2 | no suitable footage |', (folder / 'review.md').read_text(encoding='utf-8'))
            count, _ = imageio_ffmpeg.count_frames_and_secs(str(folder / 'preview.mp4'))
            self.assertEqual(count, round(5 * 30000 / 1001))
            final = checkpoint.call_args
            self.assertEqual(final.args, ('review', 'needs_review'))
            self.assertEqual(final.kwargs['technical_status'], 'rendered');self.assertTrue(final.kwargs['verification'])
            uploaded = {c.args[0].name for c in store.upload.call_args_list}
            self.assertTrue({'review.json', 'review.md', 'preview.mp4', 'editable-package.zip'} <= uploaded)


if __name__ == '__main__':
    unittest.main()
