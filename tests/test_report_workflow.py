"""Result selection must never confuse a short clip, changed profile or stale avatar."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from notebooks import report_workflow as report

ROOT = Path(__file__).resolve().parents[1]


class ReportSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {trial: report.read_json(ROOT / 'configs/trials' / f'{trial}.json')
                       for trial in ('trial1', 'trial2', 'trial3')}
        cls.profile = report.read_json(ROOT / 'configs/users/trial_user_175.json')

    def test_all_full_runs_match_their_exact_avatar_without_generation(self):
        with patch('beta_translator.body_model.BodyModel.fit', side_effect=AssertionError('Unexpected fit')), \
             patch('beta_translator.avatar_pipeline.run_avatar', side_effect=AssertionError('Unexpected avatar generation')), \
             patch('beta_translator.body_render.render_run', side_effect=AssertionError('Unexpected rendering')):
            for trial, config in self.configs.items():
                with self.subTest(trial=trial):
                    full = report.saved_full(ROOT, config, self.profile)
                    folder = report.matching_avatar(ROOT, full)
                    self.assertIsNotNone(folder)
                    avatar = report.validate_avatar(full, folder)
                    self.assertEqual(avatar['frame_count'], config['frame_count'])

    def test_short_clips_cannot_be_labeled_as_full(self):
        for trial, config in self.configs.items():
            with self.subTest(trial=trial):
                clip = report.saved_comparison(ROOT, config, self.profile)
                with self.assertRaisesRegex(ValueError, 'full-sequence'):
                    report.validate_motion(clip, config, self.profile, full=True)

    def test_changed_profile_is_rejected_even_when_height_is_unchanged(self):
        profile = copy.deepcopy(self.profile)
        profile['mass_kg'] += 1
        with self.assertRaisesRegex(ValueError, 'profile differs'):
            report.saved_full(ROOT, self.configs['trial1'], profile)

    def test_changed_scene_is_rejected(self):
        config = copy.deepcopy(self.configs['trial2'])
        config['scene']['middle_deg'] = 30
        with self.assertRaisesRegex(ValueError, 'scene configuration'):
            report.saved_full(ROOT, config)

    def test_avatar_matching_rejects_tampered_identity_fields(self):
        full = report.saved_full(ROOT, self.configs['trial1'], self.profile)
        avatar = report.read_json(report.matching_avatar(ROOT, full) / 'summary.json')
        variants = []
        for key, value in [('source_candidate', 'balanced'), ('last_frame', 136), ('fps', 24), ('arm_span_cm', 175)]:
            variant = copy.deepcopy(avatar)
            variant[key] = value
            variants.append(variant)
        for key in ['source_summary_sha256', 'source_trajectory_sha256', 'source_metrics_sha256']:
            variant = copy.deepcopy(avatar)
            variant['provenance'][key] = 'changed'
            variants.append(variant)
        with tempfile.TemporaryDirectory() as tmp:
            for i, variant in enumerate(variants):
                with self.subTest(variant=i):
                    (Path(tmp) / 'summary.json').write_text(json.dumps(variant))
                    with self.assertRaises(ValueError):
                        report.validate_avatar(full, tmp)

    def test_new_run_without_avatar_never_falls_back_to_an_old_run(self):
        full = report.saved_full(ROOT, self.configs['trial1'], self.profile)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'summary.json').write_text('{}\n')
            result = {**full, 'output_dir': tmp}
            self.assertIsNone(report.matching_avatar(ROOT, result))

    def test_missing_presentation_does_not_render_implicitly(self):
        full = report.saved_full(ROOT, self.configs['trial1'], self.profile)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'summary.json').write_text('{}\n')
            result = {**full, 'output_dir': tmp}
            with patch('beta_translator.body_render.render_run', side_effect=AssertionError('Unexpected rendering')):
                self.assertIsNone(report.saved_presentation(result))

    def test_saved_short_presentations_are_reused(self):
        with patch('beta_translator.body_render.render_run', side_effect=AssertionError('Unexpected rendering')):
            for config in self.configs.values():
                result = report.saved_comparison(ROOT, config, self.profile)
                self.assertIsNotNone(report.saved_presentation(result))

    def test_new_full_run_remains_avatar_source_after_short_comparison(self):
        # Execute only orchestration cells with all computational entry points mocked.
        for trial in (1, 2, 3):
            with self.subTest(trial=trial):
                notebook = report.read_json(ROOT / f'Trial{trial}_Analysis.ipynb')
                sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
                full_code = next(s for s in sources if s.startswith('if RUN_FULL_SEQUENCE:'))
                short_code = next(s for s in sources if s.startswith('COMPARISON_RESULT = None'))
                avatar_code = next(s for s in sources if s.startswith('report.validate_motion(FULL_RESULT'))
                full = {'output_dir': str(ROOT / 'outputs' / f'trial{trial}' / 'new-full-run')}
                short = {'output_dir': str(ROOT / 'outputs' / f'trial{trial}' / 'new-short-run')}
                model = SimpleNamespace(user_profile=self.profile, fit=MagicMock(side_effect=[full, short]))
                views = SimpleNamespace(validate_motion=MagicMock(), show_motion=MagicMock(),
                                        show_comparison=MagicMock(), show_avatar=MagicMock())
                namespace = dict(ROOT=ROOT, TRIAL=f'trial{trial}', model=model, config=self.configs[f'trial{trial}'],
                                 Path=Path, report=views, ITERATIONS=1, RUN_FULL_SEQUENCE=True,
                                 RUN_COMPARISON=True, RUN_AVATAR=True, SHOW_DETAILS=False,
                                 SHOW_COMPARISON_VIDEO=False, RENDER_COMPARISON=False)
                destination = ROOT / 'outputs' / f'trial{trial}' / 'new-avatar'
                with patch('beta_translator.avatar_pipeline.run_avatar', return_value={'output_dir': str(destination)}) as generate:
                    for src in (full_code, short_code, avatar_code):
                        exec(compile(src, 'notebook-orchestration', 'exec'), namespace)
                    generate.assert_called_once_with(ROOT, Path(full['output_dir']))
                self.assertIs(namespace['FULL_RESULT'], full)
                self.assertIs(namespace['COMPARISON_RESULT'], short)
                self.assertEqual(model.fit.call_args_list[0].kwargs['frames'], 'all')
                self.assertEqual(model.fit.call_args_list[1].kwargs['frames'], 'verification')
                self.assertFalse(model.fit.call_args_list[0].kwargs['render'])


if __name__ == '__main__':
    unittest.main()
