import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from beta_translator.portable_paths import artifact_paths, project_path, set_artifacts
from beta_translator.project_io import file_sha256
from beta_translator.trial_pipeline import checkpoint, prepare_video, valid_cache
from scripts.prepare_assets import POINTER, restore_asset
from beta_translator.target_demo import build_target_demo


class PortableSetupTests(unittest.TestCase):
    def test_cached_target_demo_returns_paths_in_the_current_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder).resolve()
            (project/'beta_translator').mkdir()
            root = project/'outputs/trial3'
            (root/'state').mkdir(parents=True)
            destination = root/'automatic/ik/move_006'
            (destination/'state').mkdir(parents=True)
            (destination/'state/motion_ik_summary.json').write_text(json.dumps(
                {'metrics': {'max_angle_step_deg': 1.}}))
            (destination/'target.mp4').write_bytes(b'clip')
            old = '/old/computer/Beta_Translator/'
            saved = {key: old+'outputs/trial3/'+name for key, name in
                     dict(video_file='videos/demo.mp4', poster_file='images/demo.png',
                          source_clip='automatic/ik/move_006/target.mp4', comparison_video='videos/comparison.mp4').items()}
            (root/'state/target_ik_demo.json').write_text(json.dumps(saved))
            plan = dict(body={'height_cm': 180, 'ratios': {}}, moves=[dict(status='generated',
                output_dir=old+'outputs/trial3/automatic/ik/move_006',
                target_video_file=saved['source_clip'])])
            with patch('beta_translator.target_demo.current_motion_outputs'), \
                 patch('beta_translator.target_demo.valid_cache', return_value={'verified': True}):
                result = build_target_demo(project/'videos/trial3.mp4', root, plan)
            for key, value in result.items():
                self.assertEqual(Path(value), project/Path(saved[key]).relative_to(old))

    def test_legacy_cache_checks_new_checkout_even_when_old_copy_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = [Path(folder)/x for x in ('author', 'teammate')]
            for root in (old, new):
                (root/'beta_translator').mkdir(parents=True)
                (root/'outputs/state').mkdir(parents=True)
                (root/'outputs/result').write_bytes(b'original')
            manifest = new/'outputs/state/manifest.json'
            saved = dict(inputs={'v': 1}, artifacts={str(old/'outputs/result'): file_sha256(old/'outputs/result')})
            manifest.write_text(json.dumps(saved))
            self.assertIsNotNone(valid_cache(manifest, {'v': 1}))
            (new/'outputs/result').write_bytes(b'changed')
            self.assertIsNone(valid_cache(manifest, {'v': 1}))
            set_artifacts(manifest, saved, {new/'outputs/result': file_sha256(new/'outputs/result')})
            manifest.write_text(json.dumps(saved))
            self.assertEqual(len(saved['artifacts']), 1)
            self.assertIsNotNone(valid_cache(manifest, {'v': 1}))
            (new/'outputs/result').unlink()
            self.assertIsNone(valid_cache(manifest, {'v': 1}))
            self.assertEqual(project_path(str(old/'data/grid.json'), manifest), (new/'data/grid.json').resolve())

    def test_new_manifest_survives_directory_move(self):
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder)/'old'; old.mkdir()
            (old/'result').write_bytes(b'ok')
            checkpoint(old/'state/cache.json', {'x': 2}, [old/'result'])
            new = Path(folder)/'new'; shutil.move(old, new)
            self.assertIsNotNone(valid_cache(new/'state/cache.json', {'x': 2}))
            saved = json.loads((new/'state/cache.json').read_text())
            self.assertIn((new/'result').resolve(), artifact_paths(new/'state/cache.json', saved))

    def test_bundled_video_requires_matching_manifest_and_frame_map(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'tables').mkdir()
            video = root/'clip.mp4'; video.write_bytes(b'video')
            table = root/'tables/source_frame_map.csv'; table.write_bytes(b'mapping')
            checkpoint(root/'state/video_manifest.json',
                       dict(fps=30, width=720, version=1, source_sha256='original'), [video, table])
            self.assertIsNotNone(prepare_video(None, video, root))
            with self.assertRaises(ValueError): prepare_video(None, video, root, fps=25)
            with self.assertRaises(FileNotFoundError): prepare_video(root/'missing.mov', video, root)
            with self.assertRaises(ValueError): prepare_video(None, root/'wrong.mp4', root)
            table.write_bytes(b'changed')
            with self.assertRaises(ValueError): prepare_video(None, video, root)

    def test_lfs_download_is_atomic_checked_and_preserves_existing_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'videos').mkdir()
            path = root/'videos/test.mp4'; path.write_bytes(POINTER+b'\n')
            content = b'actual video bytes'
            asset = dict(path='videos/test.mp4', bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            with patch('scripts.prepare_assets.urlopen', return_value=io.BytesIO(b'broken')):
                with self.assertRaises(ValueError): restore_asset(root, asset, 'https://test.invalid/video')
            self.assertEqual(path.read_bytes(), POINTER+b'\n')
            self.assertEqual(list(path.parent.glob('*.part')), [])
            with patch('scripts.prepare_assets.urlopen', return_value=io.BytesIO(content)):
                restore_asset(root, asset, 'https://test.invalid/video')
            self.assertEqual(path.read_bytes(), content)
            with patch('scripts.prepare_assets.urlopen') as download:
                restore_asset(root, asset, 'https://test.invalid/video')
                download.assert_not_called()
                path.write_bytes(b'new user input')
                with self.assertRaises(ValueError): restore_asset(root, asset, 'https://test.invalid/video')
                self.assertEqual(path.read_bytes(), b'new user input')
                (root/'outputs').mkdir(); (root/'outputs/result.mp4').write_bytes(b'user result')
                restore_asset(root, {**asset, 'path': 'outputs/result.mp4'}, 'https://test.invalid/video')
                self.assertEqual((root/'outputs/result.mp4').read_bytes(), b'user result')
                download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
