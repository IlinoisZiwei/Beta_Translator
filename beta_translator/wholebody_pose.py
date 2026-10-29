"""Frozen RTMW whole-body features with reproducible, resumable video extraction."""
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

POSE_SPEC = 'rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class WholebodyExtractor:
    """One climber, selected by box continuity; no contact labels are used."""
    def __init__(self, cache_dir, device='cpu', threads=4):
        import certifi
        import onnxruntime as ort
        from rtmlib import Wholebody
        cache = Path(cache_dir).resolve()
        os.environ['TORCH_HOME'] = str(cache)
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
        model = Wholebody(mode='performance', backend='onnxruntime', device='cpu')
        self.detector, self.pose = model.det_model, model.pose_model
        for tool in (self.detector, self.pose):
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            # RTMW declares a dynamic batch dimension. Fixing our batch of one
            # enables CoreML rather than silently falling back to CPU kernels.
            options.add_free_dimension_override_by_name('batch', 1)
            providers = ['CPUExecutionProvider']
            if device == 'coreml':
                if 'CoreMLExecutionProvider' not in ort.get_available_providers():
                    raise RuntimeError('CoreML is unavailable; select --device cpu.')
                providers.insert(0, ('CoreMLExecutionProvider', {
                    'ModelFormat': 'MLProgram', 'MLComputeUnits': 'ALL',
                    'RequireStaticInputShapes': '1',
                    'ModelCacheDirectory': str(cache / ('coreml_static_v1_' + Path(tool.onnx_model).stem))}))
            tool.session = ort.InferenceSession(tool.onnx_model, sess_options=options,
                                                providers=providers)
        self.provenance = {'model': POSE_SPEC, 'backend': device,
            'tracking_version': 2,
            'pose_sha256': sha256(self.pose.onnx_model),
            'detector_sha256': sha256(self.detector.onnx_model),
            'keypoint_format': 'COCO-WholeBody-133', 'detector_interval': 5,
            'score_semantics': 'Raw SimCC peak score, not a probability'}

    def extract(self, video, output):
        video, output = Path(video), Path(output)
        meta = {**self.provenance, 'video_sha256': sha256(video)}
        if output.exists():
            with np.load(output, allow_pickle=False) as saved:
                old = json.loads(str(saved['metadata']))
                if all(old.get(k) == v for k, v in meta.items()):
                    print(f'Using cache: {output}', flush=True)
                    return
            raise ValueError(f'Stale pose cache: {output}')
        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(3)), int(cap.get(4))
        if min(count, width, height, fps) <= 0:
            raise ValueError(f'Invalid video: {video}')
        keypoints = np.full((count, 133, 2), np.nan, np.float32)
        scores = np.zeros((count, 133), np.float32)
        boxes = np.full((count, 4), np.nan, np.float32)
        previous, missed = None, 0
        start = time.monotonic()
        try:
            for frame in range(count):
                ok, image = cap.read()
                if not ok:
                    raise IOError(f'Video ended early: frame {frame} / {count}')
                if frame % 5 == 0 or previous is None:
                    candidates = np.asarray(self.detector(image))
                    if len(candidates):
                        area = np.prod(np.maximum(candidates[:, 2:] - candidates[:, :2], 0), axis=1)
                        rank = np.sqrt(area) / max(width, height)
                        if previous is not None:
                            centers = (candidates[:, :2] + candidates[:, 2:]) / 2
                            center = (previous[:2] + previous[2:]) / 2
                            rank -= 3 * np.linalg.norm(centers - center, axis=1) / max(width, height)
                            # Never switch to a distant bystander when the climber detector misses.
                            distance = np.linalg.norm(centers-center, axis=1)
                            rank[distance > max(80, .6*(previous[3]-previous[1]))] = -np.inf
                        if np.isfinite(rank).any():
                            previous = candidates[rank.argmax()]
                            missed = 0
                        else:
                            missed += 5
                    else:
                        missed += 5
                if previous is not None:
                    points, confidence = self.pose(image, bboxes=[previous])
                    visible = (confidence[0, :23] > 1.5) & np.isfinite(points[0, :23]).all(axis=1)
                    if visible.sum() >= 8:
                        keypoints[frame], scores[frame], boxes[frame] = points[0], confidence[0], previous
                        # Continue the same track through detector misses using whole-body geometry.
                        if missed:
                            observed = points[0, :23][visible]
                            lo, hi = observed.min(axis=0), observed.max(axis=0)
                            extent = hi-lo
                            if extent[1] > 40 and extent[0] > 15:
                                tracked = np.r_[lo-.1*extent, hi+.1*extent]
                                previous = .8*previous + .2*tracked
                if (frame + 1) % 250 == 0:
                    print(f'{video.parent.name}/{video.name}: {frame+1}/{count}, '
                          f'{(frame+1)/(time.monotonic()-start):.1f} fps', flush=True)
        finally:
            cap.release()
        meta.update(fps=fps, width=width, height=height, frames=count,
                    seconds=time.monotonic()-start)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, keypoints=keypoints, scores=scores,
                                boxes=boxes, metadata=json.dumps(meta))
        temporary.replace(output)
        print(f'Saved {output} ({meta["seconds"]:.1f}s)', flush=True)
