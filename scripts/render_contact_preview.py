"""Render native-frame contact candidates and review status for an integration audit."""
import argparse
from pathlib import Path
import sys
import json
import cv2
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beta_translator.contact_features import LIMBS, LANDMARKS
from beta_translator.project_io import load_reviewed_holds


def main():
    parser = argparse.ArgumentParser()
    for name in ('video', 'poses', 'reviewed-output', 'predictions', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs=4, default=[107, 117, 128, 129])
    parser.add_argument('--reference', type=Path)
    a = parser.parse_args()
    truth = {}
    if a.reference:
        from beta_translator.motion_ik import contact_schedule
        frames, labels = contact_schedule(json.loads(a.reference.read_text()))
        truth = dict(zip(map(int, frames), labels))
    contacts = pd.read_csv(a.predictions/'limb_contacts.csv').set_index(['frame', 'limb'])
    records, _, _, selected = load_reviewed_holds(a.reviewed_output/'state/reviewed_hold_map.npz', a.video)
    centers = {r['hold_id']: np.array([r['cx_px'], r['cy_px']], np.float32) for r in records}
    with np.load(a.poses, allow_pickle=False) as pose:
        points = pose['keypoints'].copy()
    with np.load(a.reviewed_output/'state/alignment_cache.npz', allow_pickle=False) as z:
        transforms = {int(f): H for f, H in zip(z['frames'], z['matrices'])}
    cap = cv2.VideoCapture(str(a.video)); fps = cap.get(cv2.CAP_PROP_FPS)
    panels = []
    for frame in a.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, im = cap.read()
        if not ok: raise ValueError(f'Cannot read frame {frame}')
        header = np.full((128, 360, 3), 250, np.uint8)
        cv2.putText(header, f'Frame {frame} | {frame/fps:.2f}s', (10, 22), 0, .6, (35, 35, 35), 1, cv2.LINE_AA)
        for limb, name in enumerate(LIMBS):
            row = contacts.loc[(frame, name)]; accepted = row['status'] == 'accepted_model'
            color = (55, 145, 35) if accepted else (15, 135, 225)
            xy = np.mean(points[frame, LANDMARKS[limb][1:]], axis=0)
            if np.isfinite(xy).all():
                cv2.circle(im, tuple(np.rint(xy).astype(int)), 7, color, -1, cv2.LINE_AA)
            hold = str(row['candidate_hold'])
            if hold in centers and frame in transforms:
                target = cv2.perspectiveTransform(centers[hold][None, None], transforms[frame])[0, 0]
                target = tuple(np.rint(target).astype(int))
                cv2.circle(im, target, 13, color, 2, cv2.LINE_AA)
                cv2.putText(im, hold, (target[0]+12, target[1]), 0, .5, color, 2, cv2.LINE_AA)
            short = ['LH', 'RH', 'LF', 'RF'][limb]
            x, y = 10+(limb % 2)*175, 47+(limb//2)*22
            cv2.putText(header, f'{short}: {hold} {row["confidence"]:.2f}', (x, y), 0, .43, color, 1, cv2.LINE_AA)
        cv2.putText(header, 'Green: confident | Orange: review', (10, 93), 0, .4, (65, 65, 65), 1, cv2.LINE_AA)
        if frame in truth:
            expected = truth[frame]['left_hand']
            match = expected == contacts.loc[(frame, 'left_hand'), 'candidate_hold']
            label = f'Manual LH: {expected} | '+('match' if match else 'MISMATCH')
            cv2.putText(header, label, (10, 117), 0, .43, (65, 65, 65) if match else (30, 40, 190), 1, cv2.LINE_AA)
        image = cv2.resize(im, (360, int(im.shape[0]*360/im.shape[1])))
        panels.append(np.vstack([header, image]))
    cap.release()
    result = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(a.output), result): raise IOError('Could not save the contact preview')


if __name__ == '__main__': main()
