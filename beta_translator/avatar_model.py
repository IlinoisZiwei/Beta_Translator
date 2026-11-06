"""Anny surface bound to the shared anatomical frames, in project centimetres.

This is an appearance adapter, not a new motion/force solver. Native Anny assets
only (Apache-2.0 code / CC0 MakeHuman data). Finger pose is the fixed template
pose: no observed or mechanically validated grasp is claimed.
"""
from pathlib import Path
import os
import numpy as np
from .body_user import load_user, body_lengths
from .body_legs import kinematics

# Anny: X left, Y back, Z up (metres). Project: X right, Y up, Z out.
ANNY_TO_PROJECT = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
GROUPS = ('torso', 'head', 'upper_arm.L', 'forearm.L', 'hand.L',
          'upper_arm.R', 'forearm.R', 'hand.R', 'thigh.L', 'shank.L',
          'foot.L', 'thigh.R', 'shank.R', 'foot.R')


def unit(x):
    return x / np.linalg.norm(x)


def segment_basis(start, end, back=np.array([0., 0., 1.])):
    """A fixed bind-pose frame; never recomputed from animated endpoints."""
    y = unit(start - end)
    z = unit(back - y * np.dot(y, back))
    return np.column_stack([np.cross(y, z), y, z])


def neutral_geometry(profile):
    g = kinematics(np.zeros((1, 32)), body_lengths(profile), profile['height_cm'])
    d = profile['dimensions_cm']
    for side in range(2):
        H = g['orientations'][:, 3 + 3 * side]
        W = g['joints'][:, side, 2]
        g['palm_contacts'][:, side] = W - d['hand_patch_distance'] * H[:, :, 1] - d['hand_patch_depth'] * H[:, :, 2]
    return g


def group_frames(g):
    """Map saved world frames, including twist, without fitting/swivel choices."""
    j, O = g['joints'], g['orientations']
    if O.shape[1:] != (15, 3, 3):
        raise ValueError('Avatar requires the shared 32-coordinate anatomical model')
    origins = [j[:, 2:, 0].mean(1)] * 2
    rotations = [O[:, 0]] * 2
    for s in range(2):
        origins.extend([j[:, s, 0], j[:, s, 1], j[:, s, 2]])
        rotations.extend([O[:, 1 + 3*s], O[:, 2 + 3*s], O[:, 3 + 3*s]])
    for s in range(2):
        origins.extend([j[:, s+2, 0], j[:, s+2, 1], g['foot_origins'][:, s]])
        rotations.extend([O[:, 7+4*s], O[:, 8+4*s], g['foot_frames'][:, s]])
    return np.stack(origins, 1), np.stack(rotations, 1)


def bone_group(name):
    side = name[-1] if name.endswith(('.L', '.R')) else None
    if name.startswith(('neck', 'head', 'eye')):
        return 1
    for prefix, group in [('upperarm', 'upper_arm'), ('lowerarm', 'forearm'),
                          ('wrist', 'hand'), ('finger', 'hand'), ('metacarpal', 'hand'),
                          ('upperleg', 'thigh'), ('lowerleg', 'shank'),
                          ('foot', 'foot'), ('toe', 'foot')]:
        if name.startswith(prefix):
            return GROUPS.index(f'{group}.{side}')
    return 0


def skin(vertices, weights, origins, rotations, bind_origins, bind_rotations):
    """Linear blend skinning with rigid, source-driven segment transforms."""
    delta = rotations @ np.swapaxes(bind_rotations, -1, -2)
    translation = origins - np.einsum('...gij,gj->...gi', delta, bind_origins)
    return (np.einsum('vg,...gij,vj->...vi', weights, delta, vertices, optimize=True)
            + np.einsum('vg,...gi->...vi', weights, translation, optimize=True))


class AnnyAvatar:
    """Profile-calibrated Anny mesh and persistent surface contact anchors."""
    def __init__(self, user_profile=None, cache_dir=None):
        if cache_dir is not None:
            os.environ['ANNY_CACHE_DIR'] = str(Path(cache_dir).resolve())
        try:
            import anny
            import torch
        except ImportError as error:
            raise RuntimeError('Install the repository requirements in the Beta Translator environment') from error
        self.profile = load_user(user_profile)
        self.version = anny.__version__
        # CPU LBS avoids a CUDA/MPS/graphics-context requirement on teammates' Macs.
        model = anny.Anny(rig='anny', topology='anny', local_changes='none',
                          facial_actions='none', skinning_method='lbs').float()
        with torch.no_grad():
            result = model()
        vertices = result['rest_vertices'][0].numpy() @ ANNY_TO_PROJECT.T * 100
        heads = result['rest_bone_heads'][0].numpy() @ ANNY_TO_PROJECT.T * 100
        self.faces = model.faces.cpu().numpy().astype(np.int32)
        self.bone_labels = tuple(model.bone_labels)
        groups = np.array([bone_group(n) for n in self.bone_labels])
        sparse_w = model.vertex_bone_weights.cpu().numpy()
        sparse_i = model.vertex_bone_indices.cpu().numpy()
        self.weights = np.zeros((len(vertices), len(GROUPS)))
        for k in range(sparse_w.shape[1]):
            np.add.at(self.weights, (np.arange(len(vertices)), groups[sparse_i[:, k]]), sparse_w[:, k])
        self.weights /= self.weights.sum(1, keepdims=True)
        self.neutral = neutral_geometry(self.profile)
        self.bind_origins, self.bind_rotations = (a[0] for a in group_frames(self.neutral))
        self.vertices, self.measurements = self._calibrate(vertices, heads)
        self.anchors, self.calibration = self._surface_anchors()
        self.measurements.update(self._measure())

    def _calibrate(self, vertices, heads):
        d = self.profile['dimensions_cm']; j = self.neutral['joints'][0]
        head = dict(zip(self.bone_labels, heads))
        source_hip = (head['upperleg01.L'] + head['upperleg01.R']) / 2
        source_shoulder = (head['upperarm01.L'] + head['upperarm01.R']) / 2
        A = np.zeros((len(GROUPS), 3, 3)); b = np.zeros((len(GROUPS), 3))
        sy = d['torso'] / (source_shoulder[1] - source_hip[1])
        sx = d['shoulder_w'] / np.linalg.norm(head['upperarm01.L'] - head['upperarm01.R'])
        A[0] = A[1] = np.diag([sx, sy, sy])
        b[0] = b[1] = -A[0] @ source_hip
        for s, side in enumerate(('L', 'R')):
            for group, a, z, start, end in [
                ('upper_arm', 'upperarm01', 'lowerarm01', j[s, 0], j[s, 1]),
                ('forearm', 'lowerarm01', 'wrist', j[s, 1], j[s, 2]),
                ('thigh', 'upperleg01', 'lowerleg01', j[s+2, 0], self.neutral['femur_distal'][0, s]),
                ('shank', 'lowerleg01', 'foot', j[s+2, 1], j[s+2, 2])]:
                k = GROUPS.index(f'{group}.{side}')
                a, z = head[f'{a}.{side}'], head[f'{z}.{side}']
                source = segment_basis(a, z)
                target = segment_basis(start, end, self.bind_rotations[k, :, 2])
                length_scale = np.linalg.norm(end-start) / np.linalg.norm(z-a)
                A[k] = target @ np.diag([sy, length_scale, sy]) @ source.T
                b[k] = start - A[k] @ a
            k = GROUPS.index(f'hand.{side}'); wrist = head[f'wrist.{side}']
            # Middle MCP fixes the distal axis; index/little MCPs fix the palm plane.
            middle = head[f'finger3-1.{side}']; index = head[f'finger2-1.{side}']; little = head[f'finger5-1.{side}']
            y = unit(wrist-middle); z = unit(np.cross(index-wrist, little-wrist))
            if np.dot(z, np.array([0., 0., 1.])) < 0: z = -z
            x = unit(np.cross(y, z)); z = np.cross(x, y)
            source = np.column_stack([x, y, z])
            local = (vertices-wrist) @ source
            mask = self.weights[:, k] > .99
            scale = d['hand_length'] / (-local[mask, 1].min())
            A[k] = self.bind_rotations[k] @ (scale * source.T)
            b[k] = j[s, 2] - A[k] @ wrist
            k = GROUPS.index(f'foot.{side}'); ankle = head[f'foot.{side}']
            toe = head[f'toe2-1.{side}'] - ankle; toe[1] = 0; toe = unit(toe)
            up = np.array([0., 1., 0.]); source = np.column_stack([toe, np.cross(up, toe), up])
            local = (vertices-ankle) @ source; mask = self.weights[:, k] > .99
            lo, hi = local[mask].min(0), local[mask].max(0)
            scale = np.array([d['foot_length']/(hi[0]-lo[0]), d['foot_width']/(hi[1]-lo[1]), sy])
            shift = np.array([-lo[0]*scale[0], -(hi[1]+lo[1])*scale[1]/2, -d['foot_depth']-lo[2]*scale[2]])
            A[k] = self.bind_rotations[k] @ np.diag(scale) @ source.T
            b[k] = self.bind_origins[k] + self.bind_rotations[k] @ shift - A[k] @ ankle
        V = (np.einsum('vg,gij,vj->vi', self.weights, A, vertices, optimize=True)
             + self.weights @ b)
        # A flat sole is an explicit footwear proxy, not an inferred bare foot.
        for s in range(2):
            k = 10+3*s; mask = self.weights[:, k] > .99
            local = (V[mask]-self.bind_origins[k]) @ self.bind_rotations[k]
            bottom = np.quantile(local[:, 2], .22)
            local[:, 2] = np.where(local[:, 2] <= bottom, -d['foot_depth'],
                                   local[:, 2] - bottom - d['foot_depth'])
            V[mask] = local @ self.bind_rotations[k].T + self.bind_origins[k]
        # Height fixes unmeasured head/neck extent, never the solver's limb lengths.
        y0 = d['torso']; desired = V[:, 1].min() + self.profile['height_cm']
        for _ in range(3):
            top = np.argmax(V[:, 1]); denom = self.weights[top, 1] * (V[top, 1]-y0)
            if abs(denom) < 1: raise ValueError('Cannot calibrate head/neck height')
            V[:, 1] += self.weights[:, 1] * (V[:, 1]-y0) * (desired-V[top, 1]) / denom
        return V, {'head_neck_extent': 'inferred to complete requested stature',
                   'surface_shape': 'Anny template with segment calibration; no identity/sex inference',
                   'shoe_shape': 'Anny foot with a flat inferred sole',
                   'hand_pose': 'fixed template fingers; grasp unobserved and unvalidated'}

    def _surface_anchors(self):
        import trimesh
        anchors = []; report = []
        d = self.profile['dimensions_cm']
        for s in range(2):
            k = 4+3*s
            target = self.neutral['palm_contacts'][0, s]
            mask = np.all(self.weights[self.faces, k] > .999, axis=1)
            faces = np.flatnonzero(mask)
            if not len(faces): raise ValueError('No rigid palm surface in Anny mesh')
            closest, distance, index = trimesh.proximity.closest_point_naive(
                trimesh.Trimesh(self.vertices, self.faces[faces], process=False), [target])
            face = int(faces[index[0]])
            bary = trimesh.triangles.points_to_barycentric(self.vertices[self.faces[[face]]], closest)[0]
            # One bind-pose palm-pad adjustment, shared by every frame and trial.
            # Never snap a moving surface to a hold to hide a source contact gap.
            delta = target-closest[0]
            local_dist = np.linalg.norm(self.vertices-closest[0], axis=1)
            influence = np.exp(-(local_dist/4.)**2) * (self.weights[:, k] > .999)
            influence[self.faces[face]] = 1.
            self.vertices += influence[:, None]*delta
            anchors.append((face, bary, k, 'palm'))
            report.append({'limb': ('left_hand', 'right_hand')[s], 'pad_adjustment_cm': float(distance[0])})
        for s in range(2):
            k = 10+3*s
            for mode, fraction in [('ground', .45), ('hold', .65)]:
                target = self.bind_origins[k]+self.bind_rotations[k] @ np.array([fraction*d['foot_length'], 0, -d['foot_depth']])
                faces = np.flatnonzero(np.all(self.weights[self.faces, k] > .999, axis=1))
                closest, distance, index = trimesh.proximity.closest_point_naive(
                    trimesh.Trimesh(self.vertices, self.faces[faces], process=False), [target])
                face = int(faces[index[0]])
                bary = trimesh.triangles.points_to_barycentric(self.vertices[self.faces[[face]]], closest)[0]
                anchors.append((face, bary, k, mode))
                report.append({'limb': ('left_foot', 'right_foot')[s], 'mode': mode, 'bind_surface_error_cm': float(distance[0])})
        return anchors, report

    def _measure(self):
        d = self.profile['dimensions_cm']
        origins = self.bind_origins.copy(); rotations = self.bind_rotations.copy()
        # Measure the rendered surface in an extended-arm calibration pose.
        for s, sign in enumerate((-1., 1.)):
            G = np.array([[0., -sign, 0.], [sign, 0., 0.], [0., 0., 1.]])
            start = self.bind_origins[2+3*s]
            for offset, distance in enumerate((0, d['upper_arm'], d['upper_arm']+d['forearm'])):
                k = 2+3*s+offset
                origins[k] = start + [sign*distance, 0, 0]
                rotations[k] = G @ self.bind_rotations[k]
        extended = skin(self.vertices, self.weights, origins, rotations, self.bind_origins, self.bind_rotations)
        return {'height_cm': float(np.ptp(self.vertices[:, 1])),
                'arm_span_cm': float(np.ptp(extended[:,0])),
                'rig_arm_span_cm': d['shoulder_w']+2*(d['upper_arm']+d['forearm']+d['hand_length']),
                'rig_dimensions_cm': {k: d[k] for k in ('upper_arm','forearm','thigh','shank','torso','shoulder_w','hip_w')},
                'vertices': len(self.vertices), 'triangles': len(self.faces)}

    def pose(self, geometry):
        origins, rotations = group_frames(geometry)
        if not np.isfinite(origins).all() or not np.isfinite(rotations).all():
            raise ValueError('Non-finite source pose')
        if np.max(abs(np.swapaxes(rotations,-1,-2)@rotations-np.eye(3))) > 1e-5 or np.min(np.linalg.det(rotations)) < .99999:
            raise ValueError('Source segment frames must be proper rotations')
        return skin(self.vertices, self.weights, origins, rotations, self.bind_origins, self.bind_rotations)

    def contact_points(self, vertices, ground):
        points = np.stack([np.einsum('tvi,v->ti', vertices[:,self.faces[f]], b) for f,b,_,_ in self.anchors], 1)
        return np.stack([points[:,0], points[:,1],
                         np.where(ground[:,2,None], points[:,2], points[:,3]),
                         np.where(ground[:,3,None], points[:,4], points[:,5])], 1)
