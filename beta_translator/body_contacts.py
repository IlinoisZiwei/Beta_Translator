"""Shared explicit hand/foot contact geometry. Foot orientation is conditional."""
import numpy as np


class ContactMetadata:
    """Only scene/contact metadata; no trial-specific optimizer or skeleton."""
    def __init__(self, data, scene, regions):
        eps=regions['episodes'];count=len(eps);mapping=regions['mapping']
        if not count:raise ValueError('At least one located contact episode is required')
        ix=np.maximum(mapping,0);known=data['known']
        box=np.array([e['rectangle_px'] for e in eps],float)
        anchor=np.array([e['anchor_px'] for e in eps],float)
        uv=np.divide(anchor-box[:,:2],box[:,2:]-box[:,:2],out=np.full((count,2),.5),where=box[:,2:]!=box[:,:2])
        frozen=(box[:,2:]==box[:,:2]).ravel()
        self.lower=np.zeros(count*2);self.upper=np.ones(count*2)
        self.lower[frozen]=uv.ravel()[frozen];self.upper[frozen]=uv.ravel()[frozen]
        self.rect=box[ix];self.panels=np.array([e['panel'] for e in eps])[ix]
        self.normals=scene.normals[self.panels]
        self.offset=np.array([e['offset_cm'] for e in eps])[ix]*known[...,None]
        panels=np.array([e['panel'] for e in eps])
        self.prior=scene.intersect(anchor,panels)[ix]
        self.seed_panel=scene.panel(data['observed_px'][:,2:,0].mean(1))
        samples=np.bincount(mapping[mapping>=0],minlength=count)
        self.evidence_scale=1/np.sqrt(np.maximum(samples[ix],1))
        tangent=np.broadcast_to([1.,0.,0.],self.normals.shape).copy()
        tangent-=np.sum(tangent*self.normals,axis=-1)[...,None]*self.normals
        if np.any(np.linalg.norm(tangent,axis=-1)<1e-8):raise ValueError('Contact normal parallel to tangent seed')
        tangent/=np.linalg.norm(tangent,axis=-1)[...,None]
        self.tangents=(tangent,np.cross(self.normals,tangent))

def foot_frames(data, scene, regions, height):
    """One explicit shoe frame per stance episode, no independently fitted offset.

    Columns are toe direction, transverse direction, outward surface normal.
    Surface-aligned sole is a declared hypothesis; toe direction is estimated
    from ankle/contact evidence or heel/toe image evidence on the floor.
    """
    n = len(data['frames']); mapping = regions['mapping']
    frames = np.broadcast_to(np.eye(3), (n, 2, 3, 3)).copy()
    offsets = np.zeros((n, 2, 3)); modes = np.full((n, 2), 'unobserved', dtype='U32')
    evidence = []
    for e in regions['episodes']:
        limb = e['limb_index']
        if limb < 2:
            continue
        active = mapping[:, limb] == e['episode']
        if not active.any():
            continue
        normal = scene.normals[e['panel']].copy()
        toe = -np.asarray(e['offset_cm']); toe -= toe@normal*normal
        source = 'episode ankle-to-contact image direction'
        if e['ground_proposal']:
            heel = 19 if limb == 2 else 22
            toes = [17, 18] if limb == 2 else [20, 21]
            samples = []
            for t in np.flatnonzero(active):
                ids = [heel, *toes]
                if np.min(data['scores'][t, ids]) <= .5:
                    continue
                pts = scene.intersect(data['points'][t, ids], 3)
                v = pts[1:].mean(0)-pts[0]; v -= v@normal*normal
                if np.linalg.norm(v) > 1e-6:
                    samples.append(v/np.linalg.norm(v))
            toe = np.median(samples, axis=0) if samples else np.array([0., 0., -1.])
            source = 'floor heel/toe image hypothesis' if samples else 'floor forward-direction fallback'
        if np.linalg.norm(toe) < 1e-6:
            toe = np.array([0., 1., 0.]); toe -= toe@normal*normal
            if np.linalg.norm(toe) < 1e-6:
                toe = np.array([0., 0., -1.]); toe -= toe@normal*normal
            source = 'surface tangent fallback: sparse evidence'
        toe /= np.linalg.norm(toe)
        frame = np.stack([toe, np.cross(normal, toe), normal], axis=1)
        # Local ankle-to-patch vector. Ground uses the sole under the ankle;
        # wall holds use a forefoot proxy. The constants remain assumptions.
        local = height*np.array([0., 0., -.035] if e['ground_proposal'] else [.045, 0., -.025])
        frames[active, limb-2] = frame
        offsets[active, limb-2] = frame@local
        modes[active, limb-2] = 'ground_sole' if e['ground_proposal'] else 'wall_forefoot'
        evidence.append(dict(episode=e['episode'], limb=e['limb'], frame_world=frame.tolist(),
            ankle_to_patch_local_cm=local.tolist(), direction_source=source,
            sole_normal_status='assumed_from_scene_surface', ankle_rom_status='not_checked',
            measured=False, evidence_frames=e['evidence_frames']))
    return frames, offsets, modes, evidence




def contact_geometry(scene,regions,uv,profile):
    episodes=regions['episodes'];box=np.array([e['rectangle_px'] for e in episodes])
    center=np.array([e['center_px'] for e in episodes]);panel=np.array([e['panel'] for e in episodes])
    xy=center if profile=='center' else box[:,:2]+uv*(box[:,2:]-box[:,:2])
    c=scene.intersect(xy,panel)
    offsets=np.array([e['offset_cm'] for e in episodes]) if profile!='center' else np.zeros_like(c)
    mapping=regions['mapping'];ix=np.maximum(mapping,0);active=mapping>=0
    return c[ix]*active[...,None],offsets[ix]*active[...,None],scene.normals[panel[ix]],c


