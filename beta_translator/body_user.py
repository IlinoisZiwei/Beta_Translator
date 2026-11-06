"""Versioned, editable user geometry and explicitly assumed capability profiles."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from .inverse_kinematics import RATIOS

DEFAULT_PROFILE = Path(__file__).resolve().parents[1]/'configs/users/trial_user_175.json'


def load_user(profile=None, height_cm=None, ratios=None):
    """Resolve a complete snapshot; never silently rescale an explicit user file."""
    explicit = profile is not None
    if profile is None: profile = DEFAULT_PROFILE
    p = deepcopy(profile) if isinstance(profile, dict) else json.loads(Path(profile).read_text())
    if not isinstance(p.get('user_id'),str) or not p['user_id'].strip(): raise ValueError('Nonempty user_id required')
    if p.get('schema_version') != 1: raise ValueError('Unsupported user profile schema')
    if height_cm is not None and height_cm != p['height_cm']:
        if explicit: raise ValueError('Height conflicts with user profile; edit the profile explicitly')
        scale = height_cm/p['height_cm']
        p['height_cm'] = height_cm
        p['arm_span_cm'] *= scale
        p['dimensions_cm'] = {k:v*scale for k,v in p['dimensions_cm'].items()}
        p['evidence']['geometry'] = 'project_assumption: scaled demonstration profile'
    if ratios is not None:
        if set(ratios) != set(RATIOS): raise ValueError('Supply every body ratio')
        p['dimensions_cm'].update({k:float(v*p['height_cm']) for k,v in ratios.items()})
        d=p['dimensions_cm']
        p['arm_span_cm']=d['shoulder_w']+2*(d['upper_arm']+d['forearm']+d['hand_length'])
        p['evidence']['geometry']='user_supplied_ratios; arm span derived'
    validate_user(p)
    return p


def validate_user(p):
    d=p['dimensions_cm']
    if set(d) != set(RATIOS)|{'hand_length','hand_patch_distance','hand_patch_depth','foot_length','foot_width','foot_depth'}:
        raise ValueError('Unexpected or missing body dimension; use the template field names')
    values=[p['height_cm'],p['arm_span_cm'],p['mass_kg'],*d.values()]
    if not np.isfinite(values).all() or min(values)<=0: raise ValueError('Positive finite dimensions and mass required')
    span=d['shoulder_w']+2*(d['upper_arm']+d['forearm']+d['hand_length'])
    if abs(span-p['arm_span_cm'])>.1: raise ValueError(f'Inconsistent arm span: dimensions imply {span:.2f} cm')
    if d['hand_patch_distance']>d['hand_length']: raise ValueError('Hand patch lies beyond fingertip')
    b=p['biomechanics'];e=b['shoulder_envelope']
    grid=np.asarray(e['elevation_deg'],float);plane=np.asarray(e['plane_deg'],float)
    lower=np.asarray(e['lower_deg'],float);upper=np.asarray(e['upper_deg'],float)
    if not np.isfinite(grid).all() or not np.isfinite(plane).all(): raise ValueError('Finite envelope coordinates required')
    if len(grid)<2 or len(plane)<2 or np.any(np.diff(grid)<=0) or np.any(np.diff(plane)<=0): raise ValueError('Ordered envelope grids required')
    if lower.shape!=(len(plane),len(grid)) or upper.shape!=lower.shape or not np.isfinite([lower,upper]).all() or np.any(lower>=upper): raise ValueError('Invalid shoulder envelope')
    if plane[0]>-90 or plane[-1]<130 or grid[0]>0 or grid[-1]<180: raise ValueError('Envelope must cover adopted shoulder coordinate domain')
    for key in ['capsule_radii_cm','torque_positive_nm','torque_negative_nm']:
        x=np.asarray(b[key],float)
        if not np.isfinite(x).all() or np.any(x<=0): raise ValueError('Invalid '+key)
    if len(b['capsule_radii_cm'])!=14 or np.shape(b['torque_positive_nm'])!=(12,3) or np.shape(b['torque_negative_nm'])!=(12,3): raise ValueError('Use 14 segment radii and 12 joint capacity triplets')
    for key in ['mass_fractions','com_fractions']:
        x=np.asarray(b[key],float)
        if x.shape!=(14,) or not np.isfinite(x).all() or np.any(x<=0): raise ValueError('Invalid '+key)
    if not np.isclose(sum(b['mass_fractions']),1) or max(b['com_fractions'])>1: raise ValueError('Mass/COM fractions inconsistent')
    for key in ['fatigue_rate_s','recovery_rate_s','activation_rate_s','deactivation_rate_s','velocity_scale_rad_s','force_tolerance_bw','moment_tolerance_bw_m']:
        if not np.isfinite(b[key]) or b[key]<=0: raise ValueError('Positive '+key+' required')
    state=np.asarray(b['initial_fatigue_state'],float)
    if state.shape!=(3,) or not np.isfinite(state).all() or np.any(state<0) or not np.isclose(state.sum(),1): raise ValueError('Fatigue state is [active, resting, fatigued], sum one')
    if not 0<b['max_fatigued_fraction']<1 or not np.isfinite(b['collision_clearance_cm']) or b['collision_clearance_cm']<0: raise ValueError('Invalid fatigue/clearance threshold')
    if not isinstance(p.get('evidence'),dict) or not p['evidence']: raise ValueError('Parameter provenance required')


def body_lengths(p):
    return {k:p['dimensions_cm'][k] for k in RATIOS}


def profile_table(p):
    """Notebook-friendly measurements; bilateral dimensions are symmetric assumptions."""
    return [dict(parameter=k,value=v,unit=unit) for k,v,unit in
            [('height',p['height_cm'],'cm'),('arm_span',p['arm_span_cm'],'cm'),('mass',p['mass_kg'],'kg')]+
            [(k,v,'cm') for k,v in p['dimensions_cm'].items()]]
