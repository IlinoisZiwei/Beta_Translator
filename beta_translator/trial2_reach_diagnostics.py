"""Necessary reach bounds for chosen contact targets, not human capacity estimates."""
import itertools
import numpy as np
import pandas as pd
from .inverse_kinematics import make_body

NAMES=('left_hand','right_hand','left_foot','right_foot')


def target_reach_diagnostics(fit,height_cm=180.,tolerance_cm=.1):
    """Triangle inequality: target separation <= limb + rigid trunk + limb.

    This deliberately relaxes all joint-angle limits. Violating it proves that
    the chosen pair of endpoint targets cannot both meet the tolerance in this
    fixed-length model. It does not prove infeasibility over entire hold regions,
    alternative contact assignments, calibrated scenes, or actual human motion.
    """
    body=make_body(height_cm);n=len(fit['joints'])
    roots=np.array([[-body['shoulder_w']/2,body['torso'],0],
                    [body['shoulder_w']/2,body['torso'],0],
                    [-body['hip_w']/2,0,0],[body['hip_w']/2,0,0]])
    limbs=np.array([body['upper_arm']+body['forearm']]*2+[body['thigh']+body['shank']]*2)
    excess=np.zeros(n);pairs=np.full(n,'none',dtype='<U40');distance=np.zeros(n);bound=np.zeros(n)
    for a,b in itertools.combinations(range(4),2):
        limit=limbs[a]+np.linalg.norm(roots[a]-roots[b])+limbs[b]
        length=np.linalg.norm(fit['targets'][:,a]-fit['targets'][:,b],axis=-1)
        candidate=np.where(fit['known'][:,a]&fit['known'][:,b],length-limit,-np.inf)
        better=candidate>excess
        excess[better]=candidate[better];distance[better]=length[better];bound[better]=limit;pairs[better]=NAMES[a]+' / '+NAMES[b]
    return pd.DataFrame(dict(frame=fit.get('frames',np.arange(n)),target_reach_conflict=excess>2*tolerance_cm,
        target_span_excess_cm=excess,minimum_possible_max_endpoint_error_cm=excess/2,
        limiting_pair=pairs,target_distance_cm=distance,relaxed_body_path_bound_cm=bound))
