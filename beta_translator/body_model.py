"""Reusable public entry point for Trial1, Trial2, Trial3 and trial configs.

    model = BodyModel(height_cm=180)
    result = model.fit('trial2', project=Path.cwd(), frames='verification')

The complete specification is always reported. Missing physiological models or
parameters remain unknown; selecting a profile cannot manufacture a pass.
"""
from dataclasses import dataclass
from pathlib import Path
import json
from .body_constraints import coverage_table
from .inverse_kinematics import make_body


@dataclass(frozen=True)
class BodyModel:
    height_cm: float | None = None
    ratios: dict | None = None

    user_profile: dict | str | Path | None = None
    advanced_constraints: bool = True

    def __post_init__(self):
        from .body_user import load_user,body_lengths
        p=load_user(self.user_profile,self.height_cm,self.ratios)
        object.__setattr__(self,'user_profile',p)
        object.__setattr__(self,'height_cm',p['height_cm'])
        object.__setattr__(self,'ratios',{k:v/p['height_cm'] for k,v in body_lengths(p).items()})
        make_body(self.height_cm,self.ratios)
        object.__setattr__(self,'_profile_snapshot',json.dumps(p,sort_keys=True))

    @property
    def constraints(self):return coverage_table()

    def fit(self, trial, project=None, frames='verification', iterations=150,
            candidates=('anatomical_control','balanced','steady'), output_root=None, render=True, progress=print):
        """Fit one continuous interval; pass frames='all' for the full sequence."""
        if json.dumps(self.user_profile,sort_keys=True)!=self._profile_snapshot:
            raise ValueError('Profile changed after construction; create a new BodyModel from the updated profile')
        from .body_user import body_lengths
        if make_body(self.height_cm,self.ratios)!=body_lengths(self.user_profile):
            # Conversion through ratios can differ by roundoff, so compare tightly.
            import numpy as np
            if not np.allclose(list(make_body(self.height_cm,self.ratios).values()),list(body_lengths(self.user_profile).values()),rtol=0,atol=1e-10):
                raise ValueError('Body ratios changed after profile construction')
        from .body_pipeline import run
        return run(self,Path.cwd() if project is None else project,trial,frames,
                   iterations,candidates,output_root,render,progress)
