#!/usr/bin/env python3
"""Check the stored Stage1 numeric reference against the pinned original .osim."""
from pathlib import Path
import argparse,hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_translator.trial2_stage1 import source_reference,verify_chain
from beta_translator.anatomical_arms import ARM_NAMES,ARM_LIMITS_DEG

def verify_file(path):
    ref=source_reference();path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['source_sha256']:
        raise ValueError('File is not the pinned source snapshot')
    root=ET.parse(path).getroot()
    for expected in ref['joints']:
        joint=next(j for j in root.findall('.//CustomJoint') if j.get('name')==expected['name'])
        for tag in ['location_in_parent','orientation_in_parent','location','orientation']:
            np.testing.assert_array_equal(np.fromstring(joint.findtext(tag),sep=' '),expected[tag])
        axes=[a for a in joint.findall('./SpatialTransform/TransformAxis')
              if a.get('name').startswith('rotation') and a.find('function/Constant') is None]
        assert len(axes)==len(expected['rotations'])
        for axis,want in zip(axes,expected['rotations']):
            np.testing.assert_array_equal(np.fromstring(axis.findtext('axis'),sep=' '),want['axis'])
            assert axis.findtext('coordinates')==want['coordinate']
            fun=list(axis.find('function'))[0];assert fun.tag==want['function']
            for child in fun:np.testing.assert_array_equal(np.fromstring(child.text,sep=' '),want[child.tag])
    for want in ref['couplers']:
        c=next(c for c in root.findall('.//CoordinateCouplerConstraint') if c.findtext('dependent_coordinate_name')==want['coordinate'])
        assert c.findtext('independent_coordinate_names')==want['independent']
        fun=list(c.find('coupled_coordinates_function'))[0];assert fun.tag==want['function']
        for child in fun:np.testing.assert_array_equal(np.fromstring(child.text,sep=' '),want[child.tag])
    for name,limits in zip(ARM_NAMES,ARM_LIMITS_DEG):
        coord=next(c for c in root.findall('.//Coordinate') if c.get('name')==name)
        np.testing.assert_allclose(np.fromstring(coord.findtext('range'),sep=' '),np.deg2rad(limits),atol=1e-8,rtol=0)
    return {'coordinate_ranges_verified':True,'pinned_file_verified':True,'source_sha256':ref['source_sha256'],**verify_chain()}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('osim',type=Path)
    print(json.dumps(verify_file(parser.parse_args().osim),indent=2))
