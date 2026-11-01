"""Check projective grid recovery independently of the Trial 3 image."""
import unittest

import numpy as np

from beta_translator.plane_calibration import fit_grid, project


class PlaneCalibrationTests(unittest.TestCase):
    def test_perspective_grid_with_noise_and_false_candidates_recovers_metric_coordinates(self):
        rng=np.random.default_rng(41)
        matrix=np.array([[23., .4, 400.], [4., 22., 230.], [.0015, -.001, 1.]])
        grid=np.array([[x,y] for y in range(-3,28) for x in range(-4,15)],float)
        pixels=project(grid,matrix)+rng.normal(0,.1,grid.shape)
        seeds=np.array([[x,y] for y in range(3) for x in range(4)],float)
        seed_pixels=project(seeds,matrix)
        false=rng.uniform([280.,160.],[710.,930.],size=(80,2))
        fitted,metrics,_,_,_=fit_grid(np.vstack([pixels,false]),seed_pixels,seeds,15.)
        probes=np.array([[-2.,5.],[3.,16.],[10.,24.]])
        recovered=project(project(probes,matrix),fitted)
        np.testing.assert_allclose(recovered,probes*[15.,-15.],atol=.3)
        self.assertGreater(metrics['reserved_holes'],50)
        self.assertLess(metrics['reserved_rmse_px'],.35)
        np.testing.assert_allclose(project(recovered,np.linalg.inv(fitted)),project(probes,matrix),atol=1e-7)

    def test_grid_scale_is_physical_input_and_not_inferred_from_pixel_pitch(self):
        grid=np.array([[x,y] for y in range(12) for x in range(10)],float)
        matrix=np.array([[20.,1.,100.],[2.,18.,100.],[.001,.002,1.]])
        pixels=project(grid,matrix)
        seeds=np.array([[x,y] for y in range(3) for x in range(4)],float)
        seed_pixels=project(seeds,matrix)
        a,*_=fit_grid(pixels,seed_pixels,seeds,15.)
        b,*_=fit_grid(pixels,seed_pixels,seeds,30.)
        np.testing.assert_allclose(project(pixels,b),2*project(pixels,a),atol=1e-7)

    def test_invalid_scale_and_collinear_seed_grid_are_rejected(self):
        grid=np.array([[x,0.] for x in range(12)],float)
        with self.assertRaisesRegex(ValueError,'positive'):
            fit_grid(grid,grid,grid,0)
        with self.assertRaisesRegex(ValueError,'two-dimensional'):
            fit_grid(grid,grid,grid,15)


if __name__=='__main__':
    unittest.main()
