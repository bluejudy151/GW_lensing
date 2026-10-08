"""Analytic and invariance checks; does not run the catalogue experiment."""
import unittest
import numpy as np
import pandas as pd
from scipy.special import i0e
from sis_bayesian_primary import posterior, selected_prior


class LikelihoodChecks(unittest.TestCase):
    def test_angular_integral(self):
        phi = np.linspace(0, 2*np.pi, 100000, endpoint=False)
        for z in [0., 1., 10., 100.]:
            numerical = np.mean(np.exp(z*(np.cos(phi)-1)))
            self.assertAlmostEqual(numerical, i0e(z), places=11)

    def test_translation_rotation_and_grid(self):
        rng = np.random.default_rng(123)
        train = pd.DataFrame({"y":rng.uniform(.05,.98,1750), "theta_E(arcsec)":np.exp(rng.normal(0,.3,1750))})
        obs = pd.DataFrame({"image_x_0_observed":[1.3], "image_y_0_observed":[0.],
                            "image_x_1_observed":[-.7], "image_y_1_observed":[0.],
                            "lens_center_x_observed":[0.], "lens_center_y_observed":[0.]})
        def infer(o, size):
            y,t,lp,_ = selected_prior(train,size,size,1.)
            return posterior(o,y,t,lp,.01,.02)
        p = infer(obs,512)
        self.assertLess(abs(p[0,0]-(1+1/.3)),.03)
        moved = obs.copy()
        for prefix in ["image", "lens_center"]:
            for suffix in (["_0_observed","_1_observed"] if prefix == "image" else ["_observed"]):
                x,y = prefix+"_x"+suffix,prefix+"_y"+suffix
                vx,vy = moved[x].copy(),moved[y].copy()
                moved[x],moved[y] = -vy+3.,vx-2.
        np.testing.assert_allclose(p,infer(moved,512),atol=1e-10)
        np.testing.assert_allclose(p,infer(obs,1024),atol=.01)


if __name__ == "__main__":
    unittest.main()
