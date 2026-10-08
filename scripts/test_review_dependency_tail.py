"""Small scientific guards; full experiments are intentionally not run."""
import unittest
import numpy as np
import pandas as pd
from review_dependency_tail import representative, reduced_noise, sis_load, sis_fit, GEOMETRY, ROOT, summarize


class Guards(unittest.TestCase):
    def test_representative_ignores_labels(self):
        d = pd.DataFrame(dict(identity_key=["a","a","b"],event_key=["a1","a2","b1"],score=[1,99,2]))
        a = representative(d,4).event_key.tolist()
        d.score = [999,-3,0]
        self.assertEqual(a,representative(d,4).event_key.tolist())

    def test_cluster_coverage(self):
        d = pd.DataFrame(dict(identity_key=["a","a","b"],truth=[1.,5.,1.],pred=[1.,1.,1.],lo=[0.,0.,0.],hi=[2.,2.,2.]))
        m = summarize(d,20,1)
        self.assertAlmostEqual(m["coverage90"],2/3)
        self.assertEqual(m["equal_identity_coverage"],.75)
        self.assertEqual(m["simultaneous_identity_coverage"],.5)

    def test_saved_noise_and_oracle(self):
        d,r = sis_load(ROOT/"data_generation/data_lens_sis_gw_physics_baseline","test:")
        for col in GEOMETRY:
            self.assertIn(col,d)
        z = reduced_noise(d,r,0.)
        np.testing.assert_allclose(z.obs__image_position_asymmetry_observed,d.y,atol=1e-12)
        one = reduced_noise(d,r,1.)
        np.testing.assert_allclose(one[GEOMETRY],d[GEOMETRY],atol=1e-10)
        fn,q = sis_fit(d.iloc[:100],d.iloc[100:120],GEOMETRY,"log",1.)
        self.assertTrue(np.isfinite(fn(d.iloc[120:125])).all())
        self.assertTrue(np.isfinite(q))

    def test_primary_point_reproduction(self):
        d,_ = sis_load(ROOT/"data_generation/data_lens_sis_gw_physics_baseline","original:")
        protocol = ROOT.parent/"review_followup_20261006/results_main/contracts"
        m = pd.read_csv(protocol/"split_manifest.csv").query("split_seed == 11")
        d = d.merge(m[["event_id","split"]],on="event_id",validate="one_to_one")
        tr,ca,te = [d[d.split==s] for s in ["train","calibration","test"]]
        fn,q = sis_fit(tr,ca,GEOMETRY,"log",1.)
        r = pd.read_csv(protocol/"predictions.csv").query("split_seed == 11 and contract == 'geometry'").set_index("event_id").loc[te.event_id]
        np.testing.assert_allclose(fn(te),r.mu0_pred,atol=1e-8)
        np.testing.assert_allclose(fn(te)+q,r.hi,atol=1e-8)


if __name__=="__main__": unittest.main()
