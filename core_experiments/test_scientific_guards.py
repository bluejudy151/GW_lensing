"""Focused guards against data leakage, false independence and spurious conditioning claims."""
import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from common import audit,summarize,load_sis,identity_split,assert_disjoint,Fit,normalized_geometry,GEOMETRY,FAMILIES
from matched_transfer import matched_sample,train_subset
from diagnostics import relation
from astrometry import observe_sis

class ScientificGuards(unittest.TestCase):
 def test_event_join_is_by_identity_not_row_position(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);pd.DataFrame({'event_id':[10,20],'mu_0':[3.,5.],'y':[.5,.25]}).to_csv(p/'lens.csv',index=False)
   pd.DataFrame({'event_id':[20,10],'feature':[200.,100.]}).to_csv(p/'observable_features.csv',index=False)
   pd.DataFrame({'event_id':[20,10],'x':[2.,1.]}).to_csv(p/'lens_params.csv',index=False)
   d,r=load_sis(p);self.assertEqual(d['obs__feature'].tolist(),[100.,200.]);self.assertEqual(r.x.tolist(),[1.,2.])
 def test_truth_and_simulator_distance_are_forbidden(self):
  for col in ['mu0_true','obs__source_luminosity_distance','obs__snr_pair','obs__y_true']:
   with self.assertRaises(ValueError):audit([col],pd.DataFrame({col:[1.]}))
 def test_all_repeated_identities_share_one_partition(self):
  d=pd.DataFrame({'identity_key':np.repeat(np.arange(50).astype(str),3)});split=identity_split(d,9)
  self.assertTrue((split.groupby('identity_key').split.nunique()==1).all())
  with self.assertRaises(ValueError):assert_disjoint(split,split.iloc[:2],split.iloc[:0])
 def test_replicated_model_rows_do_not_shrink_uncertainty(self):
  d=pd.DataFrame({'identity_key':np.arange(30).astype(str),'event_key':np.arange(30).astype(str),'abs_error':np.linspace(.1,3,30),'covered':np.linspace(0,1,30)})
  a=summarize(d,100,10);b=summarize(pd.concat([d]*32),100,10)
  for k in ['n_events','n_identities','mae','mae_lo95','mae_hi95','coverage90_lo95']:self.assertAlmostEqual(a[k],b[k])
 def test_sparse_all_failures_have_nontrivial_cluster_bound(self):
  d=pd.DataFrame({'identity_key':['a','b','c'],'event_key':['a','b','c'],'abs_error':[1.,2.,3.],'covered':[0.,0.,0.]})
  r=summarize(d,100,1);self.assertEqual(r['status'],'sparse_diagnostic');self.assertEqual(r['coverage90'],0);self.assertGreater(r['coverage_hoeffding_hi95'],.5)
 def test_determinant_has_no_independent_partial_rank_signal(self):
  mu=np.linspace(2,20,80);r=relation(1/mu,mu**2,mu,np.arange(80),20,1)
  self.assertTrue(np.isnan(r['partial_spearman_given_logmu']));self.assertEqual(r['partial_status'],'not_identifiable_collinear_or_constant')
 def test_matching_edges_ignore_test_labels(self):
  rng=np.random.default_rng(9);parts=[]
  for fam in FAMILIES:
   for sp in ['train','calibration','test']:
    n=100;parts.append(pd.DataFrame({'lens_family':fam,'split':sp,'mu0_true':np.exp(rng.uniform(1,3,n)),'obs__image_separation_observed':np.exp(rng.uniform(-1,1,n)),'event_key':[f'{fam}:{sp}:{i}' for i in range(n)],'identity_key':[f'{sp}:{i}' for i in range(n)]}))
  d=pd.concat(parts,ignore_index=True);a,c,info=matched_sample(d,'mu_separation',5,2)
  changed=d.copy();changed.loc[changed.split=='test','mu0_true']*=1000;_,_,other=matched_sample(changed,'mu_separation',5,2)
  self.assertEqual(info['log_edges'],other['log_edges']);self.assertEqual(info['training_common_cells'],other['training_common_cells'])
  for _,g in a.groupby(['split','match_cell']):self.assertEqual(len(set(g.lens_family.value_counts())),1)
  tr=a[a.split=='train'];self.assertEqual(len(train_subset(tr,False,2)),len(train_subset(tr,True,2)))
 def test_normalization_invariant_to_translation_and_dilation(self):
  d=pd.DataFrame({c:[1.] for c in GEOMETRY});d['obs__image_separation_observed']=4.
  n,cols=normalized_geometry(d);changed=d.copy()
  for c in GEOMETRY:
   if 'asymmetry' not in c:changed[c]*=3
  for ax in ['x','y']:
   for c in [f'obs__lens_center_{ax}_observed',f'obs__image_{ax}_0_observed',f'obs__image_{ax}_1_observed']:changed[c]+=8
  m,_=normalized_geometry(changed);np.testing.assert_allclose(n[cols],m[cols])
 def test_nominal_ideal_sis_reobservation(self):
  base=pd.DataFrame({'event_id':[0]});ref=pd.DataFrame({'image_x_0':[1.2],'image_y_0':[0.],'image_x_1':[-.8],'image_y_1':[0.],'lens_center_x':[0.],'lens_center_y':[0.]})
  z=np.zeros((1,6));d=observe_sis(base,ref,z,0,0,0)
  self.assertAlmostEqual(d.obs__image_position_asymmetry_observed.iloc[0],.2)
  self.assertAlmostEqual(d.obs__image_separation_observed.iloc[0],2.)
 def test_calibration_identity_leakage_rejected(self):
  n=20;d=pd.DataFrame({'identity_key':np.arange(n).astype(str),'mu0_true':np.linspace(2,3,n),'obs__x':np.linspace(0,1,n)})
  with self.assertRaises(ValueError):Fit(d,d.iloc[:10],['obs__x'],1,max_iter=2,members=1)

if __name__=='__main__':unittest.main(verbosity=2)
