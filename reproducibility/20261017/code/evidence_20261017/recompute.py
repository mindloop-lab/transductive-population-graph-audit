#!/usr/bin/env python3
"""CPU recomputation of retained new evidence. No training or network access.
Run: python evidence_20261017/recompute.py [--controls]
Outputs go to a separate recomputed/ folder. Inputs are never overwritten.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--controls',action='store_true');args=ap.parse_args()
O=R/'recomputed';O.mkdir(exist_ok=True)
def auc(y,s):
 y=np.asarray(y,dtype=int);n=int(y.sum());n0=len(y)-n
 if n==0 or n0==0:return float('nan')
 return float((rankdata(s)[y==1].sum()-n*(n+1)/2)/(n*n0))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
m=pd.read_csv(R/'split_masks/SPLIT_MASKS.csv',dtype={'subject_id':str,'site':str})
assert not m.duplicated(['fold','subject_id']).any()
assert m.groupby('subject_id').label.nunique().max()==1
assert m.groupby('subject_id').site.nunique().max()==1
d=pd.read_csv(R/'site_prior_main/SITE_PRIOR_MAIN_PER_SUBJECT.csv',dtype={'subject_id':str,'site':str})
for k,g in d.groupby('fold'):
 tr=m[(m.fold==k)&(m.role=='optimization_train')]
 for _,r in g.iterrows():
  ss=tr[tr.site==r.site];score=ss.label.mean() if len(ss) else tr.label.mean()
  assert abs(score-r.score)<1e-12
  assert r.n_opt_train_at_site==len(ss) and r.n_opt_train_at_site_asd==ss.label.sum()
site_order=d.site.drop_duplicates().tolist();chunks=[g[['label','score']].to_numpy() for s in site_order for g in [d[d.site==s]]]
rng=np.random.default_rng(20261015);boots=[]
for _ in range(10000):
 v=np.concatenate([chunks[i] for i in rng.choice(len(chunks),len(chunks),replace=True)])
 boots.append(auc(v[:,0],v[:,1]))
prior={'pooled_auc':auc(d.label,d.score),'ci95':np.percentile(boots,[2.5,97.5]).tolist()}
assert np.isclose(prior['pooled_auc'],0.5056228924094929,atol=1e-12,rtol=0)
assert np.allclose(prior['ci95'],[0.4176388002243165,0.5464389246726991],atol=1e-12,rtol=0)
z=np.load(R/'split_masks/seed666_fast_oof.npz',allow_pickle=False)
main_auc=auc(z['oof_label'],softmax(z['oof_logits'],axis=1)[:,1])
assert np.isclose(main_auc,0.9396089160357151,atol=1e-12,rtol=0)
checks['prior_and_main']='PASS'

# Verify the separate paired-prior count ratios and per-arm uniqueness.
p=pd.read_csv(R/'site_prior_paired/SITE_PRIOR_PAIRED_PER_SUBJECT.csv',dtype={'subject_id':str,'site':str})
assert not p.duplicated(['arm','subject_id']).any()
for arm,g in p.groupby('arm'):
 assert len(g)==836
 expected=np.where(g.n_arm_train_at_site>0,
                   g.n_arm_train_at_site_asd/g.n_arm_train_at_site.replace(0,np.nan),
                   g.arm_train_asd/g.n_arm_train)
 assert np.allclose(g.score,expected,atol=1e-12,rtol=0)
checks['paired_prior_count_ratios']='PASS (arithmetic only; see original protocol manifest)'

e=pd.read_csv(R/'E3_predictions.csv',dtype={'subject_id':str,'site':str})
assert not e.duplicated(['arm','seed','subject_id']).any()
assert e.groupby('subject_id').label.nunique().max()==1
assert e.groupby('subject_id').site.nunique().max()==1
rows=[]
for (arm,seed),g in e.groupby(['arm','seed']):
 assert len(g)==871 and g.site.nunique()==20
 for site,h in g.groupby('site'):rows.append({'arm':arm,'seed':int(seed),'site':site,'auc':auc(h.label,h.score)})
t=pd.DataFrame(rows).pivot(index=['seed','site'],columns='arm',values='auc')
delta=t.ON-t.OFF;site_delta=delta.groupby('site').mean().sort_index().to_numpy()
rng=np.random.default_rng(20261016)
bs=site_delta[rng.integers(0,20,size=(5000,20))].mean(axis=1)
e3={'ON_macro':float(t.ON.mean()),'OFF_macro':float(t.OFF.mean()),'delta':float(delta.mean()),
    'ci95':np.percentile(bs,[2.5,97.5]).tolist(),'B':5000,'seed':20261016,'unit':'site'}
ref=json.loads((R/'E3_recomputed_summary.json').read_text())
assert np.isclose(e3['delta'],ref['point'],atol=1e-12,rtol=0)
assert np.allclose(e3['ci95'],ref['ci95'],atol=1e-12,rtol=0)
checks['E3_endpoint_and_interval']='PASS'

control_results=[]
if args.controls:
 sm=m.drop_duplicates('subject_id').set_index('subject_id')
 ref=json.loads((R/'control_contrasts.json').read_text())
 for c in ref['contrasts']:
  pp=[R/'control_predictions'/c[a]['archive'] for a in ['arm_a','arm_b']]
  for path,arm in zip(pp,['arm_a','arm_b']):assert digest(path)==c[arm]['sha256']
  a,b=[np.load(path,allow_pickle=False) for path in pp]
  for key in ['subject_ids','oof_label','oof_fold']:assert np.array_equal(a[key],b[key])
  y=a['oof_label'].astype(int);ids=a['subject_ids'].astype(str)
  assert len(set(ids))==871 and np.array_equal(sm.loc[ids].label,y)
  sites=sm.loc[ids].site.to_numpy();units=sorted(set(sites));chunks=[np.flatnonzero(sites==s) for s in units]
  sa,sb=[softmax(v['oof_logits'],axis=1)[:,1] for v in [a,b]]
  rng=np.random.default_rng(20261015);boots=[]
  for _ in range(10000):
   ix=np.concatenate([chunks[i] for i in rng.choice(20,20,replace=True)])
   boots.append(auc(y[ix],sa[ix])-auc(y[ix],sb[ix]))
  point=auc(y,sa)-auc(y,sb);ci=np.percentile(boots,[2.5,97.5])
  assert np.isclose(point,c['pooled_delta_auc'],atol=1e-12,rtol=0)
  assert np.allclose(ci,c['ci_95_site_cluster'],atol=1e-12,rtol=0)
  control_results.append({'contrast':c['contrast'],'delta':point,'ci95':ci.tolist()})
  print('PASS',c['contrast'],flush=True)
 checks['control_contrasts']='PASS (9/9)'
out={'checks':checks,'site_prior':prior,'main_model_auc':main_auc,'E3':e3,'controls':control_results}
(O/'verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
