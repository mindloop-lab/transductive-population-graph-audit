#!/usr/bin/env python3
"""Rebuild the new LOSO figure from retained seed endpoints and paired summary."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
d=pd.read_csv(R/'evidence_20261017/E3_seed_endpoints.csv')
j=json.loads((R/'evidence_20261017/E3_recomputed_summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(9,3.4),gridspec_kw={'width_ratios':[1.3,1]})
for seed,c in zip([666,2025,2026],['#1f77b4','#bc6c25','#2d6a4f']):
 vals=[d[(d.seed==seed)&(d.arm==a)].site_macro_auc.iloc[0] for a in ['ON','OFF']]
 axs[0].plot([0,1],vals,'o-',color=c,label=str(seed),lw=1.5,ms=6)
axs[0].set(xticks=[0,1],xticklabels=['Access','No access'],ylabel='Site-macro AUC',ylim=(.49,.64),title='(a) Fixed training seeds')
axs[0].axhline(.5,color='.7',ls='--',lw=1)
axs[0].legend(title='Seed',frameon=False,loc='upper center',ncol=3,fontsize=8)
point=j['point'];lo,hi=j['ci95']
axs[1].errorbar([point],[0],xerr=[[point-lo],[hi-point]],fmt='o',color='#183153',capsize=5,ms=7)
axs[1].axvline(0,color='.6',ls='--')
axs[1].set(yticks=[],ylim=(-.6,.8),xlim=(-.025,.02),xlabel='Access minus no access: macro AUC',title='(b) Paired site-bootstrap interval')
axs[1].text(.5,.18,f'{point:.4f}\n95% interval [{lo:.4f}, {hi:.4f}]',transform=axs[1].transAxes,ha='center',fontsize=10)
for a in axs:a.spines[['top','right']].set_visible(False)
fig.tight_layout(w_pad=2.5)
for ext in ['pdf','svg','png']:fig.savefig(R/f'figures/fig_loso_access_20261017.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
