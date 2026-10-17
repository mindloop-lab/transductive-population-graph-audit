#!/usr/bin/env python3
"""Figures 6/7 from packaged frozen summaries; no fitting or new inference.
Run from any directory. SVG text remains editable; PDF/SVG/PNG are generated together.
Use --output-dir DIR to render elsewhere without overwriting the packaged figures.
Requires the versions in requirements_fig6_fig7.txt for matching rendering.
"""
from pathlib import Path
import argparse
import csv,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Patch
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'figure_sources'/'inputs'/'fig6_fig7'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=ROOT/'figures',
                    help='Output directory (default: packaged figures directory).')
args=parser.parse_args()
OUT=args.output_dir.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'paperm-fig6-fig7','pdf.fonttype':42})
rows=list(csv.DictReader((DATA/'major1_site_level.csv').open()))
pop=json.loads((DATA/'major1_population_inference.json').read_text())
sym={r['site']:float(r['symmetrised_site_macro']) for r in csv.DictReader((DATA/'SYMMETRISED_VS_CANONICAL.csv').open())}
manifest=json.loads((DATA/'SPLIT_AND_MASK_MANIFEST.json').read_text())
sites=[r['site'] for r in rows]
all_sites={c['site'] for c in manifest['cells']}
eligible={c['site'] for c in manifest['cells'] if c['feasible']}
excluded=sorted(all_sites-eligible)  # all cells absent, not any one failed cell
partial=[r['site'] for r in rows if int(r['folds_used'])<5]
assert len(sites)==18 and set(sites)==eligible==set(sym)
assert sum(int(r['folds_used']) for r in rows)==88
assert math.isclose(sum(float(r['primary']) for r in rows)/18,pop['population_macro'],abs_tol=1e-12)
for r in rows:
    assert math.isclose(sum(json.loads(r['per_seed']).values())/3,float(r['primary']),abs_tol=1e-12)
def save(fig,name):
    for ext in ('svg','pdf','png'):
        # Fixed SVG ids and omitted timestamps make repeat exports stable.
        metadata = {'Date': None} if ext == 'svg' else (
            {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {})
        fig.savefig(OUT/f'{name}.{ext}',dpi=220,facecolor='white',metadata=metadata)
    plt.close(fig)
# Figure 6: roles -> intervention -> fixed information-access contract.
fig=plt.figure(figsize=(8.4,4.8))
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
def text(x,y,s,size=10,**kw):
    return ax.text(x,y,s,fontsize=size,transform=ax.transAxes,**kw)
def box(x,y,w,h,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.012',facecolor=color,edgecolor='#adb7c2',lw=.8,transform=ax.transAxes))
text(.025,.94,'A   Subject roles in one paired site–fold cell',12,weight='bold')
roles=[('T','Target-site test','Labels sealed','#dbe9f6'),('S','Target-site support','Supervised in ON','#e3efdc'),('R','Other-site support','Supervised in OFF','#fce9d9'),('C','Common trainable','Supervised in both','#e9edf0'),('V','Fixed validation','Model selection only','#f1e7f3')]
for i,(symbol,role,note,color) in enumerate(roles):
    x=.025+i*.193
    box(x,.675,.178,.205,color)
    text(x+.089,.82,symbol,18,ha='center',weight='bold')
    text(x+.089,.756,role,9,ha='center')
    text(x+.089,.707,note,8,ha='center',color='#465463')
text(.025,.61,'B   Change support location; match the supervision budget',12,weight='bold')
box(.025,.365,.456,.19,'#f1f7ed');box(.52,.365,.456,.19,'#fff5ec')
text(.047,.51,'ON   Target-site support',11,weight='bold',color='#37602c')
text(.047,.445,'Supervised labels:  C ∪ S',12)
text(.047,.39,'R remains in the graph, without training labels',8.5)
text(.542,.51,'OFF   Other-site support',11,weight='bold',color='#8c5422')
text(.542,.445,'Supervised labels:  C ∪ R',12)
text(.542,.39,'S remains in the graph, without training labels',8.5)
text(.5,.32,'R matches S in total size and per-class counts; C = O ∖ R',10,ha='center')
text(.025,.245,'C   Fixed across ON and OFF',12,weight='bold')
text(.025,.19,'Population edges: same-site pairs among T ∪ S ∪ R ∪ C; V has no population edges.',9.5)
text(.025,.135,'Same graph, validation set, initialisation and training schedule in each paired comparison.',9.5)
text(.025,.08,'Preprocessing fit: S ∪ O; T and V are transform-only. Test labels never supervise training.',9.5)
save(fig,'F1_onoff_protocol_schematic')
# Figure 7: concise key outside axes; method details in manuscript caption.
fig,ax=plt.subplots(figsize=(7.6,6.8))
fig.subplots_adjust(left=.17,right=.98,bottom=.24,top=.9)
lo,hi=pop['ci95']
ax.axvspan(lo,hi,color='#cfe3f7',alpha=.6,zorder=0)
ax.axvline(pop['population_macro'],color='#1f4e79',lw=1.4,zorder=1)
ax.axvline(0,color='#888',lw=1,ls='--',zorder=1)
for i,r in enumerate(rows):
    values=[json.loads(r['per_seed'])[s] for s in ('666','2025','2026')]
    ax.scatter(values,[i-.08,i,i+.08],s=15,color='#777',alpha=.8,zorder=3)
    ax.plot([float(r['ci_lo']),float(r['ci_hi'])],[i,i],color='#444',lw=1.2,zorder=2)
    ax.plot(float(r['primary']),i,marker='D',ms=5.6,color='#c0392b',zorder=4)
    ax.plot(sym[r['site']],i,marker='s',ms=5.1,mfc='none',mec='#1f7a4d',mew=1.3,zorder=5)
ax.set_yticks(range(len(sites)));ax.set_yticklabels([s+' *' if s in partial else s for s in sites],fontsize=9)
ax.set_ylim(-.8,len(sites)-.2);ax.set_xlim(-.31,.84)
ax.set_xlabel('ΔAUC (ON − OFF)',fontsize=11,labelpad=9)
ax.spines[['top','right']].set_visible(False)
fig.suptitle('Supervision-location effect by site',y=.975,fontsize=13,weight='bold')
fig.text(.575,.929,f'Directed population mean {pop["population_macro"]:+.3f}  |  95% CI [{lo:+.3f}, {hi:+.3f}]',ha='center',fontsize=9,color='#1f4e79')
handles=[Line2D([],[],marker='D',ls='none',color='#c0392b',label='Directed site mean'),Line2D([],[],marker='s',ls='none',mfc='none',mec='#1f7a4d',mew=1.3,label='Symmetrised site mean'),Line2D([],[],marker='o',ls='none',color='#777',ms=4,label='Seed effect (directed)'),Line2D([],[],color='#444',lw=1.2,label='Site 95% CI (directed)'),Line2D([],[],color='#1f4e79',lw=1.4,label='Population mean (directed)'),Patch(facecolor='#cfe3f7',alpha=.6,label='Population 95% CI (directed)')]
legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.035),ncol=2,fontsize=9,frameon=False,columnspacing=1.7,handlelength=2.1,labelspacing=.7)
fig.canvas.draw()
assert legend.get_window_extent().y1<ax.get_window_extent().y0
save(fig,'F2_site_delta_auc_forest')
print('18 sites; 88 eligible folds; incomplete-fold sites:',partial,'; no eligible cells:',excluded)

print('Figures written to:', OUT)
