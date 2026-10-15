"""Regenerate assets/running_example.png (requires matplotlib)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parents[1]
from running_example import make_data

a = make_data(500, 7, contaminated=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(11,4.4),layout='constrained')
for ax,(x,y,title) in zip(axs,[(0,1,'X → Y\nBoth deviate'),(0,3,'X → W\nNeither deviates'),(1,2,'Y → Z\nOnly measurement deviates')]):
 ax.scatter(a[2:,x],a[2:,y],s=10,c='#b2bbc7',alpha=.55,rasterized=True)
 for row,color,marker,label in [(0,'#0072B2','o','m'),(1,'#D55E00','D','g')]:
  ax.scatter(a[row,x],a[row,y],s=85,c=color,marker=marker,edgecolors='white',linewidths=1,zorder=3)
  ax.annotate(label,(a[row,x],a[row,y]),xytext=(-16, 8) if row == 0 else (8, -14),textcoords='offset points',color=color,weight='bold',fontsize=12)
 ax.set(xlabel=['X','Y','Z','W'][x],ylabel=['X','Y','Z','W'][y],title=title)
 ax.margins(.16)
fig.suptitle('Same root cause Y, different anomaly types\nW ← X → Y → Z',fontsize=16,weight='bold')
fig.legend(handles=[plt.Line2D([],[],color='#0072B2',marker='o',ls='',label='m: measurement error (row 0)'),plt.Line2D([],[],color='#D55E00',marker='D',ls='',label='g: mechanism shift (row 1)')],loc='outside lower center',ncol=2,frameon=False)
p=root/'examples/assets';p.mkdir(exist_ok=True)
fig.savefig(p/'running_example.png',dpi=180,facecolor='white')
