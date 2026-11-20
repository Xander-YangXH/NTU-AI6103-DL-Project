"""Plot actual exported initialization ranks and first500 optimizer losses."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'results'
TYPES=['attention.self.query','attention.self.key','attention.self.value','attention.output.dense','intermediate.dense','output.dense']
SHORT=['Query','Key','Value','Attn out','FFN in','FFN out']


def save(fig,name):
    fig.savefig(ROOT/(name+'.png'),dpi=170);fig.savefig(ROOT/(name+'.svg'));plt.close(fig)
    svg=ROOT/(name+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n',encoding='utf-8')


def main():
    fig,axes=plt.subplots(3,2,figsize=(12,13),layout='constrained')
    for i,task in enumerate(['rte','mrpc','sst2']):
        for j,group in enumerate(['G5','G6']):
            matrices=[np.array([[r['ranks'][f'roberta.encoder.layer.{depth}.{kind}'] for kind in TYPES] for depth in range(12)]) for r in [json.loads(p.read_text()) for p in (ROOT/'main').glob(f'main_{group}_{task}_*.json')]]
            data=np.mean(matrices,axis=0);ax=axes[i,j]
            image=ax.imshow(data,vmin=0,vmax=16,cmap='magma',aspect='auto')
            ax.set_xticks(range(6),SHORT,rotation=30,ha='right');ax.set_yticks(range(12),range(12));ax.set_ylabel('Encoder layer')
            ax.set_title(f'{task.upper()} / {group} / mean of {len(matrices)} seeds')
            fig.colorbar(image,ax=ax,label='Allocated rank')
    fig.suptitle('G5/G6 rank allocation / independent completed study');save(fig,'g56_rank_heatmap')
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,task in zip(axes,['rte','mrpc','sst2']):
        for group in ['G5','G6']:
            runs=[json.loads(p.read_text()) for p in (ROOT/'main').glob(f'main_{group}_{task}_*.json')]
            # Smooth each seed first; shading is descriptive seed sample SD.
            values=np.array([np.convolve(r['loss_first_steps'],np.ones(25)/25,mode='valid') for r in runs])
            mean=values.mean(0);std=values.std(0,ddof=1);x=np.arange(25,501)
            ax.plot(x,mean,label=group);ax.fill_between(x,mean-std,mean+std,alpha=.15)
        ax.set_title(task.upper());ax.set_xlabel('Optimizer step');ax.set_ylabel('Training loss (25-step mean)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('First500 steps / mean +/- sample SD / task-group-specific selected LR');save(fig,'first500-loss')
    print('Wrote rank and loss PNG/SVG figures')


if __name__=='__main__':main()
