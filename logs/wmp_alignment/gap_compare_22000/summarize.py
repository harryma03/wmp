import json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
results=[];initial={}
for f in sorted(p.glob('*.jsonl')):
 label,mode,seed=f.stem.split('_');seed=int(seed)
 try:rows=[json.loads(l) for l in f.open()]
 except ValueError:continue
 if len(rows)!=601:continue
 initial[label,mode,seed]=rows[0]['root']
 first=[r for r in rows if r['episode']==0];a=np.array([r['root'][:3] for r in first])
 crossing=np.flatnonzero((a[:,0]>6.8)&(a[:,2]>.25)&(abs(a[:,1]-4)<.8))
 passed=bool(len(crossing));deviation=None
 if passed:
  approach=a[:crossing[0]+1];approach=approach[approach[:,0]>=5.5]
  deviation=float(abs(approach[:,1]-4).max())
 results.append(dict(model=label,mode=mode,seed=seed,pass_previous_criterion=passed,direct_pass=passed and deviation<.45,approach_max_lateral=deviation,first_reset_step=next((r['step'] for r in rows if r['episode']>0),None),max_x=float(a[:,0].max())))
for mode in ('isolated','flat'):
 for label in ('old','new'):
  subset=[r for r in results if r['mode']==mode and r['model']==label]
  print(label,mode,'complete',len(subset),'direct passes',sum(r['direct_pass'] for r in subset),'previous criterion',sum(r['pass_previous_criterion'] for r in subset))
diffs=[]
for mode in ('isolated','flat'):
 for seed in range(1,6):
  if ('old',mode,seed) in initial and ('new',mode,seed) in initial:
   diffs.append(float(np.max(np.abs(np.array(initial['old',mode,seed])-initial['new',mode,seed]))))
print('maximum paired initial-state difference:',max(diffs) if diffs else None)
(p/'summary.json').write_text(json.dumps(dict(results=results,paired_initial_max_difference=max(diffs) if diffs else None),indent=2))
