import json
from pathlib import Path
import numpy as np
base=Path(__file__).resolve().parent/'gap_ab'
summary=[]
for seed in range(1,6):
 pair=[]
 for mode in ['isolated','flat']:
  path=base/f'{mode}_{seed}.jsonl'
  if not path.exists():continue
  rows=[]
  for line in path.open():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:break
  if not rows:continue
  xyz=np.array([r['root'][:3] for r in rows]);quat=np.array([r['root'][3:7] for r in rows]);pair.append(rows[0]['root'])
  # First forward gap: x=6.0..6.2 in the first terrain tile.
  # Body beyond x=6.8 leaves rear wheels clearance; check before any reset.
  reset=np.flatnonzero(np.linalg.norm(np.diff(xyz,axis=0),axis=1)>1.)
  end=int(reset[0]+1) if len(reset) else len(rows);xyz=xyz[:end]
  crossed=np.flatnonzero((xyz[:,0]>6.8)&(xyz[:,2]>.25)&(abs(xyz[:,1]-4)<.8))
  item=dict(seed=seed,mode=mode,complete=len(rows)==601,first_gap_pass=bool(len(crossed)),first_reset_step=int(reset[0]+1) if len(reset) else None,max_x=float(xyz[:,0].max()),min_height=float(xyz[:,2].min()),max_side_offset=float(abs(xyz[:,1]-4).max()))
  summary.append(item);print(item)
 if len(pair)==2:print('paired initial max difference',seed,float(np.max(abs(np.array(pair[0])-np.array(pair[1])))))
(base/'summary.json').write_text(json.dumps(summary,indent=2))
