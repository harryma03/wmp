import os,subprocess,concurrent.futures
from pathlib import Path
root=Path('/home/aiseon/wmp');out=Path(__file__).resolve().parent

def run(job):
 label,run,ckpt,mode,seed=job
 stem=f'{label}_{mode}_{seed}'
 env=dict(os.environ,LD_LIBRARY_PATH='/home/aiseon/anaconda3/envs/wmp/lib:'+os.environ.get('LD_LIBRARY_PATH',''),WMP_SMALL_PHYSX='1',WMP_SKIP_EXPORT='1',WMP_GAP_AB=mode,WMP_PLAY_TRACE_PATH=str(out/(stem+'.jsonl')))
 with (out/(stem+'.log')).open('w') as f:
  p=subprocess.run(['/home/aiseon/anaconda3/envs/wmp/bin/python','-u','legged_gym/scripts/play.py','--task=titatit','--sim_device=cuda:0','--terrain=gap',f'--load_run={run}',f'--checkpoint={ckpt}','--num_envs=1','--headless',f'--seed={seed}'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 print(stem,'exit',p.returncode,flush=True)
 return p.returncode
jobs=[(label,run,ckpt,mode,seed) for seed in range(1,6) for mode in ('flat','isolated') for label,run,ckpt in [('old','WMP',21000),('new','gap_generalization_20261031_175729',22000)]]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(run,jobs))
assert not any(results),results
