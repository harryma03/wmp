import sys,json,numpy as np,mujoco,onnxruntime as ort,yaml
from pathlib import Path
root=Path('/home/aiseon/wmp/ddt_ros2_control/ddt_ros2_control')
c=yaml.safe_load((root/'controller/rl_controller/config/titatit/controllers_wmp.yaml').read_text())['/**']['titatit_rl_controller']['ros__parameters'];cfg=c['rl_wmp']
o=ort.SessionOptions();o.intra_op_num_threads=1
policy=ort.InferenceSession(str(root/'controller/rl_controller/config/titatit/policy.onnx'),o);wm=ort.InferenceSession(str(root/'controller/rl_controller/config/titatit/world_model.onnx'),o)
m=mujoco.MjModel.from_xml_path(str(Path('/home/aiseon/wmp_sim2sim_overlay/install/titatit_description/share/titatit_description/mujoco')/sys.argv[1]));d=mujoco.MjData(m)
jids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in c['joints']]);qi=m.jnt_qposadr[jids];vi=m.jnt_dofadr[jids];idx=np.array(cfg['reindex']);sgn=np.array(cfg['re_sign']);default=np.array(cfg['default_joint_angles']);scales=np.array(cfg['action_scales']);kp=np.array(cfg['joint_kp']);kd=np.array(cfg['joint_kd']);wheel=np.array([3,7,11,15]);lim=np.array([55,55,55,10]*4)
d.qpos[:3]=[0,0,.5];d.qpos[3:7]=[1,0,0,0];d.qpos[qi]=default;mujoco.mj_forward(m,d)
base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'base_link');cam=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'depth_camera')
gyroid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SENSOR,'trunk_gyro');gyroadr=m.sensor_adr[gyroid]
history=np.zeros((5,54),np.float32);last=np.zeros(16,np.float32);ah=np.zeros((5,16),np.float32);stoch=np.zeros((1,32,32),np.float32);feature=np.zeros((1,512),np.float32);first=True
previous_image=np.zeros((1,64,64,1),np.float32)
y,x=np.mgrid[:64,:64];rays=np.stack([(x-32)/32,-(y-32)/32,-np.ones_like(x)/np.tan(np.deg2rad(58/2))],-1).reshape(-1,3);rays/=np.linalg.norm(rays,axis=1,keepdims=True);forward=-rays[:,2];groups=np.array([1,1,0,0,1,1],np.uint8);ids=np.zeros(4096,np.int32);dist=np.zeros(4096)
records=[]
for t in range(600):
 q=d.qpos[qi].copy();q[wheel]=(q[wheel]+2*np.pi)%(4*np.pi)-2*np.pi
 pos=((q-default)[idx]*sgn).astype(np.float32);vel=(d.qvel[vi][idx]*sgn*.05).astype(np.float32);gravity=(d.xmat[base].reshape(3,3).T@np.array([0,0,-1])).astype(np.float32);ang=(d.sensordata[gyroadr:gyroadr+3]*.25).astype(np.float32)
 yaw=np.arctan2(d.xmat[base].reshape(3,3)[1,0],d.xmat[base].reshape(3,3)[0,0]);cmd=np.array([1.2 if t>=25 else 0,0,np.clip(-yaw*.5,-1,1)*.25],np.float32)
 frame=np.r_[ang,gravity,pos,vel,last];history=np.concatenate([history[1:],frame[None]])
 if (t+1)%5==0:
  prop=np.clip(np.r_[ang,gravity,cmd,pos,vel],-100,100)[None]
  stoch,feature=wm.run(None,dict(prop=prop,image=previous_image,prev_stoch=stoch,prev_deter=feature,action_history=ah.reshape(1,80),is_first=np.array([first],np.float32)));first=False
  dirs=rays@d.cam_xmat[cam].reshape(3,3).T
  mujoco.mj_multiRay(m,d,d.cam_xpos[cam],dirs.ravel(),groups,1,-1,ids,dist,4096,5)
  depth=np.where(dist<0,2,dist*forward);previous_image=(np.clip(depth,0,2)/2-.5).astype(np.float32).reshape(1,64,64,1)
 obs=np.zeros((1,273),np.float32);obs[0,35:38]=cmd
 raw=policy.run(None,dict(observation=obs,history=np.clip(history,-100,100).reshape(1,270),wm_feature=feature))[0][0];ah=np.concatenate([ah[1:],raw[None]]);last=np.clip(raw,-6,6);act=last[idx]*sgn
 for _ in range(4):
  mujoco.mj_step1(m,d)
  tau=kp*(act*scales+default-d.qpos[qi])-kd*d.qvel[vi];tau[wheel]=kd[wheel]*(act[wheel]*scales[wheel]-d.qvel[vi][wheel]);d.qfrc_applied[vi]=np.clip(tau,-lim,lim);mujoco.mj_step(m,d)
 if t%50==0:print(sys.argv[1],t,np.round(d.qpos[:3],3),'gravity',np.round(gravity,2),flush=True)
 records.append(dict(step=t,root=d.qpos[:7].tolist(),action=raw.tolist()))
 if d.qpos[2]<.1:print('fallen',t,flush=True);break
Path('/tmp/wmp_ab_'+Path(sys.argv[1]).parent.name+'.rollout.json').write_text(json.dumps(records))
