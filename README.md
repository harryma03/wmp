<h1>WMP</h1>

Code for the paper: 
### World Model-based Perception for Visual Legged Locomotion
[Hang Lai](https://apex.sjtu.edu.cn/members/laihang@apexlab.org), [Jiahang Cao](https://apex.sjtu.edu.cn/members/jhcao@apexlab.org), [JiaFeng Xu](https://scholar.google.com/citations?user=GPmUxtIAAAAJ&hl=zh-CN&oi=ao), [Hongtao Wu](https://scholar.google.com/citations?user=7u0TYgIAAAAJ&hl=zh-CN&oi=ao), [Yunfeng Lin](https://apex.sjtu.edu.cn/members/yflin@apexlab.org), [Tao Kong](https://www.taokong.org/), [Yong Yu](https://scholar.google.com.hk/citations?user=-84M1m0AAAAJ&hl=zh-CN&oi=ao), [Weinan Zhang](https://wnzhang.net/) 

### [🌐 Project Website](https://wmp-loco.github.io/) | [📄 Paper](https://arxiv.org/abs/2409.16784)
   
## Requirements
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch:
    - `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`
3. Install Isaac Gym
    - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
    - `cd isaacgym/python && pip install -e .`
4. Install other packages:
    - `sudo apt-get install build-essential --fix-missing`
    - `sudo apt-get install ninja-build`
    - `pip install setuptools==59.5.0`
    - `pip install ruamel_yaml==0.17.4`
    - `sudo apt install libgl1-mesa-glx -y`
    - `pip install opencv-contrib-python`
    - `pip install -r requirements.txt`

pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python==4.6.0.66

python -c "import isaacgym; import torch; import cv2; import ruamel.yaml; import legged_gym; import rsl_rl; print('isaacgym:', isaacgym.__file__); print('torch:', torch.__version__); print('cv2:', cv2.__version__); print('cuda:', torch.cuda.is_available(), torch.cuda.device_count()); print('WMP imports OK')"

## Training
```
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0
```
Training takes about 23G GPU memory, and at least 10k iterations recommended.

## Visualization
**Please make sure you have trained the WMP before**
```
python legged_gym/scripts/play.py --task=zg_m1_amp --sim_device=cuda:0 --terrain=climb
```

For Titatit, `python legged_gym/scripts/play.py --task=titatit --sim_device=cuda:0 --terrain=climb`
also exports `logs/titatit_wmp/exported/policies/policy.onnx` and
`world_model.onnx`. Install `onnx` in the training environment first.

`world_model.onnx` runs once per depth update (every 5 control steps by default).
Inputs are `prop [B,41]`, `image [B,64,64,1]`, `prev_stoch [B,32,32]`,
`prev_deter [B,512]`, `action_history [B,80]`, and `is_first [B]` (1 on
episode reset, otherwise 0). Feed `next_stoch` and `wm_feature` back as
`prev_stoch` and `prev_deter` on the next update. The exported RSSM uses the
most likely latent category for deterministic deployment; this differs from
the sampled latent used in Python playback. `policy.onnx` takes
`observation [B,273]`, `history [B,270]`, and the latest `wm_feature [B,512]`,
and returns `action [B,16]`. Maintain the observation, history, and action
history preprocessing used in `play.py` when running the models externally.

## github
cd /home/yons/harryma/WMP
git status
git add .
git commit -m "Update wrap"
git push origin main

## Acknowledgments

We thank the authors of the following projects for making their code open source:

- [leggedgym](https://github.com/leggedrobotics/legged_gym)
- [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)
- [parkour](https://github.com/ZiwenZhuang/parkour/tree/main)
- [extreme-parkour](https://github.com/chengxuxin/extreme-parkour)



## Citation

If you find this project helpful, please consider citing our paper:
```
@article{lai2024world,
  title={World Model-based Perception for Visual Legged Locomotion},
  author={Lai, Hang and Cao, Jiahang and Xu, Jiafeng and Wu, Hongtao and Lin, Yunfeng and Kong, Tao and Yu, Yong and Zhang, Weinan},
  journal={arXiv preprint arXiv:2409.16784},
  year={2024}
}
```
