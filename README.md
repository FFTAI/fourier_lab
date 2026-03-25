## Overview

This repository shows the training code for GRX Humanoid with IsaacLab.

This readme is ok for Isaaclab v2.3.0 and isaacsim v5.1.0

## Installation

```
# install isaacsim
conda create -n lab230 python=3.11
conda activate lab230
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
isaacsim
# mkdir and clone code
mkdir GRX_humanoid && cd GRX_humanoid
git clone https://github.com/FFTAI/fourier_lab.git
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
# Switch IsaacLab to tag v2.3.0
git checkout v2.3.0
# install isaaclab 
./isaaclab.sh -i
# install rsl_rl from grx_humanoid project file
cd ../fourier_lab
../IsaacLab/isaaclab.sh -p -m pip install -e rsl_rl
# test the rsl_rl location(option)
../IsaacLab/isaaclab.sh -p -m pip show rsl-rl-lib 
# install robot project
cd exts/GRX_humanoid
python -m pip install -e .

if there are other tasks, do the same thing
```

## Add your robot model and task

```
If youwant to add new robot
1.add models in {your workspace}/models
2.use convert tool get you usd model
3.add model config in asset
4.add new env config in locomotion/velocity(for example)
5.register new task
```

## Tips before you start training

1.Before run the robot train, make sure you are in the right folder.

2.Ignore tasks that are not in the list below.

## Run For GR2T2V2 WBC LOWER

```
# convert robot model from urdf to usd
python scripts/tools/convert_urdf.py models/gr2t2v2/urdf/GR2T2V2_lower.urdf  exts/GRX_humanoid/GRX_humanoid/assets/Robots/gr2t2v2_humanoid_lower.usd --merge-joints
# run script for training
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task GR2T2V2HumanoidRoughEnvCfg_WBC_LOWER --headless
# run script for playing
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task GR2T2V2HumanoidRoughEnvCfg_WBC_LOWER_Play
# log
tensorboard --logdir logs/rsl_rl/gr2t2v2_humanoid_rough_wbc_lower/
```

## Run For WBC LOWER (PPV211, PPV222, PPV233 & PPV224)

```
# convert robot model from urdf to usd
python scripts/tools/convert_urdf.py models/gr3v2_1_1/basic_urdf/gr3v2_1_1_lower.urdf  exts/GRX_humanoid/GRX_humanoid/assets/Robots/ppv211_humanoid_lower.usd --merge-joints
# run script for training
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task PPV211HumanoidRoughEnvCfg_WBC_LOWER  --headless
# run script for playing
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task PPV211HumanoidRoughEnvCfg_WBC_LOWER_Play
# log
tensorboard --logdir  logs/rsl_rl/ppv211_humanoid_rough_wbc_lower/
```

## Run For WBC FULL(PPV211, PPV222, PPV233 & PPV224)

```
# convert robot model from urdf to usd
python scripts/tools/convert_urdf.py models/gr3v2_1_1/basic_urdf/gr3v2_1_1_noArmColli.urdf  exts/GRX_humanoid/GRX_humanoid/assets/Robots/ppv211_noArmCollision.usd --merge-joints
# run script for training
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task PPV211HumanoidRoughEnvCfg_WBC_FULL  --headless
# run script for playing
../IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task PPV211HumanoidRoughEnvCfg_WBC_FULL_Play
# log
tensorboard --logdir  logs/rsl_rl/ppv211_humanoid_rough_wbc_full/
```

## Debug
```
# If raise error: 
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /usr/local/anaconda3/envs/lab230/lib/python3.11/lib-dynload/../.././libicui18n.so.78)
# Do it in the conda enviroment
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```