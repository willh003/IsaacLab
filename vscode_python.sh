#!/bin/bash
cd /home/will/IsaacLab
source _isaac_sim/setup_python_env.sh
export CARB_APP_PATH=/home/will/IsaacLab/_isaac_sim/kit
export ISAAC_PATH=/home/will/IsaacLab/_isaac_sim
export EXP_PATH=/home/will/IsaacLab/_isaac_sim/apps
exec /home/will/IsaacLab/_isaac_sim/kit/python/bin/python3 "$@"