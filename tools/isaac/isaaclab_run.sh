#!/usr/bin/env bash
# Run a script with Isaac Lab 3.0 (/ml/IsaacLab, its uv venv) on top of the Isaac Sim 6.0.1 binary install.
# The venv has no isaacsim package, so Isaac Sim's paths come from its setup_conda_env.sh; EXP_PATH points the
# AppLauncher at Isaac Sim's app (.kit) files. Any active virtualenv (e.g. .venv-laya) is ignored.
#   tools/isaac/isaaclab_run.sh tools/isaac/g1_live.py --scene office ...
unset VIRTUAL_ENV CONDA_PREFIX
source /ml/isaac-sim/setup_conda_env.sh
export CARB_APP_PATH=/ml/isaac-sim/kit
export ISAAC_PATH=/ml/isaac-sim
export EXP_PATH=/ml/isaac-sim/apps
export RESOURCE_NAME="IsaacSim"
export LD_PRELOAD=/ml/isaac-sim/kit/libcarb.so${LD_PRELOAD:+:$LD_PRELOAD}
exec /ml/IsaacLab/env_isaaclab/bin/python "$@"
