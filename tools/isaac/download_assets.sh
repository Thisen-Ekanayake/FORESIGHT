#!/usr/bin/env bash
# Download the NVIDIA Isaac Sim 6.0 assets for the G1 demo (~2.8 GB) into /ml/FORESIGHT/isaac_assets/.
# Resumable: run it again after an interruption and it skips finished files.
# Afterwards Isaac Sim loads these with --/persistent/isaac/asset_root/default=/ml/FORESIGHT/isaac_assets
# (no internet needed at render time).
set -euo pipefail
cd "$(dirname "$0")/../.."
DEST="$PWD/isaac_assets"
mkdir -p "$DEST"

PREFIXES=(
  Isaac/Environments/Office/                     # 0.74 GB  glass meeting rooms (hero scene)
  Isaac/Environments/Hospital/                   # 1.10 GB
  Isaac/Environments/Simple_Warehouse/           # 0.54 GB
  Isaac/Environments/Digital_Twin_Warehouse/     # 0.05 GB
  Isaac/IsaacLab/Robots/Unitree/G1/              # 0.08 GB  G1 the Isaac Lab walking policy was trained on
  Isaac/Robots/Unitree/G1/                       # 0.18 GB  G1 29-dof with hand variants (Inspire, 3-finger)
  Isaac/Robots/Unitree/H1/                       # 0.09 GB  fallback: H1 (hand payloads included)
  Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/
  Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-G1-v0/
)

echo "== sizes"
python3 -I tools/isaac/s3_assets.py list "${PREFIXES[@]}"
echo "== downloading to $DEST"
python3 -I tools/isaac/s3_assets.py fetch "$DEST" "${PREFIXES[@]}"

echo "== checking that every referenced file is present (fetches shared materials/props the scenes point to)"
/ml/isaac-sim/python.sh tools/isaac/check_asset_deps.py "$DEST" \
  Isaac/Environments/Office/office.usd \
  Isaac/Environments/Hospital/hospital.usd \
  Isaac/Environments/Simple_Warehouse/full_warehouse.usd \
  Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd \
  Isaac/IsaacLab/Robots/Unitree/G1/g1.usd \
  Isaac/Robots/Unitree/G1/g1.usd \
  Isaac/Robots/Unitree/H1/h1.usd

du -sh "$DEST"
echo "done"
