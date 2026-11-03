#!/usr/bin/env bash
# Stop the depth-error sweep at a given time (default 09:00 today). Resume later with scripts/run_depth_error_sweep.sh
# (finished runs are skipped; only the episodes in flight at the stop are redone).
T=${1:-09:00}
sleep $(( $(date -d "$T" +%s) - $(date +%s) ))
P=$(ps -eo pid,cmd | grep -E '[r]un_depth_error_sweep.sh' | awk '{print $1}')
[ -n "$P" ] && kill -9 $P          # the driver first, so it can't start the next stage
P=$(ps -eo pid,cmd | grep -E '[r]un_pointnav_eval.py' | awk '{print $1}')
[ -n "$P" ] && kill -9 $P          # the workers ignore SIGTERM (seen 2026-11-02)
echo "sweep stopped at $(date +%T); $(find results/runs/pointnav/A results/runs/pointnav/S -name '*.json' 2>/dev/null | wc -l) runs done" >> results/runs/pointnav/sweep_A.log
