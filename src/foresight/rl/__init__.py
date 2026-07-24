"""RL package for FORESIGHT PointGoal navigation.

Importing this package registers the custom habitat components (measure + task-action) with habitat's
registry and Hydra ConfigStore, so training/rollout scripts only need `import foresight.rl`.
"""
from foresight.rl import smooth_reward, velocity_action  # noqa: F401
