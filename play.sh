# Usage: bash play.sh [num_envs] [terrain_level] [checkpoint]
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
num_envs=${1:-1}
terrain_level=${2:-0}
checkpoint=${3:-checkpoints/two_hurdle_gait/model_500.pt}

PYTHONPATH="$PWD" .venv/bin/python scripts/rsl_rl/play.py \
  --task Isaac-Rough-Walk-Two-Hurdle-Unitree-Go2-Play-v0 \
  --checkpoint "$checkpoint" \
  --num_envs "$num_envs" --terrain_level "$terrain_level" --real-time

# Extreme parkour mixed terrain (8 robots)
# python scripts/rsl_rl/play.py \
#   --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
#   --checkpoint logs/rsl_rl/unitree_go2_parkour/2026-11-20_12-03-48_stepping_stones_only/model_3999.pt \
#   --num_envs 8 \
#   --real-time \
#   env.scene.terrain.terrain_generator.sub_terrains.parkour.proportion=0.2 \
#   env.scene.terrain.terrain_generator.sub_terrains.parkour_gap.proportion=0.2 \
#   env.scene.terrain.terrain_generator.sub_terrains.parkour_hurdle.proportion=0.2 \
#   env.scene.terrain.terrain_generator.sub_terrains.parkour_flat.proportion=0.2 \
#   env.scene.terrain.terrain_generator.sub_terrains.parkour_step.proportion=0.2
