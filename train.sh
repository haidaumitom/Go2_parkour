# Usage: bash train.sh [num_envs] [max_iterations] [checkpoint]
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
num_envs=${1:-2048}
max_iterations=${2:-5000}
checkpoint=${3:-logs/rsl_rl/unitree_go2_rough_walk/2026-11-21_12-22-19_rough_walk_gait/model_1999.pt}

PYTHONPATH="$PWD" .venv/bin/python scripts/rsl_rl/train.py \
  --task Isaac-Rough-Walk-Two-Hurdle-Unitree-Go2-v0 \
  --num_envs "$num_envs" --max_iterations "$max_iterations" \
  --seed 1 --headless --resume --reset_iteration \
  --checkpoint "$checkpoint" --run_name two_hurdle_gait

# Extreme parkour teacher on hurdle terrain
# python scripts/rsl_rl/train.py \
#   --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
#   --num_envs 2048 --max_iterations 4000 \
#   --seed 1 --headless --run_name stepping_stones_only \
#   env.scene.terrain.terrain_generator.sub_terrains.parkour.proportion=0.0 \
#   env.scene.terrain.terrain_generator.sub_terrains.parkour_gap.proportion=0.0 \
#   env.scene.terrain.terrain_generator.sub_terrains.parkour_hurdle.proportion=1.0 \
#   env.scene.terrain.terrain_generator.sub_terrains.parkour_flat.proportion=0.0 \
#   env.scene.terrain.terrain_generator.sub_terrains.parkour_step.proportion=0.0
