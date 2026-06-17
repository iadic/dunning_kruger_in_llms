#!/bin/bash
#SBATCH --job-name=dke-eval
#SBATCH --partition=gpu               # <-- change to your cluster's GPU partition
#SBATCH --gres=gpu:1                  # 1 GPU; request more only if you shard a 70B model
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out      # %x = job name, %j = job id

# =============================================================================
# Example SLURM submission script for the DKE-in-LLMs evaluation pipeline.
#
# This is a TEMPLATE. Adjust the #SBATCH directives above to match your
# cluster (partition names, GPU type, account/project, module system, etc.),
# then submit with:
#
#     sbatch run_example.slurm
#
# To launch many configurations, copy this script or submit it in a loop
# overriding MODEL_ID / QUANT / SUBJECT_GROUP via --export, e.g.:
#
#     for m in meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B google/gemma-2-9b; do
#         sbatch --export=ALL,MODEL_ID=$m,QUANT=False,SUBJECT_GROUP=math run_example.slurm
#     done
# =============================================================================

set -euo pipefail

# ---- Environment ------------------------------------------------------------
# Point the HF cache at fast scratch storage (read by every script):
export HF_CACHE="${HF_CACHE:-$HOME/hf_cache}"
# Optional: path to your .env holding HF_TOKEN. Defaults to ".env" in the CWD.
export DOTENV_PATH="${DOTENV_PATH:-$PWD/.env}"

# Activate your Python environment (edit to match your setup):
# module load python/3.11
# source ~/envs/llm_env/bin/activate
# or: conda activate llm_env

# ---- Run configuration (override via --export when submitting) ---------------
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
QUANT="${QUANT:-False}"               # True = NF4 4-bit (use for large models)
SUBJECT_GROUP="${SUBJECT_GROUP:-math}"

# ---- Choose the matching script ---------------------------------------------
# MMLU (easier benchmark):       mmlu_base.py   / mmlu_instruct.py
# MMLU-Pro (harder benchmark):   mmlupro_base.py / mmlupro_instruct.py
SCRIPT="Python HPC versions/mmlu_instruct.py"

echo "Model:   $MODEL_ID"
echo "Quant:   $QUANT"
echo "Subjects:$SUBJECT_GROUP"
echo "Script:  $SCRIPT"

python "$SCRIPT" "$MODEL_ID" "$QUANT" "$SUBJECT_GROUP"
