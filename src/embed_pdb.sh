""""
Persistent Sheaf Laplacian (PSL)

Copyright (c) 2026 Mushal Zia

"""

#!/bin/bash --login
#SBATCH --job-name=esm_v2007_t33
#SBATCH --constraint=amd24
#SBATCH --array=0-129
#SBATCH --time=01:00:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/scratch/ziamusha/v2007_fasta/logs/esm_%A_%a.out
#SBATCH --error=/mnt/scratch/ziamusha/v2007_fasta/logs/esm_%A_%a.err

set -euo pipefail

LOGDIR=/mnt/scratch/ziamusha/v2007_fasta/logs
mkdir -p "$LOGDIR"

module purge
module load Python/3.11.3-GCCcore-12.3.0

source /mnt/home/ziamusha/PSL/venv_psl/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1

export TORCH_HOME=/mnt/home/ziamusha/PSL/torch-cache-t33
export XDG_CACHE_HOME=/mnt/scratch/ziamusha/.cache
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"

export OPENBLAS_CORETYPE=generic
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DEBUG_CPU_TYPE=5

CHUNK=10
START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))

echo "Running on host: $(hostname)"
echo "Python: $(which python)"
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import esm; print('esm ok')"
echo "TORCH_HOME=$TORCH_HOME"
echo "Processing rows $START to $END"

python /mnt/home/ziamusha/PSL/embed_pdb.py "$START" "$END"
