"""
Persistent Sheaf Laplacian (PSL)

Copyright (c) 2026 Mushal Zia

"""

#!/bin/bash --login
#SBATCH --job-name=psl_v2013
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/%%%%%%%%%%%%%%%/psl_%A_%a.out
#SBATCH --error=/%%%%%%%%%%%%%%%/psl_%A_%a.err
#SBATCH --array=0-268

set -euo pipefail

PY="/mnt/home/ziamusha/PSL/PSL.py"

#BASE_DIR="/%%%%%%%%%%%%%%%/PDBbind_datasets/v2007"
PROTEIN_PQR_DIR="/%%%%%%%%%%%%%%%"
PROTEIN_PQR_SUFFIX="protein"

OUT_DIR="/%%%%%%%%%%%%%%%/psl_v2013_CS"

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}"

# -------- knobs --------
#   ES = element-specific, CS = category-specific, BOTH = run both ES and CS
SCHEMA="CS"

# selection mode only. "P-L" means protein-ligand point cloud, not edge restriction
BLOCKS="P-L"

# Save eigenvalue spectra text files
#   0 = do not write spectra .txt files, #   1 = write spectra .txt files under: <OUT_DIR>/<pdbid>/spectra/
SAVE_SPECTRA=1

# Laplacian features
#   VR0    = dim-0 (bipartite cross-group edges only), #   Alpha1 = dim-1 (all edges)
DO_VR0=1
DO_ALPHA1=1

INTERFACE_CUTOFF="10" # for CS
#INTERFACE_CUTOFF="12" # for ES
BINS="1,2,3,4,5,6,7,8,9,10,11,12"

PRO_ELEMENTS="C,N,O,S"
# match PSL.py default ordering
LIG_ELEMENTS="C,N,O,F,P,S,Cl,Br,I"

# how many pdbids per array task
UNIT=11
START=$(( SLURM_ARRAY_TASK_ID * UNIT ))
END=$(( START + UNIT ))

# -------- derived flags (from knobs) --------
VR0_FLAGS=""
ALPHA_FLAGS=""
SPECTRA_FLAGS=""

if [[ "${DO_VR0}" -eq 0 ]]; then
  VR0_FLAGS="--no_vr0"
fi

if [[ "${DO_ALPHA1}" -eq 0 ]]; then
  ALPHA_FLAGS="--no_alpha1"
fi

if [[ "${SAVE_SPECTRA}" -eq 1 ]]; then
  SPECTRA_FLAGS="--save_spectra"
fi

python "${PY}" \
  --all --start "${START}" --end "${END}" \
  --base_dir "${BASE_DIR}" \
  --protein_pqr_dir "${PROTEIN_PQR_DIR}" \
  --protein_pqr_suffix "${PROTEIN_PQR_SUFFIX}" \
  --out_dir "${OUT_DIR}" \
  --schema "${SCHEMA}" \
  --blocks "${BLOCKS}" \
  --interface_cutoff "${INTERFACE_CUTOFF}" \
  --bins "${BINS}" \
  --pro_elements "${PRO_ELEMENTS}" \
  --lig_elements "${LIG_ELEMENTS}" \
  ${VR0_FLAGS} \
  ${ALPHA_FLAGS} \
  ${SPECTRA_FLAGS}

echo "Done array task ${SLURM_ARRAY_TASK_ID} (pdbids ${START}:${END})"
