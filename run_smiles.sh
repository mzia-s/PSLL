#!/bin/bash --login
#SBATCH --job-name=run_smiles
#SBATCH --array=0-268
#SBATCH --time=00:30:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/home/ziamusha/PDBbind_datasets/v2013_SMILES/logs/run_smiles_%A_%a.out
#SBATCH --error=/mnt/home/ziamusha/PDBbind_datasets/v2013_SMILES/logs/run_smiles_%A_%a.err

set -euo pipefail

BASE_DIR="/mnt/home/ziamusha/PDBbind_datasets/v2013_SMILES"
CSV_FILE="${BASE_DIR}/v2013_lp_smiles.csv"
LOG_DIR="${BASE_DIR}/logs"
OUT_DIR="${BASE_DIR}/v2013_smi_features"
TMP_DIR="${BASE_DIR}/tmp_smiles_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${TMP_DIR}"

module purge
module load Python/3.11.3-GCCcore-12.3.0

cd /mnt/home/ziamusha/PretrainModels

CHUNK_SIZE=11
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE - 1))

echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Processing rows ${START} to ${END}"

export CSV_FILE
export OUT_DIR
export TMP_DIR
export START
export END

python - <<'PY'
import os
import csv
import subprocess
import sys

csv_file = os.environ["CSV_FILE"]
out_dir = os.environ["OUT_DIR"]
tmp_dir = os.environ["TMP_DIR"]
start = int(os.environ["START"])
end = int(os.environ["END"])

model_name_or_path = "./chembl27_pubchem_zinc_512"
checkpoint_file = "checkpoint_best.pt"
data_name_or_path = "./chembl27_pubchem_zinc_512"
dict_file = "./chembl27_pubchem_zinc_512/dict.txt"

rows = []
with open(csv_file, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

n = len(rows)
print(f"Total complexes in CSV: {n}")

if start >= n:
    print(f"Start index {start} is beyond total rows {n}. Nothing to do.")
    sys.exit(0)

end = min(end, n - 1)

for idx in range(start, end + 1):
    row = rows[idx]
    pdbid = row["pdbid"].strip()
    smiles = row["SMILES"].strip()

    if not pdbid or not smiles:
        print(f"[{idx}] Skipping empty entry")
        continue

    tmp_smi = os.path.join(tmp_dir, f"{pdbid}.smi")
    out_file = os.path.join(out_dir, f"{pdbid}.npy")

    if os.path.exists(out_file):
        print(f"[{idx}] {pdbid}: output already exists, skipping")
        continue

    with open(tmp_smi, "w") as f:
        f.write(smiles + "\n")

    cmd = [
        "python", "./bt_pro/generate_bt_fps.py",
        "--model_name_or_path", model_name_or_path,
        "--checkpoint_file", checkpoint_file,
        "--data_name_or_path", data_name_or_path,
        "--dict_file", dict_file,
        "--target_file", tmp_smi,
        "--save_feature_path", out_file,
    ]

    print(f"[{idx}] Processing {pdbid}")
    try:
        subprocess.run(cmd, check=True)
        print(f"[{idx}] Saved: {out_file}")
    except subprocess.CalledProcessError as e:
        print(f"[{idx}] FAILED for {pdbid}: {e}")

PY

rm -rf "${TMP_DIR}"
echo "Done."