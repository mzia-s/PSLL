#!/usr/bin/env python3

"""
Persistent Sheaf Laplacian (PSL)

Copyright (c) 2026 Mushal Zia

"""

import os
import sys
import torch
import esm
import numpy as np
import pandas as pd

# Load ESM-2 t33 model -> 1280-dim embeddings
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()


batch_converter = alphabet.get_batch_converter()
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Paths
csv_path = "/%%%%%%%%%%%%%%%/v2016_fasta.csv"
#csv_path = "/%%%%%%%%%%%%%%%/v2007_fasta_test.csv"
#output_dir = "%%%%%%%%%%%%%%%/v2016_prot33"
output_dir = "/%%%%%%%%%%%%%%%/v2016_prot33"
#output_dir = "/%%%%%%%%%%%%%%%/v2007_prot33_test"
os.makedirs(output_dir, exist_ok=True)

# Command-line args
if len(sys.argv) != 3:
    print("Usage: python embed_pdb.py <start_idx> <end_idx>")
    sys.exit(1)

start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])

# Load CSV
df = pd.read_csv(csv_path)

required_cols = {"PDBID", "FASTA_Sequence"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

n_rows = len(df)

start_idx = max(0, start_idx)
end_idx = min(end_idx, n_rows)

if start_idx >= end_idx:
    print(f"Nothing to process: start={start_idx}, end={end_idx}, total={n_rows}")
    sys.exit(0)

valid_chars = set("ACDEFGHIKLMNPQRSTVWYXUO")

print(f"Processing rows {start_idx} to {end_idx - 1} out of {n_rows}")

for idx in range(start_idx, end_idx):
    row = df.iloc[idx]
    pdb_id = str(row["PDBID"]).strip()
    sequence = str(row["FASTA_Sequence"]).strip()

    if not sequence or sequence.lower() == "nan":
        print(f"SKIP row {idx}: empty FASTA_Sequence for PDBID {pdb_id}")
        continue

    sequence = sequence.replace(" ", "").replace("\n", "").upper()

    bad_chars = sorted(set(sequence) - valid_chars)
    if bad_chars:
        print(f"SKIP row {idx}: invalid residues for PDBID {pdb_id}: {bad_chars}")
        continue

    out_filename = f"{str(idx + 1).zfill(len(str(n_rows)))}_embedding_{pdb_id}.npy"
    out_path = os.path.join(output_dir, out_filename)

    if os.path.exists(out_path):
        print(f"SKIP row {idx}: already exists {out_filename}")
        continue

    print(f"Row {idx}: processing {pdb_id} | length={len(sequence)}")

    data = [(pdb_id, sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[33])

    reps = results["representations"][33]
    embedding = reps[0, 1:len(sequence) + 1].mean(0).cpu().numpy()

    np.save(out_path, embedding)
    print(f"Saved {out_path} | shape={embedding.shape}")

