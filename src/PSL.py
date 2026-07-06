#!/usr/bin/env python3

"""
Persistent Sheaf Laplacian (PSL)

Copyright (c) 2026 Mushal Zia

"""

import argparse
import os
import math
import numpy as np
import gudhi as gd

from readin import pqr_to_pqre, mol2_to_pqre

# ----------------------- Defaults -----------------------
PRO_ELEMENTS_DEFAULT = ["C", "N", "O", "S"]
LIG_ELEMENTS_DEFAULT = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

# Residue categories (peer-style)
CATEGORIES_P = [
    ["GLY", "ALA", "VAL", "LEU", "ILE", "MET", "PRO", "PHE", "TRP"],  # hydrophobic
    ["SER", "THR", "ASN", "GLN", "TYR", "CYS"],                       # polar
    ["ASP", "GLU"],                                                   # acidic
    ["LYS", "ARG", "HIS"],                                            # basic
]


# ----------------------- Helpers -----------------------
def fmt_float(x, decimals=16):
    s = f"{float(x):.{int(decimals)}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def fmt_bin(x):
    return f"{float(x):.2f}"


def list_pdbids_from_base_dir(base_dir):
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        return []
    ids = []
    for name in os.listdir(base_dir):
        if name.startswith("."):
            continue
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            ids.append(name)
    return sorted(ids)


def symmetric_interface_crop_masks(pro_xyz, lig_xyz, cutoff, block_size=256):
    """
    Returns boolean masks keep_pro, keep_lig indicating atoms within cutoff of at least one partner.
    Uses blockwise squared-distance computation to save memory.
    """
    cutoff2 = float(cutoff) ** 2
    P = pro_xyz.shape[0]
    L = lig_xyz.shape[0]
    keep_pro = np.zeros(P, dtype=bool)
    keep_lig = np.zeros(L, dtype=bool)
    if P == 0 or L == 0:
        return keep_pro, keep_lig

    bs = int(block_size)
    for ps in range(0, P, bs):
        pe = min(P, ps + bs)
        blk = pro_xyz[ps:pe]
        diffs = blk[:, None, :] - lig_xyz[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        close = (d2 <= cutoff2)
        keep_pro[ps:pe] |= np.any(close, axis=1)
        keep_lig |= np.any(close, axis=0)
    return keep_pro, keep_lig


# ----------------------- PSL Matrices -----------------------
def compute_manual_laplacian_L0(st, cutoff, values, num_pts, pt_group, min_dist=1e-9):
    """
    VR0: L0 = D^T D, using ONLY cross-group edges (pt_group differs).
    Edge weights use charges / distance:
      row r for edge (u,v): D[r,u] = -q_v/d, D[r,v] = +q_u/d
    """
    active_edges = []
    cut = float(cutoff)

    for simplex, filt in st.get_filtration():
        if len(simplex) != 2:
            continue
        dist = float(filt)
        if dist > cut:
            continue
        u, v = sorted(simplex)
        if pt_group[u] == pt_group[v]:
            continue  # drop pro-pro and lig-lig
        active_edges.append((u, v, dist))

    if not active_edges:
        return None

    D = np.zeros((len(active_edges), num_pts), dtype=float)
    for r, (u, v, dist) in enumerate(active_edges):
        q_u = float(values[u])
        q_v = float(values[v])
        d = dist if dist > min_dist else min_dist
        D[r, u] = -q_v / d
        D[r, v] =  q_u / d

    return D.T @ D


def compute_alpha_L1_matrix(stA, cutoff_sq, values, pts, num_pts, pt_group, min_dist=1e-9):
    """
    Alpha1 edge-Laplacian: L1 = D0 D0^T + D1^T D1

    - Keep ALL edges (pro-pro, lig-lig, pro-lig).
    - Alpha filtration uses (d/2)^2 for edges, so pass cutoff_sq = (cut/2)^2.
    - Recover true edge length as d_true = 2*sqrt(filt).
    """
    cutoff_sq = float(cutoff_sq)

    active_edges = []
    edge_to_idx = {}
    triangles_raw = []

    for simplex, filt in stA.get_filtration():
        ff = float(filt)
        if ff > cutoff_sq:
            continue

        if len(simplex) == 2:
            u, v = sorted(simplex)
            edge_to_idx[(u, v)] = len(active_edges)
            d_true = 2.0 * math.sqrt(max(ff, 0.0))
            active_edges.append((u, v, d_true))

        elif len(simplex) == 3:
            u, v, w = sorted(simplex)
            triangles_raw.append((u, v, w))

    m = len(active_edges)
    if m == 0:
        return None

    # D0: edges x vertices
    D0 = np.zeros((m, num_pts), dtype=float)
    for ei, (u, v, d) in enumerate(active_edges):
        q_u = float(values[u])
        q_v = float(values[v])
        dist = d if d > min_dist else min_dist
        D0[ei, u] = -q_v / dist
        D0[ei, v] =  q_u / dist

    # keep triangles whose 3 edges exist
    active_triangles = []
    for (u, v, w) in triangles_raw:
        e_uv = tuple(sorted((u, v)))
        e_uw = tuple(sorted((u, w)))
        e_vw = tuple(sorted((v, w)))
        if (e_uv in edge_to_idx) and (e_uw in edge_to_idx) and (e_vw in edge_to_idx):
            active_triangles.append((u, v, w))

    t = len(active_triangles)
    D1 = np.zeros((t, m), dtype=float)
    pts_arr = np.array(pts, dtype=float)

    def dist(i, j):
        dv = pts_arr[i] - pts_arr[j]
        return float(np.linalg.norm(dv))

    # D1: triangles x edges
    for ti, (u, v, w) in enumerate(active_triangles):
        e_uv = tuple(sorted((u, v)))
        e_uw = tuple(sorted((u, w)))
        e_vw = tuple(sorted((v, w)))

        idx_uv = edge_to_idx[e_uv]
        idx_uw = edge_to_idx[e_uw]
        idx_vw = edge_to_idx[e_vw]

        d_uv = dist(u, v) or min_dist
        d_uw = dist(u, w) or min_dist
        d_vw = dist(v, w) or min_dist

        q_u = float(values[u])
        q_v = float(values[v])
        q_w = float(values[w])

        D1[ti, idx_uv] =  q_w / (d_uw * d_vw)
        D1[ti, idx_uw] = -q_v / (d_uv * d_vw)
        D1[ti, idx_vw] =  q_u / (d_uv * d_uw)

    return (D0 @ D0.T) + (D1.T @ D1)


def eig_stats(eigs, eig_tol=1e-8):
    """
    9 stats on positive eigenvalues:
      [sum, min, max, mean, std, var, sumsq, pos_count, zero_count]
    """
    if eigs is None:
        return np.zeros(9, dtype=float)

    arr = np.array(eigs, dtype=float).ravel()
    if arr.size == 0:
        return np.zeros(9, dtype=float)

    tol = float(eig_tol)
    pos = arr[arr > tol]
    zero_count = float(arr.size - pos.size)

    out = np.zeros(9, dtype=float)
    if pos.size == 0:
        out[8] = float(arr.size)
        return out

    out[0] = float(pos.sum())
    out[1] = float(pos.min())
    out[2] = float(pos.max())
    out[3] = float(pos.mean())
    out[4] = float(pos.std())
    out[5] = float(pos.var())
    out[6] = float(np.dot(pos, pos))
    out[7] = float(pos.size)
    out[8] = zero_count
    return out


# ----------------------- Point cloud builders -----------------------
def build_points_for_block_ES(block, pro_df, lig_df, pe, le):
    """
    Build ES point clouds for P-L only.
    Returns (pts, values, pt_group).
    """
    if block != "P-L":
        raise ValueError(f"Only P-L is supported. Got block={block}")

    pro_ele = pro_df["element"].astype(object).to_numpy()
    lig_ele = lig_df["element"].astype(object).to_numpy()

    pro_mask = (pro_ele == pe)
    lig_mask = (lig_ele == le)

    pro_xyz = pro_df.loc[pro_mask, ["x", "y", "z"]].to_numpy(float)
    pro_q = pro_df.loc[pro_mask, "charge"].to_numpy(float)
    lig_xyz = lig_df.loc[lig_mask, ["x", "y", "z"]].to_numpy(float)
    lig_q = lig_df.loc[lig_mask, "charge"].to_numpy(float)

    if pro_xyz.shape[0] == 0 or lig_xyz.shape[0] == 0:
        return None, None, None

    pts = np.vstack([pro_xyz, lig_xyz])
    values = np.concatenate([pro_q, lig_q]).astype(float)

    pt_group = np.zeros(pts.shape[0], dtype=int)
    pt_group[pro_xyz.shape[0]:] = 1
    return pts, values, pt_group


def build_points_for_block_CS(block, pro_df, lig_df, category_list, le):
    """
    Category-specific builder for P-L only:
      protein side selected by residue category, using pro_df["resname"] from PQR.

    Returns (pts, values, pt_group).
    """
    if block != "P-L":
        raise ValueError(f"Only P-L is supported. Got block={block}")

    if "resname" not in pro_df.columns:
        return None, None, None

    pro_mask = pro_df["resname"].astype(str).isin(category_list).to_numpy()
    lig_ele = lig_df["element"].astype(object).to_numpy()
    lig_mask = (lig_ele == le)

    pro_xyz = pro_df.loc[pro_mask, ["x", "y", "z"]].to_numpy(float)
    pro_q = pro_df.loc[pro_mask, "charge"].to_numpy(float)
    lig_xyz = lig_df.loc[lig_mask, ["x", "y", "z"]].to_numpy(float)
    lig_q = lig_df.loc[lig_mask, "charge"].to_numpy(float)

    if pro_xyz.shape[0] == 0 or lig_xyz.shape[0] == 0:
        return None, None, None

    pts = np.vstack([pro_xyz, lig_xyz])
    values = np.concatenate([pro_q, lig_q]).astype(float)

    pt_group = np.zeros(pts.shape[0], dtype=int)
    pt_group[pro_xyz.shape[0]:] = 1
    return pts, values, pt_group


# ----------------------- Core computation per pdbid -----------------------
def compute_block_features_for_pdbid(
    pdbid,
    schema,
    block,
    base_dir,
    protein_pqr_dir,
    protein_pqr_suffix,
    pro_elements,
    lig_elements,
    categories_p,
    interface_cutoff,
    bins,
    eig_tol,
    block_size,
    compute_vr0=True,
    compute_alpha1=True,
    save_spectra=False,
    spectra_root=None,
):
    """
    Returns:
      vr_stats: (nBins, nPairs, 9) or None
      al_stats: (nBins, nPairs, 9) or None
    """
    if block != "P-L":
        raise ValueError(f"Only P-L is supported. Got block={block}")

    # protein and ligand with charges (protein now also includes resname via readin.py)
    pro_df = pqr_to_pqre(
        pdbid,
        suffix=protein_pqr_suffix,
        base_dir=protein_pqr_dir,
        keep_hetatm=False,
        drop_hydrogens=True,
    )
    lig_df = mol2_to_pqre(pdbid, "ligand", base_dir=base_dir)

    bins = [float(b) for b in bins]
    nB = len(bins)

    # enumerate pairs like peer pipeline (P-L only)
    pair_keys = []
    if schema == "ES":
        for pe in pro_elements:
            for le in lig_elements:
                pair_keys.append(("pe", pe, "le", le))
    elif schema == "CS":
        for ci, _cat in enumerate(categories_p):
            for le in lig_elements:
                pair_keys.append(("ci", ci, "le", le))
    else:
        raise ValueError(schema)

    nPairs = len(pair_keys)
    vr_stats = np.zeros((nB, nPairs, 9), float) if compute_vr0 else None
    al_stats = np.zeros((nB, nPairs, 9), float) if compute_alpha1 else None

    # optional spectra dumps
    vr_lines = {b: [] for b in bins} if (save_spectra and compute_vr0) else None
    al_lines = {b: [] for b in bins} if (save_spectra and compute_alpha1) else None

    max_bin = float(max(bins)) if bins else float(interface_cutoff)

    for j, key in enumerate(pair_keys):
        label = None

        if schema == "ES":
            pe = key[1]
            le = key[3]
            pts, values, pt_group = build_points_for_block_ES("P-L", pro_df, lig_df, pe, le)
            label = f"{pe}-{le}"
        else:  # CS
            ci = key[1]
            le = key[3]
            pts, values, pt_group = build_points_for_block_CS("P-L", pro_df, lig_df, categories_p[ci], le)
            label = f"cat{ci}-{le}"

        if pts is None or values is None or pts.shape[0] < 2:
            continue

        # interface crop (your choice): filter protein only, keep all ligand
        Pn = int(np.sum(pt_group == 0))
        Ln = int(np.sum(pt_group == 1))
        if Pn == 0 or Ln == 0:
            continue

        pro_pts = pts[:Pn]
        lig_pts = pts[Pn:]
        keepP, _keepL = symmetric_interface_crop_masks(
            pro_pts, lig_pts, cutoff=float(interface_cutoff), block_size=int(block_size)
        )
        if not np.any(keepP):
            continue

        pro_pts = pro_pts[keepP]
        pro_vals = values[:Pn][keepP]
        lig_vals = values[Pn:]

        pts = np.vstack([pro_pts, lig_pts])
        values = np.concatenate([pro_vals, lig_vals]).astype(float)
        pt_group = np.zeros(pts.shape[0], dtype=int)
        pt_group[pro_pts.shape[0]:] = 1

        num_pts = pts.shape[0]
        if num_pts < 2:
            continue

        stR = None
        if compute_vr0:
            rips = gd.RipsComplex(points=pts.tolist(), max_edge_length=float(max_bin) + 1e-6)
            stR = rips.create_simplex_tree(max_dimension=1)
            stR.make_filtration_non_decreasing()

        stA = None
        if compute_alpha1 and num_pts >= 3:
            ac = gd.AlphaComplex(points=pts.tolist())
            stA = ac.create_simplex_tree()
            stA.make_filtration_non_decreasing()

        for ib, cut in enumerate(bins):
            if compute_vr0 and stR is not None:
                L0 = compute_manual_laplacian_L0(stR, float(cut), values, num_pts, pt_group)
                if L0 is not None:
                    eigs0 = np.linalg.eigvalsh(L0)
                    vr_stats[ib, j, :] = eig_stats(eigs0, eig_tol=eig_tol)
                    if save_spectra:
                        eig_str = " ".join(fmt_float(x, 16) for x in eigs0.tolist())
                        vr_lines[float(cut)].append(f"{label}\t{eig_str}")

            if compute_alpha1 and stA is not None:
                L1 = compute_alpha_L1_matrix(
                    stA=stA,
                    cutoff_sq=(float(cut) * 0.5) ** 2,
                    values=values,
                    pts=pts,
                    num_pts=num_pts,
                    pt_group=pt_group,
                )
                if L1 is not None:
                    eigs1 = np.linalg.eigvalsh(L1)
                    al_stats[ib, j, :] = eig_stats(eigs1, eig_tol=eig_tol)
                    if save_spectra:
                        eig_str = " ".join(fmt_float(x, 16) for x in eigs1.tolist())
                        al_lines[float(cut)].append(f"{label}\t{eig_str}")

    # spectra writeout per pdbid
    if save_spectra and spectra_root:
        os.makedirs(spectra_root, exist_ok=True)

        if vr_lines is not None:
            for b in bins:
                path = os.path.join(spectra_root, f"vr0_{schema}_P-L_bin{fmt_bin(b)}.txt")
                with open(path, "w") as f:
                    f.write(f"# pdbid={pdbid} schema={schema} block=P-L bin={fmt_bin(b)} dim=vr0\n")
                    f.write("pair\teigs\n")
                    for line in vr_lines[float(b)]:
                        f.write(line + "\n")

        if al_lines is not None:
            for b in bins:
                path = os.path.join(spectra_root, f"alpha1_{schema}_P-L_bin{fmt_bin(b)}.txt")
                with open(path, "w") as f:
                    f.write(f"# pdbid={pdbid} schema={schema} block=P-L bin={fmt_bin(b)} dim=alpha1\n")
                    f.write("pair\teigs\n")
                    for line in al_lines[float(b)]:
                        f.write(line + "\n")

    return vr_stats, al_stats


def main():
    parser = argparse.ArgumentParser(description="PSL feature generation (VR0 + Alpha1) for P-L only (ES/CS)")

    parser.add_argument("--base_dir", default="/%%%%%%%%%%%%%%%/PDBbind_datasets/v2007")
    parser.add_argument("--protein_pqr_dir", default="/%%%%%%%%%%%%%%%")
    parser.add_argument("--protein_pqr_suffix", default="protein")

    parser.add_argument("--schema", default="ES", choices=["ES", "CS", "BOTH"])
    parser.add_argument("--blocks", default="P-L", help="Comma-separated. Only P-L is supported here.")

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Writes <out_dir>/<pdbid>/<pdbid>_features.npy (and spectra in <out_dir>/<pdbid>/spectra/ if enabled).",
    )

    parser.add_argument("--pdbid", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)

    parser.add_argument("--interface_cutoff", type=float, default=12.0)

    parser.add_argument("--bins", default="3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12")
    parser.add_argument("--eig_tol", type=float, default=1e-8)
    parser.add_argument("--block_size", type=int, default=256)

    parser.add_argument("--pro_elements", default="C,N,O,S")
    parser.add_argument("--lig_elements", default="C,N,O,F,P,S,Cl,Br,I")

    parser.add_argument("--no_vr0", action="store_true")
    parser.add_argument("--no_alpha1", action="store_true")

    parser.add_argument("--save_spectra", action="store_true")

    args = parser.parse_args()

    pro_elements = [x.strip() for x in args.pro_elements.split(",") if x.strip()]
    lig_elements = [x.strip() for x in args.lig_elements.split(",") if x.strip()]
    bins = [float(x.strip()) for x in args.bins.split(",") if x.strip()]
    blocks = [x.strip() for x in args.blocks.split(",") if x.strip()]

    # enforce P-L only
    blocks = [b for b in blocks if b == "P-L"]
    if not blocks:
        raise SystemExit("ERROR: Only block P-L is supported in this PSL.py")

    compute_vr0 = not args.no_vr0
    compute_alpha1 = not args.no_alpha1

    # ---- Choose pdbids ----
    if args.all:
        pdbids = list_pdbids_from_base_dir(args.base_dir)
        s = 0 if args.start is None else int(args.start)
        e = len(pdbids) if args.end is None else int(args.end)
        pdbids = pdbids[s:e]
    else:
        if not args.pdbid:
            raise SystemExit("ERROR: provide --pdbid or use --all with --start/--end")
        pdbids = [args.pdbid]

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Schemas to run ----
    schemas = ["ES", "CS"] if args.schema == "BOTH" else [args.schema]

    # ---- Run per pdbid ----
    for i, pdbid in enumerate(pdbids):
        print(f"[{i+1}/{len(pdbids)}] {pdbid}")

        out_pdb_dir = os.path.join(args.out_dir, pdbid)
        os.makedirs(out_pdb_dir, exist_ok=True)

        # spectra_root is exactly: <out_dir>/<pdbid>/spectra
        spectra_root = None
        if args.save_spectra:
            spectra_root = os.path.join(out_pdb_dir, "spectra")
            os.makedirs(spectra_root, exist_ok=True)

        feat_parts = []

        for schema in schemas:
            for block in blocks:
                try:
                    vr_stats, al_stats = compute_block_features_for_pdbid(
                        pdbid=pdbid,
                        schema=schema,
                        block=block,
                        base_dir=args.base_dir,
                        protein_pqr_dir=args.protein_pqr_dir,
                        protein_pqr_suffix=args.protein_pqr_suffix,
                        pro_elements=pro_elements,
                        lig_elements=lig_elements,
                        categories_p=CATEGORIES_P,
                        interface_cutoff=float(args.interface_cutoff),
                        bins=bins,
                        eig_tol=float(args.eig_tol),
                        block_size=int(args.block_size),
                        compute_vr0=compute_vr0,
                        compute_alpha1=compute_alpha1,
                        save_spectra=bool(args.save_spectra),
                        spectra_root=spectra_root,
                    )
                except Exception as e:
                    print("  FAILED block:", schema, block, "err:", repr(e))
                    feat_parts = None
                    break

                if compute_vr0 and vr_stats is not None:
                    feat_parts.append(np.asarray(vr_stats, dtype=float).reshape(-1))
                if compute_alpha1 and al_stats is not None:
                    feat_parts.append(np.asarray(al_stats, dtype=float).reshape(-1))

            if feat_parts is None:
                break

        if feat_parts is None or len(feat_parts) == 0:
            print("  SKIP (no features written)")
            continue

        feat = np.concatenate(feat_parts, axis=0).astype(float)

        feat_path = os.path.join(out_pdb_dir, f"{pdbid}_features.npy")
        np.save(feat_path, feat)
        print("  wrote:", feat_path, "dim=", feat.size)

    print("DONE")


if __name__ == "__main__":
    main()
