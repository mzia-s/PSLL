import os
import pandas as pd


def pqr_to_pqre(
    pdbid,
    suffix="protein",
    base_dir="/mnt/home/ziamusha/PDBbind_datasets/v2007_AMBER_pqr",
    keep_hetatm=False,
    drop_hydrogens=True,
):
    """
    Read a PQR file and return a DataFrame with columns:
      x, y, z, element, charge, resname, chain, resseq

    Supported file layouts:
      full   -> <base_dir>/<pdbid>.pqr
      pocket -> <base_dir>/<pdbid>_pocket.pqr

    The file naming is decided automatically from base_dir.
    """

    base_dir_lower = base_dir.lower()

    if "pocket" in base_dir_lower:
        filename = os.path.join(base_dir, f"{pdbid}_pocket.pqr")
    else:
        filename = os.path.join(base_dir, f"{pdbid}.pqr")

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Could not find PQR file: {filename}")

    rows = []
    with open(filename, "r") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if (not keep_hetatm) and rec == "HETATM":
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                charge = float(parts[-2])
            except ValueError:
                continue

            resname = line[17:20].strip() if len(line) >= 20 else ""
            chain = line[21].strip() if len(line) >= 22 else ""
            resseq_raw = line[22:26].strip() if len(line) >= 26 else ""

            try:
                resseq = int(resseq_raw) if resseq_raw else None
            except ValueError:
                resseq = None

            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                atom_name = line[12:16].strip()
                if atom_name and len(atom_name) >= 2 and atom_name[1].islower():
                    element = atom_name[:2]
                else:
                    element = atom_name[:1] if atom_name else ""

            element = element.capitalize() if element else element

            if drop_hydrogens and element == "H":
                continue

            rows.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "element": element,
                    "charge": charge,
                    "resname": resname,
                    "chain": chain,
                    "resseq": resseq,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["x", "y", "z", "element", "charge", "resname", "chain", "resseq"],
    )


def mol2_to_pqre(pdbid, suffix, base_dir="/mnt/home/ziamusha/PDBbind_datasets/v2007"):
    """
    Read <base_dir>/<pdbid>/<pdbid>_<suffix>.mol2 and return:
      x, y, z, element, charge
    """
    filename = f"{base_dir}/{pdbid}/{pdbid}_{suffix}.mol2"

    with open(filename, "r") as file:
        lines = file.readlines()

    index_meta = lines.index("@<TRIPOS>MOLECULE\n") + 2
    meta = lines[index_meta].split()

    index_atoms_start = lines.index("@<TRIPOS>ATOM\n")
    index_atoms_end = lines.index("@<TRIPOS>BOND\n")
    filtered_lines = [line.split() for line in lines[index_atoms_start + 1:index_atoms_end]]

    if len(filtered_lines) != int(meta[0]):
        raise Exception("incorrect number of atoms")

    df = pd.DataFrame(filtered_lines)
    df = df.iloc[:, [2, 3, 4, 5, 8]]
    df.columns = ["x", "y", "z", "element", "charge"]
    df["element"] = df["element"].str.split(".").str[0]
    df[["x", "y", "z", "charge"]] = df[["x", "y", "z", "charge"]].apply(pd.to_numeric)

    return df