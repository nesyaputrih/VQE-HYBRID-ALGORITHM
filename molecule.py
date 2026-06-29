"""
molecule.py
===========
Definisi geometri molekul dan optimasi geometri menggunakan PySCF.
Mendukung H2, H2O, dan CH4 dengan scan panjang ikatan untuk PES.
"""

import numpy as np
from pyscf import gto, scf, geomopt
import config


# ─────────────────────────────────────────────
# Geometri default setiap molekul (Angstrom)
# ─────────────────────────────────────────────

def _h2_geometry(r_hh: float) -> str:
    """H2: dua atom H pada sumbu z, jarak r_hh."""
    return f"""
    H1  0.0  0.0  0.0
    H2  0.0  0.0  {r_hh}
    """


def _h2o_geometry(r_oh: float, angle_deg: float = 104.5) -> str:
    """
    H2O: O di pusat, dua H dengan sudut HOH.
    Sudut tetap pada 104.5° (eksperimental); hanya r_OH yang divariasikan.
    """
    angle_rad = np.deg2rad(angle_deg / 2)
    hx = r_oh * np.sin(angle_rad)
    hz = r_oh * np.cos(angle_rad)
    return f"""
    O   0.0   0.0   0.0
    H1  {hx:.6f}  0.0   {hz:.6f}
    H2  {-hx:.6f}  0.0   {hz:.6f}
    """


def _ch4_geometry(r_ch: float) -> str:
    """
    CH4: geometri tetrahedral menggunakan koordinat Kartesian.
    Empat H pada sudut tetrahedral simetris di sekitar C.
    """
    d = r_ch / np.sqrt(3)
    return f"""
    C   0.0       0.0       0.0
    H1  {d:.6f}   {d:.6f}   {d:.6f}
    H2  {-d:.6f}  {-d:.6f}  {d:.6f}
    H3  {-d:.6f}  {d:.6f}   {-d:.6f}
    H4  {d:.6f}   {-d:.6f}  {-d:.6f}
    """


# ─────────────────────────────────────────────
# Builder molekul PySCF
# ─────────────────────────────────────────────

def build_mol(molecule: str, bond_length: float, basis: str = None) -> gto.Mole:
    """
    Bangun objek pyscf.gto.Mole untuk molekul dan panjang ikatan tertentu.

    Parameters
    ----------
    molecule    : "H2" | "H2O" | "CH4"
    bond_length : panjang ikatan (Angstrom) untuk atom pasangan yang discan
    basis       : basis set (default dari config.py)

    Returns
    -------
    mol : pyscf.gto.Mole yang sudah di-build
    """
    if basis is None:
        basis = config.BASIS

    mol = gto.Mole()
    mol.basis = basis
    mol.unit  = "Angstrom"
    mol.spin  = 0       # semua molekul closed-shell
    mol.verbose = 0     # silent

    if molecule == "H2":
        mol.atom  = _h2_geometry(bond_length)
        mol.charge = 0

    elif molecule == "H2O":
        mol.atom  = _h2o_geometry(bond_length)
        mol.charge = 0

    elif molecule == "CH4":
        mol.atom  = _ch4_geometry(bond_length)
        mol.charge = 0

    else:
        raise ValueError(f"Molekul '{molecule}' tidak dikenali. Pilih: H2 | H2O | CH4")

    mol.build()
    return mol

'''
# ─────────────────────────────────────────────
# Optimasi geometri (opsional)
# ─────────────────────────────────────────────

def optimize_geometry(molecule: str, basis: str = None) -> gto.Mole:
    """
    Lakukan optimasi geometri pada level RHF menggunakan PySCF + geomopt.
    Mengembalikan objek Mole dengan geometri teroptimasi.

    Parameters
    ----------
    molecule : "H2" | "H2O" | "CH4"
    basis    : basis set

    Returns
    -------
    mol_opt : Mole dengan geometri ekuilibrium
    """
    if basis is None:
        basis = config.BASIS

    # Gunakan panjang ikatan literatur sebagai tebakan awal
    r0 = {"H2": 0.74, "H2O": 0.96, "CH4": 1.089}
    mol = build_mol(molecule, r0[molecule], basis)

    mf  = scf.RHF(mol)
    mf.kernel()

    mol_opt = geomopt.optimize(mf)
    mol_opt.verbose = 0

    print(f"[GeomOpt] {molecule} ekuilibrium (basis={basis}):")
    print(mol_opt.atom_coords(unit="Angstrom"))
    return mol_opt
'''

# ─────────────────────────────────────────────
# Scan geometri untuk PES
# ─────────────────────────────────────────────

def scan_bond_lengths(
    molecule: str,
    bond_lengths: list = None,
    basis: str = None
) -> list:
    """
    Kembalikan list objek Mole untuk setiap panjang ikatan dalam scan PES.

    Parameters
    ----------
    molecule     : "H2" | "H2O" | "CH4"
    bond_lengths : list panjang ikatan (Angstrom). Default dari config.py.
    basis        : basis set

    Returns
    -------
    mols : list of (r, mol) tuple
    """
    if bond_lengths is None:
        bond_lengths = config.BOND_LENGTHS
    if basis is None:
        basis = config.BASIS

    mols = []
    for r in bond_lengths:
        mol = build_mol(molecule, r, basis)
        mols.append((r, mol))
        print(f"  [{molecule}] r = {r:.3f} Å  →  {mol.nao_nr()} AO, {mol.nelectron} elektron")

    return mols


# ─────────────────────────────────────────────
# Utilitas: info singkat molekul
# ─────────────────────────────────────────────

def molecule_info(mol: gto.Mole) -> dict:
    """Kembalikan dict ringkasan properti Mole."""
    return {
        "n_electrons" : mol.nelectron,
        "n_ao"        : mol.nao_nr(),
        "charge"      : mol.charge,
        "spin"        : mol.spin,
        "basis"       : mol.basis,
        "atom"        : mol.atom,
    }


# ─────────────────────────────────────────────
# Test cepat
# ─────────────────────────────────────────────
if __name__ == "__main__":
    for mol_name in ["H2", "H2O", "CH4"]:
        print(f"\n{'='*40}")
        print(f"  Molekul: {mol_name}")
        r0 = {"H2": 0.74, "H2O": 0.96, "CH4": 1.089}[mol_name]
        mol = build_mol(mol_name, r0)
        info = molecule_info(mol)
        for k, v in info.items():
            print(f"  {k:15s}: {v}")
