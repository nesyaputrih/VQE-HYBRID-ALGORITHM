"""
plot_results.py
===============
Plotting kurva energi potensial (PES) dan error VQE vs FCI.
Input: file JSON hasil dari vqe_runner.py (di folder results/).
Output: gambar PNG/PDF ke folder plots/.

Plot yang dihasilkan:
  1. PES: energi (Ha) vs panjang ikatan (Å)  — VQE, FCI, CASSCF/CASCI, HF
  2. Error: |E_VQE - E_FCI| (Ha) vs panjang ikatan, dengan garis chemical accuracy
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interaktif untuk HPC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

import config


# ──────────────────────────────────────────────────────────────────────────────
# Gaya plot (style jurnal: minimalis, hitam-putih kompatibel)
# ──────────────────────────────────────────────────────────────────────────────

STYLE = {
    "figure.dpi"       : 300,
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.linewidth"   : 1.2,
    "axes.labelsize"   : 12,
    "axes.titlesize"   : 12,
    "xtick.direction"  : "in",
    "ytick.direction"  : "in",
    "xtick.major.size" : 4,
    "ytick.major.size" : 4,
    "legend.frameon"   : True,
    "legend.fontsize"  : 10,
    "lines.linewidth"  : 1.8,
    "lines.markersize" : 5,
    "savefig.bbox"     : "tight",
    "savefig.pad_inches": 0.05,
}

COLOR = {
    "vqe"    : "#1f77b4",   # biru
    "fci"    : "#d62728",   # merah
    #"cas"    : "#2ca02c",   # hijau
    #"hf"     : "#9467bd",   # ungu
    "error"  : "#ff7f0e",   # oranye
    "chem"   : "#7f7f7f",   # abu
}


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

def load_results(json_path: str) -> list:
    """Load data JSON hasil PES scan."""
    with open(json_path) as f:
        data = json.load(f)
    # Filter titik tanpa error komputasi
    valid = [d for d in data if d.get("e_vqe") is not None]
    print(f"[Plot] Memuat {len(valid)} / {len(data)} titik valid dari {json_path}")
    return valid


def _arrays(data: list):
    """
    Ekstrak array numpy dari list result.

    Catatan penting soal `err`:
    Error VQE dihitung terhadap `e_fci_active` (FCI DI DALAM active space),
    bukan `e_fci` (FCI penuh dari PySCF). `e_fci_active` adalah batas atas
    yang sebenarnya bisa dicapai VQE (lihat vqe_runner.py: `quality_ref =
    e_fci_active`), sedangkan `e_fci` penuh mengandung korelasi di luar
    active space yang memang tidak bisa diperbaiki oleh ansatz VQE apa pun
    pada CAS ini. Memakai `e_fci` penuh untuk error akan mencampur error
    VQE asli dengan gap intrinsik CAS-vs-full-FCI (yang membesar di
    geometri disosiatif), sehingga error tampak jauh lebih besar dari
    yang sebenarnya bisa dikoreksi VQE.
    """
    r    = np.array([d["bond_length"]   for d in data])
    evqe = np.array([d["e_vqe"]         for d in data])
    efci = np.array([
        d.get("e_fci") if d.get("e_fci") is not None else np.nan
        for d in data
    ])
    efci_active = np.array([
        d.get("e_fci_active") if d.get("e_fci_active") is not None else np.nan
        for d in data
    ])
    ecas = np.array([
        d.get("e_cas") if d.get("e_cas") is not None else np.nan
        for d in data
    ])
    ehf  = np.array([
        d.get("e_hf") if d.get("e_hf") is not None else np.nan
        for d in data
    ])
    # Error sekarang terhadap FCI DALAM active space (target sesungguhnya VQE)
    err  = np.abs(evqe - efci_active)
    return r, evqe, efci, efci_active, ecas, ehf, err


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: Potential Energy Surface
# ──────────────────────────────────────────────────────────────────────────────

def plot_pes(data: list, save_path: str, prefix: str = ""):
    """
    Plot PES: E (Ha) vs r (Å).
    Garis: VQE, FCI (referensi), CAS (CASCI/CASSCF), HF.
    """
    r, evqe, efci, efci_active, ecas, ehf, err = _arrays(data)
    cfg = data[0].get("config", {})

    method   = cfg.get("method",   config.ACTIVE_SPACE_METHOD)
    ansatz   = cfg.get("ansatz",   config.ANSATZ_TYPE)
    enc      = cfg.get("encoding", config.ENCODING)
    opt      = cfg.get("optimizer",config.OPTIMIZER_TYPE)
    molecule = cfg.get("molecule", config.MOLECULE)
    n_ae     = cfg.get("n_active_elec", config.N_ACTIVE_ELECTRONS)
    n_ao     = cfg.get("n_active_orbs", config.N_ACTIVE_ORBITALS)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        # ── Kurva energi ──────────────────────────────────────────────────────
        # PES sekarang pakai e_fci_active (FCI DALAM active space) sebagai
        # kurva referensi, bukan e_fci penuh — ini target yang benar2 relevan
        # untuk mengevaluasi kualitas ansatz VQE pada CAS ini.
        ax.plot(r, evqe, "o-",  color=COLOR["vqe"], label=f"VQE ({ansatz}/{enc}/{opt})")
        ax.plot(r, efci_active, "s--", color=COLOR["fci"], label="FCI (dalam active space)")
        #ax.plot(r, ecas, "^-.", color=COLOR["cas"], label=f"{method} CAS({n_ae},{n_ao})")
        #ax.plot(r, ehf,  "v:",  color=COLOR["hf"],  label="Hartree-Fock")

        ax.set_xlabel(r"Panjang Ikatan ($\AA$)")
        ax.set_ylabel(r"Energi (Ha)")
        title = (f"Potential Energy Surface — {molecule}\n"
                 f"{method}/CAS({n_ae},{n_ao}) → VQE-{ansatz} [{enc}]")
        ax.set_title(title, pad=8)
        ax.legend(loc="upper right")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", top=True, right=True)

        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True)
        fig.savefig(save_path, dpi=getattr(config, "PLOT_DPI", 300), pil_kwargs={"quality": 95} if getattr(config, "PLOT_FORMAT", "jpg") == "jpg" else {})
        plt.close(fig)
        print(f"[Plot] PES tersimpan: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: Error VQE vs FCI
# ──────────────────────────────────────────────────────────────────────────────

def plot_error(data: list, save_path: str, prefix: str = ""):
    """
    Plot error |E_VQE - E_FCI| (Ha) vs r (Å).
    Termasuk garis chemical accuracy (1.6 mHa).
    """
    r, evqe, efci, efci_active, ecas, ehf, err = _arrays(data)
    cfg = data[0].get("config", {})

    method   = cfg.get("method",   config.ACTIVE_SPACE_METHOD)
    ansatz   = cfg.get("ansatz",   config.ANSATZ_TYPE)
    enc      = cfg.get("encoding", config.ENCODING)
    opt      = cfg.get("optimizer",config.OPTIMIZER_TYPE)
    molecule = cfg.get("molecule", config.MOLECULE)
    n_ae     = cfg.get("n_active_elec", config.N_ACTIVE_ELECTRONS)
    n_ao     = cfg.get("n_active_orbs", config.N_ACTIVE_ORBITALS)

    chem_acc = config.CHEMICAL_ACCURACY

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.0))

        # err = |E_VQE - E_FCI_active|, lihat catatan di _arrays()
        ax.semilogy(r, err, "o-", color=COLOR["error"],
                    label=f"|E_VQE − E_FCI(active space)|")

        ax.axhline(chem_acc, ls="--", color=COLOR["chem"], lw=1.4,
                   label=f"Chemical accuracy ({chem_acc*1e3:.1f} mHa ≈ 1 kcal/mol)")

        # Arsir wilayah di bawah chemical accuracy
        ax.fill_between(r, 0, chem_acc, alpha=0.08, color=COLOR["chem"])

        ax.set_xlabel(r"Panjang Ikatan ($\AA$)")
        ax.set_ylabel(r"$|E_\mathrm{VQE} - E_\mathrm{FCI,\,active}|$ (Ha)")
        title = (f"Error Energi VQE terhadap FCI (active space) — {molecule}\n"
                 f"{method}/CAS({n_ae},{n_ao}) → VQE-{ansatz} [{enc}/{opt}]")
        ax.set_title(title, pad=8)
        ax.legend(loc="upper right")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", top=True, right=True)
        ax.set_ylim(bottom=max(err[~np.isnan(err)].min() * 0.5, 1e-8))

        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True)
        fig.savefig(save_path, dpi=getattr(config, "PLOT_DPI", 300), pil_kwargs={"quality": 95} if getattr(config, "PLOT_FORMAT", "jpg") == "jpg" else {})
        plt.close(fig)
        print(f"[Plot] Error plot tersimpan: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: Optimization convergence (opsional)
# ──────────────────────────────────────────────────────────────────────────────

def plot_convergence(data: list, bond_length: float, save_path: str):
    """
    Plot konvergensi optimizer (energi vs evaluasi fungsi) untuk satu titik r.
    """
    # Cari titik dengan bond_length paling dekat
    diffs = [abs(d["bond_length"] - bond_length) for d in data]
    idx   = int(np.argmin(diffs))
    d     = data[idx]
    hist  = d.get("opt_history", [])

    if not hist:
        print(f"[Plot] Tidak ada riwayat optimasi untuk r={bond_length:.3f} Å")
        return

    iters  = [h[0] for h in hist]
    energi = [h[1] for h in hist]

    cfg      = d.get("config", {})
    molecule = cfg.get("molecule", config.MOLECULE)
    opt      = cfg.get("optimizer", config.OPTIMIZER_TYPE)
    ansatz   = cfg.get("ansatz",    config.ANSATZ_TYPE)
    efci_active = d.get("e_fci_active")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))

        ax.plot(iters, energi, "-", color=COLOR["vqe"], lw=1.5,
                label=f"VQE ({opt})")
        if efci_active is not None:
            ax.axhline(efci_active, ls="--", color=COLOR["fci"], lw=1.2,
                       label="E_FCI (active space)")

        ax.set_xlabel("Evaluasi Fungsi")
        ax.set_ylabel("Energi (Ha)")
        ax.set_title(f"Konvergensi Optimizer — {molecule}  r = {d['bond_length']:.3f} Å\n"
                     f"Ansatz: {ansatz}  |  Optimizer: {opt}")
        ax.legend()
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        plt.tight_layout()
        fig.savefig(save_path, dpi=getattr(config, "PLOT_DPI", 300), pil_kwargs={"quality": 95} if getattr(config, "PLOT_FORMAT", "jpg") == "jpg" else {})
        plt.close(fig)
        print(f"[Plot] Konvergensi tersimpan: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main(json_path: str = None):
    """
    Baca file JSON PES, buat semua plot, simpan ke folder plot/.

    Parameters
    ----------
    json_path : path ke file JSON hasil PES scan.
                Jika None, otomatis dicari di output/ berdasarkan config.
    """
    if json_path is None:
        prefix    = config.output_prefix()
        json_path = Path(config.RESULTS_DIR) / f"{prefix}_pes.json"

    json_path = str(json_path)
    if not os.path.exists(json_path):
        print(f"[Plot] File tidak ditemukan: {json_path}")
        sys.exit(1)

    data   = load_results(json_path)

    if not data:
        print(f"[Plot] Tidak ada data valid di {json_path} — semua titik error.")
        print("[Plot] Periksa output VQE terlebih dahulu.")
        sys.exit(1)

    prefix = config.output_prefix()
    plots  = Path(config.PLOTS_DIR)
    plots.mkdir(parents=True, exist_ok=True)

    fmt = getattr(config, "PLOT_FORMAT", "jpg")
    dpi = getattr(config, "PLOT_DPI", 300)

    # ── Plot PES ─────────────────────────────────────────────────────────────
    plot_pes(data,
             save_path=str(plots / f"{prefix}_pes.{fmt}"),
             prefix=prefix)

    # ── Plot Error ────────────────────────────────────────────────────────────
    plot_error(data,
               save_path=str(plots / f"{prefix}_error.{fmt}"),
               prefix=prefix)

    # ── Plot konvergensi untuk titik ekuilibrium ──────────────────────────────
    r_eq = {"H2": 0.74, "H2O": 0.96, "CH4": 1.089}.get(config.MOLECULE, 1.0)
    plot_convergence(data, r_eq,
                     save_path=str(plots / f"{prefix}_convergence_req.{fmt}"))

    # Simpan juga versi PDF untuk publikasi
    plot_pes(data,
             save_path=str(plots / f"{prefix}_pes.pdf"),
             prefix=prefix)
    plot_error(data,
               save_path=str(plots / f"{prefix}_error.pdf"),
               prefix=prefix)

    print(f"\n[Plot] Semua gambar tersimpan di: {plots}/")
    print(f"  Format  : {fmt.upper()} + PDF")
    print(f"  DPI     : {dpi}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot hasil PES VQE")
    parser.add_argument("--json", type=str, default=None,
                        help="Path ke file JSON hasil (opsional)")
    args = parser.parse_args()
    main(args.json)