"""
config.py
=========
Konfigurasi terpusat untuk program VQE Hybrid Algorithm.
Semua pilihan parameter diatur di sini dan diimpor oleh modul lain.
"""

# ─────────────────────────────────────────────
# 1. PILIHAN MOLEKUL
# ─────────────────────────────────────────────
# Opsi: "H2" | "H2O" | "CH4"
MOLECULE = "H2O"

# Opsi: True | False
USE_OPTIMIZED_GEOMETRY = False

# ─────────────────────────────────────────────
# 2. ACTIVE SPACE
# ─────────────────────────────────────────────
# Metode klasik untuk bangun Hamiltonian
# Opsi: "CASCI" | "CASSCF"
ACTIVE_SPACE_METHOD = "CASCI"

# Jumlah elektron aktif dan orbital aktif
# H2   → (2e, 2o)  | H2O → (4e, 4o) atau (8e, 6o) | CH4 → (8e, 8o)
N_ACTIVE_ELECTRONS = 4
N_ACTIVE_ORBITALS  = 4

# Jumlah orbital frozen core (diabaikan dari active space)
# H2 → 0 | H2O → 1 | CH4 → 1
N_FROZEN_CORE = 1

# ─────────────────────────────────────────────
# 3. ENCODING FERMION-TO-QUBIT
# ─────────────────────────────────────────────
# Opsi: "JW" (Jordan-Wigner) | "PARITY" | "BK" (Bravyi-Kitaev)
ENCODING = "JW"

# ─────────────────────────────────────────────
# 4. ANSATZ
# ─────────────────────────────────────────────
# Opsi: "UCCSD" | "kUpCCGSD" | "ADAPT-VQE"
ANSATZ_TYPE = "ADAPT-VQE"

# Parameter khusus untuk kUpCCGSD (nilai k, jumlah layer)
K_UPCCGSD = 1

# Parameter ADAPT-VQE
ADAPT_MAX_ITER    = 50       # iterasi maksimum ADAPT
ADAPT_GRAD_TOL    = 1e-5     # toleransi gradien untuk konvergensi
ADAPT_POOL        = "UCCSD"  # operator pool: "UCCSD" | "GSD" | "QUBIT"
ADAPT_PLATEAU_N   = 5        # stop jika energi stagnan N iterasi berturut-turut
ADAPT_PLATEAU_TOL = 1e-8     # threshold plateau energi (Ha)

# ─────────────────────────────────────────────
# 5. OPTIMIZER
# ─────────────────────────────────────────────
# Opsi: "BFGS" | "L-BFGS-B" | "COBYLA" | "RMSProp" | "NFT"
OPTIMIZER_TYPE = "L-BFGS-B"

# Toleransi konvergensi optimizer
OPT_TOL      = 1e-6
OPT_MAX_ITER = 1000

# Hyperparameter RMSProp (hanya dipakai jika OPTIMIZER_TYPE = "RMSProp")
RMSPROP_LR      = 0.01
RMSPROP_DECAY   = 0.9
RMSPROP_EPSILON = 1e-8

# ─────────────────────────────────────────────
# 6. SCAN POTENTIAL ENERGY SURFACE (PES)
# ─────────────────────────────────────────────
# Rentang panjang ikatan (Angstrom) untuk scan PES
BOND_LENGTHS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Pasangan atom yang panjang ikatannya divariasikan
# H2  → ("H1", "H2")
# H2O → ("O", "H1") — O-H bond length, sudut tetap
# CH4 → ("C", "H1") — C-H bond length
SCAN_ATOM_PAIR = ("O", "H1")

# ─────────────────────────────────────────────
# 7. REFERENSI ENERGI EKSAK (FCI)
# ─────────────────────────────────────────────
# Basis set yang digunakan untuk kalkulasi PySCF
BASIS = "sto-3g"

# Apakah menjalankan FCI referensi untuk perbandingan error
RUN_FCI_REFERENCE = True

# Chemical accuracy threshold (Hartree)
CHEMICAL_ACCURACY = 1.6e-3  # ~1 kcal/mol

# ─────────────────────────────────────────────
# 8. OUTPUT
# ─────────────────────────────────────────────
# Struktur folder output:
#   output/  ← hasil JSON & CSV energi VQE
#   plot/    ← gambar JPG & PDF kurva PES
#   error/   ← log SLURM stdout & stderr (diatur via #SBATCH)
#
# Path dibaca dari env var jika tersedia (diset oleh run_hpc.sh),
# fallback ke subfolder lokal jika dijalankan langsung/notebook.
import os as _os

RESULTS_DIR = _os.environ.get("VQE_OUTPUT_DIR", "output")
PLOTS_DIR   = _os.environ.get("VQE_PLOT_DIR",   "plot")

# Format gambar output
PLOT_FORMAT = "jpg"          # "jpg" | "png" | "pdf"
PLOT_DPI    = 300

# Nama file output (otomatis dilengkapi suffix molekul & metode)
def output_prefix():
    return f"{MOLECULE}_{ACTIVE_SPACE_METHOD}_CAS({N_ACTIVE_ELECTRONS},{N_ACTIVE_ORBITALS})_{ANSATZ_TYPE}_{ENCODING}_{OPTIMIZER_TYPE}"