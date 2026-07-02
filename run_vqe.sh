#!/bin/bash
#SBATCH --job-name=vqe_hybrid
#SBATCH --partition=medium-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mgpfs/home/nhandayani/vqe_hybrid/error/slurm_%j.out
#SBATCH --error=/mgpfs/home/nhandayani/vqe_hybrid/error/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nhandayani@brin.go.id

# ============================================================
# SLURM Job Script — VQE Hybrid Algorithm
# Kluster HPC BRIN (login2.brin.go.id)
# User: nhandayani
#
# Struktur output:
#   vqe_hybrid/
#   ├── output/   ← hasil JSON & CSV energi VQE
#   ├── plot/     ← gambar PNG & PDF kurva PES
#   └── error/    ← log SLURM stdout & stderr
#
# Cara submit:
#   sbatch run_hpc.sh
#
# Override parameter via argumen:
#   sbatch --export=ALL,MOL=CH4,ANSATZ=UCCSD,ENC=JW,OPT=COBYLA run_hpc.sh
# ============================================================

set -euo pipefail

# ── Direktori kerja & output ─────────────────────────────────
WORKDIR="/mgpfs/home/nhandayani/vqe_hybrid"
OUTPUT_DIR="$WORKDIR/output"
PLOT_DIR="$WORKDIR/plot"
ERROR_DIR="$WORKDIR/error"

cd "$WORKDIR"
mkdir -p "$OUTPUT_DIR" "$PLOT_DIR" "$ERROR_DIR"

# ── Info awal ────────────────────────────────────────────────
echo "========================================================"
echo "  VQE Hybrid Algorithm — HPC BRIN"
echo "  User      : nhandayani"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Node      : $SLURMD_NODENAME"
echo "  CPUs      : $SLURM_CPUS_PER_TASK"
echo "  Mem       : $SLURM_MEM_PER_NODE MB"
echo "  Mulai     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output    : $OUTPUT_DIR"
echo "  Plot      : $PLOT_DIR"
echo "  Log/Error : $ERROR_DIR"
echo "========================================================"

# ── Aktivasi conda environment mycenvn ───────────────────────
source /mgpfs/shared/apps/app/Anaconda/3-2023.9-0/etc/profile.d/conda.sh
conda activate mycenvn

# ── Verifikasi environment ────────────────────────────────────
echo ""
echo "[ENV] Python  : $(python --version)"
echo "[ENV] Lokasi  : $(which python)"
echo "[ENV] Conda   : $(conda info --envs | grep '*')"

python -c "import pyscf;         print('[ENV] PySCF       :', pyscf.__version__)"
python -c "import openfermion;   print('[ENV] OpenFermion :', openfermion.__version__)"
python -c "import qiskit;        print('[ENV] Qiskit      :', qiskit.__version__)"
python -c "import qiskit_nature; print('[ENV] Qiskit-Nat  :', qiskit_nature.__version__)"

# ── Set variabel lingkungan untuk performa ────────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE

# Teruskan path folder ke Python via env var
export VQE_OUTPUT_DIR="$OUTPUT_DIR"
export VQE_PLOT_DIR="$PLOT_DIR"

# ── Override config dari argumen sbatch (opsional) ────────────
if [ -n "${MOL:-}" ]; then
    echo "[CONFIG] Override MOLECULE = $MOL"
    sed -i "s/^MOLECULE = .*/MOLECULE = \"$MOL\"/" config.py
fi
if [ -n "${ANSATZ:-}" ]; then
    echo "[CONFIG] Override ANSATZ_TYPE = $ANSATZ"
    sed -i "s/^ANSATZ_TYPE = .*/ANSATZ_TYPE = \"$ANSATZ\"/" config.py
fi
if [ -n "${ENC:-}" ]; then
    echo "[CONFIG] Override ENCODING = $ENC"
    sed -i "s/^ENCODING = .*/ENCODING = \"$ENC\"/" config.py
fi
if [ -n "${OPT:-}" ]; then
    echo "[CONFIG] Override OPTIMIZER_TYPE = $OPT"
    sed -i "s/^OPTIMIZER_TYPE = .*/OPTIMIZER_TYPE = \"$OPT\"/" config.py
fi

# ── Tampilkan konfigurasi saat ini ───────────────────────────
echo ""
echo "[CONFIG] Konfigurasi aktif:"
grep -E "^(MOLECULE|ANSATZ_TYPE|ENCODING|OPTIMIZER_TYPE|ACTIVE_SPACE_METHOD|N_ACTIVE_ELECTRONS|N_ACTIVE_ORBITALS|BASIS)" config.py
echo ""

# ── Jalankan PES scan (VQE utama) ────────────────────────────
echo "[RUN] Memulai VQE PES scan ..."
python vqe_runner.py
echo "[RUN] VQE selesai."
echo "[RUN] Cek output: ls $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"

# ── Buat plot ─────────────────────────────────────────────────
echo ""
echo "[PLOT] Membuat plot PES dan error ..."
python plot_results.py
echo "[PLOT] Selesai."
echo "[PLOT] Cek plot: ls $PLOT_DIR"
ls -lh "$PLOT_DIR"

# ── Info selesai ──────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Selesai  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output   : $OUTPUT_DIR"
echo "  Plot     : $PLOT_DIR"
echo "  Log      : $ERROR_DIR/slurm_${SLURM_JOB_ID}.out"
echo "========================================================"
