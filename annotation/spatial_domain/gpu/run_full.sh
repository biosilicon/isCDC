#!/usr/bin/env bash
# One entry point for the isolated, resumable full SpatialGLUE run.
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
conda_script="${ISCDC_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
glue_python="${ISCDC_GLUE_PYTHON:-$project_root/temp/environments/iscdc-spatial-domain-gpu/bin/python}"
if [[ ! -f "$conda_script" || ! -x "$glue_python" ]]; then
  printf '%s\n' 'Missing Conda setup or locked GPU Python. Set ISCDC_CONDA_SH / ISCDC_GLUE_PYTHON.' >&2
  exit 1
fi
source "$conda_script"
conda activate iscdc
source "$(dirname -- "$glue_python")/activate"
cd -- "$project_root"
export PYTHONPATH="$project_root/src" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
exec "$glue_python" -u -m iscdc.spatialglue_full "$@"
