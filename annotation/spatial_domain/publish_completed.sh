#!/usr/bin/env bash
# Run from any working directory. Activates the website environment, not the inference runtime.
set -Eeo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_setup="${ISCDC_CONDA_SH:-/home1/shezixi/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$conda_setup" ]]; then
    printf 'Conda initialization not found: %s (set ISCDC_CONDA_SH)\n' "$conda_setup" >&2
    exit 1
fi
source "$conda_setup"
conda activate iscdc
set -u
cd -- "$project_root"
export PYTHONPATH="$project_root/src"
exec python -u -m iscdc.spatial_domain_publish "$@"
