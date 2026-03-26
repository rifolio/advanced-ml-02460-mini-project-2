#!/usr/bin/env bash
# Part B sweep: D decoders per VAE × M independent retrainings.
# Folder layout:  ${EXPERIMENTS_ROOT}/d<D>_r<MM>/model.pt + run_meta.json
# Figures:         ${IMAGE_OUTPUT_ROOT}/<figure_prefix>_e<epochs>_<kind>.png
#                  figure_prefix defaults to the experiment folder basename (e.g. d3_r07).
#
# Usage:
#   ./scripts/run_partb_experiments.sh train
#   ./scripts/run_partb_experiments.sh sample
#   ./scripts/run_partb_experiments.sh geodesics
#   DEVICE=mps EPOCHS_PER_DECODER=50 NUM_RERUNS=3 ./scripts/run_partb_experiments.sh train
#
# Environment (all optional):
#   PYTHON              python executable (default: python3)
#   DEVICE              cpu | cuda | mps (default: cpu)
#   EPOCHS_PER_DECODER  training epochs per run (default: 150)
#   NUM_RERUNS          M — number of VAE retrainings per D (default: 10)
#   SEED_BASE           base for --training-seed (default: 0)
#   DECODER_SWEEP       space-separated D values (default: 1 2 3)
#   EXPERIMENTS_ROOT    root dir for model folders (default: experiments/partb)
#   IMAGE_OUTPUT_ROOT   PNG output root (default: report_images/partb)
#   GEODESIC_PAIRS_DIR  shared geodesic_pairs.pt location (default: ${EXPERIMENTS_ROOT}/_shared_pairs)
#   EXTRA_ARGS          extra args passed to every ensemble_vae.py invocation (quoted string)
#   SKIP_IF_EXISTS      if 1, skip train when model.pt already exists (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cpu}"
EPOCHS_PER_DECODER="${EPOCHS_PER_DECODER:-150}"
NUM_RERUNS="${NUM_RERUNS:-10}"
SEED_BASE="${SEED_BASE:-0}"
DECODER_SWEEP="${DECODER_SWEEP:-1 2 3}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-experiments/partb}"
IMAGE_OUTPUT_ROOT="${IMAGE_OUTPUT_ROOT:-report_images/partb}"
GEODESIC_PAIRS_DIR="${GEODESIC_PAIRS_DIR:-${EXPERIMENTS_ROOT}/_shared_pairs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

RUN_MODE="${1:-}"
if [[ -z "${RUN_MODE}" ]]; then
  echo "Usage: $0 {train|sample|geodesics}" >&2
  exit 1
fi
shift || true

case "${RUN_MODE}" in
  train|sample|geodesics) ;;
  *)
    echo "Unknown mode: ${RUN_MODE}. Use: train | sample | geodesics" >&2
    exit 1
    ;;
esac

mkdir -p "${EXPERIMENTS_ROOT}" "${IMAGE_OUTPUT_ROOT}" "${GEODESIC_PAIRS_DIR}"

for D in ${DECODER_SWEEP}; do
  for ((r = 0; r < NUM_RERUNS; r++)); do
    R="$(printf '%02d' "${r}")"
    EXP="${EXPERIMENTS_ROOT}/d${D}_r${R}"
    mkdir -p "${EXP}"

    case "${RUN_MODE}" in
      train)
        if [[ "${SKIP_IF_EXISTS}" == "1" ]] && [[ -f "${EXP}/model.pt" ]]; then
          echo "skip ${EXP} (model.pt exists)"
          continue
        fi
        # shellcheck disable=SC2086
        ${PYTHON} ensemble_vae.py train \
          --device "${DEVICE}" \
          --epochs-per-decoder "${EPOCHS_PER_DECODER}" \
          --experiment-folder "${EXP}" \
          --num-decoders "${D}" \
          --rerun-index "${r}" \
          --num-reruns "${NUM_RERUNS}" \
          --training-seed $((SEED_BASE + r * 100003)) \
          ${EXTRA_ARGS} \
          "$@"
        ;;
      sample)
        # shellcheck disable=SC2086
        ${PYTHON} ensemble_vae.py sample \
          --device "${DEVICE}" \
          --experiment-folder "${EXP}" \
          --image-output-dir "${IMAGE_OUTPUT_ROOT}" \
          ${EXTRA_ARGS} \
          "$@"
        ;;
      geodesics)
        # shellcheck disable=SC2086
        ${PYTHON} ensemble_vae.py geodesics \
          --device "${DEVICE}" \
          --experiment-folder "${EXP}" \
          --image-output-dir "${IMAGE_OUTPUT_ROOT}" \
          --geodesic-pairs-dir "${GEODESIC_PAIRS_DIR}" \
          ${EXTRA_ARGS} \
          "$@"
        ;;
    esac
  done
done

echo "Done: ${RUN_MODE} (D in {${DECODER_SWEEP}}, M=${NUM_RERUNS})"
