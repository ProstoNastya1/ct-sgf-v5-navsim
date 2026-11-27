#!/usr/bin/env bash
set -euo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
navhard_root=${CT_SGF_NAVSIM_ROOT:-"$HOME/ct_sgf_navsim"}
python="$navhard_root/env/bin/python"
checkpoint="$navhard_root/checkpoints/ego_status_mlp_seed_0.ckpt"
export OPENSCENE_DATA_ROOT="$navhard_root/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/nuplan-maps-v1.0"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_DEVKIT_ROOT="$navhard_root/navsim"
export NAVSIM_EXP_ROOT="$navhard_root/exp"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

output="$NAVSIM_EXP_ROOT/ct_sgf_navhard_comparison"
cache="$NAVSIM_EXP_ROOT/ct_sgf_navhard_cache"
mkdir -p "$output"

await_file() {
    local path=$1
    printf 'Waiting for %s\n' "$path"
    until [[ -s "$path" || -f "$path" ]]; do sleep 15; done
}

case "${1:-all}" in
    base)
        await_file "$OPENSCENE_DATA_ROOT/downloads/.ct_sgf_test_extracted"
        await_file "$OPENSCENE_DATA_ROOT/navhard_two_stage/.ct_sgf_verified"
        "$python" -B "$project/ct_sgf_navhard_experiment.py" \
            --export-base "$checkpoint" --output "$output/ego_status_mlp.pkl"
        ;;
    bank)
        await_file "$OPENSCENE_DATA_ROOT/downloads/.ct_sgf_trainval_extracted"
        "$python" -B "$project/ct_sgf_navhard_bank.py" --checkpoint "$checkpoint" \
            --cache "$NAVSIM_EXP_ROOT/ct_sgf_navtrain_bank_cache" --output "$output/navtrain_bank.json" \
            --workers 1 --label-workers "${CT_SGF_NAVHARD_LABEL_WORKERS:-3}"
        ;;
    all)
        await_file "$output/ego_status_mlp.pkl"
        await_file "$output/navtrain_bank.json"
        await_file "$cache/metadata/ct_sgf_navhard_cache_metadata_node_0.csv"
        if [[ ! -s "$output/ct_sgf.pkl" || ! -s "$output/learned_scorer.pkl" ]]; then
            "$python" -B "$project/ct_sgf_navhard_experiment.py" \
                --bank "$output/navtrain_bank.json" --base "$output/ego_status_mlp.pkl" \
                --cache "$cache" --output "$output" --workers "${CT_SGF_NAVHARD_POLICY_WORKERS:-5}"
        fi
        "$python" -B "$project/ct_sgf_navhard_score.py" --cache "$cache" --output "$output/scores" \
            --workers "${CT_SGF_NAVHARD_SCORE_WORKERS:-5}" \
            --arm "EgoStatusMLP=$output/ego_status_mlp.pkl" --arm "CT-SGF=$output/ct_sgf.pkl" \
            --arm "LearnedScorer=$output/learned_scorer.pkl"
        cp -- "$output/scores/compare.json" "$project/navhard_two_stage_comparison.json"
        cp -- "$output/shared_candidate_diagnostics.json" "$project/navhard_two_stage_candidate_diagnostics.json"
        ;;
    *)
        printf 'Usage: %s {base|bank|all}\n' "$0" >&2
        exit 2
        ;;
esac
