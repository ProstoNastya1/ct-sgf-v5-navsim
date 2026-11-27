#!/usr/bin/env bash
set -euo pipefail

navhard_root=${CT_SGF_NAVSIM_ROOT:-"$HOME/ct_sgf_navsim"}
navhard_python="$navhard_root/env/bin/python"
navhard_commit=0a380a9063d7162ec93d0f51e9990ebac585f720
export OPENSCENE_DATA_ROOT="$navhard_root/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/nuplan-maps-v1.0"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_DEVKIT_ROOT="$navhard_root/navsim"
export NAVSIM_EXP_ROOT="$navhard_root/exp"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

test "$(git -C "$NAVSIM_DEVKIT_ROOT" rev-parse HEAD)" = "$navhard_commit"
test -x "$navhard_python"
test -d "$NUPLAN_MAPS_ROOT"
downloads="$OPENSCENE_DATA_ROOT/downloads"
checkpoint="$navhard_root/checkpoints/ego_status_mlp_seed_0.ckpt"
mkdir -p "$downloads" "$NAVSIM_EXP_ROOT" "$(dirname "$checkpoint")" "$OPENSCENE_DATA_ROOT/navsim_logs"
openscene=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main

fetch() {
    local url=$1 destination=$2 checksum=$3
    if [[ -f "$destination" ]] && printf '%s  %s\n' "$checksum" "$destination" | sha256sum --check --status; then
        return
    fi
    curl --fail --location --silent --show-error --retry 4 --continue-at - \
        --output "$destination.part" "$url"
    printf '%s  %s\n' "$checksum" "$destination.part" | sha256sum --check
    mv "$destination.part" "$destination"
}

extract_logs() {
    local split=$1 file=$2
    local marker="$downloads/.ct_sgf_${split}_extracted"
    if [[ ! -f "$marker" ]]; then
        tar -xzf "$file" -C "$OPENSCENE_DATA_ROOT/navsim_logs" --strip-components=2 \
            "openscene-v1.1/meta_datas/$split"
        local -a logs=("$OPENSCENE_DATA_ROOT/navsim_logs/$split/"*.pkl)
        test -f "${logs[0]}"
        # NAVSIM tries to unpickle every entry within the log split directory.
        touch "$marker"
    fi
}

case "${1:-}" in
    benchmark-data)
        fetch "$openscene/openscene-v1.1/openscene_metadata_test.tgz" \
            "$downloads/openscene_metadata_test.tgz" 871ba786da584e03f5151924674ab006e83782f796a979ee5dd144b5c6dca1ab &
        original_job=$!
        fetch "$openscene/navsim-v2/navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz" \
            "$downloads/navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz" \
            a2e429a01e9a93afd9d1f753ed306f469c094292627467182d35fae9fb2ecad7 &
        synthetic_job=$!
        wait "$original_job"
        extract_logs test "$downloads/openscene_metadata_test.tgz"
        wait "$synthetic_job"
        marker="$OPENSCENE_DATA_ROOT/navhard_two_stage/.ct_sgf_verified"
        if [[ ! -f "$marker" ]]; then
            tar -xzf "$downloads/navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz" -C "$OPENSCENE_DATA_ROOT"
            test -d "$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles"
            touch "$marker"
        fi
        ;;
    training-data)
        fetch "$openscene/openscene-v1.1/openscene_metadata_trainval.tgz" \
            "$downloads/openscene_metadata_trainval.tgz" \
            193138ac68b29d977d4438a95dfda35a07ff4ad3efb1800b3ee47d6e1dcb3764
        extract_logs trainval "$downloads/openscene_metadata_trainval.tgz"
        ;;
    cache|baseline)
        test -f "$OPENSCENE_DATA_ROOT/navhard_two_stage/.ct_sgf_verified"
        test -f "$downloads/.ct_sgf_test_extracted"
        fetch https://huggingface.co/autonomousvision/navsim_baselines/resolve/main/ego_status_mlp/ego_status_mlp_seed_0.ckpt \
            "$checkpoint" 87d75b0f43d077ac3531370d7cccac98656d4e9b5ce5fa6618e28b7358b3a86b
        cache="$NAVSIM_EXP_ROOT/ct_sgf_navhard_cache"
        worker=(worker=single_machine_thread_pool "worker.max_workers=${CT_SGF_NAVHARD_WORKERS:-2}" gpu=false)
        cd "$NAVSIM_DEVKIT_ROOT"
        "$navhard_python" -m navsim.planning.script.run_metric_caching \
            train_test_split=navhard_two_stage "metric_cache_path=$cache" force_feature_computation=false "${worker[@]}"
        if [[ "$1" == baseline ]]; then
            "$navhard_python" -m navsim.planning.script.run_pdm_score \
                train_test_split=navhard_two_stage "metric_cache_path=$cache" agent=ego_status_mlp_agent \
                "agent.checkpoint_path=$checkpoint" experiment_name=ct_sgf_navhard_frozen_mlp "${worker[@]}"
        fi
        ;;
    *)
        printf 'Usage: %s {benchmark-data|training-data|cache|baseline}\n' "$0" >&2
        exit 2
        ;;
esac
