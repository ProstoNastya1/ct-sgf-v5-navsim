# CT-SGF v5

CT-SGF is a post-planning risk check for a frozen driving planner. It uses past simulations to estimate the risk of a first failure at each step. It checks nearby alternatives and keeps the original plan when it has too little evidence or finds no supported improvement.

The [one-page Word report](output/CT-SGF_v5_Report_EN.docx) explains the method and the two completed pilot runs.

## Quick start

The standalone example uses the Python standard library. Run it from the repository root with Python 3.9 or later.

```bash
python -B ct_sgf_demo.py
python -B ct_sgf_demo.py --json
```

The example uses a simple driving model and a speed limit at a crossing line. In one pair, the early part of the path changes while the state at the crossing stays the same. In the other pair, the entry speed changes while the position and action at the crossing stay the same. Both pairs are constructed examples for checking how the implementation handles these differences.

## How it works

1. `Rollout.at_risk()` keeps observed steps up to and including the first event. If there is no observed event, it keeps the observed prefix.
2. `HazardModel.predict()` compares each planned transition with transitions in the evidence bank. Equation (3) estimates step risk as `(alpha + D) / (alpha + beta + A)`. Here, `D` is the weighted count of first events, and `A` is the weighted count of steps at risk. Survival over the whole horizon is the product of `1 - h` across steps.
3. `correct_plan()` recomputes features for each candidate. It scores the candidate using Equation (5), which combines its normalized distance from the original plan with `eta * (-log_survival)`.
4. The search checks control bounds, distance from the original plan, and evidence support. Support includes the total weight of similar transitions, the effective sample size, and the number of groups with sufficient weight.

The prototype uses one event type at a time and a finite candidate set. The current NAVSIM comparison runs with CBF disabled. Shared latent context, correction inside a diffusion sampler, matched CBF on/off runs, and evaluation with a stronger planner remain separate checks.

## Verified results

### NAVSIM mini

The mini test used kinematic plans, the official NAVSIM tracker and scorer, and non-reactive traffic replay. The bank had 96 candidates across 12 scenes in Singapore and Las Vegas. It contained 12 first at-fault events. Evaluation used four scenes from separate Boston logs.

| Metric | Result |
| --- | --- |
| Steps with sufficient evidence | 80 of 160 (50%) |
| First at-fault collisions | 0 of 4 before and after correction |
| Mean raw single-stage PDM score | 0.399 before and after correction |

CT-SGF kept all four original plans. This test confirms the simulator integration and measures coverage. [Mini results](navsim_mini_report.json) · [Mini calibration](navsim_mini_calibration.json)

### Short two-stage NAVSIM pilot

The short run used four original and eight synthetic scenes from `navhard_two_stage`, selected by cache availability before scoring. It used 53 geographically separated training scenes and the official two-stage scoring code restricted to the selected branches.

| Method | Original scenes with an at-fault collision | Synthetic scenes with an at-fault collision | Two-stage EPDMS on the subset |
| --- | ---: | ---: | ---: |
| EgoStatusMLP | 0 / 4 | 2 / 8 | 0.00 / 100 |
| CT-SGF | 0 / 4 | 2 / 8 | 0.00 / 100 |
| Learned trajectory scorer | 0 / 4 | 2 / 8 | 0.00 / 100 |

All three methods returned identical trajectories. The separate stage scores were 41.93 and 27.38. NAVSIM multiplies the original-stage score by the weighted follow-up score within each branch, then averages the products. In this subset, a drivable-area violation gave one stage a zero in every branch. This yields the observed combined score of zero. The short test verifies the three-way pipeline and provides no evidence of collision reduction. [Short-run results](output/navhard_short_comparison.json) · [Exact evaluation subset](output/navhard_short_subset.json) · [NAVSIM aggregation code](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/navsim/planning/script/run_pdm_score_from_submission.py#L184-L228)

## Evaluate a trajectory bank

```bash
python -B ct_sgf_evaluate.py bank.json
python -B ct_sgf_evaluate.py bank.json --compare-baselines --bank-sizes 2 8 32
```

This is a minimal example of the expected JSON format.

```json
{
  "horizon": 1,
  "bandwidths": [0.5, 1.0],
  "bank": [
    {"group": "train-location-a", "features": [[0.5, 5.0]], "observed_steps": 1, "first_event": 0},
    {"group": "train-location-b", "features": [[0.5, 5.0]], "observed_steps": 1, "first_event": null}
  ],
  "evaluation": [
    {"group": "held-out-location-c", "features": [[0.5, 5.0]], "observed_steps": 1, "first_event": null}
  ]
}
```

Use the same feature order, time step, and event definition in the bank and evaluation sets. `first_event` is a zero-based index or `null` if no event was observed. For full-horizon evaluation, supply features for the complete planned trajectory. The bank still uses observations only through the first event. Put dependent trajectories from the same log or location in the same group.

The evaluator reports support, calibration, Brier score, and log loss on known outcomes. `--compare-baselines` adds log-KDE and learned trajectory and stepwise scores. `--event-only-features 0 1 2` adds a comparator restricted to the selected feature indices. Coverage can change with the bank size, so read the score and coverage together.

## Run NAVSIM

Use Linux or Ubuntu in WSL, Python 3.9, the OpenScene annotations, and the nuPlan maps. The project uses NAVSIM v2 at commit `0a380a9063d7162ec93d0f51e9990ebac585f720`. Each four-second plan is simulated at 10 Hz. The standalone demo and the JSON evaluator can run without installing NAVSIM. [NAVSIM installation](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/docs/install.md) · [Dataset splits](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/docs/splits.md)

### Mini pilot

Set `OPENSCENE_DATA_ROOT`, `NUPLAN_MAPS_ROOT`, `NUPLAN_MAP_VERSION=nuplan-maps-v1.0`, and `NAVSIM_EXP_ROOT`. The following command selects scenes from different maps and asks the official runner to create their metric caches.

```bash
python -B ct_sgf_navsim_mini.py "$OPENSCENE_DATA_ROOT/navsim_logs/mini" \
  "$NAVSIM_EXP_ROOT/ct_sgf_mini_cache" --manifest mini_manifest.json --cache-missing
python -B ct_sgf_navsim.py --manifest mini_manifest.json \
  --output navsim_pilot.json --export-evidence navsim_evidence.json
python -B ct_sgf_evaluate.py navsim_evidence.json --compare-baselines
```

With an existing cache, `ct_sgf_navsim.py --cache-root /path/to/metric_cache` can split a small pilot by log. For an explicit geographic split, use the manifest with `bank` and `evaluation` entries. Each scene needs a trusted `cache` path, its official `log` name, and a geographic `group`. Optional `base` and `proposals` entries use eight ego-relative `[x, y, yaw]` waypoints at 0.5-second intervals. The mini pilot uses current object annotations and the route map as features.

### Full two-stage comparison

The full comparison uses the frozen public EgoStatusMLP checkpoint, seed 0. It uses ego and candidate motion as features, with no object annotations in the input. The setup script verifies the checkpoint and dataset archives by SHA-256. EgoStatusMLP runs without camera or LiDAR files. Use the scripts from the repository root in a Linux or WSL shell with the NAVSIM environment prepared. [Official checkpoint](https://huggingface.co/autonomousvision/navsim_baselines/blob/main/ego_status_mlp/ego_status_mlp_seed_0.ckpt) · [Official agent run script](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/scripts/evaluation/run_ego_mlp_agent_pdm_score_evaluation.sh)

```bash
bash ct_sgf_navhard_setup.sh benchmark-data
bash ct_sgf_navhard_setup.sh training-data
bash ct_sgf_navhard_setup.sh cache
bash ct_sgf_navhard_run.sh base
bash ct_sgf_navhard_run.sh bank
bash ct_sgf_navhard_run.sh all
```

The full pipeline selects 64 `navtrain` scenes for the bank and 16 from separate recordings for development. It removes bank scenes within 150 m of development or evaluation scenes on the same map. CT-SGF and the trained scorer use the same candidate set, constraints, budget, and support gate. The official reactive traffic policy is used for collision labels.

The full comparison writes `navhard_two_stage_comparison.json` after validating all 450 original and 5,462 synthetic scenes for each method. It reports raw first at-fault collisions, official EPDMS, and paired differences. The candidate diagnostics report includes development-only calibration, learning curves, and an estimate of the minimum detectable effect when the data supports one.
