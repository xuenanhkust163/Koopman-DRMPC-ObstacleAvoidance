# Serpentine Robustness Experiment Plan

## Goal

Compare robust and non-robust paper-QP controllers on the same clockwise Serpentine scenario, starting at the northwest corner and initially heading east. The primary question is whether robust control improves full-lap completion and closed-loop safety under additive process disturbance, not just whether one plotted trajectory looks better.

## Fixed Protocol

- Track: `SerpentineTrack`, northwest start, clockwise direction, 13 physical rectangular obstacles.
- Controller: paper QP with OSQP; `theta=0.10`, `epsilon=0.10`, control update every step.
- Nominal plant disturbance: additive process disturbance, `sigma=0.5`.
- Pairing: for each seed, robust and non-robust runs must use the same disturbance realization. Keep all other settings fixed within a comparison block.
- Do not change track geometry, obstacle placement, or controller tuning during a block. Record any diagnostic or engineering-tuning run separately from paper-aligned results.

## Run Plan

| Stage | Experiment                                                  |                                         Seeds / pairs |                 Individual simulations | Status / gate                                                                                                                                                                                                  |
| ----- | ----------------------------------------------------------- | ----------------------------------------------------: | -------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | Northwest-start nominal pilot, `sigma=0.5`                |                                1 paired seed (`42`) |                                      2 | Already run; exploratory only and excluded from confirmatory statistics because its outcome has already been inspected.                                                                                        |
| 1     | No-disturbance diagnostic, `sigma=0`, full obstacle scene |                                1 paired seed (`42`) |                                      2 | Completed with 100-step stall stop. Robust stopped at 421.03 m after 1436 steps; non-robust at 421.11 m after 1394 steps. QP failures: 246 and 334; robust half-space fallbacks: 178. Neither completed a lap. |
| 2     | Obstacle-free diagnostic, `sigma=0`, same track geometry  |                                1 paired seed (`42`) |                                      2 | Completed 3000 steps per controller. Both passed the ~421 m stall section and reached 78.7% / 79.2% lap progress. QP failures: 1467 / 3000 and 1472 / 3000; neither completed a lap.                           |
| 3     | Nominal robustness study, `sigma=0.5`                     |           20 paired seeds (`1001` through `1020`) |                                     40 | Main confirmatory comparison. Use fresh predeclared seeds; start only after Stages 1-2 identify no blocking implementation defect.                                                                             |
| 4     | Statistical extension, `sigma=0.5`                        | Extend to 30 paired seeds (`1001` through `1030`) | 60 total; 20 additional beyond Stage 3 | Conditional: add these 10 pairs if the paired confidence interval is inconclusive or completion outcomes vary substantially across seeds.                                                                      |
| 5     | Disturbance-shift check, `sigma=0.75` and `1.0`         |                              5 paired seeds per level |                               20 total | Optional follow-up after the nominal block. Keep robust ambiguity samples at the nominal scale as required by the paper protocol.                                                                              |

## Budget

- **Formal nominal cohort:** 20 paired seeds = 40 individual simulations. Seed 42 is excluded from confirmatory statistics because it is an inspected pilot.
- **Diagnostics completed:** 4 individual simulations across the paired zero-noise obstacle and obstacle-free blocks. These seed-42 results are diagnostic only and excluded from confirmatory statistics.
- **Remaining execution before a confirmatory conclusion:** 40 individual simulations: 20 fresh nominal seeds, each with a robust/non-robust pair. The diagnostic pairs are not part of the 20-seed cohort.
- **If extending to 30 nominal seeds:** add 20 individual simulations beyond the 20-seed cohort.
- **With the optional disturbance-shift check:** add 20 individual simulations.
- A paired seed means two simulations: one robust and one non-robust. The 20-seed nominal study is 20 pairs, not 20 individual runs.

## Metrics To Report

1. Lap completion rate and crash rate; report the number of successful pairs and seeds.
2. Progress at termination and crash location, including track arclength and lateral error.
3. For completed laps, lap time and lateral-error statistics. For failed runs, report errors only over the traveled portion and label them as partial-trajectory metrics.
4. Minimum track-boundary margin and minimum physical-obstacle clearance.
5. QP solve success, fallback and failure counts; median and 95th-percentile solve time.
6. Paired robust-minus-non-robust differences by seed, with 95% confidence intervals. Do not infer a general advantage from one seed or one trajectory overlay.

## Decision Gates

- **After Stage 1:** if the zero-disturbance runs fail at the same section, treat it as a nominal tracking/geometry/controller issue; do not launch the 20-seed block until the cause is understood.
- **Diagnostic evidence so far:** with obstacles on at `sigma=0`, both controllers stopped at the same `s≈421 m` section. At termination, robust/non-robust were at `(-34.21, -210.60)` / `(-34.28, -210.61)` m, with lateral errors `4.88` / `4.90` m and boundary margins `5.12` / `5.10` m. Their last recovery controls were approximately `[-0.357, -0.100]` / `[-0.361, -0.100]`. The nearest physical rectangular obstacle had vehicle-clearance about 35 m, so this was not a physical collision. With obstacles off at `sigma=0`, both passed that section and continued to ~79% progress. This points to an obstacle-constraint/half-space interaction at the first stall, while solver stability remains a separate unresolved issue.
- **Observed solver error:** `RuntimeError` from CasADi `paper_qp_qpsol` / `OsqpInterface`: `conic process failed`; the full unique exception text is stored in each new JSON summary. In the obstacle-free block, mean solve times were 84.8 ms / 67.1 ms and P95 times were 137.0 ms / 122.6 ms for robust / non-robust.
- The completed seed-42 artifacts predate QP-input snapshots. Future failures record sampled state, full reference trajectory, previous control, nearby obstacles, active obstacle pairs and constraint bounds on the first and every 100th QP failure; compact error/status counts remain available for every step. Raw Hessian/Jacobian matrices are not serialized.
- **Single-factor recovery-bias ablation, seed 1001:** keeping `sigma=0.5`, additive process noise, obstacles, QP/OSQP settings, and the two-lap target fixed, disabling only the recovery steering-away obstacle bias increased robust progress from 451.5 m (0.316 laps) to 514.2 m (0.360 laps), and non-robust progress from 452.6 m (0.317 laps) to 473.9 m (0.332 laps). Both still crossed the same outer track boundary before one lap. Mean absolute heading error fell from 18.3° to 4.2° (robust) and 17.8° to 7.6° (non-robust), but QP failures increased from 854 to 1444 and 935 to 1357, respectively. This supports the recovery steering bias as an aggravating factor, not a complete root-cause fix; the `conic process failed` solver issue remains unresolved. Treat this as a diagnostic ablation, not part of the nominal robustness cohort.
- **After the first 5 pairs in Stage 3:** check artifact integrity, common-noise pairing, solver diagnostics and failure locations. Continue unchanged unless there is a reproducibility or implementation defect; do not tune parameters mid-block.
- **After 20 pairs:** report the predeclared metrics. Extend to 30 pairs only if uncertainty remains material; otherwise proceed to the optional shift check or conclude the nominal comparison.
- Any controller or geometry change starts a new experiment block with a new suffix and fresh seeds; never merge tuned and untuned results.

## Reproduction Command

Run from the repository root. The paired runner executes both controllers for the selected seed and writes a summary plus trajectory artifacts. This example starts the fresh confirmatory cohort at seed 1001:

```bash
./.venv/bin/python -u script/run_noise_comparison_paper.py \
  --track-name SerpentineTrack \
  --controller-formulation qp \
  --qp-solver osqp \
  --sigma 0.5 \
  --theta 0.10 \
  --epsilon 0.10 \
  --seed 1001 \
  --steps 8000 \
  --control-update-interval 1 \
  --name-suffix northwest_clockwise_nominal_s1001
```

Diagnostic commands (seed 42 is intentionally excluded from the confirmatory cohort):

```bash
# Zero disturbance, physical obstacles enabled; stop after a sustained stall.
./.venv/bin/python -u script/run_noise_comparison_paper.py \
  --track-name SerpentineTrack --controller-formulation qp --qp-solver osqp \
  --sigma 0 --theta 0.10 --epsilon 0.10 --seed 42 --steps 3000 \
  --stop-after-stall-steps 100 --control-update-interval 1 \
  --name-suffix northwest_clockwise_zero_noise_obstacles_stallstop100_s42

# Zero disturbance, physical obstacles disabled; preserve identical track geometry.
./.venv/bin/python -u script/run_noise_comparison_paper.py \
  --track-name SerpentineTrack --controller-formulation qp --qp-solver osqp \
  --sigma 0 --theta 0.10 --epsilon 0.10 --seed 42 --steps 3000 \
  --no-obstacles --control-update-interval 1 \
  --name-suffix northwest_clockwise_zero_noise_no_obstacles_s42
```

## Scope Note

`run_noise_comparison_paper.py` enables obstacles by default. The opt-in `--no-obstacles` switch is reserved for the paired obstacle-free diagnostic and does not change standard experiment behavior. The runner records whether obstacles were enabled and aggregates solver status, failure, fallback, and full unique error details in its summary.
