# Article 1 — Experiment 2: who contributes versus what they contribute

This is the Article 1 mechanistic experiment.  It has exactly four KD arms:

| Method | Teacher selection | Per-teacher distribution |
| --- | --- | --- |
| `feddf` | all teachers | full vector |
| `all_teachers_support` | all teachers | expertise support, then renormalize |
| `expert_full` | experts in the proxy true class | full vector |
| `expert_support` | experts in the proxy true class | expertise support, then renormalize |

`expert_full` and `expert_support` are a causal pair: the runner writes their
selection matrices and fails if they differ.  `feddf` and
`all_teachers_support` are likewise required to select every teacher.  Support
is applied independently to each teacher as `A[k] * p[k]`, followed by
renormalization, before the selected teacher distributions are averaged.

The names are intentionally explicit.  The older `feddf_support_only` remains
available only as a compatibility alias in the shared target builder; it is not
an Experiment 2 artifact name.

## Clean expertise contract

`experiment_2` consumes Experiment 1's verified holdout-derived authority
artifact through `--authority-npz`; it never reads a local-test mask.  The
artifact and its sibling `provenance.json` must contain:

```text
authority                  uint8[K, C]
source_proxy_sha256         scalar SHA-256 matching the proxy-logit artifact
provenance.json             frozen `mask_source=holdout_acc_per_class>=<dataset threshold>`
```

The runner verifies the authority shape, source hash, nonempty teacher support,
and holdout provenance.  The partition protocol is:

```text
teacher train → optimization holdout (expertise) → final test
```

The runner checks the authority shape, source linkage, and holdout provenance
before target construction.  Experiment 1 has already audited the partition
and authority artifacts.  The legacy local-test mask stored inside the Study-I
proxy cache is ignored; only its teacher logits, proxy indices, and labels are
reused.

## Recommended first pilot

Do not launch the former full Study-I matrix.  First use CIFAR-10, K=10,
seed 42, a balanced 1,000-example proxy subset, and two already-established
regimes: `iid` (low heterogeneity) and `single` (high heterogeneity).  This is
eight paired student runs, plus cheap target construction.  Keep 30 epochs at
the chosen fixed proxy size; every arm in a regime receives the same resulting
update count.  Once targets, provenance, and paired student invariants are
reviewed, expand to the Article-1 replication seeds/regimes—not proxy-size,
ORACLE, supervised-proxy, or threshold-sensitivity experiments.

Run these commands from the repository root.  First build and inspect the
targets for both regimes; only then launch the paired students.  Each source
must be a fresh teacher-logit artifact with its *matching* Experiment-1
holdout-authority artifact:

```bash
OUT=OUTPUTS/experiments/article1_experiment2_v1
IID_SOURCE=OUTPUTS/experiments/study_i/seed_42/raw_work/runs/cifar10/K10/iid__40k-10k-0p15/proxy_analysis/proxy_analysis.npz
IID_AUTHORITY=OUTPUTS/experiments/article_1_server_expertise/experiment_1/holdout_authority/a3c87968752a24794e758a6313c20c33e49b86a85591bb62aa7789204f2bd35e/authority.npz
SINGLE_SOURCE=OUTPUTS/experiments/study_i/seed_42/raw_work/runs/cifar10/K10/single__40k-10k-0p15/proxy_analysis/proxy_analysis.npz
SINGLE_AUTHORITY=OUTPUTS/experiments/article_1_server_expertise/experiment_1/holdout_authority/5d6bc02900323f7e7d01a1f72ee0489432399b0fe00b1eb065fb311adf554ac5/authority.npz

cd ..

# 1. Cheap target-level validation: low heterogeneity (iid), then high (single).
conda run -n FLWR python -m experiments.article_1_server_expertise.experiment_2 \
  --stage targets --source-proxy-analysis "$IID_SOURCE" --authority-npz "$IID_AUTHORITY" \
  --output-root "$OUT" --dataset cifar --seed 42 --regime iid --proxy-size 1000

conda run -n FLWR python -m experiments.article_1_server_expertise.experiment_2 \
  --stage targets --source-proxy-analysis "$SINGLE_SOURCE" --authority-npz "$SINGLE_AUTHORITY" \
  --output-root "$OUT" --dataset cifar --seed 42 --regime single --proxy-size 1000

# Inspect target diagnostics/provenance here.  Then train the four paired students
# in each regime using the same initialization and batch order within that regime.
conda run -n FLWR python -m experiments.article_1_server_expertise.experiment_2 \
  --stage students --source-proxy-analysis "$IID_SOURCE" --authority-npz "$IID_AUTHORITY" \
  --output-root "$OUT" --dataset cifar --seed 42 --regime iid --proxy-size 1000 \
  --student-init-seed 42 --batch-order-seed 42 --training-mode fixed_epochs --device auto

conda run -n FLWR python -m experiments.article_1_server_expertise.experiment_2 \
  --stage students --source-proxy-analysis "$SINGLE_SOURCE" --authority-npz "$SINGLE_AUTHORITY" \
  --output-root "$OUT" --dataset cifar --seed 42 --regime single --proxy-size 1000 \
  --student-init-seed 42 --batch-order-seed 42 --training-mode fixed_epochs --device auto

# Write one machine-readable target/student table for each completed cell.
conda run -n FLWR python -m experiments.article_1_server_expertise.experiment_2 \
  --stage aggregate --source-proxy-analysis "$IID_SOURCE" --authority-npz "$IID_AUTHORITY" \
  --output-root "$OUT" --dataset cifar --seed 42 --regime iid --proxy-size 1000

conda run -n FLWR python -m experiments.article_1_server_expertise.experiment_2 \
  --stage aggregate --source-proxy-analysis "$SINGLE_SOURCE" --authority-npz "$SINGLE_AUTHORITY" \
  --output-root "$OUT" --dataset cifar --seed 42 --regime single --proxy-size 1000
```

`--stage targets` is the cheap validation phase; `--stage students` is
intentionally separate.  A target-only smoke check may use a tiny synthetic
cache in tests.  Runtime smoke training is available with `--smoke` and is not
evidence.

## Full matrix

Experiment 1 has already built and audited the 54 required holdout-authority
cells: CIFAR-10, MNIST, and Fashion-MNIST × seeds 42/43/44 × `iid`,
`alpha0p1`, `alpha0p5`, `alpha1p0`, `multi`, and `single`.  Their proxy hashes
match their authority artifacts.  Build all Experiment 2 targets first:

```bash
bash experiments/article_1_server_expertise/run_experiment2_full_matrix.sh
```

After reviewing the target diagnostics, launch all paired students and write
their per-cell CSVs:

```bash
RUN_STUDENTS=1 bash experiments/article_1_server_expertise/run_experiment2_full_matrix.sh
```

The full-matrix notebook is
`experiments/article_1_server_expertise/article1_experiment2_full_matrix.ipynb`.
It requires all 54 completed `results.csv` cells and reports paired seed-level
student contrasts and target-level changes.

## Outputs and analysis

Every cell writes immutable artifacts below:

```text
<out>/experiment_2/<dataset>/seed_<seed>/<regime>/N<size>_balanced_<hash>/
  protocol.json
  expertise.npz
  targets/<method>/target.npz
  targets/<method>/diagnostics.json
  students/<method>/metrics.json
  students/<method>/test_outputs.npz
  results.csv
```

Target diagnostics include normalized targets, teacher-selection/fallback
counts, selected-teacher cardinality, support mass removed, support size,
true-class mass (mean and median), NLL, Brier score, entropy, maximum
wrong-class mass, true-class margin, wrong-argmax rate, and each arm's L1/KL/JS
change versus the same FedDF target.  This exposes target effects before
training and can be grouped by heterogeneity regime.

Student metrics include global-test accuracy and NLL, update count, complete
initial-state hash, full batch-order hash, target SHA-256, optimization budget,
and a four-way paired-run ID.  The runner fails unless all four students in a
cell share initial state, batch order, and update count.  It deliberately
rebuilds the FedDF student rather than importing an older result, because an
historical student cannot prove this pairing contract.

## Reuse boundary

Valid reuse: teacher proxy logits, proxy indices/labels, partitions, model and
student-training utilities, and target construction utilities, when they were
created under the clean protocol.  Historical raw Study-I logits can help test
the code path, but their local-test mask is never read by this experiment.
Historical targets/students are left untouched and are not merged into
Experiment 2 results.
