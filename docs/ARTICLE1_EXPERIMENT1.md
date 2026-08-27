# Article 1 — Experiment 1: out-of-expertise probability mass

This package implements the diagnostic experiment only. It does not train a
student and does not implement EXPERT-support, ORACLE, supervised-proxy,
proxy-size, or threshold-sensitivity arms.

## Scientific unit and measurements

For proxy example `i`, selected teacher `k`, teacher probabilities
`p_ik = softmax(z_ik / T)`, and binary expertise mask `M_kc`, the individual
teacher diagnostic is:

`outside_teacher(i,k) = sum_c p_ik(c) * (1 - M_kc)`.

It is summarized only over the teachers selected by current EXPERT
(`M_k,y_i = 1`). The analysis also records support size/fraction for each
selected teacher, the number selected per proxy example, and the historical
empty-selection fallback rate.

At the final-target level, `q_i` is the current probability-space
`expert_full` target. Its collective unsupported mass is:

`outside_union(i) = sum_{c: no selected teacher has M_kc=1} q_i(c)`.

This is deliberately stronger than the individual diagnostic: it counts only
mass on classes supported by none of the teachers that actually contributed.
The output additionally reports target mass outside each selected teacher's
own support, which can remain high even when union-supported mass is zero.

Target quality includes argmax accuracy, NLL, Brier score, entropy, and true
class probability, plus per-proxy correlations between both noise measures and
NLL, true-class probability, and target correctness.

Each condition result is identified by dataset, seed, regime, temperature,
source-cache SHA-256, and authority-mask provenance. The per-proxy NPZ adds
the proxy index and true label. Regime metadata identifies IID, Dirichlet
alpha, and the multi/single specialization arms so figures can compare the
heterogeneity axes without relying on path names.

## Provenance gate

The historical Study-I `teacher_knows_class_mask` is exactly reproducible from
`local_test_acc_per_class`, not from holdout per-class accuracy. It therefore
fails this experiment's protocol: local test is evaluation-only. The default
diagnostic mode rejects it.

The partitions already contain disjoint `train_idx`, `holdout_idx`, and
`local_test_idx`. The historical trainer early-stopped on `holdout_idx`, but
did not persist `holdout_acc_per_class`; it persisted only local-test per-class
accuracy. Since the retained checkpoints exist, no teacher retraining is
needed for this diagnostic: run the holdout-authority preparation step below.
It evaluates the frozen checkpoints on holdout, verifies a deterministic
32-example proxy-logit probe against the saved cache, and creates a
content-addressed authority sidecar. If a checkpoint is missing or fails the
probe, that condition must be regenerated (teacher plus proxy logits) before
it can be interpreted.

## Commands

Run from the repository root in the FLWR environment.

```bash
# Read-only audit: current historical sources will be labelled legacy/local-test.
conda run -n FLWR python -m experiments.article_1_server_expertise.experiment_1 \
  --stage audit

# Produce valid holdout expertise masks from frozen teachers (inference only).
conda run -n FLWR python -m experiments.article_1_server_expertise.prepare_holdout_authority \
  --output-root OUTPUTS/experiments/article_1_server_expertise/experiment_1 \
  --device auto

# Run Experiment 1 on every available compatible dataset × seed × regime.
conda run -n FLWR python -m experiments.article_1_server_expertise.experiment_1 \
  --stage analyze \
  --authority-root OUTPUTS/experiments/article_1_server_expertise/experiment_1
```

The analysis writes a compact `results.csv` and `results.json`, an auditable
`provenance_audit.csv/json`, and one compressed per-proxy observation NPZ per
dataset/seed/regime under a content-addressed `diagnostic_<hash>/` directory.

For a non-publishable mechanistic check of the legacy masks only, append
`--allow-legacy-exploratory`. Every row and NPZ remains labelled
`legacy_local_test_leakage_affected`; it must not be mixed with the valid
holdout analysis.
