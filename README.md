# Article 1: selection and content in ensemble distillation

This repository is a deliberately small experimental pipeline, not a federated-learning framework. It asks two paired questions under class specialization: (1) which teachers should contribute to a public-proxy ensemble, and (2) once they contribute, whether the complete distribution or only their evidenced class support should be distilled.

The active scope is MNIST, Fashion-MNIST and CIFAR-10; `K=10`; IID, `alpha1p0`, `alpha0p5`, `alpha0p1`, `multi` and `single`; and seeds 42, 43 and 44. CIFAR-100, personalization and the other articles are outside the active pipeline.

## Data roles and stages

`article1.partitioning` reserves the 10,000-example proxy from the *training* set before assigning any private examples. Every client then owns disjoint `train_idx`, `holdout_idx`, and `test_idx`: train updates parameters only; holdout selects the sole saved checkpoint and estimates competence; local test is post-hoc evaluation only. The official test set is only student evaluation. Holdout, local test, and proxy use deterministic evaluation transforms. Reusing the holdout for checkpoint selection and competence estimation can be optimistic due to selection; it is not an independent calibration set.

`article1.local_training` trains each teacher once for a dataset/seed/regime, selects one holdout checkpoint, and writes one float32 proxy-logit cache with indices, proxy labels, `M`, holdout counts/accuracy, and compact provenance. No proxy or test labels can affect `M`:

`M[k,c] = 1{n_holdout[k,c] > 0 and acc_holdout[k,c] >= threshold_dataset}`.

Initial thresholds are MNIST 0.90, Fashion-MNIST 0.80, and CIFAR-10 0.70. A teacher with an all-zero row is valid and is never selected by EXPERT.

`article1.distillation` is the only target implementation. It rebuilds targets from the shared cache and the one KD loop consumes probabilities directly:

`L = T² KL(q || softmax(student_logits/T))`.

## Methods

For all logit methods, `q(x) = softmax(sum_k w_k(x) z_k(x) / T)`. Weights are non-negative, sum to one per sample, and are scalar per teacher (never per class). No teacher logit calibration or normalization is implicit.

| Identifier | Teacher selection / weights | Operator |
|---|---|---|
| `feddf_logit` | all, uniform | mean logits |
| `confidence_logit` | all, `softmax_teachers(MSP)` | weighted logits |
| `consensus_logit` | hard prediction equals `argmax mean_k softmax(z_k)` | mean selected logits |
| `energy_logit` | all, `softmax_teachers(logsumexp(z_k))` | weighted logits |
| `expert_logit` | `M[k,y]=1` | mean selected logits |
| `oracle_logit` | `argmax(z_k)=y` | mean selected logits |
| `expert_prob` | same EXPERT selection | mean `softmax(z_k/T)` |
| `expert_prob_sr` | same EXPERT selection | restrict each `softmax(z_k/T)` to `M[k,:]`, renormalize, mean |
| `oracle_prob` | same ORACLE selection | mean `softmax(z_k/T)` |

Confidence fixes `T_weight=tau=1`, so it is softmax across teacher MSP values, **not** MSP divided by their sum. Energy fixes both parameters to one and is sensitive to offsets between teachers. Consensus is soft-vote routing, not a mode of hard labels. EXPERT and ORACLE use the proxy label; EXPERT is not label-free, and ORACLE is not a mathematical upper bound on the student.

The main ablation is `expert_logit` vs `expert_prob` vs `expert_prob_sr`: first changes HOW outputs aggregate, then changes WHAT class mass remains. SR does no logit masking. For empty EXPERT, ORACLE, or Consensus selection, every arm (including probability arms) uses the same explicit fallback: `softmax(mean_k z_k/T)`. We report the fallback rate separately. For EXPERT-SR we additionally report selected-teacher mass outside support before restriction, excluding fallback rows.

## Recipe, artifacts, and analysis

All methods in a condition share proxy/index files, cached teachers, full student initialization, batch order, updates and recipe: `T=8`, AdamW, `lr=1e-3`, `weight_decay=1e-4`, batch 256, 30 epochs. The result identity hashes the method, temperature, recipe, cache, proxy, and `M`; incompatible artifacts are not reused. Student checkpoints and per-method targets are not stored by default.

The retained artifacts are partitions, one selected checkpoint per teacher, one `teacher_cache.npz`, `metadata.json`, and one safely-upserted canonical `results.csv`. Metrics per run are student test accuracy/NLL, target argmax accuracy/NLL/entropy at the actual temperature, selected-teacher count, fallback count/rate, and the stated EXPERT-SR support-mass diagnostic. `article1.analysis.paired_effects` summarizes deltas paired within dataset/regime/seed and reports mean/SD across seeds; clients and regimes are not treated as independent replicates.

## Commands

```bash
python -m article1.runner partition --dataset mnist --regime alpha0p5 --seed 42 --output OUTPUTS/article1/partitions/mnist-42-alpha0p5
python -m article1.runner teachers --dataset mnist --regime alpha0p5 --seed 42 --partitions OUTPUTS/article1/partitions/mnist-42-alpha0p5 --output OUTPUTS/article1/sources/mnist-42-alpha0p5
python -m article1.runner distill --dataset mnist --seed 42 --method expert_prob_sr --cache OUTPUTS/article1/sources/mnist-42-alpha0p5/teacher_cache.npz --results OUTPUTS/article1/results.csv
python -m article1.analysis OUTPUTS/article1/results.csv
python -m article1.audit OUTPUTS/article1/results.csv
pytest -q tests/test_article1_v2.py
```

The full grid is available as a Python launcher:

```bash
python run_article1_grid.py --device cuda
python run_article1_grid.py --stage distill --methods expert_logit expert_prob expert_prob_sr
python run_article1_grid.py --dry-run
```

For a smoke run, use a small proxy in `partition` and `--epochs 1` in `teachers` and `distill`; it is an execution check, not a scientific result.

## Literature boundary

`feddf_logit` is a one-shot adaptation of the logit-ensemble step in Lin et al., *Ensemble Distillation for Robust Model Fusion in Federated Learning* (NeurIPS 2020). The source method is an iterative federated model-fusion protocol and its official code uses server distillation data; this repository does **not** claim a full FedDF reproduction. See the [paper](https://arxiv.org/abs/2006.07242) and [official code](https://github.com/epfml/federated-learning-public-code/tree/master/codes/FedDF-code).

Selective-FD was checked but is excluded: its client-side selector uses density-ratio/OOD estimation and its server selector filters ensemble outputs during an iterative client/server protocol, so it is not a compatible output-only, one-shot comparator without a protocol change. See the [Selective-FD paper](https://www.nature.com/articles/s41467-023-44383-9). No further SOTA comparator is currently implemented; the benchmark is therefore not scientifically closed as a SOTA survey.

Interpret cautiously: EXPERT-SR's renormalization guarantees an increase in `q_y` and reduction in NLL whenever selected teachers have `M[k,y]=1`; it does not itself demonstrate removal of harmful knowledge. Probability aggregation remains sensitive to logit scale. Historical outputs using earlier definitions do not validate these methods automatically.
