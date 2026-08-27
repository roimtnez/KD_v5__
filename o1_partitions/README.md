# Client partitioning

`partitioner.py` is the single implementation used by the stable Estudio-I CLI. It creates the
seed-specific client allocation and the three disjoint surfaces stored in each
`clients/cXXX.npz`:

- `train_idx`: teacher SGD;
- `holdout_idx`: early stopping;
- `local_test_idx`: competence measurement only.

For seeds other than 42, the execution layout is:

```text
OUTPUTS/experiments/phase_a/seed_<s>/raw_work/
└── partitions/<dataset>/K10/<regime>/
    ├── clients/c000.npz ... c009.npz
    ├── config.yaml
    ├── metadata.json
    └── report/
```

## Derived report contract

`partition_reports.py` deterministically derives the report from the persisted client indices and
the canonical dataset labels. It never changes a partition. A complete report contains:

```text
report/
├── train/{counts.csv,percent.csv}
├── holdout/{counts.csv,percent.csv}
├── local_test/{counts.csv,percent.csv}
├── summary.csv
├── metadata.json
├── distribution_heatmap.png
└── dist_counts.png
```

Every report CSV repeats `seed`, `dataset`, `regime` and `split` provenance columns. Matrix CSVs
then retain the historical `Client ID, 0, ..., 9` payload expected by the heatmap notebook.

Creation and reuse through `DirichletPartitioner.ensure()` both check this contract. Therefore an
old partition with valid clients but missing reports is backfilled without retraining teachers or
changing indices.

For seeds 43/44, reuse also requires the seed-specific proxy filename and the exact SHA-256 of
`proxy_idx` recorded in `config.yaml`. A partition made from the seed-42 pool is rejected before
teacher training, even if its directory itself contains `seed_43` or `seed_44`. Seeded proxy splits
are created once, without overwrite, by `data/proxy_splits.py`.

Existing reports can also be backfilled explicitly:

```bash
conda run -n FLWR python -m o1_partitions.partition_reports \
  --partitions-root OUTPUTS/experiments/phase_a/seed_43/raw_work/partitions/mnist/K10 \
  --dataset mnist --data-dir data
```

The diagnostic consumer is outside the final three-study narrative. Partition validation is
performed from persisted reports and manifests; no final notebook consumes this diagnostic.

## Canonical execution

Partition creation is stage 1 of the stable pipeline and should normally be invoked through:

```bash
conda run -n FLWR python -m oracle_distillation.cli.run_dirichlet_alphas_methods \
  --work_root OUTPUTS/experiments/phase_a/seed_43/raw_work \
  --data_dir data --dataset mnist --alphas 0.1 0.5 1.0 iid single multi \
  --methods feddf energy confidence consensus expert oracle \
  --num_clients 10 --holdout_frac 0.15 --local_test_frac 0.2 \
  --seed 43 --skip_if_done
```

`phase_a` es la ruta heredada que contiene los outputs del Estudio I; no debe interpretarse como
la nomenclatura científica actual ni renombrarse en ejecuciones ya existentes. Dataset transforms,
class count, architecture and training recipes remain centralized in `data/dataset_config.py`.
