# Artículo 1: selección y contenido en destilación de ensembles

Este repositorio mantiene la base experimental del Artículo 1: bajo especialización por clase, compara quién contribuye a un ensemble de teachers y qué distribución se destila. El protocolo activo es `article1-v2`.

## Contrato validado

El flujo es estrictamente: **partición → teachers → targets/KD → análisis**. La partición reserva primero el proxy público y separa, para cada cliente, `train_idx`, `holdout_idx` y `test_idx`. Solo `train` actualiza el teacher; `holdout` escoge el checkpoint y construye `M`; el test local y el test oficial son solo evaluación. Por tanto no hay test leakage hacia `M`, la selección de checkpoints ni los targets.

Los nueve métodos son `feddf_logit`, `confidence_logit`, `consensus_logit`, `energy_logit`, `expert_logit`, `oracle_logit`, `expert_prob`, `expert_prob_sr` y `oracle_prob`. Los métodos `*_logit` aplican `softmax(sum_k w_k z_k / T)`; los métodos `*_prob` promedian `softmax(z_k / T)` con el mismo routing. EXPERT usa `M[k,y]`; ORACLE selecciona teachers que predicen `y`; `expert_prob_sr` restringe la distribución de cada teacher seleccionado a su soporte `M[k,:]` antes de normalizar.

El grid principal es únicamente **T=8** y vive en `OUTPUTS/article1/results.csv`. Las sensibilidades de temperatura deben usar un CSV distinto, por ejemplo `OUTPUTS/article1/results_temperature.csv`; nunca se mezclan con el grid principal. La identidad de una ejecución incluye dataset, régimen, semilla, método y temperatura.

Una selección vacía (EXPERT, ORACLE o Consensus) usa exactamente el fallback FedDF: `softmax(mean_k z_k / T)`. La fila conserva el fallback explícito y sus métricas; no es una observación eliminada ni un cambio de método.

Los artefactos actuales `article1-v2` son particiones, checkpoints de teachers, un cache inmutable de logits y CSV canónicos. No se usan resultados ni notebooks históricos de `protocol_v1` como evidencia o validación de este contrato.

## Validación y reproducción

Audita el grid principal y sus caches así:

```bash
python -m article1.audit OUTPUTS/article1/results.csv \
  --source-root OUTPUTS/article1/sources \
  > OUTPUTS/article1/audit_report.json
```

Repite una sola celda KD, siempre fuera del CSV principal:

```bash
python -m article1.reproduce \
  --dataset cifar --seed 42 --method expert_logit \
  --cache OUTPUTS/article1/sources/cifar-seed42-alpha0p1/teacher_cache.npz \
  --results OUTPUTS/article1/reproducibility_check.csv \
  --device cuda
```

El único notebook definitivo activo es [`notebooks/article1_definitive_analysis.ipynb`](notebooks/article1_definitive_analysis.ipynb). Lee exclusivamente los CSV canónicos, valida el esquema y analiza de forma explícita T=8; no recalcula ni modifica resultados.
