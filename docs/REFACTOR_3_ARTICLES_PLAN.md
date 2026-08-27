# Plan concreto de migración a tres artículos

## Arquitectura objetivo

Se evoluciona `oracle_distillation/` como core compartido. Crear otro package
core duplicaría loaders, checkpoints y training ya estabilizados. Los entrypoints
legacy permanecen como compatibility layer hasta cerrar reproducción.

```text
oracle_distillation/                 # shared scientific core
  data/                              # futuro facade sobre data/dataset_config.py
  partitions/                        # futuro facade sobre o1_partitions/
  teachers/                          # trainer + artifact contracts
  proxy/
    splits.py                        # ya añadido: nested deterministic splits
  logits/                            # futuro alignment/cache contracts
  competence/                       # legacy_mask / clean_mask / authority
  targets/
    expertise.py                     # ya añadido: FULL/SUPPORT puro
    personalization.py               # futuro, desde target_builders.py
  distill/
    student_training.py              # futuro engine sin I/O
  metrics/
  reproducibility/

experiments/
  article_1_server_expertise/        # ya creado: piloto
  article_2_label_light/             # wrapper futuro sobre estudio_ii
  article_3_personalization/         # wrapper futuro sobre estudio_iii

analysis/
  article_1/
  article_2/
  article_3/

OUTPUTS/experiments/
  article1_support_v1/               # nuevo e inmutable
  article2_label_light_v1/           # futuro
  article3_personalization_v1/       # futuro
  study_i|phase_a12|phase_b/         # legacy, read-only
```

## Stage 0 — freeze y provenance

**Objetivo:** impedir colisiones y registrar qué evidencia existe.

- Nuevos: `docs/REFACTOR_3_ARTICLES_AUDIT.md`, validator de
  `experiments/article_1_server_expertise/artifacts.py`.
- Afectados: ninguno de `OUTPUTS/study_i`, `phase_a12`, `phase_b`.
- Compatibility: lectura por referencia; no symlinks ni copias.
- Tests: shape/finite/mask reconstruction/proxy equality/SHA-256.
- Migrado: solo metadata de referencia.
- Legacy intencional: todos los checkpoints, CSVs y notebooks históricos.

Gate de salida: inventario estable y `legacy_mask` etiquetado como leakage.

## Stage 1 — shared target core

**Objetivo:** separar teacher logits → target artifact → student.

- Nuevos: `oracle_distillation/targets/expertise.py`,
  `oracle_distillation/proxy/splits.py`.
- Modificados: ninguno de los builders legacy; evita alterar Study II/III.
- Compatibility: `T*log(q)` permite usar el loss KD existente sin cambiar su
  contrato.
- Tests: full mask, partial support, normalización, masa exterior, selección
  EXPERT, single one-hot, zero-support fail-closed, ORACLE gate/fallback.
- Artefactos: NPZ de target con probabilities, pseudo_logits, selection matrix,
  proxy indices, labels y hashes.

Gate de salida: causal pairs tienen selection matrix byte-identical.

## Stage 2 — Article 1

**Objetivo:** ejecutar solo el piloto CIFAR seed 42.

- Nuevos: `experiments/article_1_server_expertise/{config,artifacts,training,run}.py`.
- Outputs nuevos: `OUTPUTS/experiments/article1_support_v1/`.
- Compatibility: FedDF histórico se referencia si la receta es idéntica;
  EXPERT/ORACLE nuevos se reentrenan solo desde targets nuevos.
- Tests/smoke: multi, 32 proxy samples, un update por brazo en root temporal.
- Migrado: ninguna masa binaria; referencias por path+hash.
- Legacy intencional: Energy y los students EXPERT/ORACLE históricos.

Antes de matriz final: Stage 2b debe crear `clean_mask`. Sin ello, el piloto es
exploratorio.

## Stage 3 — Article 2

**Objetivo:** migrar label-light después de evaluar el piloto.

- Crear `experiments/article_2_label_light/` como protocolo; conservar
  `estudios/estudio_ii` como launcher legacy.
- Mover primitives puras de `core.py` (authority estimation, router, class
  targets) al core compartido.
- Reemplazar `build_class_targets` full-vector por la primitive de authority
  support, bajo un **nuevo** `experiment_version`; no cambiar resultados v3.
- Centralizar training/control supervised; split existente debe importarse con
  hash, no regenerarse silenciosamente.
- Tests: no U/test labels en fit, soft authority, A=ones parity, A=zero policy,
  nested budgets, matched updates.

Legacy intencional: `study_ii_v3_simple`, auto_v2 y hard-vote ablation.

## Stage 4 — Article 3

**Objetivo:** personalización local independiente de Article 2.

- Crear `experiments/article_3_personalization/` que acepte un
  `global_student_artifact` genérico.
- Extraer math de `analysis/target_builders.py` a
  `targets/personalization.py`.
- Extraer `_eval_local_surface`, `_synth_analysis` y training de helpers privados
  de CLI a módulos públicos.
- Protocolo mínimo: global, fine-tuning, local teacher KD, class_mask,
  proxy_plain; LwF opcional predeclarado.
- Métricas: known/unknown/overall global y local-real, con clientes como unidades
  dependientes, no réplicas independientes.

Legacy intencional: Study IV bridge y Phase B six-arm.

## Stage 5 — análisis y limpieza

**Objetivo:** recuperar análisis por artículo sin romper evidencia.

- Recuperar primero `ANALYSIS_v2/`/`ANALYSIS/` ausentes.
- Crear notebooks/scripts bajo `analysis/article_{1,2,3}` que consuman schemas
  versionados, nunca paths Phase A hardcoded.
- Generar vistas expandidas de `regime=shared` solo en análisis.
- Marcar outputs legacy `read-only` mediante manifest, sin moverlos.
- Absorber el shim `paths.py` restaurado en el registry y retirarlo de forma
  controlada solo después de eliminar todos sus imports legacy.

## Migration map

| Old path | New role/path | Acción |
|---|---|---|
| `oracle_distillation/analysis/proxy_analysis_builder.py` | logits orchestration + legacy target wrapper | adelgazar por etapas; no romper |
| `_compute_global_targets` | legacy full-logit implementation | congelar y testear parity |
| `oracle_distillation/analysis/target_builders.py` | futuro `targets/personalization.py` | mover con re-export legacy |
| `o1_partitions/partitioner.py` | futuro `oracle_distillation/partitions/` | facade primero, movimiento después |
| `o3_local/teacher_trainer.py` | futuro `oracle_distillation/teachers/` | facade; añadir clean calibration split |
| `estudios/estudio_ii/core.py` | futuro core authority/router | Stage 3, con wrappers |
| `estudios/estudio_ii/controls.py` | shared supervised training | Stage 3; receta ya reutilizada conceptualmente |
| `estudios/estudio_iii/` | `experiments/article_3_personalization/` | wrapper nuevo; legacy queda |
| `estudios/estudio_iv/` | bridge histórico Article2→3 | archivar, no narrativa principal |
| `OUTPUTS/experiments/study_i/` | Article 1 legacy evidence/provenance | read-only |
| `OUTPUTS/experiments/phase_a12/` | Article 2 legacy evidence | read-only |
| `OUTPUTS/experiments/phase_b/` | Article 3 legacy evidence | read-only |
| `ANALYSIS(_v2)/` | análisis legacy ausente | recuperar antes de cleanup |

## Estrategia de compatibilidad

1. Ningún rename masivo.
2. Nuevas APIs puras; wrappers antiguos siguen importando sus módulos actuales.
3. Un cambio de semántica siempre incrementa `experiment_version` y cambia la
   KEY del CSV.
4. Outputs derivados pueden regenerarse; inputs históricos jamás se sobrescriben.
5. Deprecación solo después de un test de parity y de reconstruir el índice de
   artefactos/notebooks.
