# Auditoría para el refactor en tres artículos

Fecha de inspección: 2026-08-19. Repositorio inspeccionado: `KD_v5` (los
artefactos conservan rutas absolutas históricas a `KD_v4`). Esta auditoría se
basa en código y artefactos, no solo en READMEs.

## Resumen ejecutivo

El repositorio ya contiene un core parcialmente consolidado en
`oracle_distillation/`, pero los protocolos siguen nombrados por fases y los
límites entre mecanismo, protocolo e I/O son irregulares. La estrategia de
menor riesgo es **evolucionar ese package y mantener wrappers legacy**, no
crear un segundo core paralelo.

El hallazgo científico crítico es inequívoco: la máscara histórica `M*` se
deriva de `local_test_acc_per_class` y ese `local_test_idx` se usa después como
superficie de evaluación local. Esto es leakage de selección respecto a esa
evaluación. Los resultados históricos se conservan como `legacy_mask`, pero no
deben presentarse como evidencia final publicable. El protocolo limpio debe
crear un `calibration_idx`/holdout dedicado y reservar `local_test_idx` para una
única evaluación final.

Los cuatro regímenes CIFAR seed 42 del piloto conservan
`teacher_logits_cache[10000,10,10]`, máscara `[10,10]`, labels e índices proxy.
Por tanto, FULL/SUPPORT puede ejecutarse sin reentrenar teachers. Los
checkpoints también existen, pero no son necesarios para construir targets.

## 1. Arquitectura real

El checkout difiere de la descripción inicial:

- no contiene `ANALYSIS_v2/` ni `ANALYSIS/`;
- sí contiene `OUTPUTS/` (~14 GB), `oracle_distillation/`, `estudios/`,
  `o1_partitions/`, `o3_local/` y datos locales;
- no contenía un `paths.py` raíz, aunque `data/dataset_config.py` y
  `o1_partitions/partitioner.py` todavía lo importan. Esta tarea restauró un
  shim mínimo y relocatable de constantes para recuperar compatibilidad;
- `.git/` es un directorio de solo lectura incompleto en este workspace, por lo
  que no se pudo obtener status/commit desde git. La provenance histórica sí
  registra commits, por ejemplo `196a171...` para controles Phase A1/2.

Árbol funcional actual:

```text
data/
  dataset_config.py          loaders, transforms, defaults por dataset
  proxy_splits.py            split proxy/pool por seed
o1_partitions/
  partitioner.py             iid/Dirichlet/single/multi y split local 3-way
o3_local/
  teacher_trainer.py         training/evaluación/cache de logits de teachers
oracle_distillation/
  analysis/
    proxy_analysis_builder.py  teacher pass, máscara y targets globales legacy
    target_builders.py         accessors globales + targets personales
  distill/
    kd_runner.py             training global/personal y persistencia
    losses.py                KD KL y CE
    kd_metrics.py            métricas post-training
  cli/
    run_dirichlet_alphas_methods.py  pipeline Study I
    run_personal_kd.py               personalización legacy
    run_transfer_set_ablation.py     segunda KD/transfer sets
  models/, provenance.py, metrics_io.py, run_paths.py, ...
estudios/
  estudio_ii/                label-light + controles
  estudio_iii/               wrapper de personalización/headroom
  estudio_iv/                bridge shared-q → class_mask
OUTPUTS/experiments/
  study_i/                   matriz histórica y teachers/logits
  phase_a12/                 label-light y supervised históricos
  phase_b/                   personalización/transfer histórico
```

## 2. Entrypoints y runners reales

| Entrypoint | Responsabilidad real | Estado |
|---|---|---|
| `python -m oracle_distillation.cli.run_dirichlet_alphas_methods` | partición → teachers → proxy analysis → KD global | Study I reproducible; mezcla protocolo e I/O |
| `python -m oracle_distillation.cli.run_personal_kd` | KD personal por cliente | motor histórico necesario para Article 3 |
| `python -m oracle_distillation.cli.run_transfer_set_ablation` | segunda KD sobre proxy/local/mix/LwF | motor histórico, demasiado amplio para protocolo final |
| `python -m estudios.estudio_ii.run` | offline targets label-light y students | Article 2 futuro; no migrar aún |
| `python -m estudios.estudio_ii.controls` | ORACLE-LN, supervised matched, self-learning | contiene lógica supervisada reutilizable |
| `python -m estudios.estudio_ii.validate` | validación Study II | mantener legacy |
| `python -m estudios.estudio_iii.run` | wrapper `headroom/personal/transfer` | Article 3 candidato; apunta a `phase_a`, no a `study_i` |
| `python -m estudios.estudio_iv.run` | reconstruye bases Study II y personaliza | bridge histórico; narrativamente mezcla Articles 2 y 3 |
| `python -m experiments.article_1_server_expertise.run` | piloto FULL/SUPPORT nuevo | añadido por esta tarea |

`estudios/estudio_iii/run.py` construye paths bajo
`OUTPUTS/experiments/phase_a/`, mientras que los outputs presentes están bajo
`study_i/`. Debe tratarse como incompatibilidad de layout, no como ausencia de
artefactos.

## 3. Mapa del flujo científico

| Operación | Implementación real | Notas |
|---|---|---|
| cargar datasets/transforms | `data/dataset_config.py` | centralizado; algunas propiedades importan el `paths.py` ausente |
| crear proxy/pool | `data/proxy_splits.py` | 10k, determinista por seed, balanceado para datasets flat |
| particionar clientes | `o1_partitions/partitioner.py` | RNG separado para asignación y split 3-way |
| entrenar teachers | `o3_local/teacher_trainer.py:TeacherTrainer` | mismo estado inicial para todos los teachers de una réplica |
| early stopping teacher | `_train_model`, usando `holdout_idx` | correcto: holdout no es `local_test` |
| calcular logits proxy | `TeacherTrainer.train_client` o `ProxyAnalysisBuilder` | nuevos runs guardan cache por teacher; históricos guardan cache conjunto |
| construir `M*` | `build_expert_mask_from_manifest` | primero `local_test_acc_per_class`, fallback a holdout |
| targets globales legacy | `_compute_global_targets` | media de **logits**, EXPERT/ORACLE con fallback FedDF |
| targets personales | `analysis/target_builders.py` | mezcla matemática personal y dispatch; no CLI |
| training student | `distill/kd_runner.py` | core compartido, pero el path global posee demasiado I/O |
| métricas | `distill/kd_metrics.py`, `metrics.py`, `utils.py` | hay solapamiento de helpers CE/accuracy/logits |
| seeds/CRN | `utils.set_seed`, `_HashedRandomSampler`, seeds ad hoc en Phase B/Study IV | mecanismos buenos pero no centralizados en un contrato único |
| splits label-light | `estudio_ii.core.nested_stratified_order` | un orden por dataset × seed; budgets anidados |
| CSV | `metrics_io.append_row`, writers propios de Study II/IV | tres implementaciones distintas |
| checkpoints/provenance | `checkpoints.py`, `provenance.py`, manifests por runner | infraestructura valiosa pero adopción desigual |

## 4. Auditoría exacta de `M*`

### Implementación

```text
datos que determinan M*: local_test_idx de cada cliente
función de evaluación: o3_local.teacher_trainer.test_with_per_class_accuracy
persistencia raw: teachers_manifest.json → entry["crea"]["local_test_acc_per_class"]
función que umbraliza: oracle_distillation.analysis.proxy_analysis_builder.build_expert_mask_from_manifest
split preferido: local_test
fallback legacy: holdout, solo si falta local_test_acc_per_class
umbral CIFAR-10: 0.7 (data.dataset_config._CIFARConfig)
uso posterior de esos datos: evaluación local en run_personal_kd / transfer ablation
```

El particionador crea conjuntos disjuntos en este orden:

```text
client allocation
  ├── local_test_idx = primer 20% histórico
  └── resto
       ├── holdout_idx = 15% del resto
       └── train_idx
```

El teacher entrena con `train_idx`, hace early stopping con `holdout_idx`, evalúa
per-class en `local_test_idx`, y esa evaluación genera `M*`. Más tarde
`local_test_idx` vuelve a usarse para medir utilidad local. Por tanto:

- no hay leakage hacia el test global de CIFAR-10;
- sí hay leakage/model selection hacia la métrica local final;
- todos los resultados EXPERT/ORACLE-maskgated/class_mask que consuman esa
  máscara están afectados para claims sobre local test;
- también hay optimismo indirecto al describir `M*` como autoridad conocida
  independiente de evaluación.

### Semánticas que deben coexistir

- `legacy_mask`: máscara histórica desde `local_test_acc_per_class`; sirve para
  reproducibilidad y para el piloto mecanístico, siempre etiquetada como leakage-affected.
- `clean_mask`: máscara desde un `calibration_idx` dedicado, nunca reutilizado
  como test. `local_test_idx` permanece sellado hasta evaluación final.

Propuesta limpia para nuevas réplicas:

```text
client allocation
  ├── train_idx
  ├── optimization_holdout_idx   # early stopping
  ├── competence_calibration_idx # calcula accuracy teacher–clase y M*
  └── local_test_idx             # evaluación final una sola vez
```

No debe reconstruirse `clean_mask` a partir del holdout histórico si ese holdout
ya intervino en early stopping: sería menos grave, pero seguiría acoplando
selección de checkpoint y autoridad. Hay que generar un split explícito.

## 5. Targets y fallbacks históricos

`ProxyAnalysisBuilder._compute_global_targets` agrega en espacio de logits:

- FedDF: media de logits de todos los teachers;
- ORACLE: media de logits de teachers cuyo argmax coincide con `y`;
- EXPERT: media de logits de teachers con `M[k,y]=1`;
- selección vacía: `avg_logits` (FedDF histórico), no hard vote.

El piloto nuevo implementa las ecuaciones solicitadas en espacio probabilístico
para EXPERT/ORACLE y conserva como fallback exactamente
`softmax(mean(logits)/T)`. Esto implica una incompatibilidad deliberada y
documentada: `expert_full` nuevo es la media de distribuciones teacher a T=8,
no `softmax(media de logits/T)`. Es la comparación causal definida en el nuevo
protocolo; los resultados `expert` antiguos no deben renombrarse como
`expert_full` nuevo sin recomputar el target/student.

El fallback histórico es científicamente discutible porque una celda sin
autoridad pasa a una referencia no informada, pero es estable y se registra. En
CIFAR seed 42, el cache histórico reporta 3,949 fallbacks EXPERT en `iid` y cero
en `alpha0p1`, `multi` y `single`.

## 6. Código compartido, duplicado y mezclado

### Buen core ya existente

- registry de datasets/modelos;
- particiones y loaders;
- checkpoint/provenance;
- loss KD;
- caches de logits alineados por `proxy_idx`;
- sampler con hash de orden;
- builders personales matemáticos.

### Duplicación/deuda

- softmax estable y métricas NLL/accuracy aparecen en `utils.py`,
  `proxy_analysis_builder.py`, Study II y sus controles;
- training AdamW aparece en `KdRunner`, `estudio_ii.run`, `controls.py` y Study IV;
- CSV atómico/deduplicación se implementa en `metrics_io.py`, Study II y Study IV;
- resolución de paths existe en `run_paths.py`, `experiment_paths.py` y lógica
  local de cada estudio;
- `proxy_analysis_builder.py` mezcla forward de teachers, resolución de máscara,
  seis targets, diagnósticos y escritura NPZ/JSON;
- `KdRunner.distill_global` mezcla dataset, target I/O, training, evaluación,
  checkpoint, logits cache y manifests;
- `controls.py` mezcla receta científica, CLI, training, evaluación y CSV;
- Study IV importa helpers privados (`_...`) de dos CLIs, señal de límites
  inestables.

## 7. Dependencias entre áreas

```text
data + o1_partitions
        ↓
o3_local.teacher_trainer
        ↓ teachers_manifest + teacher logits/checkpoints
oracle_distillation.analysis.proxy_analysis_builder
        ↓ proxy_analysis.npz
oracle_distillation.distill.KdRunner
        ├── Study I global
        ├── personal CLI / Study III
        └── Study IV

Study II ──lee──> Study I proxy_analysis.npz
Study IV ──lee──> Study II checkpoints + Study I teachers/masks
notebooks ausentes en este checkout ──históricamente leían──> CSV/NPZ outputs
```

Article 3 no debería depender en el futuro de Study II: el bridge Study IV se
archiva como experimento específico, mientras un protocolo Article 3 acepta
cualquier `global_student_artifact` compatible.

## 8. Artefactos reutilizables

| Artifact | Existing path | Can reuse? | Reason | New reference |
|---|---|---:|---|---|
| teacher checkpoints | `.../study_i/seed_42/raw_work/runs/cifar10/K10/<regime>/teachers/cid_*.pt` | sí, condicional | 10 por régimen, arquitectura ResNet9; no hacen falta para targets | provenance only |
| teacher logits | `.../<regime>/proxy_analysis/proxy_analysis.npz::teacher_logits_cache` | **sí** | `[10000,10,10]`, finitos y alineados | target artifact guarda path + SHA-256 |
| proxy indices/labels | mismo NPZ y `data/splits/cifar10_train_proxy_10000_seed_42.npz` | **sí** | idénticos entre cuatro regímenes | split/provenance Article 1 |
| competence mask | mismo NPZ + manifest | sí como `legacy_mask` | reconstruible exactamente con threshold 0.7; leakage-affected | `mask_source=local_test...` |
| student init | implícito por `set_seed(42)` | no como artefacto | no se persistió state inicial independiente | se regenera y hashea |
| FedDF student/result | `distillation/feddf/metrics.json` | **sí** | receta exacta 30×ceil(10k/256), T=8, AdamW | nueva fila de referencia, sin copiar checkpoint |
| Energy result | `distillation/energy/` | sí como referencia histórica | mismo proxy/recipe; fuera de brazos obligatorios del primer piloto | análisis posterior |
| EXPERT/ORACLE students | `distillation/{expert,oracle}` | no para FULL nuevo | agregaron logits, no probabilidades; semántica distinta | recomputar target + student |
| supervised N=100…5000 | `phase_a12/seed_42/baselines/*.csv` | solo referencia | no se persistieron índices/order de entrenamiento verificables; standard carece de NLL | rerun canónico |
| Study-II nested splits | no presentes bajo `OUTPUTS/experiments/study_ii/` | no | documentación/código existen, artefacto no | nuevo split persistido |

Los cuatro artefactos exactos del piloto son `iid__40k-10k-0p15`,
`alpha0p1__40k-10k-0p15`, `multi__40k-10k-0p15` y
`single__40k-10k-0p15`.

## 9. Nombres de fase todavía embebidos

- paths: `phase_a12/`, `phase_b/`, referencias a `phase_a/`;
- IDs: `phase_a_methods`, `phase_a12_auto_v2`,
  `phase_b_classmask_causal` en registries/provenance;
- funciones/comentarios: “Phase-3 diagnostics”, “Extended Phase-5 metrics”,
  “Phase-B causal pair”;
- defaults de Study II: argumento `--phase-a-root`;
- `_phase_a_root` en Study III;
- columnas/configs históricos `group_alpha` y `phase_a_*`.

No deben renombrarse in-place: son claves de trazabilidad. Los protocolos nuevos
usan `article`, `experiment_version` y nombres científicos, manteniendo esas
rutas como inputs legacy.

## 10. Riesgos antes de una matriz final

1. **Mask leakage (blocker publicable):** crear clean calibration split y nuevos
   teachers/máscaras, o limitar claims del piloto a evidencia exploratoria.
2. **Espacio de agregación:** histórico = logits; protocolo nuevo = probabilidades
   a T. No mezclar filas.
3. **Temperatura:** support se calcula sobre `softmax(z/T)` con T=8 y los
   pseudo-logits se construyen como `T log q`. Aplicar T de nuevo de otra forma
   cambiaría el target.
4. **Checkpoints:** muchos students históricos no tienen `.pt`; existen logits y
   métricas. Teachers CIFAR seed 42 sí tienen `.pt`.
5. **Proxy mismatch:** el nuevo validator exige índices/labels idénticos entre
   regímenes y registra SHA-256.
6. **Paths:** el shim restaurado desbloquea imports, pero a largo plazo las
   constantes deben integrarse en el registry sin reintroducir absolutos.
7. **Notebooks ausentes:** no limpiar ni declarar migrada la capa de análisis
   hasta recuperar `ANALYSIS(_v2)` o reconstruirla desde outputs.

## 11. Qué archivar y qué convertir en core

Archivar sin borrar: Phase A1/2 `auto_v2`, hard-vote fallback, matriz histórica
Study I, Study IV bridge y Phase B six-arm. Mantener runners como wrappers de
reproducción mientras sus artefactos sean citados.

Convertir en core: support/authority probabilística, selección/aggregación,
diagnósticos de target, splits proxy anidados, training student con presupuesto
exacto, schema/provenance y loaders/cache alignment. Mantener separadas las
transformaciones server-target (Articles 1/2) de personalización client-target
(Article 3).
