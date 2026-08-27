# Artículo 1 — protocolo canónico v1

Esta carpeta es el punto de corte con `study_i`, `phase_a` y
`article_1_server_expertise`. El código legacy no se importa. La única entrada
histórica admitida son los checkpoints de teachers y sus índices originales de
partición, necesarios para identificar el holdout y el test que corresponden al
teacher entrenado.

## Contrato

Contrato exigido por cliente:

- `train_idx`: único split autorizado para ajustar parámetros del teacher;
- `holdout_idx`: único split autorizado para seleccionar el checkpoint y
  construir `M[k,c]`;
- `test_idx`: evaluación local posterior; nunca modifica `M`.

`prepare` valida la disjunción, copia los índices al árbol nuevo y vuelve a
calcular logits de holdout, test y proxy desde el checkpoint. No lee máscaras,
targets, students ni caches de logits agregados legacy.

La salida está fijada a:

```text
OUTPUTS/experiments/article_1/protocol_v1/
  sources/<dataset>/seed_<seed>/<regime>/
    source.json
    teacher_source.npz
    splits/client_XXX.npz
  runs/<dataset>/seed_<seed>/<regime>/tau_<threshold>/N_<size>/
    targets.npz
    target_metrics.json
    targets_extension__<method>.npz  # append-only target additions, if needed
    target_metrics_extension__<method>.json
    mechanism.npz
    students/<method>/metrics.json
  summary/
```

No se guardan checkpoints ni logits de test de students. Se guardan únicamente
el hash del estado final, las métricas de hipótesis y la provenance necesaria.

La reutilización de un teacher exige dos comprobaciones diferentes: integridad
del artifact migrado y evidencia de la ejecución histórica que produjo el
checkpoint. La auditoría consolidada se ejecuta con:

```bash
conda run -n FLWR python -m experiments.article_1.audit_teacher_provenance
```

Puede añadirse `--strict` en CI o antes de congelar resultados. En ese modo, la
auditoría falla si no siguen disponibles los checkpoints originales, su
`teachers_manifest.json` y su `run_config.yaml`. Los hashes y logits migrados
demuestran integridad, pero no sustituyen esos registros de entrenamiento.
La auditoría actual de `protocol_v1` conserva íntegros los artifacts migrados,
pero clasifica los teachers como `requires_regeneration_for_strict_provenance`
porque los registros legacy ya no están disponibles.

Si se acepta como premisa que esos teachers fueron generados correctamente, la
auditoría del pipeline posterior y de la cobertura experimental es:

```bash
conda run -n FLWR python -m experiments.article_1.audit_executions
```

Esta segunda auditoría recompone cada máscara desde holdout, comprueba los
subconjuntos anidados del proxy, los hashes de targets usados por los students,
la evaluación test-only, ORACLE-v2 y las comparaciones emparejadas EXPERT.

## Métodos

- `feddf`: media de logits de todos los teachers;
- `energy`: pesos adaptativos por sample basados en free energy, sin máscara;
- `support_only`: todos los teachers, restringiendo cada distribución a `M[k,:]`;
- `expert_full`: selección `M[k,y]=1`, distribución completa;
- `expert_v2`: misma selección, restringiendo cada teacher a `M[k,:]`;
- `oracle_full`: teachers cuyo argmax es la clase real, distribución completa;
- `oracle_v2`: one-hot de la clase real si algún teacher acierta; fallback FedDF si ninguno acierta;
- `supervised_matched`: one-hot con el mismo KL, temperatura, datos limpios,
  inicialización, orden y presupuesto que ORACLE-v2;
- `supervised_standard`: CE, augmentations, AdamW + cosine y 100 epochs.
- `expert_full_entropy_matched`: control mecanístico. Mantiene la selección
  EXPERT y la cola de clases de EXPERT-full, fija la probabilidad de la clase
  real a la de EXPERT-support y ajusta una potencia por ejemplo para igualar
  exactamente su entropía. Por tanto no usa test ni ajusta hiperparámetros; el
  contraste FULL→control cuantifica el cambio conjunto de
  sharpness/fidelidad de etiqueta y control→SUPPORT cuantifica retirar soporte
  con ambos valores fijados.

## Ejecución recomendada

Primero enumerar y después auditar las 54 fuentes sin GPU:

```bash
conda run -n FLWR python -m experiments.article_1.prepare --dry-run

conda run -n FLWR python -m experiments.article_1.prepare --audit-only
```

Preparar seed 42 —esto hace inferencia de los 10 teachers por condición—:

```bash
conda run -n FLWR python -m experiments.article_1.prepare \
  --seeds 42 --device cuda
```

Diagnóstico de mecanismo y comparación central, target-only:

```bash
conda run -n FLWR python -m experiments.article_1.run \
  --stage targets --preset central --seeds 42
```

Students centrales emparejados:

```bash
conda run -n FLWR python -m experiments.article_1.run \
  --stage students --preset central --seeds 42 --device cuda
```

Comparación final de apertura —FedDF / Energy / EXPERT-full / ORACLE-full,
N=10000—. Reutiliza los logits y students ya disponibles; sólo crea el
sidecar inmutable de Energy y entrena las celdas que falten:

```bash
conda run -n FLWR python -m experiments.article_1.run \
  --stage all --preset baseline --device cuda
```

ORACLE, sanity supervisado y curva de proxy:

```bash
conda run -n FLWR python -m experiments.article_1.run \
  --stage students --preset oracle --seeds 42 --device cuda

conda run -n FLWR python -m experiments.article_1.run \
  --stage all --preset curve --seeds 42 --device cuda
```

Ablación pequeña de threshold, primero target-only y después los students que se
decida confirmar:

```bash
conda run -n FLWR python -m experiments.article_1.run \
  --stage targets --preset threshold --seeds 42

conda run -n FLWR python -m experiments.article_1.run \
  --stage students --preset threshold --seeds 42 --device cuda
```

Una vez aceptado seed 42, preparar 43/44 y cerrar únicamente la comparación
central multi-seed:

```bash
conda run -n FLWR python -m experiments.article_1.prepare \
  --seeds 43 44 --device cuda

conda run -n FLWR python -m experiments.article_1.run \
  --stage all --preset central --seeds 43 44 --device cuda
```

Agregar tablas y puntos de cruce a 1/2 pp:

```bash
conda run -n FLWR python -m experiments.article_1.collect
```

### Control mecanístico FULL→SUPPORT

La mínima comparación explicativa es `expert_full`, `expert_v2` y
`expert_full_entropy_matched`, siempre con los mismos teachers, prefijo de
proxy, inicialización, orden de minibatches y presupuesto KD. Se priorizan las
celdas CIFAR-10 en que el efecto ya es visible: IID, `alpha1p0` y `alpha0p1`,
con seed 43 inicialmente. Los nuevos targets se guardan en sidecars inmutables
`targets_extension__*.npz`; el corpus principal no se reescribe.

```bash
conda run -n FLWR python -m experiments.article_1.run \
  --stage all --preset mechanism --datasets cifar --seeds 43 \
  --regimes iid alpha1p0 alpha0p1 --device cuda

conda run -n FLWR python -m experiments.article_1.collect
```

La sección 24 del notebook lee esos sidecars y exporta
`table_mechanism_entropy_control.csv`. Comprueba emparejamiento CRN, igualdad
de entropía y de probabilidad de clase real antes de mostrar los dos contrastes.
No debe interpretarse la comparación de seed único como evidencia de
generalidad; se replica sólo si los tres contrastes no dan una lectura clara.

## Eficiencia muestral del proxy (experimento final)

La curva histórica no responde esta pregunta: contiene EXPERT-support y sólo
seed 42. El análisis final usa los prefijos balanceados y anidados ya
persistidos; **cada prefijo está completamente etiquetado**. Los tamaños son
100, 250, 500, 1000, 2500, 5000 y 10000. Las únicas repeticiones son seeds 42,
43 y 44; IID, Dirichlet y especialización son condiciones de heterogeneidad y
nunca se agregan como repeticiones independientes.

El baseline principal es `feddf`, `expert_full`, `supervised_matched` y
`supervised_standard`. El control `supervised_matched` conserva init, orden,
datos limpios y presupuesto KD; `supervised_standard` es el comparador directo
CE de 100 épocas con augmentations. `oracle_full` queda como referencia
histórica en N=10000/seed 42, fuera de los efectos principales.

```bash
# Materializar los targets completos de los prefijos y entrenar sólo students
# ausentes (los artifacts compatibles existentes se omiten automáticamente).
conda run -n FLWR python -m experiments.article_1.run \
  --stage targets --preset proxy-efficiency
conda run -n FLWR python -m experiments.article_1.run \
  --stage students --preset proxy-efficiency --device cuda

# Producir los CSV/JSON autoritativos consumidos por el notebook final.
conda run -n FLWR python -m experiments.article_1.collect_proxy_efficiency
```

Energy es una extensión mask-free especificada limpiamente en este protocolo,
pero se ejecuta sólo después de cerrar el baseline anterior. Los resultados
Energy históricos no se mezclan con esta matriz: no son una curva canónica
multi-seed compatible. Si se decide ejecutar la extensión, primero actualizar
los targets para incluir su campo determinista y después entrenar Energy:

```bash
conda run -n FLWR python -m experiments.article_1.run \
  --stage targets --preset proxy-efficiency-energy --force
conda run -n FLWR python -m experiments.article_1.run \
  --stage students --preset proxy-efficiency-energy --device cuda
conda run -n FLWR python -m experiments.article_1.collect_proxy_efficiency \
  --include-energy
```

## Notebook de resultados

`notebooks/article1_main_findings.ipynb` es la fuente analítica del Artículo 1.
Lee exclusivamente `protocol_v1`, audita cobertura, leakage y emparejamiento,
organiza el análisis por preguntas científicas, clasifica los claims y genera
las figuras/tablas del paper. Está guardado con todas sus celdas ejecutadas.

Los derivados se exportan sin modificar resultados crudos a:

```text
OUTPUTS/experiments/article_1/paper_analysis/
  figures/  # PNG + PDF
  tables/   # CSV
```

Antes de una matriz costosa puede añadirse `--dry-run` a `run` para mostrar el
número exacto de conditions y trainings.
