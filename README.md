# KD_v5__ — Article 1

Repositorio pequeño de investigación sobre **one-shot federated knowledge distillation**. El objetivo es determinar qué aporta una máscara de competencia cliente–clase y cómo deben contribuir los expertos al target del estudiante global.

**Estado: 2026-09-08. Protocolo activo: `article1-v3`.** Esta versión cambia el reparto local a `train / validation / expertise = 70/10/20`, con estratificación por clase y estimación de expertise independiente de la selección del checkpoint. No contiene resultados científicos nuevos de v3. Los resultados previos de v2 son evidencia histórica; no se atribuyen al nuevo protocolo.

## 1. Alcance del artículo

- MNIST, Fashion-MNIST y CIFAR-10; 10 clientes; seeds 42, 43 y 44.
- Un entrenamiento local por teacher, un cache común de salidas sobre el proxy y un único student global. Sin rondas posteriores de comunicación.
- Proxy público etiquetado y máscara binaria disponibles. EXPERT no es label-free.
- Umbrales previos: MNIST 0.90, Fashion-MNIST 0.80, CIFAR-10 0.70. No se declaran óptimos ni se ajustan sobre test para mejorar resultados o eliminar fallbacks.
- Cuatro bloques secuenciales: selección oracular, operador, expertise y soporte; supervisado como extensión independiente.

| Pregunta | Contraste | Se mantiene fijo |
|---|---|---|
| Selección idealizada | `oracle_logit − feddf_logit` | Operador logit y presupuesto; ORACLE selecciona aciertos, no competencia acreditada |
| Operador | `feddf_prob − feddf_logit`; `oracle_prob − oracle_logit` | Selección y temperatura |
| Propuesta EXPERT | `expert_prob − feddf_prob`; gap `oracle_prob − expert_prob` | Operador probabilístico |
| Soporte | `expert_prob_sr − expert_prob` | Routing EXPERT y operador probabilístico |


Inferencia de máscaras desde modelos, proxies sin etiquetas, competencia continua y segunda destilación/personalización quedan para trabajos posteriores. Cada nueva variante necesita una pregunta independiente.

## 2. Reparto entre clientes

Se reserva primero un proxy de 10.000 ejemplos del entrenamiento oficial, exactamente 1000 por clase. Quedan 50.000 ejemplos privados en MNIST/Fashion-MNIST y 40.000 en CIFAR-10. El test oficial permanece separado.

| Régimen | Reparto privado |
|---|---|
| `iid` | Cada clase se divide casi por igual entre los 10 clientes |
| `alpha1p0`, `alpha0p5`, `alpha0p1` | Proporciones Dirichlet entre clientes para cada clase; también varían los tamaños totales de cliente |
| `multi` | Cinco pares disjuntos de clases: (0,1), (2,3), (4,5), (6,7), (8,9); dos clientes por par |
| `single` | Una clase por cliente |

El orden gráfico es **IID → α=1.0 → α=0.5 → α=0.1 → Multi → Single**, como eje categórico. No expresa distancias numéricas iguales ni aísla causalmente solo heterogeneidad de clases: Dirichlet también cambia cantidades y `multi` fija parejas concretas.

Esta revisión mantiene `balanced_order` y `_assign`: no modifica el proxy ni la asignación privada entre clientes para el mismo dataset y seed. Cambia la división dentro de cada cliente.

## 3. Reparto dentro de cada cliente: v3

| Conjunto | Objetivo por clase | Uso autorizado |
|---|---:|---|
| `train_idx` | 70% | Gradientes y actualización de parámetros del teacher |
| `validation_idx` | 10% | Accuracy global de selección de checkpoint y early stopping |
| `expertise_idx` | 20% | Accuracy y recuentos por clase del teacher ya seleccionado y congelado; construcción de M |

No existe `test_idx` local en v3. La evaluación final del student utiliza el test oficial. El reparto 70/10/20 es una decisión de diseño, no una proporción demostrada óptima.

### Estratificación y clases raras

`article1/partitioning.py::_split_client` baraja los ejemplos de cada clase local usando una RNG por seed/cliente. Calcula los tamaños enteros por **restos mayores** de 70/10/20. Los empates priorizan train, después expertise y después validation. Al terminar se baraja cada split.

- No se equilibran clases entre sí: se preserva aproximadamente la composición de cada cliente.
- El error de redondeo por clase/split es menor que un ejemplo.
- No se duplica ningún ejemplo ni se fuerza presencia de una clase en los tres conjuntos.
- Una clase con un único ejemplo queda en train. Las clases raras pueden carecer de observaciones en validation o expertise.
- Si un cliente queda con cualquiera de los tres conjuntos completamente vacío, se detiene la preparación con un error. No hay remuestreo ni transferencia silenciosa de datos. Cualquier cambio de política requiere una decisión explícita.

### Invariantes y persistencia

`validate_splits` comprueba índices enteros no negativos, conjuntos locales no vacíos, ausencia de duplicados/solapamientos entre roles, clientes y proxy, y cobertura exacta del entrenamiento oficial.

`save_partitions` guarda `proxy.npz`, diez `client_*.npz` y un `metadata.json` con protocolo, fracciones, regla de redondeo, tamaño/huella de las etiquetas oficiales, commit y manifiesto SHA256 de los archivos de índices. Rechaza directorios no vacíos.

`load_partitions` verifica ese manifiesto, la versión, las funciones de los conjuntos, la cobertura y, cuando se suministran, las etiquetas oficiales. Teachers y notebook utilizan esta misma lectura.

## 4. Entrenamiento, máscara y evaluación

El flujo en `article1/local_training.py::train_and_cache` es:

1. Verificar identidad y huellas de las particiones con las etiquetas de entrenamiento oficial.
2. Entrenar cada teacher solo con `train` mediante Adam, lr=1e-3, batch 64, máximo 50 épocas.
3. Evaluar en `validation` tras cada época. Seleccionar mejora estricta de accuracy global; en empate se conserva el checkpoint anterior. Patiencia: 5.
4. Restaurar el checkpoint seleccionado. No volver a entrenar sobre train+validation ni actualizar sus estadísticas durante las evaluaciones.
5. Evaluarlo una vez en `expertise` con transformaciones deterministas. Guardar accuracy y recuentos por clase.
6. Producir logits del mismo checkpoint sobre el proxy y construir M usando exclusivamente los estadísticos de expertise.

CIFAR-10 usa augmentation solo para entrenamiento local; proxy, validation y expertise usan la vista determinista. MNIST/Fashion-MNIST no tienen augmentation aleatoria. No se evalúa el teacher sobre el test oficial.

La máscara es:

\[
M_{k,c}=\mathbf 1[n^{E}_{k,c}>0\land Acc^{E}_{k,c}\geq\tau_d].
\]

`article1/distillation.py::authority_from_expertise` implementa esta regla. M=0 significa **competencia no acreditada**, no incompetencia demostrada. Los recuentos distinguen ausencia de observaciones y precisión insuficiente. No se añade un mínimo muestral ni un estimador continuo en esta revisión.

El cache contiene `logits[N,K,C]` float32, `labels`, `proxy_idx`, `M`, `expertise_accuracy` y `expertise_counts`. El metadata registra época seleccionada, épocas ejecutadas, accuracy de validación, hashes de checkpoints, procedencia de particiones y `M_source=expertise_accuracy_and_counts_only`. No contiene estadísticas de un test local.

### Qué significa cada evaluación

- Validation selecciona el teacher; su accuracy está condicionada por esa selección.
- Expertise estima competencia para construir el sistema. **No es un test final**.
- Las métricas del target sobre proxy describen la señal utilizada para entrenar al student.
- Accuracy y NLL del student final en el test oficial evalúan el procedimiento completo.

No se usan test ni métricas de student para seleccionar checkpoints, umbrales, temperaturas o soporte. Si en el futuro se selecciona un checkpoint del student, hará falta una validación del servidor separada. Los presupuestos actuales son fijos.

Evaluar M sobre el mismo expertise que la construye no valida independientemente su generalización. Una afirmación específica sobre esa generalización requeriría un cuarto conjunto local separado desde el principio. Reiniciar el procedimiento no convierte en desconocido un test ya consultado: se conserva el historial experimental y no se presenta como una preregistración anterior a esos resultados.

## 5. Targets canónicos

Sean `p_k = softmax(z_k/T)`, `L(w)=softmax(sum_k w_k z_k/T)` y `P(w)=sum_k w_k p_k`. Los pesos suman uno sobre teachers.

| Método | Selección/pesos | Target |
|---|---|---|
| `feddf_logit` | Uniformes, todos | L(w) |
| `feddf_prob` | Uniformes, todos | P(w) |
| `expert_logit` | Uniformes entre M[k,y]=1 | L(w) |
| `oracle_logit` | Uniformes entre argmax(z_k)=y | L(w) |
| `confidence_logit` | MSP a T=1, seguido de softmax entre teachers | L(w) |
| `consensus_logit` | Clase argmax de la media de probabilidades a T=1; teachers cuyo argmax coincide | L(w) |
| `energy_logit` | logsumexp(z_k) a T=1, seguido de softmax entre teachers | L(w) |
| `expert_prob` | Mismo routing EXPERT | P(w) |
| `oracle_prob` | Mismo routing ORACLE | P(w) |
| `expert_prob_sr` | Mismo routing EXPERT | Enmascarar y renormalizar cada p_k por M antes de promediar |

Un conjunto seleccionado vacío usa el mismo fallback **FedDF-logit**, también en brazos prob/SR. Se contabiliza y se comprueba igualdad de targets. Consensus puede quedar vacío aunque haya diez teachers.

FedDF es una adaptación one-shot de su componente de ensemble distillation, no una reproducción del protocolo iterativo completo. Confidence, Consensus y Energy son controles internos. Energy depende de escalas y offsets aditivos de logits; su peso no acredita expertise. ORACLE usa etiquetas y no es un límite superior garantizado del rendimiento del student.

EXPERT-prob es el método base propuesto: selecciona teachers con M[k,y]=1 y conserva su distribución completa. Poner clases a cero y renormalizar define EXPERT-prob-SR, una ablación distinta. En logit aggregation se ponen a cero pesos de teachers excluidos, no coordenadas de clase.

SR revisión 2 calcula softmax dentro del soporte (restando el máximo permitido); equivale matemáticamente a enmascarar probabilidades y renormalizar, pero evita descartar teachers por masa menor que EPS o por underflow. Sus filas registran `target_revision=2`, su run_id cambia y la auditoría rechaza revisiones anteriores. No se reentrenan teachers por esta corrección; solo las KD SR afectadas. Las identidades de los demás métodos se conservan.

Logits y probabilidades no son operadores equivalentes. En SR, la probabilidad normalizada de la clase verdadera no puede disminuir para un experto seleccionado; mejorar NLL del target es parcialmente consecuencia de la construcción y no prueba que el conocimiento eliminado perjudique KD. No se implementan class-supported ni ORACLE+SR.

## 6. KD, supervisado y tamaños del proxy

`article1/runner.py::_train_proxy` comparte arquitectura, inicialización, orden, transformaciones, optimizador y evaluación entre CE y KD. AdamW, lr=1e-3, weight decay=1e-4, batch solicitado 256, sin scheduler. La receta KD principal es T=8, 30 épocas: con 10.000 ejemplos son 1200 actualizaciones.

KD consume q directamente con `T² KL(q || softmax(student/T))`, sin segundo softmax del target. CE consume etiquetas enteras. El student se evalúa al terminar; su NLL usa logits sin dividir por T.

`article1/proxy.py::proxy_positions` permite curvas de tamaño N: subconjuntos anidados estratificados del proxy ya reservado, sin cambiar teachers ni datos privados. El proxy completo conserva su orden. No confundir `distill/supervised --proxy-size` con `partition --proxy-size`, que cambia la reserva de datos.

Para tamaños reducidos se exige `--updates` explícito en KD. Supervisado usa 1200 por defecto. N<256 reduce el batch efectivo a N; se conserva el batch final incompleto. Presupuesto igualado no significa convergencia óptima ni idéntico número de ejemplos vistos entre tamaños.

Se registran tamaño/hash del proxy maestro y seleccionado, etiquetas, política de selección, receta JSON, inicialización, estado final, updates, ejemplos vistos y épocas. `consumed_batches_sha256` incluye índices realmente consumidos y fronteras de batch. `batch_order_sha256` conserva la huella de permutaciones completas.

La identidad supervisada excluye régimen y teachers: puede reutilizarse entre regímenes si coinciden dataset, seed, datos, inicialización y receta. No contarlo como réplicas independientes. La identidad KD incluye cache, máscara, proxy, método, temperatura y receta. Las filas de protocolos diferentes no se mezclan. Una invocación directa repetida vuelve a entrenar y hace upsert por run_id, salvo que se indique `--skip-existing`: entonces solo reutiliza una fila completa con exactamente esa identidad.

## 7. Organización

| Archivo | Responsabilidad |
|---|---|
| `article1/__init__.py` | Versión activa, datasets, regímenes, seeds y umbrales |
| `article1/partitioning.py` | Reserva pública, reparto entre clientes, estratificación local y manifiesto |
| `article1/datasets.py`, `models.py` | Datos, transformaciones y arquitecturas: MnistNet o ResNet9 |
| `article1/local_training.py` | Train → selección en validation → expertise del checkpoint congelado |
| `article1/distillation.py` | Máscara, routing, targets y pérdida KD |
| `article1/runner.py`, `proxy.py` | CLI, runtime común CE/KD y selección N |
| `run_article1_grid.py` | Plan explícito, dry-run y reutilización de ejecuciones compatibles |
| `run_article1_pipeline.py` | Flujo completo v3 con comprobación de las 54 particiones, piloto, barridos, auditorías y notebooks |
| `article1/audit.py`, `conditions.py`, `rq2.py` | Auditoría de fuentes, tabla de condiciones y control de temperatura |
| `article1/analysis.py` | Contrastes emparejados y gráficas de resultados |
| `article1/reproduce.py`, `hashes.py` | Reproducción de una celda y huellas |
| `notebooks/article1_partition_diagnostics.ipynb` | Heatmaps de reparto real y diagnóstico de soporte |
| `notebooks/article1_definitive_analysis.ipynb` | Análisis de resultados v3 cuando esté completa la evidencia necesaria |
| `tests/` | Invariantes NumPy y comprobaciones sintéticas CPU |

`README.md` es la única documentación explicativa. Los notebooks contienen las celdas de análisis y las instrucciones necesarias para interpretarlas, sin duplicar una especificación del método.

## 8. Uso paso a paso

Python 3.11+. Instalar en un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Las dependencias están consolidadas en `requirements.txt`; no se requieren los antiguos requirements-dev/analysis. El notebook de particiones carga etiquetas de torchvision y necesita PyTorch/torchvision, además de Jupyter. Las pruebas de runtime se omiten explícitamente si faltan esas dependencias.

### Ejecución por fases: diseño mínimo

Sin argumentos, `run_article1_pipeline.py` muestra la secuencia principal. Una
fase requiere `--phase` y `--execute`; termina sin lanzar las posteriores.

| Fase principal | Ejecución nueva desde cero | Archivo T=8 |
|---|---|---|
| `partitions` | 54 particiones y heatmaps | Índices compartidos v3 |
| `pilot` | MNIST/alpha0p1/42: teachers, FedDF-logit y ORACLE-logit; dos repeticiones técnicas de ORACLE | Las dos KD se reutilizan en selección |
| `teachers` | Completar las 54 condiciones, 540 teachers; máscaras y logits | Caches compartidos v3 y conditions.csv |
| `rq1` | 108 KD: FedDF-logit y ORACLE-logit | results_selection.csv |
| `aggregation` | 108 KD: FedDF-prob y ORACLE-prob | results_pooling.csv |
| `expertise` | 54 KD: EXPERT-prob | results_expertise.csv |
| `support` | 54 KD: EXPERT-prob-SR, revisión 2 | results_support_v2.csv |

**Total principal: 324 KD científicas**, incluyendo las dos del piloto.
Comparado con el diseño anterior de ocho brazos completos (432), se ahorran
108 KD antes de añadir diagnósticos opcionales. No incluir controles opcionales
limita la comparación principal a FedDF/ORACLE/EXPERT; no es un benchmark SOTA.

```bash
python run_article1_pipeline.py
python run_article1_pipeline.py --phase partitions --execute
python run_article1_pipeline.py --phase pilot --execute --device cuda
python run_article1_pipeline.py --phase teachers --execute --device cuda
python run_article1_pipeline.py --phase rq1 --execute --device cuda
python run_article1_pipeline.py --phase aggregation --execute --device cuda
python run_article1_pipeline.py --phase expertise --execute --device cuda
python run_article1_pipeline.py --phase support --execute --device cuda
```

Los teachers usan train para gradientes, validation para seleccionar checkpoint,
y expertise para estimar M tras congelarlo. El test oficial evalúa el student
final. Este cambio no modifica particiones ni entrenamiento local.

La secuencia no decide automáticamente el operador mirando accuracy de test.
EXPERT-prob se fija como propuesta por su combinación normalizada y su extensión
natural a SR. La comparación FedDF/ORACLE caracteriza el operador; no demuestra
que EXPERT-logit sea equivalente. Ausencia de significación no prueba igualdad:
para hablar de equivalencia práctica hace falta un margen previo y precisión
suficiente. No se fija un margen arbitrario en el código. Tres seeds ofrecen
precisión limitada. Las figuras muestran media y desviación estándar de efectos
emparejados, no un test automático de equivalencia.

| Fase opcional | Coste y finalidad |
|---|---|
| `baseline` | 540 KD: los diez métodos; guarda results_baseline.csv y analiza el conjunto completo |
| `controls` | 108 KD Consensus-logit/Energy-logit, contra FedDF-logit; Confidence queda disponible solo en el runner |
| `expert-logit` | 9 KD CIFAR IID/alpha0p1/single × tres seeds a T=8; compara con EXPERT-prob existente |
| `temperature` | Completa/reutiliza esas 9 KD y añade 36 T=1/4 EXPERT-logit/prob |
| `supervised` | 9 CE con proxy completo |
| `proxy-curve` | 12 CE + 36 EXPERT-prob CIFAR N=100/500/1000/5000; reutiliza KD N=10000 y completa/reutiliza tres CE ancla |

```bash
python run_article1_pipeline.py --phase expert-logit --execute --device cuda
python run_article1_pipeline.py --phase controls --execute --device cuda
python run_article1_pipeline.py --phase supervised --execute --device cuda
python run_article1_pipeline.py --phase proxy-curve --execute --device cuda
```

`--skip-notebooks` permite el análisis manual. El notebook acepta `STAGE=baseline` (540 filas) o `STAGE=rq1`,
`aggregation`, `expertise`, `support` o `controls`; exige respectivamente
108/216/270/324/216 filas completas en los archivos de esos bloques. Exporta
figuras y tablas en `minimal_analysis/<STAGE>/`. Los diagnósticos focales exportan
sus propios efectos emparejados. El análisis gráfico supervisado/curva sigue
pendiente; sus ejecuciones no autorizan a afirmar un N óptimo.

Se conservan los CSV históricos v3 sin sobrescribirlos. El diseño nuevo usa los
archivos indicados arriba; no importa automáticamente sus filas. Caches de
teachers completos se reutilizan; las nuevas celdas se reanudan por identidad.
El informe del piloto debe corresponder a ORACLE y al cache actual.

KD base usa 30 épocas, proxy 10000 y batch 256: 1200 actualizaciones.
CE y curva fijan 1200 explícitamente. Los subconjuntos son anidados y balanceados;
N=100 usa batch efectivo 100. Se igualan actualizaciones, no ejemplos consumidos.
El fallback común permanece FedDF-logit incluso en ORACLE-prob/EXPERT-prob/SR:
el contraste ORACLE de operador no cambia las muestras con selección vacía.

### A. Crear y revisar solo particiones

```bash
python run_article1_grid.py --stage partition --datasets mnist --seeds 42 --dry-run
python run_article1_grid.py --stage partition --datasets mnist --seeds 42
jupyter notebook notebooks/article1_partition_diagnostics.ipynb
```

Los comandos usan `OUTPUTS/article1_v3/` por defecto. En el notebook elegir dataset, seed y régimen local de interés. Comprueba los seis regímenes, el proxy común, cobertura y huellas. Exporta cuatro figuras en PNG/PDF y dos tablas CSV bajo `figures/partitions/<dataset>-seed<seed>/`:

1. Recuentos cliente–clase entre regímenes.
2. Composición por cliente normalizada.
3. Recuentos de train/validation/expertise del régimen elegido.
4. Fracciones de cada split respecto al total de su pareja cliente–clase.

Repetir para seeds 43/44 y los otros datasets antes del barrido. Las figuras no infieren expertise del reparto: tener datos de una clase no garantiza competencia. Gris indica ausencia de clase y cero indica ausencia en ese split.

### B. Teacher y KD: ejecutar después de aprobar las particiones

```bash
python -m article1.runner teachers --dataset mnist --seed 42 --regime iid --partitions OUTPUTS/article1_v3/partitions/mnist-seed42-iid --output OUTPUTS/article1_v3/sources/mnist-seed42-iid --device cuda
python -m article1.runner distill --dataset mnist --seed 42 --method oracle_logit --cache OUTPUTS/article1_v3/sources/mnist-seed42-iid/teacher_cache.npz --results OUTPUTS/article1_v3/results_selection.csv --device cuda
```

Para planificar más celdas sin ejecutarlas:

```bash
python run_article1_grid.py --stage distill --datasets mnist --regimes iid --seeds 42 --methods feddf_logit oracle_logit --dry-run
```

No ejecutar el grid entero antes del control pequeño. Los outputs de teachers no vacíos se rechazan; un entrenamiento interrumpido se inspecciona antes de decidir cómo repetirlo. No hay sobrescritura automática de checkpoints ni resampling de particiones.

### C. Supervisado y un tamaño del proxy

```bash
python -m article1.runner supervised --dataset mnist --seed 42 --cache OUTPUTS/article1_v3/sources/mnist-seed42-iid/teacher_cache.npz --results OUTPUTS/article1_v3/results_supervised_proxy.csv --updates 1200 --device cuda
python -m article1.runner supervised --dataset cifar --seed 42 --cache OUTPUTS/article1_v3/sources/cifar-seed42-iid/teacher_cache.npz --results OUTPUTS/article1_v3/results_proxy_size.csv --proxy-size 1000 --updates 1200 --device cuda
python -m article1.runner distill --dataset cifar --seed 42 --cache OUTPUTS/article1_v3/sources/cifar-seed42-iid/teacher_cache.npz --results OUTPUTS/article1_v3/results_proxy_size_expert_prob.csv --method expert_prob --temperature 8 --proxy-size 1000 --updates 1200 --device cuda
```

N=1000 es ejemplo de sintaxis, no una orden para lanzar un barrido. Estos brazos rechazan una salida llamada `results.csv`. KD a temperaturas distintas de 8 debe ir también a un CSV separado; el lanzador lo exige.

### D. Comprobar código y resultados

```bash
python -m compileall -q article1 run_article1_grid.py
python -m pytest -q
python -m article1.conditions
python -m article1.audit OUTPUTS/article1_v3/results_selection.csv
jupyter notebook notebooks/article1_definitive_analysis.ipynb
```

Conditions exige las 54 fuentes. El notebook definitivo exige los bloques completos de la fase seleccionada (108/216/270/324 filas T=8): no se presenta como ejecutable con resultados v3 aún ausentes. Para verificación completa de caches activar `VERIFY_CACHES`; comprobar CSV y huellas registradas no sustituye reconstruir los targets con las fuentes reales.

## 9. Estado histórico y reinicio

La versión anterior a este cambio es el commit `653a145ba4cc2367820d1be5bdf05e1029d83ef3` (protocolo v2, train/holdout/test local, 65/20/15). Su código sigue disponible en Git. Sus artefactos permanecen en `OUTPUTS/article1/`, sin migración ni borrado.

Evidencia histórica disponible: 486 ejecuciones T=8 (54 condiciones × 9 métodos), experimento focal de temperatura y análisis de soporte. Mostraban ventaja creciente de EXPERT con especialización, cercanía descriptiva a ORACLE, dependencia de temperatura del contraste logits/probabilidades y casos donde mejorar NLL del target con SR empeoraba al student. Estas observaciones no se convierten en resultados v3 por cambiar nombres de archivos.

El nuevo reparto exige nuevos teachers, máscaras y KD. El proxy y la asignación entre clientes se mantienen algorítmicamente; los supervisados históricos pueden ser numéricamente reutilizables si se verifica la identidad completa, pero no se importan ni reetiquetan automáticamente como v3. Los caches v2 son rechazados por el runtime activo.

Los caches históricos con commit desconocido conservan `legacy_unverified`. El control MNIST-IID de identidad de targets y pequeñas diferencias del student sigue pendiente de aclaración sobre las fuentes históricas; no justifica mezclar ambos protocolos.

## 10. SIGUIENTES PASOS

### 10.1 Cerrar el labeled baseline que está en ejecución

La fase `baseline` del commit remoto e311558 ejecuta **54 condiciones × 10 métodos = 540 KD** a T=8, no las 324 del diseño mínimo. Su salida es `OUTPUTS/article1_v3/results_baseline.csv`. Dentro de cada condición empieza con FedDF-prob, EXPERT-prob, EXPERT-prob-SR y ORACLE-prob; después ejecuta seis métodos logit. No ejecuta primero todas las probabilidades de todas las condiciones. Teachers y máscaras se reutilizan.

No actualizar el checkout desde el que está corriendo el proceso: el pipeline lanza subprocesses que importan archivos de disco y una actualización intermedia podría mezclar versiones. No se requiere interrumpir una KD sana por esta revisión. Si el proceso termina fallando solo en el notebook con `Unknown analysis stage: baseline`, sus filas KD ya escritas se conservan; actualizar y ejecutar el análisis basta, sin relanzar baseline.

Se han corregido dos problemas de integración del estado remoto: faltaba `run_article1_grid.py` pese a ser importado y ejecutado por el pipeline, y `baseline` no estaba admitido por el cargador de análisis. Un checkout limpio de aquel commit no podía arrancar el pipeline sin el grid. Si una ejecución estaba avanzando, su copia local conservaba ese archivo o difería del snapshot remoto. Los targets y la receta KD no cambian en esta revisión.

Después de que termine el entrenamiento:

```bash
# Desde cleanup/article1-base, con el proceso ya terminado:
git pull --ff-only
python -m article1.conditions
python -m article1.audit OUTPUTS/article1_v3/results_baseline.csv --methods feddf_prob expert_prob expert_prob_sr oracle_prob feddf_logit confidence_logit consensus_logit energy_logit expert_logit oracle_logit

# Validar las 540 filas y materializar los bloques/focal sin entrenar:
python -m article1.baseline_results
jupyter notebook notebooks/article1_definitive_analysis.ipynb
```

Seleccionar `STAGE="baseline"` en el notebook para revisar el conjunto completo. `article1.baseline_results` conserva el CSV maestro, exige integridad y crea los CSV de selección, pooling, expertise, soporte, controles y nueve EXPERT-logit focales con los mismos run_id. Si existe un destino diferente se detiene antes de escribir; no sobreescribe resultados conflictivos. La validación de CSV no sustituye la auditoría de caches anterior. No ejecutar el exportador mientras se escribe baseline, ni si faltan filas.

- [ ] Verificar procedencia, presupuesto, emparejamiento por seeds y fallbacks; comprobar SR revisión 2. No afirmar que una ejecución remota está completa solo por lanzar el comando.
- [ ] Analizar primero ORACLE-logit − FedDF-logit; después los contrastes de operador FedDF y ORACLE; después EXPERT-prob − FedDF-prob y gap a ORACLE-prob; finalmente SR − full support.
- [ ] Con las 540 filas completas no repetir `rq1`, `aggregation`, `expertise`, `support` o `controls`: sus celdas ya están presentes. Tampoco hace falta entrenar de nuevo EXPERT-logit focal.
- [ ] Ejecutar `--phase supervised --execute --device cuda` (nueve CE N=10000) y revisar la comparación con EXPERT-prob.
- [ ] Después, ejecutar `--phase proxy-curve --execute --device cuda`: CIFAR, IID/alpha0p1/single, seeds 42/43/44 y N=100/500/1000/5000, con EXPERT-prob. Son 12 CE + 36 KD nuevas si no existen; reutiliza los puntos N=10000. Añadir su análisis gráfico al notebook antes de afirmar un cruce o un N suficiente.
- [ ] `--phase temperature` queda como robustez focal opcional: reutiliza los nueve EXPERT-logit T=8 exportados y añade 36 KD T=1/4. No selecciona automáticamente una temperatura por test.
- [ ] Cerrar las afirmaciones del estudio labeled antes de lanzar el experimento unlabeled descrito abajo. Clientes no son réplicas independientes; falta de significación no prueba equivalencia.

### 10.2 Justificación del espacio probabilístico: alcance de las garantías

Para el labeled proxy se mantiene `S_E(x)={k:M[k,y(x)]=1}` y `q_full=sum_k w_k p_k`, con pesos uniformes en S_E. Se conserva la distribución completa de cada experto para mantener relaciones interclase potencialmente útiles. Esto no demuestra que todo valor fuera de soporte sea dark knowledge válido ni que el enmascaramiento cause necesariamente peor generalización; esa es la pregunta de la ablación SR.

Con pesos fijos, reemplazar la distribución de un teacher de peso w cambia cada coordenada del target como máximo w y su distancia L1 como máximo 2w. Acotar cada distribución evita que su norma de logits aumente directamente ese peso. **No proporciona invariancia a escala ni calibración**: escalar logits cambia su softmax, y un teacher muy confiado pero erróneo aún puede perjudicar el target. La cota no afirma que todos los gradientes de los parámetros del student estén acotados: también interviene su Jacobiano.

El producto M*p requiere normalización para ser una distribución. Para SR labeled se usa softmax estable dentro del soporte; no se inyectan infinitos ni se descartan teachers por masa pequeña. El uso conceptual de -infinito como máscara antes de softmax no es intrínsecamente incorrecto o inestable. KL y entropía son aplicables también al target obtenido agregando logits y aplicando softmax; no constituyen una ventaja exclusiva de la media de probabilidades.

Los marcadores `[cite: ...]` del texto de propuesta no identifican referencias verificables y no se incorporan como citas. El fundamento matemático anterior es una derivación; las afirmaciones empíricas quedan por contrastar.

### 10.3 Propuesta futura: síntesis por clase con proxy sin etiquetas

**Estado: propuesta a evaluar, no implementada ni activada en baseline.** Es una regla class-supported de contribución por coordenada, diferente del routing labeled por muestra. Usa M construida con etiquetas privadas de expertise, pero no la etiqueta pública y(x). Por tanto es "proxy unlabeled", no un sistema sin ninguna etiqueta.

Definir `p_k(x;T)=softmax(z_k(x)/T)` con la misma temperatura para todos los teachers. Para cada clase c:

\[
E_c=\{k:M_{k,c}=1\},\qquad n_c=|E_c|.
\]

Cuando hay varios expertos, la propuesta base es promediar sus probabilidades **completas antes de la selección de coordenadas**:

\[
s_c(x)=
\begin{cases}
\frac{1}{n_c}\sum_{k\in E_c}p_{k,c}(x;T), & n_c>0,\\
0,&n_c=0,
\end{cases}
\qquad
q_c(x)=\frac{s_c(x)}{\sum_j s_j(x)}.
\]

Con un único experto por clase recupera la fórmula propuesta originalmente. Se denominan s_c "puntuaciones" antes de normalizar: no forman todavía una distribución común. La media evita introducir por construcción un prior proporcional al número de expertos: si A tiene dos expertos que emiten 0.8 y B uno que emite 0.8, sumar produciría (2/3,1/3); promediar por clase produce (1/2,1/2). Esto tampoco demuestra calibración entre clases.

**No aplicar antes el SR por teacher del experimento labeled.** Si cada teacher solo conoce una clase, su softmax restringido es siempre one-hot, independiente de x. Al combinar un experto por clase se obtiene exactamente una distribución uniforme. Deben conservarse las probabilidades completas para extraer cada coordenada. Incluso así, si cada teacher ya emite aproximadamente 1 para su clase sobre cualquier entrada, la propuesta vuelve a producir casi uniforme: normalizar no crea información discriminativa.

Casos que deben fijarse antes de implementar:

- Si n_c=0, s_c=0 expresa falta de evidencia acreditada; no demuestra que la clase sea imposible. La versión estricta no puede darle masa al target. Para presentar una solución sobre todas las clases hay que exigir cobertura completa o declarar una variante explícita que complete clases sin expertos con una referencia; no hacerlo silenciosamente.
- Si todas las puntuaciones son cero (M vacía o underflow), usar como fallback propuesto la media de probabilidades de todos los teachers y contabilizar el evento. Este sería un nuevo método con nueva identidad; no cambia el fallback FedDF-logit de los métodos labeled existentes.
- Calcular las puntuaciones y la normalización con estabilidad numérica (por ejemplo logsumexp/log-softmax); no recuperar ceros numéricos con etiquetas reales. Una suma pequeña hace sensibles los cocientes a perturbaciones relativas. L1 asegura suma uno, no calibración ni calidad del target.

**Limitación principal de M:** la accuracy por clase actual estima `P(pred=c | y=c)` sobre el expertise local. No estima `P(y=c | pred=c)`, especificidad ni rechazo OOD. Un teacher que siempre predice c puede tener 100% de aciertos en c y recibir M[k,c]=1. Eso basta para el criterio actual, pero no certifica que p[k,c](x) sea una señal fiable para distinguir c del resto. El labeled routing dispone de y(x); la nueva regla ya no tiene esa ayuda.

Tampoco está garantizada la comparabilidad de puntuaciones entre especialistas entrenados con distintos priors locales. La normalización final no corrige ese desajuste. A T muy alta, las probabilidades completas tienden a 1/C: con cobertura completa la media por clase y L1 tienden a uniforme. La temperatura del protocolo labeled no debe suponerse óptima para esta extensión.

### 10.4 Experimento de viabilidad unlabeled, después del estudio actual

- [ ] Reutilizar teachers/logits/M congelados para un diagnóstico de targets sin reentrenar; mantener el nuevo método en un archivo e identidad independientes.
- [ ] Implementar una API sin argumento labels para la construcción del target. Comprobar que permutar/eliminar y no cambia ni selección, ni target, ni fallback. Separar del código que usa etiquetas para medir calidad del target.
- [ ] Comprobar expertos únicos/múltiples, clases sin experto, M vacía, masa casi nula, pesos y normalización; incluir single como caso de posible degeneración.
- [ ] Comparar targets con FedDF-prob unlabeled y EXPERT-prob labeled como referencia informada (no como competidor de igual presupuesto informacional). Medir cobertura, fallbacks, accuracy/NLL/entropía, variación de s_c por muestra y señales constantes de cada especialista.
- [ ] Investigar discriminación fuera de la clase acreditada con datos etiquetados de desarrollo independientes si se pretende ajustar M o calibración. No reutilizar test oficial para ese ajuste ni presentar un desarrollo que usó etiquetas públicas como si nunca las hubiera utilizado.
- [ ] Solo si hay señal discriminativa, ejecutar una KD focal con seeds emparejadas; después ampliar. La evaluación final sigue en test oficial y no se modifica el experimento labeled en curso.

La propuesta tiene una definición matemática válida una vez fijados expertos múltiples, cobertura y fallback. Su utilidad en alta especialización es una hipótesis, no una consecuencia de usar probabilidades.

## 11. Validación de esta revisión

- 40 pruebas correctas con PyTorch/torchvision CPU; compilación y comprobaciones de estilo de los archivos modificados correctas.
- Una prueba sintética modifica solo las predicciones observadas en expertise: cambia M, pero conserva exactamente checkpoints seleccionados, registros de selección y logits del proxy. No es un resultado científico de los datasets.
- Se generaron y validaron particiones reales de MNIST, seed 42, en los seis regímenes. Frente al código anterior, coinciden exactamente el proxy y la asignación privada a cada cliente; cambia solo la división local.
- Las seis celdas de código del notebook de particiones se ejecutaron secuencialmente en proceso, generando cuatro figuras PNG/PDF y dos tablas. El entorno impidió iniciar el kernel Jupyter por restricciones de sockets; queda comprobar esa vía de ejecución en el entorno del investigador.
- No se entrenaron teachers ni students sobre datasets reales. No se ha validado CUDA ni completado una cuadrícula científica v3.

Revisión del lanzador completo: 44 pruebas correctas, incluidos modo plan sin ejecución, parada antes de entrenamiento en `--partitions-only` y reutilización CE/KD únicamente por identidad. Se reconstruyeron sin errores las 36 condiciones de MNIST/Fashion-MNIST con etiquetas oficiales; mínimos locales de validation: 80 y 17 ejemplos respectivamente. Las 18 configuraciones CIFAR se comprobaron con una fixture de sus recuentos de clase (mínimo 12 en validation), no con los archivos de imágenes: la descarga completa no se terminó en este entorno. El lanzador exige la comprobación con etiquetas oficiales de los tres datasets antes de entrenar. Los tamaños pequeños indican poca precisión estadística, no un solapamiento ni un defecto del reparto.

Validación de la unificación por fases: 52 pruebas correctas; compilación del código y celdas del notebook correcta; lint correcto en los archivos modificados. No se han ejecutado entrenamientos reales en esta revisión.

Validación del diseño mínimo probabilístico: 57 pruebas correctas y 8 omitidas por ausencia de PyTorch/torchvision; lint y compilación correctos. Se comprobaron los planes de todas las fases sin entrenar. Los tests nuevos cubren FedDF-prob, soporte de masa diminuta/underflow, fallback, revisión de identidades SR y máscaras no binarias. No se han ejecutado entrenamientos reales ni medido equivalencia empírica entre operadores.

Validation de la revisión unlabeled/baseline: 59 pruebas correctas y 8 omitidas por ausencia de PyTorch/torchvision; compilación y lint correctos. Se verifican las 540 filas sintéticas, exportación idempotente sin entrenamiento, rechazo de destinos conflictivos y revisión SR por método. No se ha ejecutado ni implementado una KD unlabeled.

### Rendimiento CUDA y MSI Pulse GL76

Configuración confirmada por las salidas del equipo del 9 de septiembre de 2026:

| Componente | Valor observado |
|---|---|
| Portátil | MSI Pulse GL76 (modelo indicado por el usuario) |
| GPU | NVIDIA GeForce RTX 3060, 6144 MiB de VRAM |
| Driver NVIDIA | 595.84 |
| PyTorch | 2.9.1+cu128 |
| Runtime CUDA de PyTorch | 12.8 |
| cuDNN | 91002, valor devuelto por PyTorch |
| Límite de potencia actual informado | 80 W |

La cabecera CUDA 13.2 de `nvidia-smi` corresponde al soporte del driver;
no sustituye al runtime CUDA 12.8 de PyTorch. Conservar este entorno que ya
está entrenando. Estas consultas no confirman la RAM, el SSD ni el SKU completo
del portátil. El pipeline imprime GPU, VRAM y versiones al ejecutar una fase CUDA.

**Estado del código:** están aplicadas las optimizaciones conservadoras de
transporte para esta GPU, compartidas entre teachers, CE y KD. Esto no certifica
un máximo rendimiento ni equivalencia numérica entre commits en CUDA: ambos
requieren medición local. El cambio de transporte pasó 67 pruebas con PyTorch
2.9.1 CPU y torchvision 0.24.1 CPU, además de lint. No se ejecutó entrenamiento
científico ni se midió CUDA desde el entorno de revisión.

**Prioridad actual: estabilizar la refrigeración.** En las capturas del equipo
se observaron 88–89 °C, objetivo térmico informado de 87 °C,
`SW Thermal Slowdown: Active` y frecuencia SM/Graphics de 232 MHz. Esto confirma
limitación térmica en ese instante; no cuantifica la pérdida de rendimiento
sostenida. La primera captura mostró 99 % de utilización y 3089/6144 MiB ocupados:
una GPU puede permanecer ocupada trabajando a frecuencias reducidas. Los
contadores de eventos son acumulados y no deben atribuirse íntegramente al
experimento actual. Véase la [documentación de NVIDIA](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

Comprobar ventilación, entradas y salidas de aire y funcionamiento de ventiladores;
medir con condiciones térmicas estables. Mantener un entrenamiento simultáneo.
No aumentar potencia ni batch para ocupar la VRAM libre. La limitación térmica
observada no exige por sí sola descartar resultados; registrar sus efectos al
comparar tiempos de ejecución. Resolver esta limitación antes de atribuir una
mejora de velocidad a workers, pinning u otros cambios de software.

Optimizaciones de transporte aplicadas al runtime compartido CE/KD y teachers:

| Archivo | Cambio | Motivo |
|---|---|---|
| `article1/runner.py` | `pin_memory` en el loader del proxy; copias H2D con `non_blocking=True` | Permitir transferencias desde memoria fijada sin esperar en el host tras cada copia |
| `article1/runner.py` | Conservar solo la pérdida final separada del grafo; convertirla a `float` al terminar | Evitar una sincronización CPU/GPU por actualización |
| `article1/local_training.py` | Copias H2D no bloqueantes; pinning según el dispositivo solicitado | Aplicar la misma política al entrenamiento y evaluación de teachers |
| `article1/datasets.py` | Pinning del test solo cuando se evalúa en CUDA | Evitar fijar RAM al ejecutar explícitamente en CPU |
| `run_article1_pipeline.py` | Mostrar dispositivo y versiones | Identificar el entorno real en el registro de ejecución |

Los targets completos ya residen en el dispositivo durante KD. Se mantienen
`num_workers=0`, batches, precisión FP32, RNG, optimizadores y algoritmos
deterministas. Las copias GPU→CPU de logits siguen siendo bloqueantes antes de
leer NumPy: convertirlas sin esperar podría leer datos incompletos.
`pin_memory` más `non_blocking` no garantiza solapar transferencia y cómputo:
se usa el mismo stream, y fijar memoria también tiene coste. No hay una mejora
porcentual medida en el portátil. Véase la [guía oficial de PyTorch](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

Siguientes optimizaciones, por orden de evaluación:

1. Medir tiempo por celda y utilización GPU en el equipo, separando carga inicial,
   construcción de targets, entrenamiento y evaluación. Comparar condiciones
   idénticas y varias repeticiones, con calentamiento y sincronización CUDA en los
   límites de las mediciones. No comparar tiempos de métodos distintos como si
   midieran el efecto de una optimización.
2. Probar 0/2/4 workers en un benchmark separado. Más procesos no implica mayor
   velocidad; CIFAR aplica transformaciones por muestra y el lanzamiento de workers
   tiene coste. `persistent_workers` puede ahorrar reinicios, pero cambia el ciclo
   de vida del RNG. En teachers también afecta al augmentation; en students hay
   dropout. Antes de adoptarlo exigir las mismas huellas de batches y estados, o
   declarar un cambio de receta. No activar workers globalmente a mitad del estudio.
3. Evitar crear la vista aumentada de entrenamiento en cada ejecución del student
   (`datasets_for` la construye y el runner la descarta). Después valorar cachear
   el proxy ya transformado en RAM: 10.000 × 3 × 32 × 32 float32 son unos 117 MiB,
   sin contar estructuras auxiliares. Esto requiere conservar índices, etiquetas,
   transformaciones y huellas, y medir el coste inicial frente al ahorro por época.
4. Reducir sincronizaciones por batch en evaluación y agrupar carga/inicialización
   entre métodos solo si el perfil lo justifica. El aislamiento actual por proceso
   simplifica la reanudación y evita estado compartido accidental.
5. AMP, TF32, `channels_last`, `torch.compile`, optimizadores fusionados o cambios
   de batch son pruebas posteriores de rendimiento. Pueden cambiar resultados
   numéricos o la receta experimental; no se activan en las comparaciones actuales.

Para identificar y observar tu GPU sin modificar el experimento:

```bash
nvidia-smi
nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER,CLOCK
nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,clocks.sm,power.draw --format=csv -l 2
```

La disponibilidad de sensores depende del driver. Medir conectado a corriente y
con condiciones térmicas estables; no lanzar varios entrenamientos simultáneos
sin medir contención de GPU/VRAM. Conservar el entorno CUDA que ya funciona.

Si `--phase baseline --execute` sigue activo, **esperar a que termine antes de
actualizar su checkout**: el pipeline arranca subprocesos que leerían el nuevo
código. Después de actualizar, comprobar reproducción en un CSV separado antes
de continuar. La comprobación siguiente repite una celda; no demuestra por sí
sola equivalencia con el commit anterior. Para comparar commits, ejecutar esa
misma celda con el mismo entorno en dos checkouts y comparar las huellas finales:

```bash
python -m article1.reproduce --dataset mnist --seed 42 --method expert_prob --cache OUTPUTS/article1_v3/sources/mnist-seed42-iid/teacher_cache.npz --results OUTPUTS/article1_v3/cuda_io_repro.csv --device cuda
```

No se han medido tiempos ni equivalencia CUDA en el MSI desde este entorno.

## Cerrar el principal y actualizar la curva sin interferir con entrenamiento

[Informe de cierre, tabla de preguntas y evidencia revisable](docs/article1_closure/report.md).

La instantánea final `20260909T214849851047Z` verifica 324 filas del baseline de seis
métodos, nueve supervisados y la curva completa: 60 ejecuciones únicas y 45 pares.
Los tres notebooks se ejecutaron en CPU con modo definitivo, sin exclusiones en la curva.

Usar **otro checkout**, en la rama de análisis `analysis/article1-main-closure`.
No editar ni actualizar `cleanup/article1-base` mientras su cola utiliza ese
checkout. El programa rechaza usar el mismo checkout y escribir análisis bajo
el directorio del entrenamiento. No instala paquetes, no entrena, no usa CUDA,
no carga modelos ni cambia los originales. Se limita a un hilo y usa los caches
existentes, una condición cada vez, para comprobar los diagnósticos del target.

Comando exacto utilizado en este equipo (desde `/tmp/article1-close`):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 15 /home/roi/miniconda3/envs/FLWR/bin/python run_article1_analysis.py \
  --source-root /home/roi/PycharmProjects/KD_v5/OUTPUTS/article1_v3 \
  --data-dir /home/roi/PycharmProjects/KD_v5/data \
  --analysis-root /tmp/article1-closure-output \
  --curve-mode definitive
```

Jupyter requiere sockets locales. Se ejecutan, en orden, el notebook definitivo,
el de expertise y el de presupuesto. El notebook de progreso se conserva como
vía distinta para un baseline incompleto; no duplica la ejecución principal.
Cada ejecución crea una nueva instantánea UTC con `manifest.json`, `snapshots/`,
`tables/`, `figures/` PNG/PDF, `notebooks/` ejecutados y `report.md`. Para abrir un
notebook manualmente, exportar `ARTICLE1_SNAPSHOT` con la ruta absoluta de esa
instantánea. Las actualizaciones usan **otra instantánea**, nunca una nueva lectura
del CSV vivo dentro de un notebook.

El runner escribe mediante `os.replace` bajo bloqueo. El analizador copia una
versión completa, comprueba estructura, inode/tamaño/fecha y dos hashes; reintenta
si cambia el archivo. El manifiesto conserva rutas, SHA256, fechas, filas, versiones,
commit del análisis, código utilizado y procedencia declarada de los artefactos.
No se presupone que el proceso activo utiliza el commit actualmente en disco.

### Diseños explícitos y cierre

`article1.analysis.load_results(..., stage="baseline", design="auto")` reconoce
el diseño observado de seis o diez métodos. `design="six"` exige las 324 identidades
y `design="ten"` las 540; ninguna opción admite una cuadrícula incompleta o descarta
métodos extra para ajustarla al cargador. En el notebook definitivo, `DESIGN` permite
fijar la elección explícita. Se verifican protocolo v3, presupuesto, temperatura,
revisión 2 únicamente para SR, fuentes y CRN, incluidos los batches consumidos.

El principal puede cerrarse con sus seis métodos completos aunque el respaldo de
controles siga parcial. Los CSV maestro y derivados se deduplican por identidad,
rechazando discrepancias. Las cabeceras de controles o EXPERT-logit sin filas no
son evidencia de experimentos ejecutados. El exportador baseline valida el diseño
completo, omite bloques opcionales vacíos y rechaza destinos discrepantes; solo
debe usarse sobre una copia validada y estable, nunca el CSV activo.

`scientific_questions.csv` enlaza pregunta, contraste, artefacto, observación,
interpretación, limitación y estado cerrado/provisional/pendiente. Accuracy y NLL
se presentan por dataset y régimen, con puntos por seed, diferencias emparejadas,
media y SD muestral. No se agregan datasets como réplicas intercambiables ni clientes
como réplicas. Las relaciones de cobertura/soporte con los efectos son descriptivas.

### Máscaras y selección

El notebook de expertise verifica las 54 fuentes y los recuentos de las particiones
con etiquetas locales, sin descargar ni inferir. Produce M, accuracy y recuentos
cliente–clase, tamaños efectivos, expertos por clase, clases por teacher, densidad,
soporte exacto y descriptivo, selección de checkpoints y fallback ponderado por
etiquetas del proxy. Gris distingue count=0 de accuracy observada cero.

Validation selecciona el checkpoint; expertise construye M. No hay test local
independiente v3. M acredita accuracy condicional por clase, no rechazo OOD ni
validación independiente de sus celdas seleccionadas. Los hashes de bytes de
checkpoints y el fingerprint declarado se registran; no se presentan hashes de
estados como recalculados, porque no se deserializan modelos.

### CE y curva

EXPERT-prob−CE se evalúa a N=10000 para los tres datasets, seis regímenes y tres
seeds: 54 pares frente a nueve CE únicos. CE no requiere igualdad de M ni cache
privado, pero sí proxy/etiquetas, inicialización, batches, optimizador y presupuesto
compatibles. Su repetición visual entre regímenes no crea réplicas nuevas.

La curva lee exclusivamente `results_proxy_size_expert_prob.csv` como archivo
activo de tamaños: contiene CE y EXPERT-prob. Se añaden las referencias N=10000 de
baseline y supervisado, sin duplicarlas cuando aparecen también en bloques.
El diseño completo CIFAR tiene 15 CE y 45 KD (60 ejecuciones únicas); 12 CE y 36 KD
son de tamaños inferiores. Se exigen subconjuntos anidados y 45 pares completos.

Modo provisional: inventario antes de las figuras, puntos solo de pares compatibles,
SD indefinida con una seed y exclusiones explícitas. Las líneas de seeds se cortan
en huecos; no se unen medias calculadas con seeds diferentes. Las etiquetas indican
N, n y seeds. `examples_consumed.csv` conserva el presupuesto y ejemplos consumidos.
Un cruce se acota entre N evaluados; no demuestra un óptimo ni mínimo suficiente.

### Procedimiento final y futuras actualizaciones

1. Repetir el comando anterior con **`--curve-mode definitive`**: nueva instantánea
   y auditoría completa. Ese modo rechaza filas ausentes, identidades inesperadas
   o pares incompatibles; no reduce la cuadrícula para poder ejecutarse.
2. Regenerar figuras de accuracy, NLL y KD−CE por N/régimen.
3. Revisar conclusiones y captions según la cobertura y replicación verificadas.
4. Identificar preguntas realmente pendientes, separadas del principal cerrado.
5. Decidir posteriormente si estudiar temperatura, controles o viabilidad unlabeled.
   Esta tarea no implementa ni lanza unlabeled ni ninguna fase de entrenamiento.

Pruebas CPU de integridad y diseños:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/article1-matplotlib \
  /home/roi/miniconda3/envs/FLWR/bin/python -m pytest -q \
  tests/test_analysis_stages.py tests/test_progress.py \
  tests/test_budget_analysis.py tests/test_probability_design.py
```
