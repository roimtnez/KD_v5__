# KD_v5 — Artículo 1: expertise en one-shot federated distillation

**Estado actualizado: 7 de septiembre de 2026.** Este README es la única documentación del repositorio. El notebook contiene análisis ejecutable; las decisiones científicas, instrucciones y tareas pendientes se mantienen aquí.

El proyecto estudia cómo construir un ensemble teacher útil cuando los modelos locales se especializan en distintas clases. La hipótesis de partida es que disponer de información de competencia cliente–clase puede mejorar la destilación frente a agregar todos los teachers uniformemente. El éxito consiste en responder la pregunta con comparaciones controladas, no en conseguir que EXPERT gane siempre.

Título provisional: **Who and What to Teach: Expertise-Aware Ensemble Distillation under Client Specialization**.

## 1. Alcance acordado del Artículo 1

Se estudia una **versión base y simple** de la propuesta:

- One-shot: un entrenamiento local por teacher, sin rondas federadas posteriores.
- 10 clientes, un proxy público y un único estudiante global.
- Se asume disponibilidad de **etiquetas del proxy y máscara de competencia**. EXPERT utiliza ambas; no es label-free.
- Máscara binaria estimada en holdout, con umbrales fijados previamente. No se eligen a partir del test ni se ajustan para eliminar fallbacks.
- Se mantienen tres contrastes principales: selección de teachers, logits frente a probabilidades y distribución completa frente a soporte restringido.
- Se añade al alcance científico la comparación con **entrenamiento supervisado sobre el mismo proxy** y su dependencia del número de muestras. Está pendiente de diseño e implementación; no existen resultados supervisados canónicos en este repositorio.

La inferencia de la máscara a partir del modelo, el routing con proxy sin etiquetas, los estimadores continuos de competencia y la segunda destilación/personalización se reservan para trabajos posteriores. Una comprobación básica de cobertura y sensibilidad de los umbrales puede formar parte de las limitaciones o robustez de esta versión base; no es una nueva búsqueda para optimizar los resultados del test.

## 2. Preguntas y contrastes controlados

| Pregunta | Comparación principal | Qué se mantiene fijo |
|---|---|---|
| RQ1: ¿quién debe contribuir? | `expert_logit − feddf_logit` | Operador de logits, teachers, proxy, inicialización, batches, T y presupuesto KD |
| RQ2-A: ¿cómo combinar expertos? | `expert_prob − expert_logit` | Routing EXPERT, soporte completo y receta |
| RQ2-B: ¿qué distribución transmitir? | `expert_prob_sr − expert_prob` | Routing EXPERT, pooling de probabilidades y receta |

`oracle_logit − expert_logit` contextualiza la información adicional de selección correcta por muestra. ORACLE es una referencia informada, **no un límite superior garantizado del rendimiento del estudiante**. `oracle_prob − oracle_logit` es un control secundario del espacio de agregación.

EXPERT-logit frente a FedDF-logit separa la regla de selección del operador, pero no separa el valor de la máscara del valor de disponer de etiquetas para consultarla. El presupuesto de información debe ser explícito en el artículo.

## 3. Datasets, especialización y papeles de los datos

| Dataset | Identificador CLI | Modelo local y estudiante | Umbral de expertise |
|---|---|---|---|
| MNIST | `mnist` | `MnistNet` | 0,90 |
| Fashion-MNIST | `fmnist` | `MnistNet` | 0,80 |
| CIFAR-10 | `cifar` | `ResNet9` | 0,70 |

Semillas: **42, 43 y 44**. Eje categórico ordenado: **IID → α=1.0 → α=0.5 → α=0.1 → Multi → Single**. Su separación gráfica uniforme no implica distancias cuantitativas iguales de heterogeneidad. Las figuras conservan una trayectoria continua en los seis regímenes.

`multi` asigna dos clases a cada cliente; `single`, una. Los tres regímenes Dirichlet usan los α indicados. La partición IID reparte cada clase entre clientes. Estas son configuraciones experimentales concretas, no una caracterización universal de toda heterogeneidad.

### Separación de datos

1. Se reserva un proxy balanceado de **10.000 ejemplos del conjunto de entrenamiento oficial**, antes de repartir datos privados.
2. Cada cliente recibe subconjuntos disjuntos: aproximadamente 65% `train`, 20% `holdout` y 15% `test`, con redondeo de tamaños. La separación local es aleatoria y no estratificada por clase.
3. `train` optimiza el teacher; `holdout` selecciona su checkpoint y estima competencia; `test` local solo evalúa posteriormente.
4. El test oficial solo evalúa el estudiante. No interviene en el target, el umbral, KD ni selección de modelos.

Reutilizar el holdout para escoger checkpoint y medir competencia puede producir optimismo por selección. No equivale a disponer de una calibración independiente.

### Preprocesado y modelos

- MNIST y Fashion-MNIST se transforman a 32×32 y tres canales. Ambos usan en el código las constantes de normalización `(0.1307, 0.3081)` por canal; no se estiman nuevas constantes en esta limpieza.
- CIFAR-10 usa normalización `(0.4914, 0.4822, 0.4465)` / `(0.2470, 0.2435, 0.2616)`. Solo el entrenamiento local añade crop aleatorio y flip horizontal.
- Holdout, test y proxy usan transformaciones deterministas. KD utiliza la vista de evaluación del proxy.
- `MnistNet`: tres convoluciones con batch normalization, dos max-pools y clasificador de 256 unidades con dropout 0,3. `ResNet9`: bloques convolucionales y dos bloques residuales. Las definiciones únicas están en `article1/models.py`.

## 4. Máscara: qué representa y qué no

Para teacher k y clase c:

\[
M_{k,c}=\mathbf 1[n^{holdout}_{k,c}>0\;\land\;Acc^{holdout}_{k,c}\geq\tau_d].
\]

La métrica es el acierto entre ejemplos cuya clase real es c. No es la precisión entre todas las predicciones emitidas como c. La máscara representa **competencia demostrada según ese criterio**, no exposición exacta durante entrenamiento ni corrección de cada muestra.

Un cero no prueba que toda probabilidad relativa a esa clase sea ruido. Puede haber datos insuficientes, rendimiento por debajo del umbral o información interclase útil que el criterio binario no recoge. Los umbrales son supuestos previos de esta versión, no valores óptimos demostrados.

Es válido que un teacher no tenga ninguna clase experta o que una clase no tenga experto. Se informan cobertura, seleccionados y fallback. No se rebaja retrospectivamente el umbral para mejorar accuracy o forzar cobertura total.

## 5. Métodos: flujo exacto hasta el target

Sean z_k(x) los logits, y(x) la etiqueta del proxy y p_k^T(x)=softmax(z_k(x)/T). Se definen dos operadores:

\[
L_T(w)=\operatorname{softmax}\left(\frac{\sum_k w_k z_k}{T}\right),\qquad
P_T(w)=\sum_k w_k\operatorname{softmax}(z_k/T).
\]

| Identificador | Selección o pesos | Target |
|---|---|---|
| `feddf_logit` | Todos, pesos 1/K | L_T(w) |
| `confidence_logit` | MSP de cada teacher a T=1; softmax entre teachers | L_T(w) |
| `consensus_logit` | Clase ganadora del promedio de probabilidades a T=1; selecciona teachers cuyo argmax coincide | L_T(w), uniforme entre seleccionados |
| `energy_logit` | `logsumexp(z_k)` a T=1; softmax entre teachers | L_T(w) |
| `expert_logit` | Selecciona k si M[k,y]=1 | L_T(w), uniforme entre seleccionados |
| `oracle_logit` | Selecciona k si argmax(z_k)=y | L_T(w), uniforme entre seleccionados |
| `expert_prob` | Mismo routing EXPERT | P_T(w) |
| `oracle_prob` | Mismo routing ORACLE | P_T(w) |
| `expert_prob_sr` | Mismo routing EXPERT | Máscara y renormalización por teacher antes del promedio de probabilidades |

Para SR:

\[
p^{SR}_{k,c}=\frac{M_{k,c}p^T_{k,c}}{\sum_jM_{k,j}p^T_{k,j}},\qquad
q^{SR}=\frac{1}{|S_E|}\sum_{k\in S_E}p^{SR}_k.
\]

«EXPERT-full» se refiere a conservar el soporte completo; los nombres inequívocos en código son `expert_logit` o `expert_prob`. El contraste full/support usa **`expert_prob` y `expert_prob_sr`**.

Detalles que no deben confundirse:

- Confidence no usa MSP dividido por su suma, sino softmax entre teachers.
- Consensus no es una moda de etiquetas duras y puede producir una selección vacía.
- Energy es sensible a escala y offsets aditivos por teacher; sus pesos no demuestran expertise.
- No hay calibración o normalización implícita de logits. Normalizar a probabilidades no elimina toda sensibilidad a su escala.
- SR actúa sobre probabilidades individuales; no pone logits a cero. Enmascarar con −∞ antes del softmax es equivalente por teacher, manteniendo después el mismo pooling.
- No están implementados `class-supported`, ORACLE+SR ni variantes adicionales de selección. `article1-v2` es el identificador del protocolo, no otra familia de métodos.

### Fallback y pérdida

Si EXPERT, ORACLE o Consensus no seleccionan teachers, **todas sus variantes** utilizan el mismo fallback: softmax(mean_k z_k / T), incluido el brazo de probabilidades y SR. Se registra el número de muestras afectadas; no se eliminan.

`article1/distillation.py` es la única implementación de targets. Normaliza el resultado, devuelve probabilidades float32 y la KD las consume directamente:

\[
\mathcal L_{KD}=T^2D_{KL}\left(q\parallel\operatorname{softmax}(s/T)\right).
\]

No se aplica un segundo softmax a q. La NLL del estudiante se evalúa con sus logits sin dividir por T. Las métricas del target corresponden a la temperatura de KD.

## 6. Receta y reproducibilidad

Teachers: Adam, lr=1e−3, batch 64, hasta 50 épocas y early stopping con paciencia 5 según accuracy de holdout. Se guarda un checkpoint seleccionado por cliente y un cache común por condición.

KD principal: **T=8**, AdamW, lr=1e−3, weight decay=1e−4, batch 256, 30 épocas. Con proxy de 10.000 ejemplos son **1200 actualizaciones**. Los métodos comparten cache, máscara, índices, inicialización y orden de batches. El código configura algoritmos deterministas y registra hashes de artefactos e identidad de ejecución.

La identidad KD incluye método, temperatura, receta y hashes del cache, máscara y proxy. El lanzador reutiliza una fila únicamente si su run_id coincide con la receta solicitada y el cache actual. Cambiar épocas, temperatura o cache deja de ser la misma ejecución. La existencia de una fila reutilizable no sustituye una auditoría de sus métricas.

La semilla es la unidad de réplica dentro de cada dataset–régimen. Se informan media, SD y puntos emparejados de tres seeds. Ni los clientes ni las combinaciones de datasets/regímenes se tratan como réplicas IID para inferencia.

## 7. Organización del repositorio

| Ruta | Responsabilidad |
|---|---|
| `README.md` | Única explicación y registro de decisiones/siguientes pasos |
| `article1/partitioning.py` | Proxy y particiones disjuntas |
| `article1/datasets.py`, `models.py` | Datos, transforms y arquitecturas |
| `article1/local_training.py` | Entrenamiento local, checkpoint y cache |
| `article1/distillation.py` | Máscara, routing, target y pérdida KD |
| `article1/runner.py` | CLI de las tres etapas: partition, teachers, distill |
| `run_article1_grid.py` | Lanzador explícito, dry-run y reutilización por identidad |
| `article1/hashes.py` | Huellas comunes y procedencia |
| `article1/conditions.py` | Tabla de cobertura/procedencia por condición |
| `article1/audit.py` | Comprobación de caches, máscaras, emparejamiento e invariantes de targets |
| `article1/rq2.py` | Verificador del experimento de temperatura ya existente |
| `article1/reproduce.py` | Dos ejecuciones de una celda para comprobar reproducción exacta |
| `article1/analysis.py` | Validación CSV, tres contrastes, tablas y figuras compartidas |
| `notebooks/article1_definitive_analysis.ipynb` | Secuencia corta de análisis y exportación |
| `tests/test_article1.py` | Pruebas rápidas, principalmente NumPy/CPU |

Se mantiene esta estructura pequeña para conservar rutas y caches existentes. No se añade un framework de experimentos.

### Datos y artefactos locales

`data/` y `OUTPUTS/` están excluidos de Git. El repositorio por sí solo **no contiene** los datos, caches ni CSV experimentales. Las rutas canónicas son:

- `OUTPUTS/article1/partitions/<dataset>-seed<seed>-<regime>/`: `proxy.npz`, `client_*.npz`, `metadata.json`.
- `OUTPUTS/article1/sources/<condición>/`: `teacher_cache.npz`, `metadata.json`, `teachers/teacher_*.pt`.
- `OUTPUTS/article1/results.csv`: resultados principales; el análisis selecciona explícitamente T=8.
- `OUTPUTS/article1/conditions.csv`: 54 filas, sin repetición por método.
- `OUTPUTS/article1/results_rq2_temperature.csv`: ejecuciones adicionales del estudio focal de temperatura.
- `OUTPUTS/article1/tables/` y `figures/`: exportaciones derivadas, regenerables.

El cache contiene logits float32 `[N,K,C]`, etiquetas e índices del proxy, M, aciertos/recuentos de holdout y evaluación local posterior. Las salidas históricas sin commit verificable conservan `legacy_unverified`; no se les atribuye el commit actual.

## 8. Instalación y uso

Python 3.11 o posterior. Crear un entorno virtual antes de instalar:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
```

`requirements.txt` mantiene las versiones de referencia del runtime de entrenamiento (incluido PyTorch). Para trabajar solo con CSV y pruebas rápidas, sin PyTorch:

```bash
python -m pip install -r requirements-dev.txt
```

`requirements-analysis.txt` reúne NumPy, pandas y matplotlib. El notebook necesita además Jupyter, incluido en `requirements.txt`; en el entorno ligero se puede instalar `jupyter>=1.1,<2`. La prueba de determinismo del runtime se omite explícitamente si PyTorch/torchvision no están disponibles. No se presenta esa omisión como validación de CUDA.

### Inspeccionar un plan sin entrenar

```bash
python run_article1_grid.py --dry-run
python run_article1_grid.py --stage distill --datasets cifar --regimes iid alpha0p1 single --seeds 42 --methods expert_logit expert_prob expert_prob_sr --dry-run
```

El grid T=8 existente está completo: **no debe repetirse sin una razón metodológica concreta**. Eliminar `--dry-run` ejecuta las etapas pendientes. No hay ninguna automatización de nuevos barridos.

### Una etapa o una celda explícita

```bash
python -m article1.runner partition --dataset mnist --regime iid --seed 42 --output OUTPUTS/article1/partitions/mnist-seed42-iid
python -m article1.runner teachers --dataset mnist --regime iid --seed 42 --partitions OUTPUTS/article1/partitions/mnist-seed42-iid --output OUTPUTS/article1/sources/mnist-seed42-iid --device cuda
python -m article1.runner distill --dataset mnist --seed 42 --method expert_logit --cache OUTPUTS/article1/sources/mnist-seed42-iid/teacher_cache.npz --results OUTPUTS/article1/results.csv --device cuda
```

Las etapas de partición y teachers ya completadas se conservan. Para un diseño distinto, usar otro `--output-root`; no sobrescribir ni reutilizar por nombre caches incompatibles.

### Temperatura: salida separada

Ejemplo de planificación del diseño existente, no de un nuevo barrido:

```bash
python run_article1_grid.py --stage distill --datasets cifar --regimes iid alpha0p1 single --seeds 42 43 44 --methods expert_logit expert_prob --temperatures 1 4 --results OUTPUTS/article1/results_rq2_temperature.csv --reuse-results OUTPUTS/article1/results.csv --dry-run
```

No se permite T≠8 hacia el CSV principal mediante el lanzador. La CLI de una sola celda permite una ruta explícita: quien la invoque debe respetar la misma separación.

### Tablas, análisis y niveles de validación

Para reconstruir `conditions.csv` se necesitan tanto caches como particiones:

```bash
python -m article1.conditions --source-root OUTPUTS/article1/sources --partition-root OUTPUTS/article1/partitions
```

Para analizar, colocar `results.csv` y `conditions.csv` en sus rutas canónicas y ejecutar el notebook desde Jupyter. Si está disponible `results_rq2_temperature.csv`, se incorpora y se exige que la comparación focal tenga 54 ejecuciones; si falta, se indica explícitamente y el análisis T=8 sigue disponible.

El notebook valida siempre cobertura, duplicados, métricas, presupuesto y coincidencia de hashes/routing **registrados en CSV**. Esto no verifica los archivos fuente ausentes. Su variable `VERIFY_CACHES=False` se imprime como validación CSV únicamente. Con `VERIFY_CACHES=True`, requiere los caches y añade las auditorías; cualquier error detiene el análisis.

Se exportan CSV derivados y figuras PNG/PDF. No se sobrescriben los resultados experimentales. Git conserva el notebook sin outputs voluminosos; sus tablas y figuras se regeneran. Los captions comunes son: líneas finas/puntos = seeds; marcadores/barras = media ± SD; diferencias = efecto emparejado en pp; eje = categorías ordenadas, sin distancias cuantitativas implícitas. Los rangos pequeños no constituyen una prueba de equivalencia.

Auditorías explícitas:

```bash
python -m article1.audit OUTPUTS/article1/results.csv --source-root OUTPUTS/article1/sources
python -m article1.rq2 --results OUTPUTS/article1/results.csv OUTPUTS/article1/results_rq2_temperature.csv --source-root OUTPUTS/article1/sources --temperatures 1 4 8 --isolated-results OUTPUTS/article1/results_rq2_temperature.csv
python -m compileall -q article1 run_article1_grid.py
python -m pytest -q
```

`audit` reconstruye targets para comprobar normalización, routing, fallback y repetibilidad, y comprueba procedencia/máscara de las fuentes. El verificador RQ2 contrasta además las métricas registradas con los targets reconstruidos de sus celdas focales. Ninguno verifica por sí solo que un estudiante histórico sea reproducible sin su ejecución y artefactos.

La comprobación de reproducción **entrena dos veces** una condición y escribe fuera del CSV principal:

```bash
python -m article1.reproduce --dataset cifar --seed 42 --method expert_logit --cache OUTPUTS/article1/sources/cifar-seed42-alpha0p1/teacher_cache.npz --results OUTPUTS/article1/reproducibility_check.csv --device cuda
```

No ejecutar este comando como una prueba rápida de limpieza. `reproduce` rechaza un destino llamado `results.csv` y guarda el informe JSON junto a su salida.

## 9. Estado científico verificado a 2026-09-07

La base de código inspeccionada antes de esta limpieza fue `baa0e2d`. Las cifras siguientes se recalcularon desde los CSV facilitados por el investigador; no son nuevas ejecuciones ni implican que esos CSV estén versionados en Git.

| Evidencia | Estado y lectura defendible |
|---|---|
| Grid principal | 486 filas T=8; 54 condiciones; nueve métodos; sin huecos ni duplicados en ese grid |
| EXPERT−FedDF | Ventaja creciente en las trayectorias medias de los tres datasets; 8/9 trayectorias por seed no decrecientes en los seis regímenes |
| Excepción a monotonía | CIFAR-10, seed 42: +40,67 → +37,49 pp entre α=0.1 y Multi |
| ORACLE−EXPERT | Media +0,2498 pp, MAE 0,5735 pp; cercanía descriptiva, no equivalencia |
| EXPERT-prob−EXPERT-logit, T=8 | Media +0,0870 pp, MAE 0,2778 pp; 50/54 pares dentro de ±1 pp |
| Temperatura focal | CIFAR-10, IID/α=0.1/Single, tres seeds, dos métodos, T=1/4/8: 54 ejecuciones completas |
| Ejemplo de dependencia de T | CIFAR-10 α=0.1: prob−logit = −3,92 ± 0,34 pp a T=1; +0,57 ± 0,35 a T=8 |
| Soporte restringido, T=8 | NLL del target mejora en 51/54 pares; accuracy del estudiante baja en 34/54; ambas cosas suceden en 32/54 |
| SR por condición | En CIFAR-10, pérdidas de 4,86–6,83 pp de media en IID/Dirichlet; Single gana 1,43 ± 0,80 pp de accuracy pero empeora NLL |
| Supervisado y tamaño del proxy | Pendiente; no se conoce aún un N suficiente ni un punto de cruce |

El CSV principal adjunto contiene 504 filas: 486 a T=8 y 18 a temperaturas inferiores, conservadas históricamente. El CSV aislado aporta 27 filas adicionales. El análisis filtra T=8 y combina las filas pertinentes para temperatura sin duplicarlas; esta limpieza no mueve ni reescribe filas originales. Nueve pares prob/SR a T inferior tienen cobertura incompleta y no se mezclan con RQ2-B a T=8.

### Límites y controles pendientes

- Las 486 filas T=8 recibidas carecen de `student_final_sha256`; las 54 condiciones declaran `legacy_unverified`. La coincidencia de hashes registrados no demuestra retrospectivamente determinismo ni procedencia completa.
- En MNIST-IID la máscara es completa, pero SR−full da −0,03, +0,07 y −0,02 pp. Debe esclarecerse con los caches y la procedencia antes de interpretar diferencias pequeñas como efectos del método.
- La normalización SR garantiza matemáticamente que q_y no disminuya cuando los seleccionados tienen M[k,y]=1. Mejorar la NLL del target no prueba que se haya eliminado conocimiento perjudicial. La accuracy del target tampoco es una garantía de utilidad KD.
- El daño por pérdida de estructura interclase es una interpretación compatible, todavía no una causa identificada. La reducción de entropía por sí sola no explica el signo de los cambios del estudiante.
- El código determina qué hace la versión actual. Una comprobación sintética o una reproducción de una celda no valida automáticamente todas las ejecuciones históricas.

## 10. Comparadores y literatura

`feddf_logit` es la adaptación one-shot del paso de ensemble distillation de FedDF, no una reproducción del protocolo federado iterativo completo. Confidence, Consensus y Energy son **controles internos**, no reproducciones externas SOTA.

Punto de partida bibliográfico: [FedDF, Ensemble Distillation for Robust Model Fusion in Federated Learning](https://arxiv.org/abs/2006.07242). La revisión focal de comparadores sigue siendo un siguiente paso: comprobar fuentes primarias y sus ecuaciones, información disponible, logits/probabilidades, etiquetas, rondas y coste. El objetivo es una selección defendible; no cumplir una cuota de métodos.

Selective-FD se dejó fuera de la comparación principal por diferencias de protocolo e información requerida en el análisis previo. Una eventual reconsideración debe justificarse con la fuente original. Los métodos generativos/data-free, class-supported y los híbridos no se incorporan automáticamente. No se afirma que el benchmark actual sea exhaustivo del SOTA.

## 11. SIGUIENTES PASOS

### A. Cerrar la evidencia base sin rehacer el grid

- [ ] Aclarar el control MNIST-IID con los caches y metadatos de seeds 42/43/44 y el informe de reproducción disponible. Primero comparar targets y procedencia; solo después decidir si hace falta una repetición mínima.
- [ ] Ejecutar la auditoría completa con las fuentes reales y registrar qué parte de la evidencia es verificable.
- [ ] Revisar las tablas/figuras regeneradas de los tres contrastes y fijar captions, claims y limitaciones con el investigador.
- [ ] Documentar cobertura/calidad de la máscara con los umbrales previos. No optimizarlos sobre el test ni perseguir fallback cero.
- [ ] Completar la revisión focal de comparadores antes de implementar nuevos métodos.

### B. Supervisado frente a destilación según tamaño del proxy

**Pregunta añadida al Artículo 1:** ¿cuánto conocimiento adicional aprovecha KD frente a entrenar el mismo modelo únicamente con las etiquetas de esos N ejemplos públicos? ¿A partir de qué N el supervisado alcanza un criterio de rendimiento útil, y dónde cambia el signo de KD−supervisado?

Esta parte aún no tiene tamaños N, presupuesto ni umbral de «rendimiento decente» congelados. No se inventan cruces a partir de los resultados T=8 actuales.

- [ ] Definir antes de ejecutar qué significa «decente»: un criterio de aplicación o referencia explícita, no un corte retrospectivo que favorezca KD. Si se seleccionan hiperparámetros o parada, usar validación separada; nunca test.
- [ ] Seleccionar una lista pequeña de tamaños N y subconjuntos anidados y reproducibles del **proxy ya reservado**, compartidos entre supervisado y KD.
- [ ] Mantener particiones privadas, teachers, máscara e índices maestros fijos al variar N. **No usar `--proxy-size` de la partición para este estudio:** cambiaría también los datos disponibles para los clientes. El futuro soporte N debe seleccionar filas del cache existente.
- [ ] Implementar un baseline supervisado desde la misma inicialización, arquitectura y muestras proxy, usando cross-entropy con sus etiquetas reales. KD usa los mismos N ejemplos y, además, el conocimiento de los teachers y la máscara.
- [ ] Acordar un presupuesto comparable antes de medir: fijar épocas no fija actualizaciones al variar N. Una referencia con updates emparejados responde a un contraste distinto de entrenar cada método hasta convergencia con validación. Decidir el principal y evitar un barrido de ambas cosas sin necesidad.
- [ ] Incorporar N, índices seleccionados, tipo de entrenamiento y receta a la identidad de las nuevas ejecuciones. Usar un CSV separado, por ejemplo `results_proxy_size.csv`; no mezclarlo con el grid histórico.
- [ ] Reutilizar el supervisado entre regímenes cuando dataset, seed, N, índices, inicialización y receta sean realmente idénticos. El supervisado no depende por sí mismo de cómo se repartieron los datos privados entre clientes; no repetirlo seis veces ni contarlo como seis réplicas independientes.
- [ ] Evaluar accuracy y NLL, diferencias emparejadas por N y seeds, y el intervalo entre tamaños ensayados en el que cambia el signo. Puede no haber cruce, haber varios o diferir por dataset/régimen; no asumir un umbral universal ni interpolar como si estuviera observado.

No están implementados todavía el brazo supervisado ni la selección N para KD. Se harán tras acordar este diseño mínimo, sin volver a entrenar teachers innecesariamente.

### C. Trabajos posteriores: fuera de la versión base

- Inferir exposición o competencia desde el modelo recibido; distinguir ambos objetivos y evaluar implicaciones de privacidad.
- Routing fiable con proxy sin etiquetas o con pocas etiquetas; posibles pseudoetiquetas, OOD o estimadores de fiabilidad por muestra.
- Estimadores de competencia continuos/incertidumbre si aportan una pregunta independiente frente a la máscara binaria.
- Segunda destilación y personalización con `class_mask`, distinta de restringir el ensemble de este artículo.

La asignación exacta de estas líneas a artículos posteriores no está fijada. No se importan resultados KD_v4 como evidencia canónica sin reproducción compatible.

## 12. Pautas de mantenimiento y cambios de esta limpieza

Pregunta científica → evidencia existente → decisión → implementación → verificación. Cada variante necesita una hipótesis independiente. Los cambios importantes de método, presupuesto y alcance se validan con el investigador antes de ejecutar. Se priorizan código legible, funciones pequeñas y pruebas de invariantes sobre frameworks y migraciones permanentes.

Esta limpieza:

- Retira `article1/backfill.py`, una migración de métricas ya realizada, y su prueba específica. Su versión histórica sigue en Git.
- Elimina del lanzador el modo especial `--rq2-temperature-sweep` y el permiso `--allow-nondefault-main-results`. Se utiliza el comando explícito de temperatura anterior y la reutilización general por identidad. Se mantiene el verificador `article1.rq2` porque sigue siendo útil para auditar evidencia existente.
- Centraliza el análisis compartido en `article1.analysis` y rehace el notebook sin definiciones duplicadas ni documentación científica paralela. Integra el contraste de soporte ya analizado y exporta PNG/PDF.
- Mantiene las fórmulas, modelos, particiones, umbrales y receta numérica. La configuración KD usada para identidad y ejecución tiene una sola definición.
- Separa dependencias ligeras de análisis/pruebas y entrenamiento; elimina la dependencia no utilizada de seaborn.
- Hace explícitas las rutas locales de particiones al generar condiciones; no calcula una huella aparentemente válida de un directorio inexistente.
- Corrige el destino por defecto de reproducción para proteger el CSV principal.

Validación de esta revisión: **20 pruebas correctas y una omitida** (determinismo del runtime, por ausencia de PyTorch), compilación correcta y **27 comparaciones de targets antes/después exactamente iguales** (nueve métodos × T=1,4,8 con entradas sintéticas). Las seis celdas de código del notebook se ejecutaron en proceso con los CSV facilitados, generando 13 tablas y cinco figuras en PNG/PDF sin modificar los CSV fuente. El entorno impidió arrancar un kernel Jupyter por restricciones de sockets; la verificación se hizo ejecutando las celdas secuencialmente. La ejecución de entrenamiento/CUDA y la auditoría con caches reales no se sustituyen por estas comprobaciones. No se ejecutó nueva KD.
