# KD_v5__ — Article 1

Repositorio pequeño de investigación sobre **one-shot federated knowledge distillation**. El objetivo es determinar qué aporta una máscara de competencia cliente–clase y cómo deben contribuir los expertos al target del estudiante global.

**Estado: 2026-09-08. Protocolo activo: `article1-v3`.** Esta versión cambia el reparto local a `train / validation / expertise = 70/10/20`, con estratificación por clase y estimación de expertise independiente de la selección del checkpoint. No contiene resultados científicos nuevos de v3. Los resultados previos de v2 son evidencia histórica; no se atribuyen al nuevo protocolo.

## 1. Alcance del artículo

- MNIST, Fashion-MNIST y CIFAR-10; 10 clientes; seeds 42, 43 y 44.
- Un entrenamiento local por teacher, un cache común de salidas sobre el proxy y un único student global. Sin rondas posteriores de comunicación.
- Proxy público etiquetado y máscara binaria disponibles. EXPERT no es label-free.
- Umbrales previos: MNIST 0.90, Fashion-MNIST 0.80, CIFAR-10 0.70. No se declaran óptimos ni se ajustan sobre test para mejorar resultados o eliminar fallbacks.
- Tres contrastes controlados y una comparación adicional frente al entrenamiento supervisado con las etiquetas del mismo proxy.

| Pregunta | Contraste | Se mantiene fijo |
|---|---|---|
| WHO: quién contribuye | `expert_logit − feddf_logit` | Agregación de logits, teachers, proxy, inicialización, orden y presupuesto |
| HOW: cómo se agrega | `expert_prob − expert_logit` | Routing EXPERT y temperatura de cada contraste |
| WHAT: qué soporte se conserva | `expert_prob_sr − expert_prob` | Routing y media de probabilidades |

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

Logits y probabilidades no son operadores equivalentes. En SR, la probabilidad normalizada de la clase verdadera no puede disminuir para un experto seleccionado; mejorar NLL del target es parcialmente consecuencia de la construcción y no prueba que el conocimiento eliminado perjudique KD. No se implementan class-supported ni ORACLE+SR.

## 6. KD, supervisado y tamaños del proxy

`article1/runner.py::_train_proxy` comparte arquitectura, inicialización, orden, transformaciones, optimizador y evaluación entre CE y KD. AdamW, lr=1e-3, weight decay=1e-4, batch solicitado 256, sin scheduler. La receta KD principal es T=8, 30 épocas: con 10.000 ejemplos son 1200 actualizaciones.

KD consume q directamente con `T² KL(q || softmax(student/T))`, sin segundo softmax del target. CE consume etiquetas enteras. El student se evalúa al terminar; su NLL usa logits sin dividir por T.

`article1/proxy.py::proxy_positions` permite curvas de tamaño N: subconjuntos anidados estratificados del proxy ya reservado, sin cambiar teachers ni datos privados. El proxy completo conserva su orden. No confundir `distill/supervised --proxy-size` con `partition --proxy-size`, que cambia la reserva de datos.

Para tamaños reducidos se exige `--updates` explícito en KD. Supervisado usa 1200 por defecto. N<256 reduce el batch efectivo a N; se conserva el batch final incompleto. Presupuesto igualado no significa convergencia óptima ni idéntico número de ejemplos vistos entre tamaños.

Se registran tamaño/hash del proxy maestro y seleccionado, etiquetas, política de selección, receta JSON, inicialización, estado final, updates, ejemplos vistos y épocas. `consumed_batches_sha256` incluye índices realmente consumidos y fronteras de batch. `batch_order_sha256` conserva la huella de permutaciones completas.

La identidad supervisada excluye régimen y teachers: puede reutilizarse entre regímenes si coinciden dataset, seed, datos, inicialización y receta. No contarlo como réplicas independientes. La identidad KD incluye cache, máscara, proxy, método, temperatura y receta. Las filas de protocolos diferentes no se mezclan. Una invocación directa repetida vuelve a entrenar y hace upsert por run_id.

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
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Solo análisis/pruebas ligeras: `pip install -r requirements-dev.txt`. El notebook de particiones carga etiquetas de torchvision y necesita PyTorch/torchvision, además de Jupyter. Las pruebas de runtime se omiten explícitamente si faltan esas dependencias.

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
python -m article1.runner distill --dataset mnist --seed 42 --method expert_logit --cache OUTPUTS/article1_v3/sources/mnist-seed42-iid/teacher_cache.npz --results OUTPUTS/article1_v3/results.csv --device cuda
```

Para planificar más celdas sin ejecutarlas:

```bash
python run_article1_grid.py --stage distill --datasets mnist --regimes iid --seeds 42 --methods feddf_logit expert_logit oracle_logit --dry-run
```

No ejecutar el grid entero antes del control pequeño. Los outputs de teachers no vacíos se rechazan; un entrenamiento interrumpido se inspecciona antes de decidir cómo repetirlo. No hay sobrescritura automática de checkpoints ni resampling de particiones.

### C. Supervisado y un tamaño del proxy

```bash
python -m article1.runner supervised --dataset mnist --seed 42 --cache OUTPUTS/article1_v3/sources/mnist-seed42-iid/teacher_cache.npz --results OUTPUTS/article1_v3/results_supervised_proxy.csv --updates 1200 --device cuda
python -m article1.runner supervised --dataset cifar --seed 42 --cache OUTPUTS/article1_v3/sources/cifar-seed42-iid/teacher_cache.npz --results OUTPUTS/article1_v3/results_proxy_size.csv --proxy-size 1000 --updates 1200 --device cuda
python -m article1.runner distill --dataset cifar --seed 42 --cache OUTPUTS/article1_v3/sources/cifar-seed42-iid/teacher_cache.npz --results OUTPUTS/article1_v3/results_proxy_size.csv --method expert_logit --temperature 8 --proxy-size 1000 --updates 1200 --device cuda
```

N=1000 es ejemplo de sintaxis, no una orden para lanzar un barrido. Estos brazos rechazan una salida llamada `results.csv`. KD a temperaturas distintas de 8 debe ir también a un CSV separado; el lanzador lo exige.

### D. Comprobar código y resultados

```bash
python -m compileall -q article1 run_article1_grid.py
python -m pytest -q
python -m article1.conditions
python -m article1.audit OUTPUTS/article1_v3/results.csv
jupyter notebook notebooks/article1_definitive_analysis.ipynb
```

Conditions exige las 54 fuentes. El notebook definitivo exige el grid de 486 filas T=8: no se presenta como ejecutable con resultados v3 aún ausentes. Para verificación completa de caches activar `VERIFY_CACHES`; comprobar CSV y huellas registradas no sustituye reconstruir los targets con las fuentes reales.

## 9. Estado histórico y reinicio

La versión anterior a este cambio es el commit `653a145ba4cc2367820d1be5bdf05e1029d83ef3` (protocolo v2, train/holdout/test local, 65/20/15). Su código sigue disponible en Git. Sus artefactos permanecen en `OUTPUTS/article1/`, sin migración ni borrado.

Evidencia histórica disponible: 486 ejecuciones T=8 (54 condiciones × 9 métodos), experimento focal de temperatura y análisis de soporte. Mostraban ventaja creciente de EXPERT con especialización, cercanía descriptiva a ORACLE, dependencia de temperatura del contraste logits/probabilidades y casos donde mejorar NLL del target con SR empeoraba al student. Estas observaciones no se convierten en resultados v3 por cambiar nombres de archivos.

El nuevo reparto exige nuevos teachers, máscaras y KD. El proxy y la asignación entre clientes se mantienen algorítmicamente; los supervisados históricos pueden ser numéricamente reutilizables si se verifica la identidad completa, pero no se importan ni reetiquetan automáticamente como v3. Los caches v2 son rechazados por el runtime activo.

Los caches históricos con commit desconocido conservan `legacy_unverified`. El control MNIST-IID de identidad de targets y pequeñas diferencias del student sigue pendiente de aclaración sobre las fuentes históricas; no justifica mezclar ambos protocolos.

## 10. SIGUIENTES PASOS

- [ ] Crear particiones v3 de los tres datasets/seeds y revisar heatmaps, recuentos escasos y ausencia de conjuntos vacíos. No usar resultados del test para modificar este reparto.
- [ ] Validar una condición pequeña: entrenamiento local, selección en validation, M desde expertise, targets y repetición exacta KD. CUDA debe comprobarse en el entorno experimental.
- [ ] Autorizar y ejecutar después el grid v3, manteniendo common random numbers y registro de procedencia. No reutilizar KD histórica como evidencia del nuevo reparto.
- [ ] Cerrar los tres contrastes y sus captions con puntos emparejados por seed, media y SD. Clients no son réplicas independientes; no inferir causalidad de correlaciones agregadas.
- [ ] Ejecutar el supervisado sobre el proxy completo cuando existan caches v3; acordar el lanzamiento de la curva focal. Propuesta previa: N=100/500/1000/5000/10000, CIFAR, IID/α=0.1/single, seeds 42/43/44, EXPERT-logit. Las cuentas de ejecuciones previas suponían reutilizar KD v2 y deben recalcularse para v3.
- [ ] Añadir al notebook las curvas KD−supervisado y cualquier intervalo de cruce observado. No se conoce un tamaño suficiente ni un cruce; «rendimiento decente» necesita un criterio previo explícito.
- [ ] Revisar posicionamiento bibliográfico y comparadores compatibles antes de añadir métodos.
- [ ] Redactar el artículo desde las figuras: observación, interpretación compatible, claim defendible y limitación. Evitar equivalencia EXPERT/ORACLE, superioridad universal de logits o atribuir causalmente a dark knowledge cualquier pérdida con SR.
- [ ] Reservar para estudios posteriores estimación de máscaras, proxy sin etiquetas y personalización con class_mask.

La separación de expertise evita su reutilización para seleccionar el checkpoint. No elimina la incertidumbre con pocos ejemplos, la dependencia de umbrales, el coste informacional de M+y ni el conocimiento previo de resultados del benchmark.

## 11. Validación de esta revisión

- 40 pruebas correctas con PyTorch/torchvision CPU; compilación y comprobaciones de estilo de los archivos modificados correctas.
- Una prueba sintética modifica solo las predicciones observadas en expertise: cambia M, pero conserva exactamente checkpoints seleccionados, registros de selección y logits del proxy. No es un resultado científico de los datasets.
- Se generaron y validaron particiones reales de MNIST, seed 42, en los seis regímenes. Frente al código anterior, coinciden exactamente el proxy y la asignación privada a cada cliente; cambia solo la división local.
- Las seis celdas de código del notebook de particiones se ejecutaron secuencialmente en proceso, generando cuatro figuras PNG/PDF y dos tablas. El entorno impidió iniciar el kernel Jupyter por restricciones de sockets; queda comprobar esa vía de ejecución en el entorno del investigador.
- No se entrenaron teachers ni students sobre datasets reales. No se ha validado CUDA ni completado una cuadrícula científica v3.
