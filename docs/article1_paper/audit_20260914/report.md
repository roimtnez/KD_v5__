# Artículo 1: auditoría experimental cerrada — 14 de septiembre de 2026

El [notebook único ejecutado](../../../notebooks/article1_paper.ipynb) verifica y
analiza las ejecuciones individuales locales. **471 students principales/acordados
válidos y emparejados; ningún pendiente requerido.** No se confundió la finalización
de procesos con la cobertura: se comprobaron las identidades previstas una a una.

| block | expected | found | valid | paired | pending |
| --- | --- | --- | --- | --- | --- |
| teachers_masks | 54 | 54 | 54 | 54 | 0 |
| baseline | 324 | 324 | 324 | 324 | 0 |
| presence | 54 | 54 | 54 | 54 | 0 |
| CE_10000 | 9 | 9 | 9 | 9 | 0 |
| curve_CE | 12 | 12 | 12 | 12 | 0 |
| curve_EXPERT | 36 | 36 | 36 | 36 | 0 |
| curve_FedDF | 36 | 36 | 36 | 36 | 0 |

Los 324 baseline son seis métodos × 54 condiciones. CE se cuenta una sola vez por
dataset/seed/N: repetirlo en paneles de regímenes no añade réplicas. La curva CIFAR utiliza **105 students únicos**: 15 CE, 45 EXPERT y 45 FedDF.
De ellos, 21 son anclas N=10000 ya contadas en baseline/CE y 84 son adicionales.
Generan 135 pares (45 por contraste); estas cifras de pares no son nuevos entrenamientos.

## Fuentes y verificación

- Selección explícita: `OUTPUTS/article1_v3/results_baseline.csv`,
  `results_supervised_proxy.csv`, `results_proxy_size_expert_prob.csv`,
  `results_presence.csv`, `results_proxy_size_feddf_prob.csv`, más
  `results_rq2_backup.csv` para opcionales compatibles.
- Se localizaron las fuentes mediante inventario de `OUTPUTS`, búsqueda en
  `/home/roi`, `/tmp`, `/mnt` y `/media`, y lectura de `launch_*.json`. Los dos
  lanzamientos del 10 de septiembre enlazan las 36 filas FedDF con las referencias
  EXPERT correctas. Su ruta efectiva es `OUTPUTS/article1_v3` en este entorno;
  el notebook la fija explícitamente.
- Se inventariaron CSV de fases e instantánea histórica
  `OUTPUTS/article1_analysis/20260909T144508572182Z/`; no se combinaron automáticamente
  con las fuentes seleccionadas. Las tablas del 9 de septiembre no suministran
  observaciones al análisis actual.
- Se verificaron 54 caches, particiones y subconjuntos anidados; etiquetas locales,
  recuentos, regla M, máscara de presencia exclusivamente de train y vínculo con M.
  540 checkpoints: SHA256 de bytes y hash de estado deserializado se comprobaron
  por separado (`weights_only=True`, CPU).
- Se comprobaron identidades, ausencia de conflictos en las fuentes seleccionadas,
  valores finitos, recetas completas, T=8, N, 1200 actualizaciones, SR revisión 2,
  inicialización y hashes de batches consumidos. Los targets y la masa previa de SR
  se reconstruyeron con los caches verificados. La masa es promedio sobre eventos
  teacher/muestra seleccionados, excluyendo fallback.
- No se regeneraron logits mediante inferencia ni se reevaluó el test de students.
  Sus métricas están auditadas por registros y procedencia; esta limitación no se
  confunde con verificación independiente de rendimiento. No se modificaron entradas
  ni se lanzaron corridas experimentales nuevas. La prueba final confirmó hashes
  originales sin cambios.

El [manifiesto](manifest.json) conserva rutas, SHA256, fechas, commits declarados,
código observado, exclusiones y alcance. Las copias de entradas permanecen locales
bajo `OUTPUTS/article1_paper/snapshots/`. [runs.csv](tables/runs.csv) conserva cada
identidad, fuente y fila; las tablas de pares conservan ambos IDs. Los archivos
publicados aquí son salidas de la ejecución, no nuevas fuentes del notebook.
Los 54 diagnósticos de máscaras se distribuyen juntos en un ZIP verificable para
mantener compacto el PR; los 108 archivos PNG/PDF individuales siguen disponibles
en `OUTPUTS/article1_paper/figures/` y se regeneran al ejecutar el notebook.

## Resultados defendibles

**Competencia frente a presencia.** En CIFAR-10, EXPERT−presencia tiene diferencias
medias de accuracy de +1.37 ± 0.15 pp (α=1), +5.25 ± 0.94 pp (α=.5) y
+3.18 ± 2.91 pp (α=.1). En α=1 y α=.1 sube el NLL: no hay mejora uniforme de ambas
métricas. En IID la diferencia media de accuracy es −0.08 ± 0.39 pp. Multi y Single
usan A=M y producen resultados coincidentes. El contraste puede cambiar también
cuántos teachers participan; no aísla un efecto exclusivo de calidad.

**SR.** En CIFAR baja el NLL del target en los seis regímenes, pero sube el del
student en los seis. La accuracy del student cae en cinco; Single es la excepción
(+0.57 ± 0.87 pp). MNIST y Fashion-MNIST tienen otras excepciones explícitas en las
tablas. Renormalizar el soporte que contiene la etiqueta verdadera explica
algebraicamente el NLL del target. No garantiza mejorar el student ni demuestra
destrucción de dark knowledge: SR cambia información y concentración.

**Presupuesto proxy.** En CIFAR IID, EXPERT−FedDF tiene accuracy media negativa en
los cinco tamaños; en α=.1 y Single es positiva. Frente a CE el signo depende de N:
para IID, EXPERT−CE pasa de +7.91 ± 0.68 pp en N=100 a −3.73 ± 0.89 pp en N=10000.
La curva usa 1200 actualizaciones, no un presupuesto fijo de ejemplos, y N=100
reduce el batch efectivo. Los cruces observados no son N óptimos ni umbrales
universales. Se muestran ambas métricas y las tres seeds de CE.

La [interpretación numérica completa](observed_results.md) se genera desde los pares;
[contrasts.csv](tables/contrasts.csv) y [curve_summary.csv](tables/curve_summary.csv)
contienen medias, SD, n y seeds. No se agrupan datasets o regímenes, ni se hacen tests
automáticos de significación. Tres seeds limitan la inferencia.

## Figuras y tablas

| Estudio | Figuras PNG (también PDF) | Tablas CSV (también LaTeX) |
|---|---|---|
| Panorama global | [Referencias](figures/overview_references.png), [ablaciones](figures/overview_ablations.png) | [Niveles](tables/absolute.csv) |
| Selección y competencia | [Contrastes](figures/selection_competence.png), [routing](figures/routing.png) | [Diferencias](tables/contrasts.csv), [presencia](tables/competence_presence_pairs.csv) |
| Espacio de agregación | [Pooling](figures/aggregation_space.png) | [EXPERT pooling](tables/pooling_expert_pairs.csv) |
| Target–student y SR | [Cinco métricas](figures/support_target_student.png), [masa real](figures/support_mass.png) | [Pares y masa](tables/support_mass_pairs.csv) |
| Etiquetas públicas y N | [Solo CE](figures/ce_only.png), [absolutas](figures/proxy_absolute.png), [diferencias](figures/proxy_differences.png) | [CE únicos](tables/ce_unique.csv), [pares](tables/curve_pairs.csv) |
| Máscaras | [Relación con ventajas](figures/mask_advantage.png), [54 diagnósticos PNG/PDF en ZIP](figures/mask_diagnostics_supplement.zip) | [5400 celdas](tables/mask_cells.csv), [expertos/clase](tables/experts_per_class.csv), [ejemplos](tables/representative_masks.csv) |

Los [captions](tables/captions.csv) detallan signo, incertidumbre y límites.
Principales: panorama, selección/competencia, pooling, target/student, CE sola y
diferencias por N. Routing ampliado, masa, curvas absolutas y máscaras completas
pueden ir al suplemento. Los ejemplos de máscaras usan la seed mediana por
recuento de discrepancias A/M, desempate por seed; no se eligieron por rendimiento.

## Opcionales y pendientes

Los bloques requeridos no tienen pendientes: [lista exacta](tables/pending_conditions.csv).
Cada uno de EXPERT-logit, confidence-logit, consensus-logit y energy-logit tiene
41/54 ejecuciones válidas recuperadas del respaldo (164 en total). Faltan, por cada
método: CIFAR seeds 43/44 en los seis regímenes y CIFAR seed 42 Single. La
[matriz completa](tables/optional_coverage.csv) conserva esos huecos.
Pooling EXPERT en CIFAR se limita a seed 42 de cinco regímenes; no existe par Single.
No se encontraron temperaturas KD distintas de T=8.

El control de entropía/concentración, unlabeled proxy, inversión de modelos y
personalización son trabajo futuro. ORACLE no es una cota garantizada. Los controles
internos no constituyen una comparación exhaustiva con SOTA. Related Work necesita
la revisión bibliográfica correspondiente antes del envío.

## Estructura propuesta

1. Introducción: pregunta de selección/competencia y contribuciones respaldadas.
2. Related Work: ensemble distillation y one-shot FL, sin vender los controles como SOTA.
3. Método: A/M, selección, pooling, fallback y SR.
4. Protocolo: splits, trazabilidad, ausencia de leakage y presupuesto/comparabilidad.
5. Resultados: selección, presencia, pooling, target–student y proxy N.
6. Discusión: heterogeneidad, alcance de n=3 y mecanismos no aislados.
7. Conclusiones y futuro: resultados condicionados al protocolo y controles pendientes.

Cada sección tiene pregunta, evidencia, figura/tabla y límite en
[article_structure.csv](tables/article_structure.csv). El mapa
[hipótesis → contraste → resultado → afirmación → limitación → estado](tables/claim_evidence.csv)
forma parte del notebook.

## Ejecución y validación

El kernel Jupyter ejecutó las 11 celdas de código completas, sin errores. El primer
arranque estuvo bloqueado por sockets locales del sandbox; se ejecutó con permiso
para kernel local, sin recurrir a una simulación de salidas. Se revisaron visualmente
panoramas, contrastes, pooling, SR, curvas y diagnósticos de máscaras.

96 pruebas pasan en la suite sin `test_proxy_training.py`; la suite analítica
focal pasa 58 pruebas (subconjunto de las anteriores). En la primera pasada general
se ejecutaron también pruebas sintéticas de pocos pasos en directorios temporales;
no produjeron corridas experimentales. Se corrigieron cuatro mocks del pipeline
para aceptar `seeds`, preservando sin cambios los scripts de ejecución preexistentes. Se sustituyó la prueba editorial obsoleta que exigía usar
medias publicadas y omitía las ejecuciones posteriores.

Los resultados originales y los informes históricos se conservaron. El esquema
nbformat, las identidades, los pares requeridos y los hashes de entradas se
comprobaron al finalizar. Las versiones y huellas de entrega están en
[validation.json](validation.json).

## Entrega local

Por indicación del usuario, se canceló la publicación en GitHub y se deshizo el
commit local conservando todos los archivos. No se creó rama remota ni PR.
Los cambios quedan disponibles para revisión y publicación manual.

Se conservó una reejecución local posterior del mismo notebook (10:44 UTC):
los hashes de todas las entradas y las 44 tablas CSV coinciden con la ejecución
validada inicial; cambian metadatos de ejecución y algunos colores automáticos.
Los 14 gráficos incrustados coinciden con las exportaciones seleccionadas para
esta entrega. Esta selección explícita se registra en `validation.json`.
