# Artículo 1: estado del análisis — 9 de septiembre de 2026

> Registro histórico de la instantánea de las 14:45 UTC. El estado posterior, con curva completa, está en [article1_closure/report.md](article1_closure/report.md); el plan actual está en [article1_paper/plan.md](article1_paper/plan.md).

Instantánea UTC: **2026-09-09 14:45:08** (16:45:08 en Madrid).
Código ejecutado: `67c328b285eebad4e203d2a57cd868b0df1fe735`, rama
`cleanup/article1-base`. Se trabajó en esta rama, sin otro checkout.

La instantánea local está en
[`OUTPUTS/article1_analysis/20260909T144508572182Z/`](../OUTPUTS/article1_analysis/20260909T144508572182Z/).
Contiene [manifiesto](../OUTPUTS/article1_analysis/20260909T144508572182Z/manifest.json),
[informe](../OUTPUTS/article1_analysis/20260909T144508572182Z/report.md), tres notebooks
 ejecutados, 55 tablas CSV y 93 figuras en PNG y PDF. `OUTPUTS/` queda fuera de Git;
este documento registra el estado y las rutas, no sustituye las entradas congeladas.

## Completitud observada

| Ámbito | Presentes y válidas | Pendientes |
|---|---:|---:|
| Baseline configurado: 54 condiciones × 6 métodos | 324 | 0 |
| Controles y EXPERT-logit en respaldo RQ2 | 164 | 52 |
| Cuadrícula original de diez métodos | 488 | 52 |
| CE supervisado N=10000 | 9 | 0 |
| Curva CIFAR N=100/500/1000/5000: 12 CE + 36 KD | 0 | 48 |

MNIST y Fashion-MNIST tienen 180 filas cada uno al incorporar el respaldo. CIFAR
tiene 128/180: sus seis métodos principales están completos. Se verificaron los
54 caches, sus particiones, etiquetas, regla M y procedencia registrada; las
488 filas KD y nueve CE utilizadas superaron las comprobaciones implementadas.

Los seis contrastes principales (selección, operadores, expertise y SR) tienen
54 pares. EXPERT-prob−EXPERT-logit y los tres controles frente a FedDF-logit tienen
41 pares cada uno, con 13 condiciones excluidas por falta de algún brazo. Se
mantienen seeds, diferencias, medias y SD muestrales por dataset/régimen; no se
consideran los clientes como réplicas.

## Resultados observados a N=10000

CE supervisado, tres seeds por dataset, media ± SD muestral:

| Dataset | Accuracy (%) | NLL |
|---|---:|---:|
| MNIST | 98.623 ± 0.241 | 0.0494 ± 0.0062 |
| Fashion-MNIST | 89.773 ± 0.360 | 0.3963 ± 0.0462 |
| CIFAR | 79.013 ± 0.340 | 0.8754 ± 0.0265 |

En CIFAR, EXPERT-prob−CE, emparejado por seed/proxy/inicialización/batches/receta
compartida, con 1200 actualizaciones y tres pares por régimen:

| Régimen | Diferencia accuracy (pp), media ± SD | Diferencia NLL, media |
|---|---:|---:|
| IID | −3.730 ± 0.885 | −0.1132 |
| alpha0p1 | −2.643 ± 1.049 | +0.1533 |
| single | −0.253 ± 1.290 | +0.8369 |

A este tamaño, las medias de accuracy de EXPERT-prob son inferiores a CE en los
regímenes focales; en IID, su NLL media es menor. Esto muestra por qué conviene
conservar ambas métricas. No es una conclusión de equivalencia o superioridad
universal, ni permite inferir qué ocurre con proxies menores. CE se reutiliza
entre regímenes: no son nueve réplicas CE nuevas.

Los resultados completos de KD se encuentran en
[`contrast_summary.csv`](../OUTPUTS/article1_analysis/20260909T144508572182Z/tables/contrast_summary.csv),
y los pares CE/KD en las tablas `proxy_<regime>_paired.csv`. No se seleccionaron
variantes, temperaturas, thresholds o tamaños mediante el test.

## Dependencias y pendientes

`results_selection.csv` original contiene las 108 identidades necesarias para
`supervised`. En la captura faltaban `results_expertise.csv` y
`results_proxy_size_expert_prob.csv` originales. Por archivos no se puede confirmar
que `proxy-curve` haya superado su auditoría inicial; la ausencia del CSV tampoco
permite afirmar el estado de un proceso.

Se prepararon cuatro bloques completos, exclusivamente bajo
[`exports/`](../OUTPUTS/article1_analysis/20260909T144508572182Z/exports/):
selection (108), pooling (108), expertise (54) y support_v2 (54). El bloque
[`results_expertise.csv`](../OUTPUTS/article1_analysis/20260909T144508572182Z/exports/results_expertise.csv)
permite preparar la dependencia sin repetir KD. No se colocó en el directorio de
la cola activa. Controles y EXPERT-logit focal siguen pendientes; no se generaron
CSV vacíos ni se ejecutó el exportador global de baseline sobre una cuadrícula
original incompleta.

No se entrenó ni se hizo inferencia, no se usó CUDA y no se modificaron el pipeline,
las colas, los CSV originales, las particiones, los caches o los checkpoints.
Los hashes de estados de checkpoints no se recalcularon porque no se cargaron;
se registraron presencia y hashes de bytes, y se comprobó la consistencia del
fingerprint declarado. Esta limitación impide presentar la revisión como una
verificación independiente de esos estados.

Validation selecciona el checkpoint; expertise construye M. La accuracy de las
celdas acreditadas no constituye una validación independiente de generalización
ni rechazo OOD. Los soportes escasos son descriptivos, sin nuevas reglas.

Validación técnica: **12 pruebas superadas** y los tres notebooks ejecutados sin
errores, en CPU con un hilo. Una nueva ejecución del comando documentado en el
[README](../README.md#actualizar-el-análisis-durante-supervised-y-proxy-curve)
crea otra instantánea para incorporar la curva cuando sus filas estén disponibles.
