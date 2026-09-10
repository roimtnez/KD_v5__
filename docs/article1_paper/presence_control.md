# Control de presencia y diagnóstico Target vs Student

## Pregunta y decisión

¿Aporta medir competencia por clase más que saber qué clases recibió cada teacher?
El baseline existente compara EXPERT con agregación uniforme. Ese contraste mezcla
conocer el soporte privado y medir accuracy. El nuevo control separa ambas reglas:

- `presence_prob`: A[k,c] = 1 si hay al menos un ejemplo de c en **train** del cliente k.
- `expert_prob`: M[k,c] = 1 si hay evidencia en expertise y su accuracy supera el umbral.

Ambos seleccionan usando la etiqueta pública y promedian vectores de probabilidades
completos a T=8. Mantienen fallback FedDF-logit, teachers, proxy, inicialización,
orden de batches y presupuesto. No se aplica SR al control de presencia. No se
interpreta ausencia de expertise como ausencia de clase en train.

La presencia se reconstruye de los índices originales y las etiquetas oficiales de
entrenamiento. Se verifican manifiestos, identidad dataset/seed/régimen, huella de
las etiquetas y coincidencia del proxy con el cache. No se modifica ningún cache ni
checkpoint. Los counts de validation/expertise y el test no construyen A.

## Ejecución

```bash
# Mostrar los 54 comandos, sin entrenar:
python run_article1_pipeline.py --phase presence

# Ejecutar solo el nuevo control y producir su comparación:
python run_article1_pipeline.py --phase presence --execute --device cuda
```

Son 3 datasets × 6 regímenes × 3 seeds: **54 KD nuevas**, 30 épocas sobre N=10000
(1200 actualizaciones, batch 256), receta del baseline existente. Antes de empezar
se exige el EXPERT completo y compatible en `results_baseline.csv` o, si este no
existe, `results_expertise.csv`. No se añade este brazo al inventario histórico de
seis métodos ni se reinician las fases ya cerradas.

Salida: `OUTPUTS/article1_v3/results_presence.csv`; análisis emparejado:
`OUTPUTS/article1_v3/presence_pairs.csv`. Se reanudan únicamente identidades
compatibles. `M_sha256` identifica la máscara **efectivamente usada** en cada nueva
fila; `expertise_M_sha256` conserva la huella del M original. La identidad del run
incorpora el método y la máscara de presencia. Los CSV antiguos no cambian.

El contraste es **EXPERT − presencia**. El análisis permite que difiera la máscara,
pero exige las mismas fuentes, receta y trazas de entrenamiento. La comparación
canónica SR − EXPERT sigue exigiendo la misma máscara y routing. No se debe pasar
`results_presence.csv` al auditor de seis brazos: tiene un contraste propio.

La selección por presencia puede incluir más teachers; el contraste mide la utilidad
del procedimiento completo de acreditación, no un efecto independiente del número
de teachers. No prueba superioridad frente a toda regla imaginable basada en soporte.
No se ha ejecutado aún este control en este entorno.

## Target vs Student

El notebook `notebooks/article1_paper.ipynb` presenta, con signo **SR − full EXPERT**:

1. Accuracy del target sobre proxy a T=8.
2. NLL del target sobre proxy a T=8.
3. Entropía del target a T=8.
4. Accuracy del student en test oficial a T=1.
5. NLL del student en test oficial a T=1.

Las barras son SD de tres diferencias emparejadas, no intervalos de confianza.
Los niveles de target y student no son comparables como un gap de generalización:
cambian modelo, conjunto y temperatura. La precisión reportada aquí es accuracy,
no precision positiva de una clase.

La masa observada antes de SR es el promedio de
`1 - sum_c M[k,c] p[k,c](x)` sobre eventos teacher/muestra seleccionados, excluyendo
fallback. Es exactamente el estimando persistido en el runner; no es necesariamente
el promedio por muestra de masa eliminada del ensemble. Se verifica su igualdad
entre brazos. No se aproxima como fracción de clases desconocidas. IID no obliga
M=1 y no implica masa cero. Las filas de ese diagnóstico requieren el snapshot
`OUTPUTS/article1_analysis/*/tables/baseline_validated.csv`; el notebook comprueba
que sus contrastes coinciden con el cierre publicado. Puede fijarse `VALIDATED`
explícitamente para elegir una instantánea concreta.

Al renormalizar cada teacher seleccionado, la probabilidad de la etiqueta verdadera
no disminuye. Por ello una bajada del NLL del target es esperable algebraicamente;
no es evidencia independiente de utilidad para el student. En CIFAR publicado baja
en 6/6 regímenes, mientras la accuracy del student baja en 5/6 y su NLL sube en 6/6.
La excepción de accuracy es `single`.

Esto es compatible con pérdida de información interclase, pero SR también cambia
concentración. No demuestra que toda masa externa sea dark knowledge útil ni un
estimador de rechazo. Un control de concentración/entropía con soporte completo
sería un estudio adicional; no se afirma haberlo realizado.

## Limpieza verificable

- Eliminadas las celdas duplicadas de SR, el signo invertido y la tabla de masa
  basada en números supuestos; sustituidas por evidencia emparejada y valores reales.
- Eliminadas salidas incrustadas antiguas del notebook para no conservar conclusiones
  o figuras anteriores tras cambiar el código. Las nuevas figuras se exportan al ejecutar.
- Sustituidas pruebas huérfanas que importaban el módulo eliminado `article1.paper`
  por ejecución del notebook real y comprobaciones sobre el cierre publicado.
- Eliminado un import sin uso en `budget_analysis.py` y métodos comentados redundantes
  en `experiments.py`. Los controles opcionales activos mantienen sus implementaciones.
- Conservados análisis históricos, manifiestos y figuras citadas: siguen siendo
  evidencia de procedencia. No se eliminan módulos solo porque no los use el baseline.

## Validación del cambio

97 pruebas correctas y 8 omitidas por falta de PyTorch; lint de los módulos
modificados y esquema nbformat correctos. Todas las celdas del notebook se ejecutaron
secuencialmente con el cierre público y se revisó visualmente la figura de cinco
métricas. Este entorno no dispone del CSV privado de resultados individuales: la
rama de masa observada queda preparada para ejecutarse sobre esa instantánea, sin
inventar datos. No se ejecutaron entrenamientos ni se verificó rendimiento CUDA.
