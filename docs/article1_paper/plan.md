# Plan de redacción v0.2 — evidencia cerrada del 14 de septiembre

## Decisión y tesis

Redactar a partir de los 471 students acordados, conservando los 164 opcionales como evidencia de alcance parcial. La base es el commit de auditoría b43da870bc5a20f7c533089b4145503fdfe995cc. Este documento sustituye la hoja de ruta histórica; no modifica el cierre ni la ejecución.

Tesis: conocer el soporte de entrenamiento, acreditar competencia y conservar una distribución completa son decisiones diferentes. Su utilidad se mide en el student y frente al uso directo de las etiquetas públicas.

No presentar como contribución nueva la mera selección de especialistas ni la observación de problemas con teachers monoclase. El resultado decisivo es A=M en Multi/Single: allí el beneficio adicional de medir competencia no está demostrado. El control Dirichlet delimita dónde sí aparece.

## Orden de lectura

| Sección | Pregunta y mensaje | Evidencia principal | Límite |
|---|---|---|---|
| Abstract / Introduction | ¿Qué aporta cada fuente de información? | Presencia, SR y curvas completas | Sin superioridad universal ni prioridad histórica |
| Related Work | ¿Qué se ha estudiado ya y con qué recursos? | Distillation, FedMD, FedDF, FedAUX, Maron et al., DENSE, FOL | Lectura completa de Maron pendiente por bloqueo de acceso |
| Methods | ¿Cómo se construyen A, M, selección y targets? | Fórmulas, fallback y código auditado | Proxy etiquetado; M no acredita rechazo |
| Protocol | ¿Son comparables las ejecuciones? | 54 condiciones, 471 students, hashes y pares | Tres seeds, T=8, receta fija |
| Results 5.1 | ¿Cuándo falla la agregación uniforme? | Panorama y ORACLE | No atribuir aún el efecto a medir accuracy |
| Results 5.2 | ¿Qué añade acreditar competencia a presencia? | Tabla 1 y Fig. 2 | Cambian también número/composición de teachers; NLL mixto |
| Results 5.3 | ¿Depende pooling del routing? | Fig. 3 | EXPERT CIFAR incompleto |
| Results 5.4 | ¿Mejor target implica mejor student? | Fig. 4 y masa suplementaria | Concentración y relaciones interclase no aisladas |
| Results 5.5 | ¿Qué añade KD frente a CE? | Figs. 5–6 | Curva CIFAR focal; updates no equivalen a ejemplos |
| Discussion / Conclusion | ¿Qué queda establecido y qué no? | Mapa de evidencia | Sin claims SOTA, OOD o causalidad no medidos |

## Figuras principales: selección cerrada para esta versión

Se reutilizan las exportaciones auditadas, sin regenerar datos ni alterar colores. La incorporación a una plantilla de revista puede requerir redimensionarlas y comprobar su legibilidad a tamaño final.

| Número | Archivo en audit_20260914/figures | Función |
|---|---|---|
| 1 | overview_references.png / .pdf | Panorama absoluto con CE |
| 2 | selection_competence.png / .pdf | Selección y presencia frente a acreditación |
| 3 | aggregation_space.png / .pdf | Pooling condicionado por routing; huecos explícitos |
| 4 | support_target_student.png / .pdf | Cinco métricas target/student, signo SR − full |
| 5 | ce_only.png / .pdf | Curva exclusivamente supervisada |
| 6 | proxy_differences.png / .pdf | EXPERT−CE, FedDF−CE, EXPERT−FedDF |

Tabla 1: todos los regímenes CIFAR, con EXPERT−uniform y EXPERT−presencia en accuracy, y EXPERT−presencia en NLL. Generada de contrasts.csv, cuyas diferencias de accuracy ya están en puntos porcentuales. No multiplicar por 100 otra vez.

Suplemento: overview_ablations, routing, support_mass, proxy_absolute, mask_advantage y los 54 diagnósticos de máscaras. Añadir tablas completas absolute.csv, contrasts.csv, curve_summary.csv, cobertura opcional y condiciones excluidas. Las figuras ya contienen todos los grupos: no elegir seeds por rendimiento.

## Mapa afirmación–evidencia

| Afirmación del manuscrito | Fuente en audit_20260914/tables | Estado |
|---|---|---|
| 471 acordados + 164 opcionales únicos | runs.csv, coverage.csv | Comprobado por conteos e identidades publicadas |
| 54 pares presencia y 135 pares curvas | competence_presence_pairs.csv, curve_pairs.csv | Diferencias comprobadas contra ambos lados |
| A=M en Multi/Single | mask_statistics.csv; mask_cells.csv | Discrepancias cero en condiciones verificadas |
| CIFAR alpha=.5: +5.25 pp sobre presencia | contrasts.csv, competence_presence | Tres seeds; accuracy no equivale a todas las métricas |
| NLL empeora en alpha=1 y .1 frente a presencia | contrasts.csv, competence_presence | Reportado junto a accuracy |
| SR mejora target NLL pero no student CIFAR | contrasts.csv, support; support_pairs.csv | No identifica mecanismo dark knowledge |
| IID N=500: FedDF supera a EXPERT, ambos superan CE | curve_summary.csv | Tres seeds; no atribuir ventaja de KD a la máscara |
| Single y alpha=.1: EXPERT mejora FedDF en todos los N | curve_summary.csv | No implica mejorar CE en todos los N |
| 540 estados deserializados verificados | teacher_states.csv; report.md | Auditoría local registrada; esta revisión no la repite |

## Bibliografía y riesgo de novedad

La revisión añade siete referencias primarias y explicita los presupuestos informativos. Ver literature_positioning.md. No se presenta como revisión sistemática ni como certificación de novedad.

Maron et al. (TMLR 2025) es un antecedente directo. Se verificaron título, autores, publicación y alcance general en el registro primario indexado. OpenReview y HAL bloquearon el texto completo en esta sesión. Antes de enviar hay que comparar sus operadores, uso de etiquetas, supervisión fuera de distribución y baselines. El manuscrito solo le atribuye su tema documentado; no inventa ausencias de ablaciones.

No se lanza un nuevo benchmark automáticamente. Tras esa lectura se decidirá si basta un contraste conceptual de protocolos para el alcance empírico propuesto o si una afirmación más ambiciosa necesita otra comparación.

## Revisión adversarial

1. Contribución: ¿se atribuye a accuracy el enorme beneficio Single? No; se explica A=M.
2. Claridad: ¿se confunden soporte y selección? No; A, M y S tienen definiciones separadas.
3. Fuerza experimental: ¿se venden 635 runs como 635 réplicas principales? No; 471 acordados y 164 opcionales.
4. Cobertura: ¿se ocultan NLL adversos o pooling CIFAR incompleto? No.
5. Diseño: ¿se identifica causalmente dark knowledge o conocimiento privado? No; se explicitan concentración, objetivo y recursos.

## Antes del envío

- Leer el texto completo del antecedente monoclase y verificar la diferenciación concreta.
- Fijar revista/conferencia, autoría y extensión; ajustar las seis figuras a su plantilla.
- Incorporar citas específicas de datasets/arquitecturas y ficha exacta del preprocesamiento según la plantilla final.
- Revisar referencias y afirmaciones con coautores.
- No afirmar robustez a temperatura, equivalencia de operadores ni superioridad SOTA sin evidencia adicional.

El manuscrito v0.2 es una primera versión narrativa completa; no está listo para envío.
