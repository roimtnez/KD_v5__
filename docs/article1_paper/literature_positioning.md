# Posicionamiento bibliográfico — revisión del 14 de septiembre de 2026

Esta revisión es dirigida, no sistemática. Se consultaron registros primarios de arXiv, NeurIPS, PMLR y registros indexados de TMLR/OpenReview. Las descripciones del manuscrito se limitan al alcance respaldado por esas fuentes. No se afirma haber leído íntegramente cada PDF ni haber reproducido sus métodos.

| Referencia del manuscrito | Relación con el artículo | Qué se comprobó | Qué no se atribuye |
|---|---|---|---|
| [1] Hinton et al., 2015 | Distillation y especialistas | Abstract original: compresión de ensemble y modelos especialistas | No es nuevo combinar especialistas |
| [2] Li y Wang, FedMD, 2019 | Colaboración mediante KD con modelos diferentes | Registro arXiv, autores y alcance | No se afirma reproducción de FedMD |
| [3] Lin et al., FedDF, 2020 | Fusión de modelos mediante distillation y datos auxiliares | Registro NeurIPS y abstract | Nuestro FedDF-logit no reproduce todo el protocolo |
| [4] Sattler et al., FedAUX, 2021 | Influencia no uniforme de teachers | Registro arXiv: pretraining auxiliar y certainty scoring | Confidence/energy internos no son FedAUX |
| [5] Maron et al., TMLR, 2025 | Antecedente directo: teachers monoclase y fragmentación | Título completo, autores, publicación y abstract indexado del PDF primario | No se afirma que falten controles concretos sin leer el texto |
| [6] Zhang et al., DENSE, 2022 | One-shot sin proxy auxiliar | Registro NeurIPS: generación y distillation | No se compara su rendimiento con nuestro labeled-proxy |
| [7] Huang y Shu, FOL, ICML 2025 | One-shot personalizado | Registro PMLR: alineamiento de modelos y KD | Nuestro global student no evalúa personalización |

Fuentes exactas: referencias [1]–[7] de manuscript.md. La búsqueda incluyó términos FedMD, FedAUX, one-shot distillation, specialists y knowledge fragmentation; no se presenta una lista exhaustiva de métodos existentes en septiembre de 2026.

## Lectura prioritaria antes del envío

Maron, Fresse y Orzalesi, *One-Shot Federated Distillation Using Monoclass Teachers: A Study of Knowledge Fragmentation and Out-of-Distribution Supervision*, TMLR 2025:
https://openreview.net/forum?id=ENdm5BM7aF

El registro primario indexado identifica explícitamente monoclase y supervisión fuera de distribución. La apertura directa de OpenReview mostró verificación de navegador y el repositorio HAL devolvió acceso denegado. Por ello no se infirieron reglas matemáticas, operadores, ablations ni presupuestos que no aparezcan en el registro consultado.

Obtener el texto completo y contrastar:
- Cómo entrenan un teacher monoclase y qué otras clases o datos de rechazo utiliza.
- Etiquetas e información de soporte disponibles al construir targets.
- Pooling, enmascaramiento, renormalización y tratamiento de muestras externas.
- Baselines supervisados, tamaño del proxy y presupuestos de optimización.
- Qué análisis de presencia/competencia realizan, si los hay.

Esta comparación puede cambiar el énfasis de novedad, pero no invalida los contrastes ya ejecutados. No lanzar nuevos experimentos antes de determinar qué afirmación concreta los necesitaría.

## Contribución defendible ahora

Un estudio controlado que separa presencia y acreditación de accuracy en diversas asignaciones, examina pooling y soporte con teachers compartidos, y compara transferencia con supervisión pública a distintos N. Se conservan resultados adversos y se reportan accuracy y NLL.

No afirmar: primera selección de especialistas; primer estudio de monoclase; superioridad SOTA; solución unlabeled; demostración causal del dark knowledge; una mejora uniforme sobre presencia o CE.

## Comprobación editorial

El manuscrito distingue referencias bibliográficas de baselines implementados. Las referencias [1]–[7] están citadas en el cuerpo; no se incorporan cifras de rendimiento de terceros como comparación cuantitativa. La versión de envío deberá añadir las citas específicas de datasets/arquitecturas y adaptar las entradas al estilo de la revista.

## Reintento de acceso — 15 de septiembre de 2026

Se volvió a consultar el índice oficial de TMLR y se intentó obtener el PDF de OpenReview, sus endpoints públicos y la copia HAL (hal-05272000). El texto completo siguió inaccesible: verificación de navegador, HTTP 403 o acceso denegado. **La lectura íntegra sigue pendiente**; no se añadió una comparación metodológica deducida del título o del abstract. Es necesario facilitar el PDF o una copia pública accesible antes de cerrar el posicionamiento de novedad.

La versión LaTeX mantiene el alcance prudente de la sección de trabajo relacionado. El empaquetado y la compilación del manuscrito no cierran este pendiente bibliográfico.
