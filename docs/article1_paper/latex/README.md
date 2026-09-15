# Article 1 — manuscript LaTeX v0.3

Texto completo en inglés, dividido por secciones, con seis figuras vectoriales y siete referencias. Deriva de `../manuscript.md` v0.2 y conserva sus resultados auditados. Evidencia: commit b43da870bc5a20f7c533089b4145503fdfe995cc, auditoría del 14 de septiembre de 2026. Preparación LaTeX: 15 de septiembre de 2026.

## Compilación

Requiere una distribución TeX con XeLaTeX, BibTeX y latexmk (TeX Live). En Overleaf, seleccionar XeLaTeX y `main.tex`. Desde esta carpeta:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Salida: `build/main.pdf`. No requiere Python, Pandoc, datasets ni checkpoints. Las figuras se incluyen en `figures/`.

## Archivos

- `main.tex`: documento principal, estilo neutral y bibliografía.
- `sections/00_abstract.tex`: resumen completo.
- `sections/01_introduction.tex`: problema, preguntas y contribuciones.
- `sections/02_related_work.tex`: antecedentes y límites del posicionamiento.
- `sections/03_methods.tex`: conjuntos, máscaras, routing, pooling, SR y objetivos.
- `sections/04_protocol.tex`: diseño experimental, entrenamiento y recuento de ejecuciones.
- `sections/05_results.tex`: cinco estudios, seis figuras y tabla CIFAR.
- `sections/06_discussion.tex`: interpretación, límites y reproducibilidad.
- `sections/07_conclusion.tex`: conclusiones y futuras líneas.
- `references.bib`: bibliografía editable.

Los archivos LaTeX son editables y no se regeneran automáticamente desde Markdown. Mantener ambos textos sincronizados en futuras revisiones; este paquete corrige además la leyenda de Figura 4: métricas en cinco columnas, datasets en tres filas.

## Pendientes antes del envío

La lectura íntegra de Maron et al. (TMLR 2025, OpenReview ENdm5BM7aF) sigue bloqueada. Se verificó su registro, pero no se afirma una comparación metodológica completa. Obtener el PDF y cerrar esa comparación antes de afirmar novedad. Véase `../literature_positioning.md`.

Añadir autores y afiliaciones, seleccionar revista y adaptar su plantilla. Completar las referencias específicas de datasets y arquitecturas; traducir al inglés las etiquetas españolas de las figuras auditadas al generar la versión final. Las cifras y figuras existentes se conservan sin reejecutar experimentos. Este es un manuscrito completo de trabajo, no una versión lista para envío.
