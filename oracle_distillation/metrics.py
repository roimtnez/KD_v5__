"""
Taxonomía de métricas y funciones de cálculo compartidas entre runners.

Tres categorías:
  MODEL_METRICS      — calidad del modelo destilado final
  DISTILL_METRICS    — diagnóstico del proceso de destilación
  METADATA_FIELDS    — trazabilidad, no son métricas comparables
"""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Funciones de cálculo de métricas de modelo
# ---------------------------------------------------------------------------

def compute_known_unknown_acc(
    per_class_acc: np.ndarray,
    known_classes: list[int],
) -> tuple[float, float]:
    """Accuracy media sobre clases known y unknown, excluyendo NaN.

    Promedia accuracies por clase (no por muestra).  Las clases sin muestras
    en el test set (NaN en per_class_acc) se excluyen del promedio en vez de
    contarse como 0.

    Returns
    -------
    (known_acc, unknown_acc)  — NaN si el conjunto correspondiente está vacío.
    """
    known_set   = set(known_classes)
    all_classes = set(range(len(per_class_acc)))
    unknown_set = all_classes - known_set

    def _mean(indices: set) -> float:
        vals = [per_class_acc[c] for c in sorted(indices)
                if not math.isnan(per_class_acc[c])]
        return float(np.mean(vals)) if vals else float("nan")

    return _mean(known_set), _mean(unknown_set)


def compute_gap_ku(known_acc: float, unknown_acc: float) -> float:
    """Gap conocido/desconocido: known_acc − unknown_acc.  NaN si alguno es NaN."""
    if math.isnan(known_acc) or math.isnan(unknown_acc):
        return float("nan")
    return known_acc - unknown_acc


# ---------------------------------------------------------------------------
# Función de derivación de columnas metadata
# ---------------------------------------------------------------------------

def parse_group_alpha(rel: str) -> str:
    """Extrae group_alpha de un rel_path.

    Soporta el esquema plano ('cifar10/K10/alpha0p1__…' -> '0.1') y el esquema
    jerárquico CIFAR-100 k10 ('cifar100/K10/<regime_tag>'):
      single_super -> 'single',  multi_super -> 'multi',  iid -> 'iid',
      dir_coarse_alpha0p5 -> 'dir_0.5',  dir_fine_alpha0p5 -> 'fine_0.5'.
    coarse y fine se mantienen distintos para no colisionar en la clave del CSV.
    """
    def _num(s: str) -> str:
        m = re.search(r"(\d+p\d+|\d+)", s)
        return m.group(1).replace("p", ".") if m else s

    parts = rel.replace("\\", "/").split("/")
    for part in parts:
        low = part.lower()
        # CIFAR-100 k10 Dirichlet tags — check coarse/fine before the generic regex
        if low.startswith("dir_fine") or low.startswith("fine_"):
            return f"fine_{_num(low)}"
        if low.startswith("dir_coarse") or low.startswith("coarse_"):
            return f"dir_{_num(low)}"
        for tag in ("iid", "single", "multi"):
            if tag in low:
                return tag
        m = re.search(r"alpha(\d+p\d+|\d+)", low)
        if m:
            return m.group(1).replace("p", ".")
    return rel  # fallback: return rel unchanged
