"""Minimal selection → pooling → expertise → support experiment."""

RQ1_METHODS = ("feddf_logit", "oracle_logit")
T8_BLOCKS = {
    "rq1": ("results_selection.csv", RQ1_METHODS),
    "aggregation": ("results_pooling.csv", ("feddf_prob", "oracle_prob")),
    "expertise": ("results_expertise.csv", ("expert_prob",)),
    "support": ("results_support_v2.csv", ("expert_prob_sr",)),
    "controls": ("results_controls.csv", ("consensus_logit", "energy_logit")),
}
ANALYSIS_BLOCKS = {
    "rq1": ("rq1",),
    "aggregation": ("rq1", "aggregation"),
    "expertise": ("rq1", "aggregation", "expertise"),
    "support": ("rq1", "aggregation", "expertise", "support"),
    "controls": ("rq1", "controls"),
}
PHASES = (
    "partitions",
    "pilot",
    "teachers",
    "rq1",
    "aggregation",
    "expertise",
    "support",
)
OPTIONAL_PHASES = (
    "controls",
    "expert-logit",
    "temperature",
    "supervised",
    "proxy-curve",
)
FOCAL_REGIMES = ("iid", "alpha0p1", "single")
