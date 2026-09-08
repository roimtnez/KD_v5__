"""Frozen scientific blocks; one CSV per question, no implicit nine-arm grid."""

RQ1_METHODS = (
    "feddf_logit",
    "expert_logit",
    "oracle_logit",
    "confidence_logit",
    "consensus_logit",
    "energy_logit",
)
T8_BLOCKS = {
    "rq1": ("results.csv", RQ1_METHODS),
    "aggregation": ("results_aggregation.csv", ("expert_prob",)),
    "support": ("results_support.csv", ("expert_prob_sr",)),
}
ANALYSIS_BLOCKS = {
    "rq1": ("rq1",),
    "aggregation": ("rq1", "aggregation"),
    "temperature": ("rq1", "aggregation"),
    "support": ("rq1", "aggregation", "support"),
}
PHASES = (
    "partitions",
    "pilot",
    "teachers",
    "rq1",
    "aggregation",
    "temperature",
    "support",
    "supervised",
    "proxy-curve",
)
