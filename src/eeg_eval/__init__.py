"""Package init for eeg_eval."""
from .features import bandpower_features, BANDS
from .probes import sleep_staging_probe, age_estimation_probe, compare_sources, ProbeResult
from .synth import make_dataset, STAGES, FS

__all__ = [
    "bandpower_features", "BANDS",
    "sleep_staging_probe", "age_estimation_probe", "compare_sources", "ProbeResult",
    "make_dataset", "STAGES", "FS",
]
