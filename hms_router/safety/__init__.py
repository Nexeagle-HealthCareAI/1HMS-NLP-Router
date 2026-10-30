"""Safety Guard: deterministic red-flag detection that runs before, and independently of, any model."""
from .detectors import AnyOf, EmergencyDetector, ModelDetector, RuleDetector
from .rules import RED_FLAGS_PATH, load_red_flags, recommended_specialties
from .self_harm import SelfHarmDetector, mentions_self_harm

__all__ = ["AnyOf", "EmergencyDetector", "ModelDetector", "RuleDetector", "RED_FLAGS_PATH", "load_red_flags",
           "recommended_specialties", "SelfHarmDetector", "mentions_self_harm"]
