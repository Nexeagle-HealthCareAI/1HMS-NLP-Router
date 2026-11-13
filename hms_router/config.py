import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # Step 1
    session_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("HMS_SESSION_TTL", "1800")))  # 30 min
    max_input_chars: int = 1000
    fasttext_model_path: str | None = field(default_factory=lambda: os.getenv("HMS_FASTTEXT_LID"))  # lid.176.bin

    # Step 2
    fuzzy_max_distance: int = 2      # RapidFuzz edit-distance <= 2 (per architecture doc)
    fuzzy_min_word_len: int = 4      # never fuzzy-match very short words ("pet" vs "pat" etc.)
    phonetic_max_distance: int = 3   # sanity bound on phonetic-key candidates
    review_queue_key: str = "review_queue"   # Step 8 uses Redis LPUSH review_queue


    # Step 4
    followup_max_turns: int = 3      # ask at most this many follow-ups before analysing what we have
    max_buffer: int = 20             # symptom records kept per session

    # Step 6
    rag_top_k: int = 10
    rag_min_confidence: float = 0.25

    # Step 8
    feedback_max_boost: float = 0.10      # cap on how far positive feedback can lift a cluster's confidence
    feedback_boost_step: float = 0.01
    novel_min_count: int = 3              # a novel token must be seen this often before it is reviewed
