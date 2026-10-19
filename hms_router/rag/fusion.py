"""Fuse specialty score maps from several scorers into one ranked list whose scores sum to 1."""
from __future__ import annotations


class WeightedFusion:
    def __init__(self, weights: list[float]):
        if not weights or any(w < 0 for w in weights) or sum(weights) == 0:
            raise ValueError("fusion weights must be non-negative and not all zero")
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def fuse(self, score_maps: list[dict[str, float]]) -> list[tuple[str, float]]:
        if len(score_maps) != len(self.weights):
            raise ValueError("one weight per score map")
        fused: dict[str, float] = {}
        live = [(w, m) for w, m in zip(self.weights, score_maps) if m]      # a scorer with no opinion drops out
        norm = sum(w for w, _ in live)
        for w, m in live:
            for sid, s in m.items():
                fused[sid] = fused.get(sid, 0.0) + (w / norm) * s
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
