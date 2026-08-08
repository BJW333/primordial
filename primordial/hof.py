"""
Hall of fame with a signal-correlation gate (#7).

GP converges to one basin: without diversity pressure every elite is a mutant
of the same rule, and "top 5" is one strategy five times. The HOF admits a
candidate only if its concatenated train-signal is < max_corr correlated with
every member already in -- so what comes out is a SET of genuinely different
surviving behaviors, which is what you ensemble (uncorrelated mediocre edges
beat one good one at the portfolio level).

Works for both engines: pass any signal_fn(df) -> array. Correlation is over
the concatenated per-name signal arrays on train data, i.e. "do these rules
fire at the same times on the same names," which is the correlation that
matters for an ensemble.
"""

from __future__ import annotations
import numpy as np


class HallOfFame:
    def __init__(self, max_size: int = 5, max_corr: float = 0.7):
        self.max_size = max_size
        self.max_corr = max_corr
        self.members = []          # [{"fitness","payload","sig"}] sorted desc

    @staticmethod
    def concat_signal(signal_fn, data, syms) -> np.ndarray:
        parts = []
        for s in syms:
            try:
                a = np.asarray(signal_fn(data[s]), float)
            except Exception:
                a = np.zeros(len(data[s]))
            parts.append(np.nan_to_num(a))
        return np.concatenate(parts) if parts else np.zeros(1)

    def _corr(self, a: np.ndarray, b: np.ndarray) -> float:
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if a.std() == 0 or b.std() == 0:
            return 1.0 if np.array_equal(a, b) else 0.0
        return float(abs(np.corrcoef(a, b)[0, 1]))

    def consider(self, fitness: float, payload, signal_fn, data, syms) -> bool:
        """Admit if fitter than the floor AND decorrelated from every member.
        If it correlates > max_corr with a member, it only REPLACES that member
        when fitter (same basin -> keep the better representative)."""
        sig = self.concat_signal(signal_fn, data, syms)
        for i, m in enumerate(self.members):
            if self._corr(sig, m["sig"]) > self.max_corr:
                if fitness > m["fitness"]:
                    self.members[i] = {"fitness": fitness, "payload": payload, "sig": sig}
                    self._sort()
                    return True
                return False
        self.members.append({"fitness": fitness, "payload": payload, "sig": sig})
        self._sort()
        if len(self.members) > self.max_size:
            self.members = self.members[: self.max_size]
        return payload in [m["payload"] for m in self.members]

    def _sort(self):
        self.members.sort(key=lambda m: m["fitness"], reverse=True)

    def payloads(self):
        return [(m["fitness"], m["payload"]) for m in self.members]

    def __len__(self):
        return len(self.members)
