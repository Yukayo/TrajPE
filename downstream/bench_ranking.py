from __future__ import annotations

import numpy as np
import torch

from downstream.metrics import RankingMetrics

KS = (1, 5, 10, 20)
NDCG_KS = (5, 10)


class BenchRanking:
    def __init__(self, to_cand: torch.Tensor, eval_lengths: np.ndarray, ks=KS, ndcg_ks=NDCG_KS):
        self.to_cand = to_cand
        L = np.asarray(eval_lengths, np.int64)
        self.n_rows = int((L - 1).sum())
        last = np.zeros(self.n_rows, bool)
        last[np.cumsum(L - 1) - 1] = True
        self.is_last = torch.as_tensor(last, device=to_cand.device)
        self.last_k = torch.as_tensor(L - 1, device=to_cand.device)
        self.trail_of_last = torch.cumsum(self.is_last.long(), 0) - 1
        self.acc = {scope: [RankingMetrics(ks=ks if i == 0 else (), ndcg_k=nk) for i, nk in enumerate(ndcg_ks)]
                    for scope in ("all", "last")}
        self.seen = 0

    def _update(self, scope, scores, target):
        for m in self.acc[scope]:
            m.update(scores, target)

    def tap(self, score_iter):
        for item in score_iter:
            scores, true_rows, k = item[:3]
            n = len(true_rows)
            target = self.to_cand[true_rows]
            self._update("all", scores, target)
            m = self.is_last[self.seen:self.seen + n]
            if m.any():
                exp = self.last_k[self.trail_of_last[self.seen:self.seen + n][m]]
                if not torch.equal(k[m].long(), exp.long()):
                    raise RuntimeError("score iterator order differs from trail order (last-step mask misaligned)")
                self._update("last", scores[m], target[m])
            self.seen += n
            yield item

    def result(self) -> dict:
        if self.seen != self.n_rows:
            raise RuntimeError(f"saw {self.seen} scored positions, expected {self.n_rows}")
        out = {}
        for scope, (main, *rest) in self.acc.items():
            r = main.result()
            for extra in rest:
                r.update({k: v for k, v in extra.result().items() if k.startswith("ndcg@")})
            out[scope] = {k: r[k] for k in sorted(r) if k.startswith(("acc@", "ndcg@", "mrr", "n_targets", "target_in"))}
        return out
