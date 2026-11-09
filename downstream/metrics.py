from __future__ import annotations

import math

import torch


class RankingMetrics:

    def __init__(self, ks=(1, 5, 10), ndcg_k: int = 10, group_names: list[str] | None = None,
                 cand_cat: torch.Tensor | None = None, n_cats: int = 0):
        self.ks, self.ndcg_k = tuple(ks), ndcg_k
        disc = [0.0] + [1.0 / math.log2(r + 1) for r in range(1, ndcg_k + 1)]
        self._cum_disc = torch.tensor(disc, dtype=torch.float64).cumsum(0)
        self.group_names = list(group_names or [])
        self.cand_cat, self.n_cats = cand_cat, n_cats
        self._sums: dict[str, torch.Tensor] = {}
        self._counts = torch.zeros(1 + len(self.group_names), dtype=torch.float64)
        self._in_vocab = 0
        self._cat_sums: dict[str, float] = {}

    def _per_row(self, scores: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        s_t = scores.gather(1, target[:, None])
        if not torch.isfinite(s_t).all():
            raise ValueError("non-finite score for a target")
        g = (scores > s_t).sum(1).double()
        n_pos = (scores == s_t).sum(1).double()
        out = {f"acc@{k}": (k - g).clamp(min=0).minimum(n_pos) / n_pos for k in self.ks}
        cum, k = self._cum_disc.to(scores.device), self.ndcg_k
        out[f"ndcg@{k}"] = (cum[(g + n_pos).clamp(max=k).long()] - cum[g.clamp(max=k).long()]) / n_pos
        out["mrr"] = (torch.digamma(g + n_pos + 1) - torch.digamma(g + 1)) / n_pos
        return out

    @torch.no_grad()
    def update(self, scores: torch.Tensor, target: torch.Tensor, group: torch.Tensor | None = None,
               target_cat: torch.Tensor | None = None) -> None:
        n_bins = 1 + len(self.group_names)
        slot = torch.zeros_like(target) if group is None else group.to(target.device).long() + 1
        inside = target >= 0
        for name, v_in in self._per_row(scores[inside], target[inside]).items():
            v = torch.zeros(len(target), dtype=torch.float64, device=target.device)
            v[inside] = v_in
            acc = torch.bincount(slot, weights=v, minlength=n_bins)
            acc[0] = v.sum()
            self._sums[name] = self._sums.get(name, 0) + acc.cpu()
        cnt = torch.bincount(slot, minlength=n_bins).double()
        cnt[0] = len(target)
        self._counts += cnt.cpu()
        self._in_vocab += int(inside.sum())
        if self.cand_cat is not None and target_cat is not None:
            cc = self.cand_cat.expand(len(target), -1)
            cat_p = torch.zeros(len(target), self.n_cats, device=scores.device).scatter_add_(1, cc, scores.exp())
            for name, v in self._per_row(cat_p, target_cat).items():
                self._cat_sums[name] = self._cat_sums.get(name, 0.0) + float(v.sum())

    def result(self) -> dict:
        def at(i):
            n = float(self._counts[i])
            res = {name: float(s[i]) / max(n, 1.0) for name, s in self._sums.items()}
            res["n_targets"] = int(n)
            return res
        res = at(0)
        res["target_in_vocab"] = self._in_vocab / max(res["n_targets"], 1)
        if self.group_names:
            res["by_group"] = {name: at(i + 1) for i, name in enumerate(self.group_names)}
        if self._cat_sums:
            res["category"] = {name: s / max(res["n_targets"], 1) for name, s in self._cat_sums.items()}
        return res
