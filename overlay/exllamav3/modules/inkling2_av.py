"""
Adaptive verification for Inkling-2 DSpark decoding (after Flare's adaptive verification and the user's DeepSeek patch 0010).

Every round drafts the full 7-token DSpark block (a block drafter: the depth costs ~nothing extra). Each request then
verifies its anchor plus a prefix of its drafts: live width w_b in 1..Wp, where Wp is the physical width of the captured
graph (rows per request). Rows past w_b are dead: their routed experts are skipped (pad expert, weight 0), so a dead
row costs only its share of the dense/attention work, and the accepted count is capped at w_b - 1.

Choice (on device, graph-safe, from draft confidence only, so speculative sampling stays exact at any temperature):
  c_bk  = prod_{i<=k} p_bi                     prefix product of the Markov-corrected draft probabilities
  f_bk  = F[tcls_b, k, bin(c_bk)]              calibrated P(first k drafts accepted | c_bk), monotone in k
  rows  = the N highest f_bk over the batch (a shared row budget: requests compete for verify rows)
  N     = argmax_N (nreal + sum of the top-N f) / cost(B, Wp, nreal + N)     expected tokens per ms
where cost(B, Wp, L) is the measured complete-round ms (verify + store + 7-draft + host overhead) of bucket B, physical
width Wp, L live rows. The same evaluation over every captured Wp picks the next round's graph (reported to the host).
"""
from __future__ import annotations
import json, math
import numpy as np
import torch

NB = 32                     # confidence bins (uniform in c on [0, 1])


# ---------------------------------------------------------------------------------------------------------- fitting
def _pava(y, w):
    """weighted isotonic (non-decreasing) regression"""
    y, w = list(map(float, y)), list(map(float, w))
    blocks = [[y[i], w[i], 1] for i in range(len(y))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            y2, w2, n2 = out.pop(); y1, w1, n1 = out.pop()
            ww = w1 + w2
            out.append([(y1 * w1 + y2 * w2) / max(ww, 1e-12), ww, n1 + n2])
    r = []
    for yv, _, n in out: r += [yv] * n
    return np.array(r)


def fit_table(conf, n, temp, prior = 20.0):
    """conf [R, 7], n [R] accepted counts (0..7) at full width, temp [R]. Returns F [2, 7, NB] (tcls 0: temp 0, 1: temp > 0):
    per bin a Beta-smoothed empirical P(n >= k) (prior = the k's overall rate, `prior` pseudo-counts), then isotonic in c"""
    F = np.zeros((2, 7, NB))
    for tc in (0, 1):
        m = (temp == 0) if tc == 0 else (temp > 0)
        if m.sum() == 0:
            m = np.ones_like(temp, dtype = bool)
        cp = np.cumprod(conf[m], 1)
        for k in range(1, 8):
            y = (n[m] >= k).astype(np.float64)
            b = np.clip((cp[:, k - 1] * NB).astype(int), 0, NB - 1)
            cnt = np.bincount(b, minlength = NB).astype(np.float64)
            s = np.bincount(b, weights = y, minlength = NB)
            base = y.mean()
            # bins with no data get the bin centre as prior (c is a probability-like quantity)
            ctr = (np.arange(NB) + 0.5) / NB
            pr = np.where(cnt > 0, base, ctr)
            est = (s + prior * pr) / (cnt + prior)
            F[tc, k - 1] = np.clip(_pava(est, cnt + prior), 0.0, 1.0)
        F[tc] = np.minimum.accumulate(F[tc], axis = 0)        # P(n >= k) non-increasing in k at equal c
    return F


def expected_tokens(F, conf, temp, w):
    """predicted tokens for rows verifying w (vector or scalar) rows per request"""
    tc = (temp > 0).astype(int)
    cp = np.cumprod(conf, 1)
    b = np.clip((cp * NB).astype(int), 0, NB - 1)
    f = F[tc[:, None], np.arange(7)[None], b]
    f = np.minimum.accumulate(f, axis = 1)
    w = np.broadcast_to(np.asarray(w), (conf.shape[0],))
    return 1 + np.array([f[i, :max(0, int(w[i]) - 1)].sum() for i in range(conf.shape[0])]), f


def _rows(table, B, ctx):
    return {(r["Wp"], r["w"]): r["ms"] for r in table if r["B"] == B and r.get("ctx", 0) == ctx}


def cost_model(table, B, Wp, host_ms = 0.0, ctx = 0):
    """table: list of {B, Wp, w, ms[, ctx]} (serving/av_offline.py cost mode). Returns cost[L] for L = 0..B*Wp live rows:
    rect(B, Wp) - live8(B, B*Wp) + live8(B, L), live8 = the Wp=8 graph at uniform live width L/B (linear in between).
    A (B, ctx) pair without measurements: the ctx-0 table of B plus the ctx delta of the largest measured B' < B, scaled by B / B'"""
    t = _rows(table, B, ctx)
    if not t or (8, 8) not in t:
        if ctx == 0:
            raise KeyError(f"no cost rows for B={B}")
        base = cost_model(table, B, Wp, 0.0, 0)
        Bm = max(b for b in {r["B"] for r in table if r.get("ctx", 0) == ctx} if b < B)
        d = cost_model(table, Bm, Wp, 0.0, ctx) - cost_model(table, Bm, Wp, 0.0, 0)      # [Bm * Wp + 1]
        L = np.arange(0, B * Wp + 1)
        return base + np.interp(L * Bm / B, np.arange(len(d)), d) * B / Bm + host_ms
    live8 = np.array([t[(8, w)] for w in range(1, 9)])
    def l8(L):
        x = np.clip(L / B, 1.0, 8.0)
        return np.interp(x, np.arange(1, 9), live8)
    rect = t.get((Wp, Wp))
    if rect is None:               # physical width not measured: interpolate the rectangular costs
        ws = sorted(w for (wp, w) in t if wp == w)
        rect = float(np.interp(Wp, ws, [t[(w, w)] for w in ws]))
    L = np.arange(0, B * Wp + 1)
    c = rect - l8(B * Wp) + l8(L)
    return c + host_ms


def cost_ctxs(table):
    return sorted({r.get("ctx", 0) for r in table})


# ---------------------------------------------------------------------------------------------------------- device side
class AVPolicy:
    def __init__(self, F, costs, device, widths, ctxs = (0,)):
        """F [2, 7, NB]; costs {(B, Wp): array [len(ctxs), live rows + 1] (or [live rows + 1])}; widths {B: sorted captured
        physical widths}; ctxs: context lengths of the cost tables (interpolated linearly on the batch's mean context)"""
        self.F = torch.tensor(np.asarray(F), dtype = torch.float32, device = device)
        self.widths = {B: sorted(ws) for B, ws in widths.items()}
        self.cost = {}
        for k, v in costs.items():
            v = np.asarray(v, dtype = np.float64)
            self.cost[k] = torch.tensor(v if v.ndim == 2 else v[None], dtype = torch.float32, device = device)
        self.ctxs = torch.tensor([float(c) for c in ctxs], dtype = torch.float32, device = device)
        self.k = torch.arange(7, device = device)
        self.dev = device
        self.fixed = None          # diagnostics: a fixed live width (int) instead of the policy

    @classmethod
    def from_files(cls, calib_path, cost_path, device, widths, host_ms = 0.0, cost_scale = None):
        F = np.load(calib_path)["F"] if calib_path.endswith(".npz") else np.array(json.load(open(calib_path))["F"])
        tab = json.load(open(cost_path))
        ctxs = cost_ctxs(tab)
        costs = {}
        for B, ws in widths.items():
            for Wp in ws:
                costs[(B, Wp)] = np.stack([cost_model(tab, B, Wp, host_ms, c) for c in ctxs])
        return cls(F, costs, device, widths, ctxs)

    def _cost(self, key, ctx):
        """cost over live rows at context ctx (None or 0-d/1-elem tensor): linear in ctx between the measured tables"""
        c = self.cost[key]
        if c.shape[0] == 1 or ctx is None:
            return c[0]
        x = ctx.reshape(1).float().clamp(self.ctxs[0], self.ctxs[-1])
        i = torch.bucketize(x, self.ctxs).clamp(1, self.ctxs.shape[0] - 1)          # upper table index
        lo, hi = self.ctxs.gather(0, i - 1), self.ctxs.gather(0, i)
        t = ((x - lo) / (hi - lo).clamp_min(1.0)).clamp(0.0, 1.0)
        return c.index_select(0, i - 1)[0] * (1 - t) + c.index_select(0, i)[0] * t

    def marginals(self, conf, temps, real):
        """f [B, 7]: calibrated P(prefix k accepted), -1 where the request is not real"""
        c = torch.cumprod(conf.clamp(0.0, 1.0), 1)
        b = (c * NB).long().clamp_(0, NB - 1)
        tc = (temps > 0).long()
        f = self.F[tc[:, None], self.k[None], b]
        f = torch.cummin(f, 1).values
        return torch.where(real[:, None], f, torch.full_like(f, -1.0))

    def _choose(self, f, real, B, Wp, cost_B = None, ctx = None):
        """-> (live widths w [B] long, expected tokens per ms (scalar tensor)); cost_B: bucket of the cost table (default B);
        ctx: mean context of the batch's real requests (tensor) for context-dependent costs"""
        cost = self._cost((cost_B or B, Wp), ctx)
        fm = torch.where(self.k[None] < Wp - 1, f, torch.full_like(f, -1.0)).reshape(-1)
        vals, _ = torch.sort(fm, descending = True)
        nreal = real.sum()
        gain = torch.cumsum(vals.clamp_min(0.0), 0)
        N = torch.arange(1, vals.shape[0] + 1, device = f.device)
        rows = (nreal + N).clamp_max(cost.shape[0] - 1)
        ratio = torch.where(vals >= 0, (nreal + gain) / cost[rows], torch.full_like(gain, -1.0))
        # every index below is a 1-element tensor (gather): a 0-d tensor index would sync the host (not capturable)
        r0 = nreal.reshape(1) / cost.gather(0, nreal.clamp_max(cost.shape[0] - 1).reshape(1))
        allr = torch.cat([r0, ratio])
        best = torch.argmax(allr).reshape(1)           # 0 = anchors only
        thr = torch.where(best > 0, vals.gather(0, (best - 1).clamp_min(0)), torch.full_like(best, 2, dtype = vals.dtype))
        cnt = (fm.view(B, 7) >= thr).sum(1)
        cnt = torch.minimum(cnt, best)                # ties at the threshold never add rows past N in total for one request
        w = 1 + cnt
        return w, allr.gather(0, best)[0]

    def choose(self, conf, temps, real, B, Wp, cost_B = None, ctx = None):
        if self.fixed is not None:
            return torch.full((B,), min(self.fixed, Wp), dtype = torch.long, device = conf.device)
        f = self.marginals(conf, temps, real)
        return self._choose(f, real, B, Wp, cost_B, ctx)[0]

    def best_width(self, conf, temps, real, B, ctx = None):
        """index (into self.widths[B]) of the physical width with the most expected tokens per ms"""
        f = self.marginals(conf, temps, real)
        rs = torch.stack([self._choose(f, real, B, Wp, None, ctx)[1] for Wp in self.widths[B]])
        return torch.argmax(rs)

    @staticmethod
    def mean_ctx(P, real):
        """mean position of the real requests (graph-safe)"""
        r = real.float()
        return ((P.float() * r).sum() / r.sum().clamp_min(1.0)).reshape(1)
