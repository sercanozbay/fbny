from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterator, Tuple

EPS = 1e-12

# ---------- basic indicators ----------
def lag(x: pd.Series, k: int) -> pd.Series:
    return x.shift(k)

def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()

def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()

def stddev(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).std()

def momentum(x: pd.Series, n: int) -> pd.Series:
    return x - x.shift(n)

def ts_sum(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).sum()

def ts_min(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).min()

def ts_max(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).max()

def clip(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.clip(lower=lo, upper=hi)

# --- ranks, zscores, decay ---
def zscore(x: pd.Series, n: int) -> pd.Series:
    m = sma(x, n)
    s = stddev(x, n)
    return (x - m) / (s + EPS)

def _pct_rank_of_last(arr: np.ndarray) -> float:
    if arr.size == 0 or np.isnan(arr[-1]):
        return np.nan
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return np.nan
    last = a[-1]
    less = np.sum(a < last)
    equal = np.sum(a == last)
    return (less + 0.5 * equal) / max(len(a), 1)

def pct_rank(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).apply(_pct_rank_of_last, raw=True)

def ts_rank(x: pd.Series, n: int) -> pd.Series:
    return pct_rank(x, n)

def decay_linear(x: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    return x.rolling(n, min_periods=n).apply(lambda a: np.dot(a, w), raw=True)

# --- correlations and beta ---
def rolling_corr(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).corr(y)

def rolling_beta(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
    cov = y.rolling(n, min_periods=n).cov(x)
    var = x.rolling(n, min_periods=n).var()
    return cov / (var + EPS)

# --- RSI (Wilder) ---
def rsi(x: pd.Series, n: int = 14) -> pd.Series:
    delta = x.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100.0 - (100.0 / (1.0 + rs))

# --- MACD / OBV / Bollinger ---
def macd_line(c: pd.Series, fast=12, slow=26) -> pd.Series:
    return ema(c, fast) - ema(c, slow)

def macd_signal(c: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    return ema(macd_line(c, fast, slow), signal)

def obv(c: pd.Series, v: pd.Series) -> pd.Series:
    delta = c.diff()
    sign = pd.Series(0.0, index=c.index)
    sign[delta > 0] = 1.0
    sign[delta < 0] = -1.0
    return (sign * v).fillna(0).cumsum()

def bollinger_bands(c: pd.Series, n=20, k=2.0) -> tuple[pd.Series, pd.Series]:
    m = sma(c, n)
    s = stddev(c, n)
    return m + k * s, m - k * s

# ---------- context ----------
@dataclass
class AlphaCtx:
    # required series (aligned index)
    c: pd.Series  # close
    o: pd.Series  # open
    v: pd.Series  # volume
    s: pd.Series  # sentiment
    h: pd.Series  # high
    l: pd.Series  # low

    # precomputed inds
    SMA_5:  pd.Series | None = None
    SMA_20: pd.Series | None = None
    EMA_10: pd.Series | None = None
    MOM_3:  pd.Series | None = None
    MOM_10: pd.Series | None = None
    RSI_14: pd.Series | None = None
    MACD:   pd.Series | None = None
    MACD_S: pd.Series | None = None
    BBU:    pd.Series | None = None
    BBL:    pd.Series | None = None
    OBV:    pd.Series | None = None

    def prepare(self) -> "AlphaCtx":
        self.SMA_5  = sma(self.c, 5)
        self.SMA_20 = sma(self.c, 20)
        self.EMA_10 = ema(self.c, 10)
        self.MOM_3  = momentum(self.c, 3)
        self.MOM_10 = momentum(self.c, 10)
        self.RSI_14 = rsi(self.c, 14)
        self.MACD   = macd_line(self.c, 12, 26)
        self.MACD_S = macd_signal(self.c, 12, 26, 9)
        self.BBU, self.BBL = bollinger_bands(self.c, n=20, k=2.0)
        self.OBV    = obv(self.c, self.v)
        return self

# ============================================================
# Reindexed ALPHAS (1..71)
#   - Paper alphas kept (41 total): original 1–6, 11–15, 21–50
#   - +30 custom alphas
# ============================================================

# ---- Paper: Momentum-based ----
def alpha1(ctx):  return (ctx.c - ctx.o) / (ctx.o + EPS) + 0.5 * ctx.MOM_3
def alpha2(ctx):  return ctx.MOM_10 * (ctx.c - ctx.SMA_5)
def alpha3(ctx):  return (ctx.MOM_3 + ctx.MOM_10) / 2.0
def alpha4(ctx):  return (ctx.c - ctx.SMA_20) * ctx.MOM_3
def alpha5(ctx):  return ctx.MOM_3 * ctx.MOM_10

# ---- Paper: Sentiment-based ----
def alpha6(ctx):  return ctx.s * (ctx.c - ctx.o) / (ctx.o + EPS)

# ---- Paper: Volume-based ----
def alpha7(ctx):  return ctx.v / (ctx.SMA_20 + EPS)                 # paper literal; swap to sma(ctx.v,20) if desired
def alpha8(ctx):  return ctx.OBV * (ctx.c - ctx.o) / (ctx.o + EPS)
def alpha9(ctx):  return ctx.v * ctx.MOM_3
def alpha10(ctx): return ctx.v * ctx.MOM_10
def alpha11(ctx): return ctx.OBV * ctx.s

# ---- Paper: Technical indicator-based ----
def alpha12(ctx): return ctx.MACD * ctx.MACD_S
def alpha13(ctx): return (ctx.MACD - ctx.MACD_S) * ctx.s
def alpha14(ctx): return ctx.RSI_14 * ctx.MOM_3
def alpha15(ctx): return ctx.RSI_14 * ctx.MOM_10
def alpha16(ctx): return ctx.BBU - ctx.BBL

# ---- Paper: Moving average-based ----
def alpha17(ctx): return (ctx.SMA_5 - ctx.SMA_20) * ctx.s
def alpha18(ctx): return (ctx.EMA_10 - ctx.SMA_20) * ctx.MOM_3
def alpha19(ctx): return ctx.SMA_5 * ctx.SMA_20
def alpha20(ctx): return ctx.EMA_10 * ctx.SMA_20
def alpha21(ctx): return (ctx.SMA_5 + ctx.SMA_20) / 2.0

# ---- Paper: Combination formulas ----
def alpha22(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + 0.5*ctx.s
def alpha23(ctx): return (ctx.c-ctx.SMA_20)/(ctx.SMA_20+EPS) + 0.5*ctx.MOM_3
def alpha24(ctx): return (ctx.c-ctx.EMA_10)/(ctx.EMA_10+EPS) + 0.5*ctx.MOM_10
def alpha25(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.SMA_20)/(ctx.SMA_20+EPS)
def alpha26(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.EMA_10)/(ctx.EMA_10+EPS)

# ---- Paper: Volatility-based ----
def alpha27(ctx): return (ctx.h - ctx.l) / (ctx.c + EPS) * ctx.s
def alpha28(ctx): return (ctx.h - ctx.l) / (ctx.c + EPS) * ctx.MOM_3
def alpha29(ctx): return (ctx.h - ctx.l) / (ctx.c + EPS) * ctx.MOM_10
def alpha30(ctx): return ctx.BBU - ctx.BBL
def alpha31(ctx): return (ctx.BBU - ctx.c) / ((ctx.BBU - ctx.BBL) + EPS)

# ---- Paper: More combinations ----
def alpha32(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.SMA_20)/(ctx.SMA_20+EPS) + ctx.s
def alpha33(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.EMA_10)/(ctx.EMA_10+EPS) + ctx.MOM_3
def alpha34(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.EMA_10)/(ctx.EMA_10+EPS) + ctx.MOM_10
def alpha35(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.SMA_20)/(ctx.SMA_20+EPS) + ctx.MOM_3
def alpha36(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.SMA_20)/(ctx.SMA_20+EPS) + ctx.MOM_10

# ---- Paper: Advanced combinations ----
def alpha37(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.SMA_20)/(ctx.SMA_20+EPS) + ctx.s + ctx.MOM_3
def alpha38(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.EMA_10)/(ctx.EMA_10+EPS) + ctx.s + ctx.MOM_3
def alpha39(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.EMA_10)/(ctx.EMA_10+EPS) + ctx.s + ctx.MOM_10
def alpha40(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.SMA_20)/(ctx.SMA_20+EPS) + ctx.s + ctx.MOM_10
def alpha41(ctx): return (ctx.c-ctx.SMA_5)/(ctx.SMA_5+EPS) + (ctx.c-ctx.SMA_20)/(ctx.SMA_20+EPS) + ctx.s + ctx.MOM_3 + ctx.MOM_10

# ============================================================
# Custom alphas (30) — reindexed to follow as 42..71
# ============================================================
def alpha42(ctx): return zscore(ctx.c - ema(ctx.c,20), 20) - zscore(ctx.v - ema(ctx.v,20), 20)
def alpha43(ctx): return (ema(ctx.c,12) - ema(ctx.c,26)) - (0.5 * stddev(ctx.c,20))
def alpha44(ctx): return (rsi(ctx.c,14) / 100.0) * zscore(ctx.s,20) - ts_rank(ctx.v,10)
def alpha45(ctx): return momentum(ctx.c,5) - 0.0001 * momentum(ctx.v,5)
def alpha46(ctx): return ts_rank(decay_linear(momentum(ctx.c,10),10),10) - ts_rank(decay_linear(ctx.v,10),10)
def alpha47(ctx): return zscore(ctx.c/ctx.o - 1.0, 20) + 0.5 * zscore(ctx.s,10)
def alpha48(ctx): return (ema(ctx.c,5) - ema(ctx.c,20)) / (stddev(ctx.c,20) + EPS)
def alpha49(ctx): return -rolling_corr(ctx.c, ctx.v, 20) * zscore(ctx.v,20)
def alpha50(ctx): return (pct_rank(ctx.c - lag(ctx.c,1),20) - 0.5) + 0.3 * pct_rank(ctx.s,20)
def alpha51(ctx): return (ts_rank(ctx.c,20) - ts_rank(ctx.v,20)) + 0.2 * ts_rank(ctx.s,20)
def alpha52(ctx): return (ctx.c - sma(ctx.c,50)) / (0.1 + stddev(ctx.c,50)) - 0.1 * rsi(ctx.v,14) / 100.0
def alpha53(ctx): return decay_linear(zscore(ctx.c,20),10) - decay_linear(zscore(ctx.v,20),10)
def alpha54(ctx): return (momentum(ema(ctx.c,10),5) / (stddev(ctx.c,10) + EPS)) * (1 - rsi(ctx.c,14)/100.0)
def alpha55(ctx): return (sma(ctx.s,5) - sma(ctx.s,20)) * (ctx.c / ema(ctx.c,30) - 1.0)
def alpha56(ctx):
    rng = ts_max(ctx.c,20) - ts_min(ctx.c,20)
    return (ts_max(ctx.c,20) - ctx.c) / (rng + EPS) - ts_rank(ctx.v,20)
def alpha57(ctx): return rolling_beta(ctx.c, ctx.v,30) * (-zscore(ctx.v,30))
def alpha58(ctx): return zscore(momentum(ctx.c,3),20) - zscore(momentum(ctx.v,3),20) + 0.2 * zscore(ctx.s,20)
def alpha59(ctx): return (rsi(ctx.c,8) - 50.0)/50.0 - 0.3 * (rsi(ctx.v,8) - 50.0)/50.0
def alpha60(ctx): return (ctx.c / (sma(ctx.c,10) + EPS) - 1.0) - (ctx.v / (sma(ctx.v,10) + EPS) - 1.0)
def alpha61(ctx): return (ts_rank(ctx.c - ctx.o,10) - 0.5) + 0.25 * ts_rank(ctx.s,10)
def alpha62(ctx): return -rolling_corr(momentum(ctx.c,5), momentum(ctx.v,5), 20)
def alpha63(ctx): return zscore(decay_linear(ctx.c/lag(ctx.c,1) - 1.0, 5), 20) + 0.2 * zscore(decay_linear(ctx.s,5), 20)
def alpha64(ctx): return (ema(ctx.c - ctx.o,5) / (stddev(ctx.c - ctx.o,20) + EPS)) - 0.1 * zscore(ctx.v,20)
def alpha65(ctx): return (pct_rank(momentum(ctx.c,10),20) - pct_rank(momentum(ctx.v,10),20)) * (1 + 0.1 * pct_rank(ctx.s,20))
def alpha66(ctx):
    r = ctx.c / (lag(ctx.c,1) + EPS) - 1.0
    return ts_sum(r,5) / (ts_sum(abs(r),5) + EPS)
def alpha67(ctx): return (ema(ctx.c,3) - sma(ctx.c,30)) / (stddev(ctx.c,30) + EPS) + 0.15 * (sma(ctx.s,3) - sma(ctx.s,15))
def alpha68(ctx): return (ts_rank(zscore(ctx.c,60),10) - 0.5) - 0.5 * ts_rank(zscore(ctx.v,60),10)
def alpha69(ctx): return (ctx.c - ctx.o) / (abs(ctx.c - lag(ctx.c,1)) + EPS) + 0.1 * zscore(ctx.s,10)
def alpha70(ctx): return clip(zscore(ctx.c - ema(ctx.c,50), 50) + 0.2*zscore(ctx.s,50) - zscore(ctx.v - ema(ctx.v,50), 50), -3, 3)
def alpha71(ctx):
    corr = rolling_corr(ctx.c, ctx.s,20).fillna(0.0)
    return corr * zscore(ctx.c,20) - 0.2 * zscore(ctx.v,20)

# ---------- registry & evaluator ----------
ALPHA_FUNCS = {
    name: fn for name, fn in dict(
        alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4=alpha4, alpha5=alpha5,
        alpha6=alpha6,
        alpha7=alpha7, alpha8=alpha8, alpha9=alpha9, alpha10=alpha10, alpha11=alpha11,
        alpha12=alpha12, alpha13=alpha13, alpha14=alpha14, alpha15=alpha15, alpha16=alpha16,
        alpha17=alpha17, alpha18=alpha18, alpha19=alpha19, alpha20=alpha20, alpha21=alpha21,
        alpha22=alpha22, alpha23=alpha23, alpha24=alpha24, alpha25=alpha25, alpha26=alpha26,
        alpha27=alpha27, alpha28=alpha28, alpha29=alpha29, alpha30=alpha30, alpha31=alpha31,
        alpha32=alpha32, alpha33=alpha33, alpha34=alpha34, alpha35=alpha35, alpha36=alpha36,
        alpha37=alpha37, alpha38=alpha38, alpha39=alpha39, alpha40=alpha40, alpha41=alpha41,
        alpha42=alpha42, alpha43=alpha43, alpha44=alpha44, alpha45=alpha45, alpha46=alpha46,
        alpha47=alpha47, alpha48=alpha48, alpha49=alpha49, alpha50=alpha50, alpha51=alpha51,
        alpha52=alpha52, alpha53=alpha53, alpha54=alpha54, alpha55=alpha55, alpha56=alpha56,
        alpha57=alpha57, alpha58=alpha58, alpha59=alpha59, alpha60=alpha60, alpha61=alpha61,
        alpha62=alpha62, alpha63=alpha63, alpha64=alpha64, alpha65=alpha65, alpha66=alpha66,
        alpha67=alpha67, alpha68=alpha68, alpha69=alpha69, alpha70=alpha70, alpha71=alpha71
    ).items()
}

def evaluate_all(ctx: AlphaCtx, dropna: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame with columns alpha1..alpha71.
    Call ctx.prepare() once before evaluation (or pass a ctx already prepared).
    """
    if ctx.SMA_5 is None:
        ctx = ctx.prepare()
    out = {name: fn(ctx) for name, fn in ALPHA_FUNCS.items()}
    df = pd.DataFrame(out, index=ctx.c.index)
    return df.dropna() if dropna else df

# alphactx_multi.py
class AlphaCtxN:
    """
    Multi-asset adapter around your single-asset AlphaCtx/evaluate_all pipeline.

    Inputs (long format):
      prices   : DataFrame indexed by (date, asset), cols=['open','high','low','close','volume']
      sentiment: Series   indexed by (date, asset)

    Usage:
      import your_alpha_module as A   # where AlphaCtx, evaluate_all live
      ctxN = AlphaCtxN(prices, sentiment)
      alphas_df = ctxN.evaluate_all(A, dropna=True)   # long-format (date, asset) x alpha1..alpha71
    """
    def __init__(self, prices: pd.DataFrame, sentiment: pd.Series):
        if not isinstance(prices.index, pd.MultiIndex) or prices.index.nlevels < 2:
            raise ValueError("prices must be indexed by (date, asset)")
        if not isinstance(sentiment.index, pd.MultiIndex) or sentiment.index.nlevels < 2:
            raise ValueError("sentiment must be indexed by (date, asset)")
        req = {"open", "high", "low", "close", "volume"}
        missing = req - set(prices.columns)
        if missing:
            raise ValueError(f"prices missing columns: {missing}")

        # align and sort
        idx = prices.index.intersection(sentiment.index)
        self.prices = prices.loc[idx].sort_index()
        self.sentiment = sentiment.loc[idx].sort_index().rename("s")

    def iter_asset_ctxs(self, eval_module) -> Iterator[Tuple[str, "eval_module.AlphaCtx"]]:
        """
        Yields (asset, prepared AlphaCtx) for each asset present in the input.
        """
        for asset, df_a in self.prices.groupby(level=1, sort=True):
            s_a = self.sentiment.xs(asset, level=1, drop_level=False)
            s_a = s_a.droplevel(1)  # -> index=dates
            # ensure sentiment is aligned to this asset's dates
            s_a = s_a.reindex(df_a.index.get_level_values(0))

            # build single-asset ctx using your module's AlphaCtx and prepare it
            ctx = eval_module.AlphaCtx(
                c=df_a["close"].droplevel(1),
                o=df_a["open"].droplevel(1),
                v=df_a["volume"].droplevel(1),
                s=s_a,
                h=df_a["high"].droplevel(1),
                l=df_a["low"].droplevel(1),
            ).prepare()
            yield asset, ctx

    def evaluate_all(self, eval_module, dropna: bool = True) -> pd.DataFrame:
        """
        Runs your module's evaluate_all(ctx) for each asset and concatenates results.

        Returns:
          DataFrame indexed by (date, asset), with columns alpha1..alpha71
        """
        frames = []
        for asset, ctx in self.iter_asset_ctxs(eval_module=eval_module):
            A = eval_module.evaluate_all(ctx, dropna=False)
            A["asset"] = asset
            A["date"] = A.index
            frames.append(A.set_index(["date", "asset"]).sort_index())

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames).sort_index()
        return out.dropna() if dropna else out

    # Convenience helpers that mirror what we added earlier
    def make_forward_returns(self, horizon: int = 1) -> pd.Series:
        """
        Forward returns R_{t->t+h} per asset from close, long format index (date, asset).
        """
        close = self.prices["close"]
        # close has (date, asset) index; group by asset and shift
        fwd = close.groupby(level=1).apply(lambda s: s.shift(-horizon) / s - 1.0)
        if isinstance(fwd.index, pd.MultiIndex) and fwd.index.nlevels == 3:
            fwd.index = fwd.index.droplevel(0)
        return fwd.rename(f"fwd_ret_h{horizon}")

    def to_wide(self, df_long: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """
        Utility: convert long (date, asset) to wide (date x asset).
        """
        if isinstance(df_long, pd.Series):
            return df_long.unstack("asset").sort_index()
        return df_long.unstack("asset").sort_index()

def test(prices, sentiment):
    # prices: long DataFrame (date, asset) with open/high/low/close/volume
    # sentiment: long Series (date, asset)
    A = AlphaCtx
    ctxN = AlphaCtxN(prices, sentiment)
    
    alphas_df = ctxN.evaluate_all(A, dropna=True)     # (date, asset) x alpha1..alpha71
    fwd_ret    = ctxN.make_forward_returns(horizon=1) # (date, asset) Series
    
    # (Optional) standardize cross-section and run checks using helpers you already have:
    from . import data
    sanity = data.sanity_check(alphas_df)
    alphas_std = data.standardize_cross_section(alphas_df, winsor=(0.01,0.99), clip_z=5.0, by_group=None, exposures=None)
