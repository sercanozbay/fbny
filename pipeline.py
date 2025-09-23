import numpy as np
import pandas as pd

# =========================================================
# 1) Forward returns (works on long or wide data)
# =========================================================
def make_forward_returns(
    close: pd.Series | pd.DataFrame,
    horizon: int = 1,
    long_format: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    Compute forward returns R_{t->t+h} = close[t+h]/close[t] - 1.

    If long_format:
      - 'close' should be a Series indexed by a MultiIndex (date, asset) OR
        a DataFrame with (date, asset) index and a 'close' column.
      - Returns a Series aligned to (date, asset), with NaN for last h rows per asset.

    Else (wide):
      - 'close' should be a DataFrame indexed by date with columns=assets/tickers.
      - Returns a DataFrame of same shape.
    """
    if long_format:
        if isinstance(close, pd.DataFrame):
            if "close" in close.columns and "asset" not in close.columns:
                s = close["close"]
            else:
                raise ValueError("For long_format=True, pass a Series with (date, asset) index or a DataFrame with a 'close' column.")
        else:
            s = close
        if not isinstance(s.index, pd.MultiIndex) or len(s.index.levels) < 2:
            raise ValueError("Long format requires a MultiIndex (date, asset).")
        fwd = s.groupby(level=1).apply(lambda x: x.shift(-horizon) / x - 1.0)
        # flatten groupby-added level if present
        if isinstance(fwd.index, pd.MultiIndex) and fwd.index.nlevels == 3:
            fwd.index = fwd.index.droplevel(0)
        return fwd.rename("fwd_ret_h{}".format(horizon))
    else:
        if not isinstance(close, pd.DataFrame):
            raise ValueError("For long_format=False, pass a wide DataFrame (date x assets).")
        return close.shift(-horizon) / close - 1.0


# =========================================================
# 2) Turnover proxy (average absolute day-over-day change)
# =========================================================
def turnover(
    signals: pd.DataFrame,
    by_asset: pd.Series | None = None,
) -> pd.Series:
    """
    signals: long-format DataFrame indexed by (date, asset) with columns=alpha names.
    Returns:
      A Series indexed by alpha name: mean over dates of cross-sectional mean |Δ signal|.
    If by_asset is None, asset is inferred from the MultiIndex second level.
    """
    if not isinstance(signals.index, pd.MultiIndex):
        raise ValueError("signals must be indexed by (date, asset) in long format.")
    # Compute per-asset diffs then absolute
    diffs = signals.groupby(level=1).diff().abs()
    # Cross-sectional mean per date, then time-average
    xs_mean = diffs.groupby(level=0).mean()
    return xs_mean.mean(axis=0).sort_values(ascending=False)


# =========================================================
# 3) End-to-end pipeline: prices + sentiment -> 71 alphas
# =========================================================
def build_alphas_pipeline(
    prices: pd.DataFrame,
    sentiment: pd.Series,
    *,
    horizon: int = 1,
    winsor: tuple[float, float] = (0.01, 0.99),
    clip_z: float | None = 5.0,
    by_group: pd.Series | None = None,      # e.g., industry code per (date, asset) row
    exposures: pd.DataFrame | None = None,  # e.g., {'beta':..., 'size':...} aligned to rows
    dropna_final: bool = True,
    eval_module=None,   # pass the module or namespace that defines AlphaCtx/evaluate_all/sanity_check/standardize_cross_section
):
    """
    Inputs (all LONG format, aligned index):
      prices: DataFrame indexed by (date, asset) with columns: ['open','high','low','close','volume']
      sentiment: Series indexed by (date, asset) with per-asset s_t values

    Returns dict with:
      - 'alphas_raw':   long-format DataFrame of 71 raw alphas
      - 'alphas_std':   standardized/winsorized/neutralized alphas (cross-section per date)
      - 'fwd_ret':      forward returns Series (horizon)
      - 'sanity':       diagnostics DataFrame from sanity_check
      - 'turnover':     per-alpha turnover proxy
      - 'train':        dict with train splits: {'alphas':..., 'fwd':...}
      - 'test':         dict with test splits: {'alphas':..., 'fwd':...}
    """
    if eval_module is None:
        raise ValueError("Please pass eval_module that exposes AlphaCtx, evaluate_all, sanity_check, standardize_cross_section")

    # --- align & basic checks ---
    req_cols = {"open","high","low","close","volume"}
    if not req_cols.issubset(prices.columns):
        missing = req_cols - set(prices.columns)
        raise ValueError(f"prices is missing columns: {missing}")
    if not isinstance(prices.index, pd.MultiIndex) or prices.index.nlevels < 2:
        raise ValueError("prices must be indexed by (date, asset)")
    if not isinstance(sentiment.index, pd.MultiIndex):
        raise ValueError("sentiment must be indexed by (date, asset)")

    # align indexes
    idx = prices.index.intersection(sentiment.index)
    P = prices.loc[idx].sort_index()
    S = sentiment.loc[idx].sort_index().rename("s")

    # --- build raw alphas per asset using AlphaCtx + evaluate_all ---
    frames = []
    for asset, df_a in P.groupby(level=1):
        s_a = S.xs(asset, level=1)
        # construct ctx
        ctx = eval_module.AlphaCtx(
            c=df_a["close"], o=df_a["open"], v=df_a["volume"],
            s=s_a.reindex(df_a.index), h=df_a["high"], l=df_a["low"]
        ).prepare()
        A = eval_module.evaluate_all(ctx, dropna=False)
        # add asset key back
        A["asset"] = asset
        A["date"] = A.index
        A = A.set_index(["date","asset"]).sort_index()
        frames.append(A)

    alphas_raw = pd.concat(frames).sort_index()

    # --- optional sanity report before standardization ---
    sanity = eval_module.sanity_check(alphas_raw)

    # --- forward returns ---
    fwd_ret = make_forward_returns(P["close"], horizon=horizon, long_format=True)

    # --- standardize cross-section per date (winsorize -> neutralize -> zscore) ---
    alphas_std = eval_module.standardize_cross_section(
        alphas_raw,
        winsor=winsor,
        clip_z=clip_z,
        by_group=by_group,
        exposures=exposures,
        min_obs_per_day=10,
    )

    # --- align final matrices (drop NaNs if requested) ---
    common_idx = alphas_std.index.intersection(fwd_ret.index)
    alphas_std = alphas_std.loc[common_idx]
    alphas_raw = alphas_raw.reindex(common_idx)
    fwd_ret = fwd_ret.loc[common_idx]

    if dropna_final:
        # drop rows where either all alphas are NaN or fwd_ret is NaN
        mask = (~alphas_std.isna().all(axis=1)) & fwd_ret.notna()
        alphas_std = alphas_std.loc[mask]
        alphas_raw = alphas_raw.loc[mask]
        fwd_ret = fwd_ret.loc[mask]

    # --- turnover proxy on standardized alphas ---
    tover = turnover(alphas_std)

    # --- time-based split (last 20% as test) ---
    dates = alphas_std.index.get_level_values(0).unique()
    split_at = int(np.floor(0.8 * len(dates)))
    train_dates = dates[:split_at]
    test_dates  = dates[split_at:]

    train = {
        "alphas": alphas_std.loc[alphas_std.index.get_level_values(0).isin(train_dates)],
        "fwd":    fwd_ret.loc[fwd_ret.index.get_level_values(0).isin(train_dates)],
    }
    test = {
        "alphas": alphas_std.loc[alphas_std.index.get_level_values(0).isin(test_dates)],
        "fwd":    fwd_ret.loc[fwd_ret.index.get_level_values(0).isin(test_dates)],
    }

    return {
        "alphas_raw": alphas_raw,
        "alphas_std": alphas_std,
        "fwd_ret": fwd_ret.rename("fwd_ret"),
        "sanity": sanity,
        "turnover": tover,
        "train": train,
        "test": test,
    }
