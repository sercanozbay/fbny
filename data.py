import numpy as np
import pandas as pd

# =========================
# 1) Sanity checks
# =========================
def sanity_check(
    alphas_df: pd.DataFrame,
    min_non_nan_ratio: float = 0.80,
    near_const_threshold: float = 1e-6,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per alpha and diagnostic columns:
      - non_nan_ratio: fraction of non-NaN observations
      - is_zero_var:    True if global std == 0 (ignoring NaNs)
      - near_const:     True if global std < near_const_threshold
      - mean, std, max_abs: global distribution quick stats
      - p_infinite:     fraction of +/-inf values
    """
    df = alphas_df.copy()

    # basic stats (ignore NaNs)
    count = df.notna().sum()
    total = df.shape[0]
    non_nan_ratio = count / np.maximum(total, 1)

    std = df.std(skipna=True)
    mean = df.mean(skipna=True)
    max_abs = df.abs().max(skipna=True)

    is_zero_var = std.fillna(0.0).eq(0.0)
    near_const = std.fillna(0.0).lt(near_const_threshold)

    # inf check
    p_inf = np.isinf(df).sum() / np.maximum(total, 1)

    report = pd.DataFrame({
        "non_nan_ratio": non_nan_ratio,
        "is_zero_var": is_zero_var,
        "near_const": near_const,
        "mean": mean,
        "std": std,
        "max_abs": max_abs,
        "p_infinite": p_inf,
        "flag_low_coverage": non_nan_ratio.lt(min_non_nan_ratio),
    }).sort_index()

    return report.sort_values(["flag_low_coverage", "is_zero_var", "near_const", "p_infinite"], ascending=False)


# =========================
# 2) Cross-sectional standardization
#    (winsorize -> optional de-mean by group/exposures -> zscore)
# =========================
def _winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    if s.notna().sum() == 0:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

def standardize_cross_section(
    alphas_df: pd.DataFrame,
    *,
    winsor: tuple[float, float] = (0.01, 0.99),
    clip_z: float | None = 5.0,
    by_group: pd.Series | None = None,
    exposures: pd.DataFrame | None = None,
    min_obs_per_day: int = 10,
) -> pd.DataFrame:
    """
    Cross-sectionally standardize each date:
      1) Winsorize each column within date (e.g., 1st-99th pct).
      2a) If by_group is provided: de-mean by group within date (industry-neutral).
      2b) Else if exposures provided: regress each alpha on exposures per date, take residuals.
      3) Z-score within date (mean=0, std=1). Optionally clip z to +/- clip_z.

    Args:
      alphas_df: Multi-asset DataFrame indexed by date with columns = alpha names.
                 Rows represent a *cross-section* (e.g., many tickers) at each date.
      winsor: (low, high) quantiles for winsorization per date.
      clip_z: final z clipping threshold; None to disable.
      by_group: pd.Series aligned to alphas_df rows giving a discrete group (e.g., industry code).
      exposures: DataFrame with K exposures (e.g., [market_beta, size]), aligned to rows.
                 If provided, we do per-date OLS to remove exposures (residualize).
      min_obs_per_day: skip standardization on dates with fewer than this many non-NaN rows;
                       those rows are left as-is.

    Returns:
      DataFrame of same shape, standardized cross-sectionally per date.
    """
    low_q, hi_q = winsor
    Z = alphas_df.copy()

    # Per-date operation
    def _process_date(block: pd.DataFrame) -> pd.DataFrame:
        # require enough obs
        if block.notna().any(axis=1).sum() < min_obs_per_day:
            return block

        # 1) winsorize each alpha
        block_w = block.apply(_winsorize_series, axis=0, lower=low_q, upper=hi_q)

        # 2) neutralize
        if by_group is not None:
            # group-wise de-mean
            g = by_group.loc[block_w.index]
            block_n = block_w.groupby(g).transform(lambda s: s - s.mean(skipna=True))
        elif exposures is not None:
            X = exposures.loc[block_w.index]
            # add intercept
            X_ = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
            # solve OLS for each alpha: y = Xb; residual = y - Xb
            # safeguard: drop rows with NaNs in X_ per column fit
            block_n = pd.DataFrame(index=block_w.index, columns=block_w.columns, dtype=float)
            for col in block_w.columns:
                y = block_w[col]
                mask = y.notna() & X_.notna().all(axis=1)
                if mask.sum() >= X_.shape[1] + 1:
                    # closed-form OLS (X'X)^{-1} X'y
                    X_m = X_.loc[mask].values
                    y_m = y.loc[mask].values
                    try:
                        beta = np.linalg.pinv(X_m.T @ X_m) @ (X_m.T @ y_m)
                        y_hat = X_m @ beta
                        resid = y_m - y_hat
                        block_n.loc[mask, col] = resid
                    except Exception:
                        # fallback: no neutralization on this alpha/date
                        block_n[col] = block_w[col]
                else:
                    block_n[col] = block_w[col]
        else:
            block_n = block_w

        # 3) z-score per date
        mu = block_n.mean(axis=0, skipna=True)
        sd = block_n.std(axis=0, skipna=True).replace(0.0, np.nan)
        out = (block_n - mu) / sd

        # optional clipping
        if clip_z is not None:
            out = out.clip(lower=-clip_z, upper=clip_z)

        return out

    # Apply by date (index assumed DateTimeIndex or PeriodIndex; works with any index grouped by level=0)
    standardized = Z.groupby(Z.index).apply(_process_date)
    # groupby adds an extra index level; drop it if present
    if isinstance(standardized.index, pd.MultiIndex):
        standardized.index = standardized.index.get_level_values(0)
    return standardized


# =========================
# (Optional) tiny evaluator
# =========================
def information_coefficient(alphas_df: pd.DataFrame, fwd_ret: pd.Series, method="spearman") -> pd.Series:
    """
    Cross-sectional IC per date, then average across dates.
    method: 'spearman' or 'pearson'
    """
    def _ic_on_date(block):
        r = fwd_ret.loc[block.index]
        if method == "spearman":
            return block.rank(axis=0).corrwith(r.rank(), method="pearson", drop=True)
        else:
            return block.corrwith(r, method="pearson", drop=True)

    per_date = alphas_df.groupby(alphas_df.index).apply(_ic_on_date)
    return per_date.mean(axis=0).sort_values(ascending=False)

def test():
      # alphas_df: rows = (date, asset) multiindex or just date if already screened to a universe per day
    # simplest: index = date, one row PER ASSET per date, with columns alpha1..alpha71
    
    report = sanity_check(alphas_df)
    print(report.head(10))
    
    # industry neutralized, winsorized, z-scored:
    std_df = standardize_cross_section(
        alphas_df,
        winsor=(0.01, 0.99),
        by_group=industry_series,      # pd.Series of industry codes per row (or None)
        exposures=None,                # or a DataFrame like pd.DataFrame({"beta": mkt_beta, "size": ln_mcap})
    )
    
    # Optional: quick IC snapshot against next-day returns
    ic = information_coefficient(std_df, fwd_ret=next_day_returns, method="spearman")
    print(ic.head(15))

