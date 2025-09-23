# ppo_alpha_weighting.py
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# Utilities: long-format -> per-day tensors
# =========================================================
def build_daily_tensors(alphas_df: pd.DataFrame, fwd_ret: pd.Series):
    """
    Returns time lists:
      A_list[t] : (N_t, K) float32 standardized alphas per stock at day t
      r_list[t] : (N_t,)    float32 next-day returns aligned to A_list[t]
      ids_list[t]: (N_t,)   asset identifiers (for your own alignment/debug)
      dates_used : DatetimeIndex of used days (after dropping empty)
    Drops rows with any NaN in alphas or NaN in returns per day.
    """
    idx = alphas_df.index.intersection(fwd_ret.index)
    A = alphas_df.loc[idx].copy()
    R = fwd_ret.loc[idx].copy()

    dates = A.index.get_level_values(0).unique()
    A_list, r_list, ids_list, used = [], [], [], []
    for dt in dates:
        block = A.xs(dt, level=0)
        r = R.xs(dt, level=0).reindex(block.index)
        mask = block.notna().all(axis=1) & r.notna()
        block = block.loc[mask]
        r = r.loc[mask]
        if len(block) == 0:
            continue
        A_list.append(torch.tensor(block.values, dtype=torch.float32))
        r_list.append(torch.tensor(r.values, dtype=torch.float32))
        ids_list.append(block.index.to_numpy())
        used.append(dt)
    return A_list, r_list, ids_list, pd.DatetimeIndex(used)


# =========================================================
# Actor: alpha-weights policy  (action in R^K)
#   - Dist over alpha weights (Gaussian in weight space)
#   - Map from alpha weights -> stock weights via A_t @ w_t
# =========================================================
class AlphaWeightPolicy(nn.Module):
    """
    Policy over alpha weights:
      Input  : summary of A_t  (K-dim per day)
      Action : w_alpha (K,) via Normal, with exploration in weight space
      Mapping: s = A_t @ w_alpha  -> portfolio over stocks
    """
    def __init__(self, K: int, hidden: int = 256, mode: str = "long_only"):
        super().__init__()
        self.mode = mode  # "long_only" or "long_short"
        # Small MLP on daily K-dim summary (mean across stocks of A_t)
        self.pi = nn.Sequential(
            nn.Linear(K, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, K)
        )
        # learnable log-std per alpha dimension
        self.log_std = nn.Parameter(torch.full((K,), -0.5))
        # temperature for stock softmax if long_only
        self.register_buffer("score_temp", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, A_t: torch.Tensor):
        """
        A_t: (N, K) alpha matrix for one day (standardized cross-sectionally)
        Returns:
          dist      : Normal over alpha weights (K,)
          w_alpha   : sampled alpha weights (K,)
          logp      : scalar log-prob of sampled w_alpha
          w_stock   : (N,) mapped portfolio weights over stocks
          s_stock   : (N,) raw composite scores A_t @ w_alpha
        """
        # summarize the day by mean alpha vector (could use other pooling/transformer)
        K = A_t.shape[1]
        feat = A_t.mean(dim=0)                 # (K,)
        mu = self.pi(feat)                     # (K,)
        std = torch.exp(self.log_std).clamp_min(1e-6)
        dist = torch.distributions.Normal(mu, std)
        w_alpha = dist.rsample()               # (K,)
        logp = dist.log_prob(w_alpha).sum()

        # Composite per-stock scores, then map to portfolio weights
        s = A_t @ w_alpha                      # (N,)
        if self.mode == "long_only":
            w_stock = torch.softmax(s / self.score_temp, dim=0)         # sum=1, >=0
        elif self.mode == "long_short":
            a = torch.tanh(s)                                         # [-1,1]
            w_stock = a / (a.abs().sum() + 1e-8)                      # sum|w|=1
        else:
            raise ValueError("mode must be 'long_only' or 'long_short'")
        return dist, w_alpha, logp, w_stock, s


# =========================================================
# Critic: value of the day (from cross-sectional alpha summary)
# =========================================================
class ValueNet(nn.Module):
    def __init__(self, K: int, hidden: int = 256):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(K, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, A_t: torch.Tensor):
        feat = A_t.mean(dim=0, keepdim=True)   # (1,K) mean-pooled alpha vector
        return self.v(feat).squeeze(-1)        # scalar


# =========================================================
# PPO trainer
# =========================================================
class AlphaWeightingPPO:
    def __init__(
        self,
        K: int,
        mode: str = "long_only",      # "long_only" or "long_short"
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 1e-3,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        tc_bps: float = 5.0           # per-side bps applied to turnover in stock weights
    ):
        self.actor  = AlphaWeightPolicy(K, hidden=256, mode=mode)
        self.critic = ValueNet(K, hidden=256)
        self.opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.tc_unit = tc_bps / 1e4

    def _roll(self, A_list, r_list):
        """
        One on-policy pass over all days.
        Returns a dict of tensors for PPO update.
        """
        logp_buf, val_buf, rew_buf = [], [], []
        walpha_buf, wstock_buf = [], []
        w_prev = None

        for t in range(len(A_list)):
            A_t = A_list[t]                     # (N_t, K)
            r_t = r_list[t]                     # (N_t,)

            dist, w_alpha, logp, w_stock, s = self.actor(A_t)
            v = self.critic(A_t)

            # realized PnL minus transaction costs on stock weights
            pnl = torch.dot(w_stock, r_t)
            if w_prev is None or w_prev.shape[0] != w_stock.shape[0]:
                turnover = torch.tensor(0.0)
            else:
                m = min(len(w_prev), len(w_stock))  # simple overlap heuristic
                turnover = torch.sum(torch.abs(w_stock[:m] - w_prev[:m]))
            reward = pnl - self.tc_unit * turnover

            # stash
            logp_buf.append(logp.detach())
            val_buf.append(v.detach())
            rew_buf.append(reward.detach())
            walpha_buf.append(w_alpha.detach())
            wstock_buf.append(w_stock.detach())
            w_prev = w_stock.detach()

        return {
            "logp": torch.stack(logp_buf),
            "value": torch.stack(val_buf),
            "reward": torch.stack(rew_buf),
            "w_alpha": walpha_buf,
            "w_stock": wstock_buf
        }

    @staticmethod
    def _gae(reward: torch.Tensor, value: torch.Tensor, gamma: float, lam: float):
        """
        Standard GAE over a single trajectory.
        """
        T = reward.shape[0]
        adv = torch.zeros_like(reward)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nextv = value[t+1] if t+1 < T else torch.tensor(0.0, dtype=value.dtype)
            delta = reward[t] + gamma * nextv - value[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam
        ret = adv + value
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv.detach(), ret.detach()

    def update(self, A_list, r_list, epochs: int = 5):
        with torch.no_grad():
            traj = self._roll(A_list, r_list)
            adv, ret = self._gae(traj["reward"], traj["value"], self.gamma, self.lam)
            old_logp = traj["logp"]

        pi_losses, v_losses = [], []
        for _ in range(epochs):
            t_ptr = 0
            for t in range(len(A_list)):
                A_t = A_list[t]

                # Rebuild the distribution and evaluate likelihood at the SAME sampled w_alpha
                # (textbook PPO ratio)
                feat = A_t.mean(dim=0)                    # (K,)
                mu = self.actor.pi(feat)
                std = torch.exp(self.actor.log_std).clamp_min(1e-6)
                dist = torch.distributions.Normal(mu, std)
                # use the old sampled alpha weights
                w_alpha_old = traj["w_alpha"][t]          # (K,)
                logp = dist.log_prob(w_alpha_old).sum()

                ratio = torch.exp(logp - old_logp[t_ptr])
                surr1 = ratio * adv[t_ptr]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[t_ptr]
                pi_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus (encourage exploration in alpha-weight space)
                ent = dist.entropy().sum()

                # Critic
                v = self.critic(A_t)
                v_clip = traj["value"][t_ptr] + (v - traj["value"][t_ptr]).clamp(-self.clip_eps, self.clip_eps)
                vf1 = (v - ret[t_ptr]).pow(2)
                vf2 = (v_clip - ret[t_ptr]).pow(2)
                v_loss = 0.5 * torch.max(vf1, vf2).mean()

                loss = pi_loss - self.ent_coef * ent + self.vf_coef * v_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.opt.step()

                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())
                t_ptr += 1

        return {"pi_loss": float(np.mean(pi_losses)), "v_loss": float(np.mean(v_losses))}

# =========================================================
# Example wiring (toy data) — replace with real bundle outputs
# =========================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Make a toy long-format panel: 1200 days, 300 stocks, 71 alphas
    dates = pd.bdate_range("2016-01-01", periods=1200)
    assets = [f"T{i:03d}" for i in range(300)]
    idx = pd.MultiIndex.from_product([dates, assets], names=["date","asset"])
    K = 71

    A_long = pd.DataFrame(rng.normal(0,1,(len(idx), K)), index=idx, columns=[f"alpha{i}" for i in range(1,K+1)])
    # forward returns with mild cross-sectional signal on alpha1 for fun
    base = rng.normal(0, 0.012, len(idx))
    signal = A_long["alpha1"].values * 0.002
    fwd = pd.Series(base + signal, index=idx, name="fwd_ret")

    # Drop some rows to simulate ragged universe
    drop_mask = rng.random(len(idx)) < 0.05
    A_long = A_long.loc[~drop_mask]
    fwd = fwd.loc[A_long.index]

    # Build per-day tensors
    A_list, r_list, ids_list, used_dates = build_daily_tensors(A_long, fwd)

    # Split train/test by days
    split = int(0.8 * len(A_list))
    A_tr, r_tr = A_list[:split], r_list[:split]
    A_te, r_te = A_list[split:], r_list[split:]

    agent = AlphaWeightingPPO(
        K=K,
        mode="long_short",  # try "long_only" too
        lr=3e-4,
        ent_coef=1e-3,
        tc_bps=5.0
    )

    # Train a few PPO passes
    for ep in range(8):
        stats = agent.update(A_tr, r_tr, epochs=4)
        if (ep+1) % 2 == 0:
            print(f"Epoch {ep+1}: pi_loss={stats['pi_loss']:.4f}  v_loss={stats['v_loss']:.4f}")

    # Evaluate on test set
    with torch.no_grad():
        daily = []
        w_prev = None
        for t in range(len(A_te)):
            dist, w_alpha, logp, w_stock, s = agent.actor(A_te[t])
            pnl = torch.dot(w_stock, r_te[t]).item()
            if w_prev is None or len(w_prev) != len(w_stock):
                to = 0.0
            else:
                m = min(len(w_prev), len(w_stock))
                to = torch.sum(torch.abs(w_stock[:m] - w_prev[:m])).item()
            daily.append(pnl - agent.tc_unit * to)
            w_prev = w_stock
        daily = pd.Series(daily, index=used_dates[split:])
        sharpe = daily.mean() / (daily.std() + 1e-12) * math.sqrt(252.0)
        cum = (1 + daily).cumprod() - 1
        print("Test Sharpe (ann):", float(sharpe))
        print("Test CumRet:", float(cum.iloc[-1]))

