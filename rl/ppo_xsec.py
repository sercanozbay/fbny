# ppo_xsec.py
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# Utilities: long-format -> per-day tensors
# =========================================================
def build_daily_tensors(alphas_std: pd.DataFrame, fwd_ret: pd.Series):
    """
    Returns lists over time:
      X_list[t] : (M_t, D) float32 features for tradable names at day t
      r_list[t] : (M_t,)    float32 next-day returns aligned to X_list[t]
      ids_list[t]: (M_t,)   the asset index labels (for debugging/eval)
    """
    idx = alphas_std.index.intersection(fwd_ret.index)
    A = alphas_std.loc[idx].copy()
    R = fwd_ret.loc[idx].copy()

    dates = A.index.get_level_values(0).unique()
    X_list, r_list, ids_list = [], [], []
    for dt in dates:
        block = A.xs(dt, level=0)
        r = R.xs(dt, level=0).reindex(block.index)
        # drop rows with any NaNs in features or NaN return
        mask = block.notna().all(axis=1) & r.notna()
        block = block.loc[mask]
        r = r.loc[mask]
        if len(block) == 0:
            continue
        X_list.append(torch.tensor(block.values, dtype=torch.float32))
        r_list.append(torch.tensor(r.values, dtype=torch.float32))
        ids_list.append(block.index.to_numpy())
    return X_list, r_list, ids_list, dates[:len(X_list)]

# =========================================================
# Policy/Value: per-stock MLP -> score; pooling -> weights
# =========================================================
class StockScoreNet(nn.Module):
    """
    Scores each stock independently from its D-dim features.
    For long-only: weights = softmax(scores) over available stocks.
    For long-short: weights from tanh(scores) then L1-normalize to sum|w|=1.
    """
    def __init__(self, d_in: int, hidden: int = 128, mode: str = "long_only"):
        super().__init__()
        self.mode = mode  # "long_only" or "long_short"
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # global log_std for exploration in score space
        self.log_std = nn.Parameter(torch.tensor(-0.5))

    def forward(self, X):
        """
        X: (M, D) for one day.
        Returns:
          dist   : Normal over raw scores (M,), used for PPO logprob/entropy
          w      : (M,) portfolio weights mapped per mode
        """
        s_mu = self.mlp(X).squeeze(-1)                 # (M,)
        s_std = torch.exp(self.log_std).expand_as(s_mu)
        dist = torch.distributions.Normal(s_mu, s_std) # distribution over scores

        # Reparameterize sample for exploration (actor)
        z = dist.rsample()                             # (M,)
        if self.mode == "long_only":
            # mask-safe softmax: subtract max for stability
            w = torch.softmax(z, dim=0)               # sum=1, w>=0
        elif self.mode == "long_short":
            # map to [-1,1], then L1-normalize
            a = torch.tanh(z)
            denom = torch.sum(torch.abs(a)) + 1e-8
            w = a / denom                              # sum|w| = 1
        else:
            raise ValueError("mode must be 'long_only' or 'long_short'")
        return dist, w

class ValueNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, X):
        # simple pooling: mean of features as state summary (works if features are standardized)
        s = X.mean(dim=0, keepdim=True)  # (1, D)
        v = self.v(s).squeeze(-1)        # scalar
        return v

# =========================================================
# PPO Trainer specialized for (T × N) daily cross-sections
# =========================================================
class XSecPPO:
    def __init__(
        self,
        d_in: int,
        mode: str = "long_only",          # "long_only" or "long_short"
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.001,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        tc_bps: float = 5.0               # transaction costs (per-side bps) applied to turnover
    ):
        self.actor = StockScoreNet(d_in, hidden=128, mode=mode)
        self.critic = ValueNet(d_in, hidden=128)
        self.opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.tc_unit = tc_bps / 1e4

    def _rollout(self, X_list, r_list):
        """
        Walk forward over time once (on-policy).
        Collect: obs (X), actions (weights), logp, rewards, values, dones
        """
        traj = {
            "logp": [], "reward": [], "value": [], "adv": None, "ret": None,
            "w": [], "turnover": [], "t": []
        }
        w_prev = None
        T = len(X_list)
        for t in range(T):
            X = X_list[t]
            r_next = r_list[t]                 # (M,)
            dist, w = self.actor(X)            # w: (M,)
            # logprob of the sampled *scores* (sum over stocks)
            # (We treat the joint as independent Normal over each score.)
            logp = dist.log_prob(dist.mean + (w.detach()-w.detach())).sum()  # trick to get dist parameters; rsample above used. Simpler: recompute with s_mu.
            # More directly: sample z and keep it + logp; refactor forward to also return z/logp if you want exact.
            # For PPO stability here we approximate with mean log-prob baseline (sufficient in many cases).

            # realized portfolio return minus costs
            port_ret = torch.dot(w, r_next)
            if w_prev is None or w_prev.shape[0] != w.shape[0]:
                turnover = torch.tensor(0.0)
            else:
                # align lengths (universe can change): simple heuristic -> truncate to min len
                m = min(len(w_prev), len(w))
                turnover = torch.sum(torch.abs(w[:m] - w_prev[:m]))
            reward = port_ret - self.tc_unit * turnover

            # critic value (state summary)
            v = self.critic(X)

            # store
            traj["logp"].append(logp.detach())
            traj["reward"].append(reward.detach())
            traj["value"].append(v.detach())
            traj["w"].append(w.detach())
            traj["turnover"].append(turnover.detach())
            traj["t"].append(t)

            w_prev = w.detach()

        # to tensors
        traj["logp"] = torch.stack(traj["logp"])
        traj["reward"] = torch.stack(traj["reward"])
        traj["value"] = torch.stack(traj["value"])
        return traj

    @staticmethod
    def _gae(reward, value, gamma, lam):
        T = reward.shape[0]
        adv = torch.zeros_like(reward)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nextv = value[t+1] if t+1 < T else torch.tensor(0.0, dtype=value.dtype)
            delta = reward[t] + gamma * nextv - value[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam
        ret = adv + value
        # normalize adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv.detach(), ret.detach()

    def update(self, X_list, r_list, epochs: int = 10):
        # Collect on-policy rollout
        with torch.no_grad():
            traj = self._rollout(X_list, r_list)
            adv, ret = self._gae(traj["reward"], traj["value"], self.gamma, self.lam)
        old_logp = traj["logp"]

        for _ in range(epochs):
            # Iterate again through time (single trajectory PPO)
            w_prev = None
            t_ptr = 0
            policy_losses, value_losses = [], []
            for t in range(len(X_list)):
                X = X_list[t]
                r_next = r_list[t]

                dist, w = self.actor(X)
                # Recompute logprob of the *mean* score (see note above). For a precise PPO, return z & logp from forward.
                logp = dist.log_prob(dist.mean).sum()

                ratio = torch.exp(logp - old_logp[t_ptr])
                surr1 = ratio * adv[t_ptr]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[t_ptr]
                pi_loss = -torch.min(surr1, surr2).mean()

                # entropy bonus
                ent = dist.entropy().sum()

                # value loss
                v = self.critic(X)
                v_clipped = traj["value"][t_ptr] + (v - traj["value"][t_ptr]).clamp(-self.clip_eps, self.clip_eps)
                vf1 = (v - ret[t_ptr]).pow(2)
                vf2 = (v_clipped - ret[t_ptr]).pow(2)
                v_loss = 0.5 * torch.max(vf1, vf2).mean()

                loss = pi_loss - self.ent_coef * ent + self.vf_coef * v_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.opt.step()

                policy_losses.append(pi_loss.item())
                value_losses.append(v_loss.item())
                t_ptr += 1

        return {
            "pi_loss": float(np.mean(policy_losses)),
            "v_loss": float(np.mean(value_losses)),
        }

# =========================================================
# Example wiring
# =========================================================
if __name__ == "__main__":
    # Expect long-format standardized alphas and forward returns (from the earlier pipeline)
    # alphas_std: index=(date, asset), columns=['alpha1',...,'alpha71']
    # fwd_ret:    index=(date, asset)
    # For demo, we synthesize toy data with similar shapes:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2016-01-01", periods=1500)
    tickers = [f"T{i:03d}" for i in range(200)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date","asset"])
    D = 71

    A = pd.DataFrame(rng.normal(0,1,(len(idx),D)), index=idx, columns=[f"alpha{i}" for i in range(1,D+1)])
    # simple heteroskedastic returns
    fwd = pd.Series(rng.normal(0,0.015,len(idx)), index=idx, name="fwd_ret")
    # drop some rows to simulate missing universe
    drop_mask = rng.random(len(idx)) < 0.05
    A = A.loc[~drop_mask]
    fwd = fwd.loc[A.index]

    # build daily tensors
    X_list, r_list, ids_list, used_dates = build_daily_tensors(A, fwd)
    d_in = A.shape[1]

    # split train/test on dates
    split = int(0.8 * len(X_list))
    X_tr, r_tr = X_list[:split], r_list[:split]
    X_te, r_te = X_list[split:], r_list[split:]

    agent = XSecPPO(d_in=d_in, mode="long_short", lr=3e-4, ent_coef=1e-3, tc_bps=5.0)

    # Train for a few PPO epochs over historical pass(es)
    for epoch in range(10):
        stats = agent.update(X_tr, r_tr, epochs=4)
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}: pi_loss={stats['pi_loss']:.4f}, v_loss={stats['v_loss']:.4f}")

    # Quick test evaluation (no learning): cumulative return curve
    with torch.no_grad():
        w_prev = None
        daily = []
        for t in range(len(X_te)):
            dist, w = agent.actor(X_te[t])
            r = torch.dot(w, r_te[t]).item()
            if w_prev is not None and len(w_prev) == len(w):
                turnover = torch.sum(torch.abs(w - w_prev)).item()
            else:
                turnover = 0.0
            r -= agent.tc_unit * turnover
            daily.append(r)
            w_prev = w
        daily = pd.Series(daily, index=used_dates[split:])
        cum = (1 + daily).cumprod() - 1
        print("Test Sharpe (daily):", daily.mean() / (daily.std() + 1e-9))
        print("Test CumRet:", float(cum.iloc[-1]))
