import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Utilities
# ============================================================
def rolling_quantile(x: pd.Series, q: float, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).quantile(q)

def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# ============================================================
# Minimal, no-gym environment faithful to the paper
#   - state s_t = [open, high, low, close, volume, p_{t-1}, regime_t, sigma_annual_t]
#   - action a_t in R^N -> clip to [-1,1], then L1-normalize -> weights
#   - composite alpha, thresholds, regime filter, vol scaling
#   - reward r_t = p_t*R_{t->t+1} - λ|Δp_t|
# ============================================================
class AlphaWeightingEnvLite:
    def __init__(
        self,
        ohlcv_df: pd.DataFrame,      # columns: open,high,low,close,volume
        future_ret: pd.Series,       # forward returns aligned with df.index
        alphas: np.ndarray,          # shape (T, N) standardized columns
        sigma_window: int = 63,
        ma_fast: int = 20,
        ma_slow: int = 100,
        q_window: int = 126,
        q_low: float = 0.25,
        q_high: float = 0.75,
        sigma_target: float = 0.15,
        tc_lambda: float = 0.001,
        action_clip: float = 1.0,
        max_steps: int | None = None,
    ):
        assert len(ohlcv_df) == len(future_ret) == alphas.shape[0], "Length mismatch"
        self.df = ohlcv_df.copy()
        self.r_fwd = future_ret.values.astype(np.float32)
        self.alphas = alphas.astype(np.float32)
        self.T, self.N = self.alphas.shape

        close = self.df["close"]

        # Regime (MA20 > MA100), annualized vol from 63d std, price quantiles over 126d
        self.ma_fast = close.rolling(ma_fast, min_periods=ma_fast).mean().values
        self.ma_slow = close.rolling(ma_slow, min_periods=ma_slow).mean().values
        self.regime = (self.ma_fast > self.ma_slow).astype(np.float32)

        rv = pd.Series(self.r_fwd, index=self.df.index).rolling(sigma_window, min_periods=sigma_window).std().values
        self.sigma_annual = (rv * math.sqrt(252.0)).astype(np.float32)

        self.tau_upper = rolling_quantile(close, q_high, q_window).values.astype(np.float32)
        self.tau_lower = rolling_quantile(close, q_low,  q_window).values.astype(np.float32)

        self.sigma_target = float(sigma_target)
        self.tc_lambda = float(tc_lambda)
        self.action_clip = float(action_clip)
        self.max_steps = max_steps or (self.T - 1)

        # Find first valid index where all rolling features exist
        idxs = []
        for arr in [self.ma_fast, self.ma_slow, self.sigma_annual, self.tau_upper, self.tau_lower]:
            idxs.append(int(np.argmax(~np.isnan(arr))))
        self.t0 = max(idxs + [1])  # need t-1 to exist

        self.reset()

    def reset(self):
        self.t = self.t0
        self.p_prev = 0.0
        self.steps = 0
        self.done = False
        return self._observe()

    def _observe(self):
        i = self.t
        row = self.df.iloc[i]
        return np.array([
            float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume),
            float(self.p_prev), float(self.regime[i]), float(self.sigma_annual[i])
        ], dtype=np.float32)

    def _l1_normalize(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a, -self.action_clip, self.action_clip)
        s = np.sum(np.abs(a))
        return a if s < 1e-8 else a / s

    def _position_from_alpha(self, alpha_comp: float, c_t: float) -> float:
        up = self.tau_upper[self.t]
        lo = self.tau_lower[self.t]
        # Gate by price quantiles (paper Eq. 7—thresholds):contentReference[oaicite:1]{index=1}
        if np.isnan(up) or np.isnan(lo):
            gate = 0.0
        elif alpha_comp > 0 and c_t > up:
            gate = min(1.0, 2.0 * (alpha_comp - up))
        elif alpha_comp < 0 and c_t < lo:
            gate = max(-1.0, 2.0 * (alpha_comp - lo))
        else:
            gate = 0.0

        # Regime filter: if bearish (MA20<=MA100) and gate>0 -> 0:contentReference[oaicite:2]{index=2}
        if (self.regime[self.t] == 0.0) and (gate > 0.0):
            gate = 0.0

        # Volatility scaling v_t=min(2, sigma_target/sigma_annual):contentReference[oaicite:3]{index=3}
        sig = self.sigma_annual[self.t]
        v = 1.0 if (sig is None or sig <= 1e-8) else min(2.0, self.sigma_target / sig)
        p_t = float(np.clip(gate * v, -1.0, 1.0))
        return p_t

    def step(self, action: np.ndarray):
        if self.done:
            return self._observe(), 0.0, True, {}

        w = self._l1_normalize(action)
        alpha_comp = float(np.dot(w, self.alphas[self.t]))
        c_t = float(self.df["close"].values[self.t])
        p_t = self._position_from_alpha(alpha_comp, c_t)

        r = float(p_t * self.r_fwd[self.t] - self.tc_lambda * abs(p_t - self.p_prev))
        self.p_prev = p_t

        self.t += 1
        self.steps += 1
        if self.t >= self.T - 1 or self.steps >= self.max_steps:
            self.done = True

        return self._observe(), r, self.done, {}

# ============================================================
# PPO (actor-critic with GAE, clipping, value clipping, entropy)
# ============================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
        # per-action log std
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def policy(self, obs: torch.Tensor):
        mu = self.pi(obs)
        std = torch.exp(self.log_std).clamp_min(1e-6)
        return mu, std

    def value(self, obs: torch.Tensor):
        return self.v(obs).squeeze(-1)

class PPO:
    def __init__(
        self,
        env: AlphaWeightingEnvLite,
        device: str = "cpu",
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rollout_steps=2048,
        update_epochs=10,
        minibatch_size=64,
    ):
        self.env = env
        self.device = device
        self.obs_dim = 8
        self.act_dim = env.N

        self.ac = ActorCritic(self.obs_dim, self.act_dim).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

    @torch.no_grad()
    def _sample_action(self, obs_t: torch.Tensor):
        mu, std = self.ac.policy(obs_t)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        # squash with tanh to (-1,1); keep proper logprob via change-of-variables
        a = torch.tanh(z)
        logp = dist.log_prob(z).sum(-1) - torch.sum(torch.log(1 - a.pow(2) + 1e-6), dim=-1)
        return a, logp

    def _compute_gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        adv = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else values[t + 1]
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_v * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    def collect_rollout(self):
        obs = self.env.reset()
        obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []

        for _ in range(self.rollout_steps):
            obs_t = to_tensor(obs, self.device).unsqueeze(0)
            with torch.no_grad():
                v = self.ac.value(obs_t).squeeze(0)
                a, logp = self._sample_action(obs_t)

            a_np = a.squeeze(0).cpu().numpy()
            next_obs, r, done, _info = self.env.step(a_np)

            obs_list.append(obs.copy())
            act_list.append(a_np.copy())
            logp_list.append(logp.item())
            rew_list.append(float(r))
            val_list.append(v.item())
            done_list.append(float(done))

            obs = next_obs
            if done:
                obs = self.env.reset()

        with torch.no_grad():
            last_value = self.ac.value(to_tensor(obs, self.device).unsqueeze(0)).item()

        batch = {
            "obs": to_tensor(np.array(obs_list), self.device),
            "act": to_tensor(np.array(act_list), self.device),
            "logp": to_tensor(np.array(logp_list), self.device),
            "rew": to_tensor(np.array(rew_list), self.device),
            "val": to_tensor(np.array(val_list), self.device),
            "done": to_tensor(np.array(done_list), self.device),
            "last_value": torch.tensor(last_value, dtype=torch.float32, device=self.device),
        }
        return batch

    def update(self, batch):
        with torch.no_grad():
            adv, ret = self._compute_gae(batch["rew"], batch["val"], batch["done"], batch["last_value"])
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = batch["obs"].shape[0]
        idx = np.arange(N)

        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                obs = batch["obs"][mb]
                act = batch["act"][mb]
                old_logp = batch["logp"][mb]
                A = adv[mb]
                R = ret[mb]

                mu, std = self.ac.policy(obs)
                dist = torch.distributions.Normal(mu, std)
                # invert tanh with atanh for correct logprob under current policy
                z = torch.atanh(torch.clamp(act, -0.999, 0.999))
                logp = dist.log_prob(z).sum(-1) - torch.sum(torch.log(1 - act.pow(2) + 1e-6), dim=-1)

                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * A
                pi_loss = -torch.min(surr1, surr2).mean()

                v = self.ac.value(obs)
                v_clipped = batch["val"][mb] + (v - batch["val"][mb]).clamp(-self.clip_eps, self.clip_eps)
                vf_loss = 0.5 * torch.max((v - R).pow(2), (v_clipped - R).pow(2)).mean()

                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()

    def train(self, total_steps=200_000, log_interval=10_000):
        steps = 0
        while steps < total_steps:
            batch = self.collect_rollout()
            self.update(batch)
            steps += self.rollout_steps
            if steps % log_interval == 0:
                print(f"[{steps}] avg rollout reward: {batch['rew'].mean().item():.6f}")

# ============================================================
# Example wiring (replace with your real data)
# ============================================================
if __name__ == "__main__":
    # Synthesize a plausible daily series (replace with real)
    T = 4000
    N = 50
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2016-01-01", periods=T)

    close = 100 + np.cumsum(rng.normal(0, 1, size=T)).astype(np.float32)
    open_ = close + rng.normal(0, 0.5, size=T).astype(np.float32)
    high  = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, size=T)).astype(np.float32)
    low   = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, size=T)).astype(np.float32)
    vol   = (1e6 + 1e5 * rng.normal(0, 1, size=T)).astype(np.float32)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates
    )

    # forward (close->close) return
    r_fwd = pd.Series(np.diff(np.append(close, close[-1])) / np.clip(close, 1e-6, None), index=dates)

    # standardized alphas (columns ~ N(0,1)); replace with your 50 cleaned alphas matrix
    alphas = rng.normal(0, 1, size=(T, N)).astype(np.float32)

    env = AlphaWeightingEnvLite(
        ohlcv_df=df,
        future_ret=r_fwd,
        alphas=alphas,
        sigma_window=63,
        ma_fast=20,
        ma_slow=100,
        q_window=126,
        sigma_target=0.15,
        tc_lambda=0.001,
        action_clip=1.0,
    )

    agent = PPO(
        env,
        device="cpu",
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        ent_coef=0.00,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rollout_steps=2048,
        update_epochs=10,
        minibatch_size=64,
    )

    agent.train(total_steps=50_000, log_interval=10_000)
