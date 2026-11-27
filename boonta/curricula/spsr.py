"""Scalable, `cell_fn`-free Prioritized State Replay (SPSR) in feature space.

Eliminates per-environment cell hand-crafting, discrete-action tables, and combinatorial
fan-out dilution while keeping PSR's exact Footprint x Relevance x Gain decomposition:

1. Transition-level Forecast Gain: g_t = u_tilde(x_t, a_t) * A_t (works on discrete & continuous
   action spaces; preserves 100% of within-cell state-action covariance).
2. Centered Feature-Space Relevance: rel(x) = phi_hat(x)^T Sigma_rho phi_hat(x), invariant to
   combinatorial state branching while filtering out dead ends abandoned by rho.
3. Hybrid Window Footprint: empirical window return blended with a fast ridge readout w_psr^T [phi_hat, 1]
   that scores unseen procedural snapshots immediately.
4. Multiplicity-Normalized, rho-Gated Water-Filling: normalizes per-slot capacity by feature
   multiplicity phi_hat_i^T Sigma_H phi_hat_i and returns unused budget to fresh rho worlds (-1)
   whenever marginal archive value drops below v(rho).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
from flax import struct

from boonta.algorithms import Algorithm
from boonta.algorithms.advantage_estimators import generalized_advantage_estimator
from boonta.curricula.psr import archive_auto_reset_state, piled
from boonta.environments.wrappers.archive_auto_reset import (
    ArchiveAutoReset,
    ArchiveAutoResetState,
    pick,
    unwrap,
)
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, PyTree, Transition
from boonta.utils.typing import Environment


def _make_projection(key: Key, in_dim: int, proj_dim: int) -> Array:
    if in_dim <= proj_dim:
        eye = jnp.eye(in_dim, proj_dim, dtype=jnp.float32)
        return eye
    z = jax.random.normal(key, (in_dim, proj_dim), dtype=jnp.float32)
    q, _ = jnp.linalg.qr(z)
    return q.astype(jnp.float32)


def _normalize_features(raw_feat: Array, mean_vec: Array, proj: Array) -> Array:
    proj_feat = jnp.matmul(raw_feat.astype(jnp.float32), proj)
    centred = proj_feat - mean_vec
    norm = jnp.linalg.norm(centred, axis=-1, keepdims=True)
    return centred / jnp.maximum(norm, 1e-6)


@struct.dataclass
class SPSRState:
    snapshots: PyTree          # Ring of `horizon` past parameter trees for forecast gain
    cursor: Array              # int32 ring index
    filled: Array              # int32 count of updates seen
    proj: Array                # (raw_dim, d) fixed orthogonal feature projection
    feat_mean: Array           # (d,) EMA mean of projected rho features
    sigma_rho: Array           # (d, d) EMA second moment of centered normalized rho features
    w_psr: Array               # (d + 1,) linear window-worth predictor in feature space
    v_rho: Array               # () EMA per-step PSR worth of from-start (rho) play
    noise_scale: Array         # () EMA noise-to-signal scale for water-filling saturation
    phi_archive: Array         # (archive_size, d) cached normalized features of archive slots
    feat_clock: Array          # (archive_size,) update index when slot's feature was refreshed
    seen_time: Array           # (archive_size,) archive timestamp when slot was last observed
    emp_sums: Array            # (archive_size,) empirical window worth sum per slot
    emp_counts: Array          # (archive_size,) empirical placement count per slot
    last_slots: Array          # (num_placed,) slots assigned at start of last rollout
    step_count: Array          # () int32 update counter


@dataclass
class SPSR(Pit):
    environment: ArchiveAutoReset
    algorithm: Algorithm
    k: float = 4.0
    restart_at: str = "rollout"
    proj_dim: int = 128
    horizon: int = 10
    kappa: float = 1.0
    clip: float = 2.0
    rate: float = 0.15
    rho_rate: float = 0.05
    refresh_chunk: int = 1024
    gate_ratio: float = 1.0
    rho_gate: bool = True
    max_per_slot: int = 8
    prior_weight: float = 2.0
    ridge: float = 10.0
    iterations: int = 50
    seed: int = 0

    def __post_init__(self):
        assert isinstance(self.environment, ArchiveAutoReset), "SPSR needs ArchiveAutoReset or SnapshotAutoReset"
        assert self.k >= 1.0
        assert self.restart_at in ("rollout", "episode")

    @property
    def num_placed(self) -> int:
        return int(round((1.0 - 1.0 / self.k) * self.environment.num_envs))

    def _extract_obs_features_and_dist(self, params: PyTree, obs: PyTree):
        """Extracts penultimate features & action distribution for a single-step batch of `obs`."""
        net = self.algorithm.network
        if net.torso is None:
            (dist, _), inter = net.apply(params, obs, temperature=1.0, mutable=["intermediates"])
            feat = inter["intermediates"]["features"]
            feat = feat[0] if isinstance(feat, tuple) else feat
            return feat, dist
        else:
            # For recurrent networks on standalone snapshots, use zero carry + dummy sequence axis
            batch = jax.tree.leaves(obs)[0].shape[0]
            carry = net.initialize_carry(jax.random.key(0), (batch, 1))
            seq_obs = jax.tree.map(lambda leaf: leaf[:, None], obs)
            act_space = self.environment.action_space()
            dummy_act = jnp.zeros((batch, 1, *act_space.shape), act_space.dtype)
            dummy_rew = jnp.zeros((batch, 1), jnp.float32)
            dummy_done = jnp.ones((batch, 1), bool)
            (_, (dist, _)), inter = net.apply(
                params, seq_obs, dummy_act, dummy_rew, dummy_done,
                carry=carry, temperature=1.0, mutable=["intermediates"]
            )
            feat = inter["intermediates"]["features"]
            feat = feat[0] if isinstance(feat, tuple) else feat
            return feat[:, 0], dist

    def init(self, key: Key, state: PyTree) -> SPSRState:
        proj_key, _ = jax.random.split(jax.random.key(self.seed))
        sample_obs = jax.tree.map(lambda leaf: leaf[:2], state.timestep.obs)
        sample_feat, _ = self._extract_obs_features_and_dist(state.algorithm_state.params, sample_obs)
        raw_dim = int(sample_feat.shape[-1])
        d = min(raw_dim, self.proj_dim)
        proj = _make_projection(proj_key, raw_dim, d)

        arch_size = self.environment.archive.archive_size
        params = state.algorithm_state.params
        return SPSRState(
            snapshots=jax.tree.map(lambda leaf: jnp.stack([leaf] * self.horizon), params),
            cursor=jnp.zeros((), jnp.int32),
            filled=jnp.zeros((), jnp.int32),
            proj=proj,
            feat_mean=jnp.zeros((d,), jnp.float32),
            sigma_rho=jnp.eye(d, dtype=jnp.float32) / float(d),
            w_psr=jnp.zeros((d + 1,), jnp.float32),
            v_rho=jnp.zeros((), jnp.float32),
            noise_scale=jnp.array(32.0, dtype=jnp.float32),
            phi_archive=jnp.zeros((arch_size, d), jnp.float32),
            feat_clock=jnp.full((arch_size,), -1000000, jnp.int32),
            seen_time=jnp.full((arch_size,), -1, jnp.int32),
            emp_sums=jnp.zeros((arch_size,), jnp.float32),
            emp_counts=jnp.zeros((arch_size,), jnp.float32),
            last_slots=jnp.full((self.num_placed,), -1, jnp.int32),
            step_count=jnp.zeros((), jnp.int32),
        )

    def update(
        self,
        spsr: SPSRState,
        key: Key,
        transitions: Transition,
        algorithm_state: PyTree,
    ) -> SPSRState:
        first, second = transitions.first, transitions.second
        values = transitions.aux["value"]
        gamma = self.algorithm.cfg.gamma
        gae_lambda = self.algorithm.cfg.gae_lambda

        advantages, _ = generalized_advantage_estimator(
            jax.tree.map(lambda leaf: leaf[:-1], transitions),
            values[:-1],
            values[-1],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Rollout policy params = latest snapshot before this PPO update; old_params = horizon ago
        latest_idx = jnp.where(spsr.filled > 0, (spsr.cursor - 1) % self.horizon, 0)
        rollout_params = jax.tree.map(lambda ring: ring[latest_idx], spsr.snapshots)
        old_params = jax.tree.map(lambda ring: ring[spsr.cursor], spsr.snapshots)

        obs_seq = jax.tree.map(lambda leaf: leaf[:-1], first.obs)
        act_seq = second.action[:-1]
        num_t = advantages.shape[0]
        step_keys = jax.random.split(key, num_t)

        # Map step-by-step across time T so peak activation VRAM stays O(num_envs), never O(T * num_envs)
        def eval_step(step_inputs):
            obs_t, act_t, logp_aux_t, logits_aux_t, sk = step_inputs
            raw_feat, now_dist = self._extract_obs_features_and_dist(rollout_params, obs_t)
            _, old_dist = self._extract_obs_features_and_dist(old_params, obs_t)
            proj_feat = jnp.matmul(raw_feat.astype(jnp.float32), spsr.proj)

            if logits_aux_t is not None:
                now_logp = jax.nn.log_softmax(logits_aux_t.astype(jnp.float32), axis=-1)
                old_logp = jax.nn.log_softmax(old_dist.logits.astype(jnp.float32), axis=-1)
                u_all = self.kappa * jnp.clip(now_logp - old_logp, -self.clip, self.clip)
                u_mean = jnp.sum(jnp.exp(now_logp) * u_all, axis=-1, keepdims=True)
                u_tilde = jnp.take_along_axis(u_all - u_mean, act_t.astype(jnp.int32)[..., None], axis=-1)[..., 0]
            else:
                act_phant = now_dist.sample(seed=sk)
                # Remove potential extra time dim if recurrent
                if act_phant.ndim > act_t.ndim:
                    act_phant = act_phant[:, 0]
                lp_now_taken = now_dist.log_prob(act_t)
                lp_old_taken = old_dist.log_prob(act_t)
                lp_now_phant = now_dist.log_prob(act_phant)
                lp_old_phant = old_dist.log_prob(act_phant)
                if lp_now_taken.ndim > 1:
                    lp_now_taken = lp_now_taken.squeeze(-1)
                    lp_old_taken = lp_old_taken.squeeze(-1)
                    lp_now_phant = lp_now_phant.squeeze(-1)
                    lp_old_phant = lp_old_phant.squeeze(-1)
                u_taken = self.kappa * jnp.clip(lp_now_taken - lp_old_taken, -self.clip, self.clip)
                u_phant = self.kappa * jnp.clip(lp_now_phant - lp_old_phant, -self.clip, self.clip)
                u_tilde = u_taken - u_phant

            return proj_feat, u_tilde.astype(jnp.float32)

        logits_seq = transitions.aux.get("logits")
        logits_in = logits_seq[:-1] if logits_seq is not None else None
        logp_in = transitions.aux["log_prob"][:-1]
        proj_feats, u_tilde = jax.lax.map(
            eval_step, (obs_seq, act_seq, logp_in, logits_in, step_keys)
        )  # proj_feats: (T-1, E, d), u_tilde: (T-1, E)

        truncated = second.truncated[:-1].astype(bool)
        valid_mask = ~truncated
        rho_mask = first.info["rho"][:-1].astype(bool) & valid_mask
        rho_count = jnp.maximum(jnp.sum(rho_mask, dtype=jnp.float32), 1.0)

        # 1. Update centered feature mean & Sigma_rho on rho transitions
        batch_mean = jnp.sum(jnp.where(rho_mask[..., None], proj_feats, 0.0), axis=(0, 1)) / rho_count
        feat_mean = jnp.where(
            spsr.filled == 0, batch_mean, (1.0 - self.rho_rate) * spsr.feat_mean + self.rho_rate * batch_mean
        )
        centred = proj_feats - feat_mean
        phi_hat = centred / jnp.maximum(jnp.linalg.norm(centred, axis=-1, keepdims=True), 1e-6)  # (T-1, E, d)

        phi_rho = jnp.where(rho_mask[..., None], phi_hat, 0.0).reshape((-1, phi_hat.shape[-1]))
        batch_sigma_rho = (phi_rho.T @ phi_rho) / rho_count
        sigma_rho = (1.0 - self.rho_rate) * spsr.sigma_rho + self.rho_rate * batch_sigma_rho

        # 2. Compute feature-space Relevance & Transition Forecast Gain
        rel = jnp.einsum("ted,df,tef->te", phi_hat, sigma_rho, phi_hat)  # (T-1, E) >= 0
        step_gain = u_tilde * advantages.astype(jnp.float32)
        step_worth = jnp.where(valid_mask, rel * step_gain, 0.0)

        # 3. Update v_rho (expected per-step worth on rho play)
        batch_v_rho = jnp.sum(jnp.where(rho_mask, step_worth, 0.0)) / rho_count
        v_rho = (1.0 - self.rate) * spsr.v_rho + self.rate * jnp.maximum(batch_v_rho, 0.0)

        # 4. Compute remaining-window average worth & fit ridge feature predictor w_psr
        # Reverse cumulative sum along rollout window gives exact suffix worth from each step
        rev_worth = jnp.cumsum(step_worth[::-1], axis=0)[::-1]
        rev_valid = jnp.cumsum(valid_mask.astype(jnp.float32)[::-1], axis=0)[::-1]
        # Use first half of window where at least half-window footprint is observed
        half_t = max(1, num_t // 2)
        target_worth = rev_worth[:half_t] / jnp.maximum(rev_valid[:half_t], 1.0)  # (half_t, E)
        fit_mask = valid_mask[:half_t] & (rev_valid[:half_t] >= max(1.0, half_t * 0.5))

        phi_fit = phi_hat[:half_t].reshape((-1, phi_hat.shape[-1]))
        ones = jnp.ones((phi_fit.shape[0], 1), jnp.float32)
        X = jnp.where(fit_mask.reshape((-1, 1)), jnp.concatenate([phi_fit, ones], axis=-1), 0.0)
        y = jnp.where(fit_mask.reshape((-1,)), target_worth.reshape((-1,)), 0.0)
        d_aug = X.shape[-1]
        gram = X.T @ X + self.ridge * jnp.eye(d_aug, dtype=jnp.float32)
        w_batch = jnp.linalg.solve(gram, X.T @ y)
        w_psr = (1.0 - self.rate) * spsr.w_psr + self.rate * w_batch

        # 5. Update per-slot empirical window worth for the `num_placed` placed environments
        num_envs = advantages.shape[1]
        placed_start = num_envs - self.num_placed
        placed_worth = step_worth[:, placed_start:]
        placed_valid = valid_mask[:, placed_start:].astype(jnp.float32)
        win_counts = jnp.sum(placed_valid, axis=0)
        win_means = jnp.sum(placed_worth, axis=0) / jnp.maximum(win_counts, 1.0)
        valid_placement = (spsr.last_slots >= 0) & (win_counts > 0.0)

        arch_size = spsr.emp_sums.shape[0]
        slot_rows = jnp.where(valid_placement, spsr.last_slots, arch_size)
        added_counts = piled(arch_size, slot_rows, valid_placement.astype(jnp.float32))
        added_sums = piled(arch_size, slot_rows, jnp.where(valid_placement, win_means, 0.0))
        decay = 1.0 - self.rate
        emp_counts = decay * spsr.emp_counts + added_counts
        emp_sums = decay * spsr.emp_sums + added_sums

        # 6. Estimate noise_scale for water-filling
        pred_all = jnp.maximum(0.0, jnp.matmul(phi_hat, w_psr[:-1]) + w_psr[-1])
        sample_power = jnp.sum(jnp.where(valid_mask, step_worth ** 2, 0.0))
        signal_power = jnp.sum(jnp.where(valid_mask, pred_all ** 2, 0.0))
        measured_scale = jnp.where(
            signal_power > 1e-20,
            jnp.clip((sample_power - signal_power) / jnp.maximum(signal_power, 1e-20), 2.0, 500.0),
            spsr.noise_scale,
        )
        noise_scale = (1.0 - self.rate) * spsr.noise_scale + self.rate * measured_scale

        # 7. Push current post-update params into forecast ring
        new_params = algorithm_state.params
        snapshots = jax.tree.map(
            lambda ring, leaf: jnp.where(
                spsr.filled == 0,
                jnp.broadcast_to(leaf, ring.shape),
                ring.at[spsr.cursor].set(leaf),
            ),
            spsr.snapshots,
            new_params,
        )

        lox.log({
            "spsr/v_rho": v_rho,
            "spsr/step_gain": jnp.sum(jnp.where(valid_mask, step_gain, 0.0)) / jnp.maximum(jnp.sum(valid_mask), 1.0),
            "spsr/rel_rho": jnp.sum(jnp.where(rho_mask, rel, 0.0)) / rho_count,
            "spsr/noise_scale": noise_scale,
        })

        return spsr.replace(
            snapshots=snapshots,
            cursor=(spsr.cursor + 1) % self.horizon,
            filled=spsr.filled + 1,
            feat_mean=feat_mean,
            sigma_rho=sigma_rho,
            w_psr=w_psr,
            v_rho=v_rho,
            noise_scale=noise_scale,
            emp_sums=emp_sums,
            emp_counts=emp_counts,
            step_count=spsr.step_count + 1,
        )

    def _refresh_archive_features(
        self, spsr: SPSRState, archive_state, params: PyTree, key: Key
    ) -> SPSRState:
        arch_size = spsr.emp_sums.shape[0]
        cur_time = archive_state.cell_states.time
        changed = archive_state.mask & (cur_time != spsr.seen_time)

        # Clear stale empirical stats on overwritten archive slots
        emp_sums = jnp.where(changed, 0.0, spsr.emp_sums)
        emp_counts = jnp.where(changed, 0.0, spsr.emp_counts)
        feat_clock = jnp.where(changed, -1000000, spsr.feat_clock)

        # Refresh `refresh_chunk` slots (prioritizing newly written/overwritten slots first)
        age_feat = jnp.where(archive_state.mask, (spsr.step_count - feat_clock).astype(jnp.float32), -1.0)
        age_feat = age_feat + jax.random.uniform(key, age_feat.shape) * 0.1
        chunk = min(self.refresh_chunk, arch_size)
        _, refresh_slots = jax.lax.top_k(age_feat, chunk)
        sub_snaps = self.environment.archive.take(archive_state, refresh_slots)
        sub_obs = super(ArchiveAutoReset, self.environment).observe(sub_snaps)
        raw_feat, _ = self._extract_obs_features_and_dist(params, sub_obs)
        phi_sub = _normalize_features(raw_feat, spsr.feat_mean, spsr.proj)

        phi_archive = spsr.phi_archive.at[refresh_slots].set(phi_sub)
        feat_clock = feat_clock.at[refresh_slots].set(spsr.step_count)
        seen_time = jnp.where(changed, cur_time, spsr.seen_time)
        return spsr.replace(
            phi_archive=phi_archive,
            feat_clock=feat_clock,
            seen_time=seen_time,
            emp_sums=emp_sums,
            emp_counts=emp_counts,
        )

    def _select_slots(
        self, spsr: SPSRState, archive_state, key: Key
    ) -> Array:
        order_key, fallback_key = jax.random.split(key)
        eligible = self.environment.archive.eligible(archive_state) & (spsr.feat_clock > -999999)
        count = self.num_placed
        arch_size = eligible.shape[0]

        # Hybrid shrinkage: empirical slot worth + feature-linear doorway predictor
        v_pred = jnp.matmul(spsr.phi_archive, spsr.w_psr[:-1]) + spsr.w_psr[-1]
        v_emp = spsr.emp_sums / jnp.maximum(spsr.emp_counts, 1e-12)
        trust = spsr.emp_counts / (spsr.emp_counts + self.prior_weight)
        v_slot = jnp.where(
            eligible,
            jnp.maximum(0.0, trust * v_emp + (1.0 - trust) * v_pred),
            0.0,
        )

        # Feature multiplicity m_i = phi_i^T (sum_{k in elig} phi_k phi_k^T) phi_i in O(|H| d^2)
        phi_elig = jnp.where(eligible[:, None], spsr.phi_archive, 0.0)
        sigma_h = phi_elig.T @ phi_elig  # (d, d)
        multiplicity = jnp.maximum(1.0, jnp.sum((spsr.phi_archive @ sigma_h) * spsr.phi_archive, axis=-1))

        cap_i = spsr.noise_scale / multiplicity
        highest = jnp.max(v_slot)

        def restarts_at(level):
            raw = jnp.where(
                v_slot > level,
                cap_i * (jnp.sqrt(v_slot / jnp.maximum(level, 1e-30)) - 1.0),
                0.0,
            )
            return jnp.minimum(raw, float(self.max_per_slot))

        def bisect(_, bounds):
            low, high = bounds
            mid = jnp.sqrt(low * high)
            too_many = jnp.sum(restarts_at(mid)) > count
            return jnp.where(too_many, mid, low), jnp.where(too_many, high, mid)

        low, _ = jax.lax.fori_loop(
            0, self.iterations, bisect, (highest * 1e-12, jnp.maximum(highest, 1e-30))
        )

        floor = jnp.where(self.rho_gate, self.gate_ratio * jnp.maximum(spsr.v_rho, 0.0), 0.0)
        effective_level = jnp.maximum(low, floor)
        restarts = restarts_at(effective_level)
        total_restarts = jnp.sum(restarts)
        admitted_count = jnp.minimum(count, jnp.round(total_restarts).astype(jnp.int32))

        boundaries = jnp.cumsum(restarts) * jnp.maximum(admitted_count, 1) / jnp.maximum(total_restarts, 1e-30)
        idx_grid = jnp.arange(count)
        filled_slots = jnp.minimum(jnp.searchsorted(boundaries, idx_grid + 0.5), arch_size - 1)
        filled_slots = jnp.where(idx_grid < admitted_count, filled_slots, -1)
        filled_slots = jax.random.permutation(order_key, filled_slots)

        # Early warmup (filled < 2): uniform over eligible slots so initial feature predictor warms up
        warmup_slots = pick(fallback_key, jnp.where(eligible, 0.0, -jnp.inf), (count,))
        chosen = jnp.where(
            spsr.filled < 2,
            jnp.where(jnp.any(eligible), warmup_slots, -1),
            jnp.where(highest > floor, filled_slots, jnp.full((count,), -1, jnp.int32)),
        )

        lox.log({
            "spsr/admitted_share": jnp.mean((chosen >= 0).astype(jnp.float32)),
            "spsr/v_max": highest,
            "spsr/water_level": effective_level,
            "spsr/mean_multiplicity": jnp.sum(jnp.where(eligible, multiplicity, 0.0)) / jnp.maximum(jnp.sum(eligible), 1.0),
        })
        return chosen

    def __call__(self, key: Key, state):
        ref_key, sel_key, assign_key = jax.random.split(key, 3)
        aar_state = archive_auto_reset_state(state.environment_state)
        spsr = self._refresh_archive_features(
            state.pit_state, aar_state.archive_state, state.algorithm_state.params, ref_key
        )
        slots = self._select_slots(spsr, aar_state.archive_state, sel_key)
        spsr = spsr.replace(last_slots=slots)

        def assign(node):
            if isinstance(node, ArchiveAutoResetState):
                return self.environment.update(
                    node, assign_key, slots=slots, preempt=self.restart_at == "rollout"
                )
            return node

        env_state = jax.tree.map(
            assign, state.environment_state, is_leaf=lambda node: isinstance(node, ArchiveAutoResetState)
        )
        return state.replace(pit_state=spsr, environment_state=env_state)


def make(
    algorithm: Algorithm,
    environment: Environment,
    k: float = 4.0,
    restart_at: str = "rollout",
    proj_dim: int = 128,
    horizon: int = 10,
    kappa: float = 1.0,
    clip: float = 2.0,
    rate: float = 0.15,
    rho_rate: float = 0.05,
    refresh_chunk: int = 1024,
    gate_ratio: float = 1.0,
    rho_gate: bool = True,
    max_per_slot: int = 8,
    **kwargs,
) -> tuple[Algorithm, Environment, SPSR, Lap | None]:
    pit = SPSR(
        environment=unwrap(environment),
        algorithm=algorithm,
        k=k,
        restart_at=restart_at,
        proj_dim=proj_dim,
        horizon=horizon,
        kappa=kappa,
        clip=clip,
        rate=rate,
        rho_rate=rho_rate,
        refresh_chunk=refresh_chunk,
        gate_ratio=gate_ratio,
        rho_gate=rho_gate,
        max_per_slot=max_per_slot,
    )
    return algorithm, environment, pit, None
