"""Density-based baselines for imitation learning. Each of these algorithms
learns a density estimate on some aspect of the demonstrations, then rewards
the agent for following that estimate."""

from typing import Sequence

from gym.spaces.utils import flatten
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv
import tensorflow as tf

from imitation.util import reward_wrapper, rollout

# Constants identifying different kinds of density we can use. Note that all
# can be augmented to depend on the time step by passing `is_stationary = True`
# to `DensityReward`.

# Density on state s
STATE_DENSITY = 'state_density'
# Density on (s,a) pairs
STATE_ACTION_DENSITY = 'state_action_density'
# Density (s,s') pairs
STATE_STATE_DENSITY = 'state_state_density'


class DensityReward:
  def __init__(self,
               *,
               trajectories,
               density_type,
               kernel,
               kernel_bandwidth,
               obs_space,
               act_space,
               is_stationary=True,
               standardise_inputs=True):
    """Reward function based on a density estimate of trajectories.

    Args:
      trajectories ([dict]): expert demonstration trajectories generated by,
        e.g., `imitation.util.rollout.generate_trajectories`.
      obs_space (gym.Space): observation space for underlying environment.
      act_space (gym.Space): action space for underlying environment.
      density_type (str): type of density to train on. Valid choices are
        defined as constants in this module, including `STATE_DENSITY` (density
        on single states), `STATE_ACTION_DENSITY` (density on state-action
        pairs), and `STATE_STATE_DENSITY` (density on state-state transitions).
      is_stationary (bool): should this trainer use the same density model for
        all time steps (True), or a different density model for each time step
        (False)? `is_stationary=False` might be useful if using
        `density_type=STATE_DENSITY`, since it encourages the agent to imitate
        entire trajectories instead of simply occupying a few states that have
        high frequency in the demonstration dataset.
      standardise_inputs (bool): if True, then the inputs to the reward model
        will be standardised to have zero mean and unit variance over the
        demonstration trajectories. Otherwise, inputs will be passed to the
        reward model with their ordinary scale.
      kernel (str): kernel to use for density estimation. Current valid options
        include `gaussian` and `exponential`.
      kernel_bandwidth (float): bandwidth of kernel. If `standardise_inputs` is
        true and you are using a Gaussian kernel, then it probably makes sense
        to set this somewhere between 0.1 and 1."""
    self.density_type = density_type
    self.is_stationary = is_stationary
    self.kernel = kernel
    self.kernel_bandwidth = kernel_bandwidth
    self.standardise = standardise_inputs
    self.obs_space = obs_space
    self.act_space = act_space
    self._fit_models(trajectories)

  def _fit_models(self, trajectories):
    flat_trajs = self._preprocess_trajectories(trajectories)

    # if requested, we'll scale demonstration transitions so that they have
    # zero mean and unit variance (i.e all components are equally important)
    flattened_dataset = np.stack(sum(flat_trajs, []), axis=0)
    self._scaler = StandardScaler(with_mean=self.standardise,
                                  with_std=self.standardise)
    self._scaler.fit(flattened_dataset)

    # now fit density model
    # TODO: add absorbing state fix that I describe in __call__
    if self.is_stationary:
      # fit to all pairs, since density model is stationary
      self._density_model = self._fit_single_density(
          self._scaler.transform(flattened_dataset))
    else:
      # fit separately for samples at each time step
      T = max(map(len, flat_trajs))
      traj_groups = [[]] * T
      for traj in flat_trajs:
        for t, flat_trans in enumerate(traj):
          traj_groups[t].append(flat_trans)
      traj_groups_scaled = [
          self._scaler.transform(np.stack(step_transitions, axis=0))
          for step_transitions in traj_groups
      ]
      self._density_models = [
          self._fit_single_density(scaled_flat_trans)
          for scaled_flat_trans in traj_groups_scaled
      ]

  def _fit_single_density(self, flat_transitions):
    # This bandwidth was chosen to make sense with standardised inputs that
    # have unit variance in each component. There might be a better way to
    # choose it automatically.
    density_model = KernelDensity(kernel=self.kernel,
                                  bandwidth=self.kernel_bandwidth)
    density_model.fit(flat_transitions)
    return density_model

  def _preprocess_trajectories(self,
                               trajectories: Sequence[rollout.Trajectory]):
    """Preprocess a list of trajectories into atomic units that we can learn a
    density function on. Depending on configuration, that could mean a sequence
    state/state pairs, or state/action pairs, or single states, etc.

    Args:
      trajectories: trajectories to process.

    Returns:
      flat_trajectories (np.ndarray): a corresponding list of "flattened"
        trajectories. Each element of a flattened trajectory is a
        one-dimensional ndarray representing a (state,state) pair at a
        particular time step, or (state,action) pair at that time step, or
        whatever the class configuration calls for.
    """
    flat_trajectories = []
    for traj in trajectories:
      obs_vec = traj.obs
      act_vec = traj.acts
      assert len(obs_vec) == len(act_vec) + 1
      flat_traj = []
      for step_num in range(len(traj.acts)):
        flat_trans = self._preprocess_transition(obs_vec[step_num],
                                                 act_vec[step_num],
                                                 obs_vec[step_num + 1])
        flat_traj.append(flat_trans)
      flat_trajectories.append(flat_traj)
    return flat_trajectories

  def _preprocess_transition(self, obs, act, next_obs):
    if self.density_type == STATE_DENSITY:
      return flatten(self.obs_space, obs)
    elif self.density_type == STATE_ACTION_DENSITY:
      return np.concatenate([
          flatten(self.obs_space, obs),
          flatten(self.act_space, act),
      ])
    elif self.density_type == STATE_STATE_DENSITY:
      return np.concatenate([
          flatten(self.obs_space, obs),
          flatten(self.obs_space, next_obs),
      ])
    else:
      raise ValueError(f"Unknown density type {self.density_type}")

  def __call__(self, obs_b, act_b, next_obs_b, steps=None):
    r"""Compute reward from given (s,a,s') transition batch. This handles
    *batches* of observations, since it's designed to work with VecEnvs.

    Args:
      obs_b (np.ndarray): current batch of observations.
      act_b (np.ndarray): batch of actions that agent took in response to those
        observations.
      next_obs_b (np.ndarray): batch of observations encountered after the
        agent took those actions.
      steps (np.ndarray): number of time steps elapsed in each environment.

    Returns:
      rew_array (np.ndarray): array of scalar rewards of the form `r_t(s,a,s')
        = \log \hat p_t(s,a,s')` (one for each environment), where `\log \hat
        p` is the underlying density model (and may be independent of s', a, or
        t, depending on options passed to constructor)."""
    rew_list = []
    assert len(obs_b) == len(act_b) and len(obs_b) == len(next_obs_b)
    for idx, (obs, act, next_obs) in enumerate(zip(obs_b, act_b, next_obs_b)):
      flat_trans = self._preprocess_transition(obs, act, next_obs)
      scaled_padded_trans = self._scaler.transform(flat_trans[np.newaxis])
      if self.is_stationary:
        rew = self._density_model.score(scaled_padded_trans)
      else:
        time = steps[idx]
        if time >= len(self._density_models):
          # Can't do anything sensible here yet. Correct solution is to use
          # hierarchical model in which we first check whether state is
          # absorbing, then assign either constant score or a score based on
          # density.
          raise Exception(
              f"Time {time} out of range (0, {len(self._density_models)}], and "
              f"I haven't implemented absorbing states etc. yet")
        else:
          time_model = self._density_models[time]
          rew = time_model.score(scaled_padded_trans)
      rew_list.append(rew)
    rew_array = np.asarray(rew_list, dtype='float32')
    return rew_array


class DensityTrainer:
  def __init__(self,
               venv: VecEnv,
               rollouts: Sequence[rollout.Trajectory],
               imitation_trainer: BaseRLModel,
               *,
               standardise_inputs: bool = True,
               kernel: str = 'gaussian',
               kernel_bandwidth: float = 0.5,
               density_type: str = STATE_ACTION_DENSITY,
               is_stationary: bool = False):
    r"""Family of simple imitation learning baseline algorithms that apply RL to
    maximise a rough density estimate of the demonstration trajectories.
    Specifically, it constructs a non-parametric estimate of `p(s)`, `p(s,s')`,
    `p_t(s,a)`, etc. (depending on options), then rewards the imitation learner
    with `r_t(s,a,s')=\log p_t(s,a,s')` (or `\log p(s,s')`, or whatever the
    user wants the model to condition on).

    Args:
      venv: environment to train on.
      rollouts: list of expert trajectories to imitate.
       imitation_trainer: RL algorithm & initial policy that will
        be used to train the imitation learner.
      kernel, kernel_bandwidth, density_type, is_stationary,
        n_expert_trajectories: these are passed directly to `DensityReward`;
        refer to documentation for that class."""
    self.venv = venv
    self.imitation_trainer = imitation_trainer
    self.reward_fn = DensityReward(trajectories=rollouts,
                                   density_type=density_type,
                                   obs_space=self.venv.observation_space,
                                   act_space=self.venv.action_space,
                                   is_stationary=is_stationary,
                                   kernel=kernel,
                                   kernel_bandwidth=kernel_bandwidth,
                                   standardise_inputs=standardise_inputs)
    self.wrapped_env = reward_wrapper.RewardVecEnvWrapper(self.venv,
                                                          self.reward_fn)
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      self.sess.run(tf.global_variables_initializer())

  def train_policy(self, n_timesteps=int(1e6), **kwargs):
    """Train the imitation policy for a given number of timesteps. Does not
    return anything.

    Args:
      n_timesteps (int): number of timesteps to train the policy for.
      kwargs (dict): extra arguments that will be passed to the `learn()`
        method of the imitation RL model. Refer to Stable Baselines docs for
        details."""
    self.imitation_trainer.set_env(self.wrapped_env)
    # FIXME: learn() is not meant to be called frequently; there are
    # significant per-call overheads (see Adam's comment in adversarial.py)
    # FIXME: the ep_reward_mean reported by SB is wrong; it comes from a
    # Monitor() that is being (incorrectly) used to wrap the underlying
    # environment.
    self.imitation_trainer.learn(n_timesteps,
                                 # ensure we can see total steps for all
                                 # learn() calls, not just for this call
                                 reset_num_timesteps=False,
                                 **kwargs)

  def test_policy(self, *, n_trajectories=10, true_reward=True):
    """Test current imitation policy on environment & give some rollout
    stats.

    Args:
      n_trajectories (int): number of rolled-out trajectories.
      true_reward (bool): should this use ground truth reward from underlying
        environment (True), or imitation reward (False)?

    Returns:
      dict: rollout statistics collected by
        `imitation.utils.rollout.rollout_stats()`.
    """
    self.imitation_trainer.set_env(self.venv)
    trajs = rollout.generate_trajectories(
        self.imitation_trainer,
        self.venv if true_reward else self.wrapped_env,
        sample_until=rollout.min_episodes(n_trajectories),
    )
    reward_stats = rollout.rollout_stats(trajs)
    return reward_stats
