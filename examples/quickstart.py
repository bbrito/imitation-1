"""Loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.
"""

import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3
from stable_baselines3.common import utils, evaluation

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util

import os
import matplotlib.pyplot as plt
from matplotlib import animation
from os import path

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

# Load pickled test demonstrations.
dir_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),os.path.pardir)),"tests/data/expert_models/cartpole_0/rollouts/final.pkl")
with open(dir_path, "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

# Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
# This is a more general dataclass containing unordered
# (observation, actions, next_observation) transitions.
transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env("CartPole-v1", n_envs=2)

tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)

print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

# Train BC on expert data.
# BC also accepts as `expert_data` any PyTorch-style DataLoader that iterates over
# dictionaries containing observations and actions.
#logger.configure(tempdir_path / "BC/")
#bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=transitions)
#bc_trainer.train(n_epochs=1)

# Train GAIL on expert data.
# GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
# iterates over dictionaries containing observations, actions, and next_observations.
dir_path = os.path.dirname(os.path.realpath(__file__))
logger.configure(os.path.join(dir_path,"GAIL/"),["tensorboard"])
gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
)
gail_trainer.train(total_timesteps=10*2048)

imitation_policy = gail_trainer.gen_algo

imitation_rewards, _ = evaluation.evaluate_policy(
       imitation_policy, venv.envs[0].env, return_episode_rewards=True
    )

observation = venv.envs[0].env.reset()
frames = []
for i in range(10000):
    frames.append(venv.envs[0].env.render(mode="rgb_array"))
    _, _, done, _ = venv.envs[0].env.step(venv.envs[0].env.action_space.sample())
    if done:
        break
venv.envs[0].env.close()
save_frames_as_gif(frames)



# Train AIRL on expert data.
#logger.configure(tempdir_path / "AIRL/")
#airl_trainer = adversarial.AIRL(
 #   venv,
 #   expert_data=transitions,
  #  expert_batch_size=32,
  #  gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
#)
#airl_trainer.train(total_timesteps=2048)
