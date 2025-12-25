import functools
import random, sys
from copy import copy

import matplotlib.pyplot as plt

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict

from pettingzoo import ParallelEnv
import logging

from zk_blind_drone.state_render import Renderer

bde_default_cfg = {
    "num_targets": 8,
    "num_agents": 4,
    "latent_dim": 2,
    "reward_noise": 0,
    "observation_noise": 0,
    "interaction": "dot_product",
    "num_render_cols": 4
}


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def rlv(dim): return normalize(np.random.uniform(-1, 1, (dim, 1)))  # random latent vector


def rlm(num, dim): return normalize(np.random.uniform(-1, 1, (num, dim)))  # random latent matrix


def rld(l, dim): return {ll: rlv(dim) for ll in l}  # ranbdom dictionary


def v2d(l, v=False): return {ll: copy(v) for ll in l}  # initialize dictionary from list


class NBox(Box):  # box space specifier with normalization
    def __init__(self, *args, **kwargs):
        self.normalize = kwargs.pop("normalize", True)
        super().__init__(*args, **kwargs)

    def sample(self): return normalize(super().sample()) if self.normalize else super().sample()


class BDEnv(ParallelEnv):
    metadata = {
        "name": "blind drone enviornment_v0",
    }

    # actions
    # part of the actions is just for tracking and observations, does not affect state or reward
    # real action is only "selected_target"
    # other actions are agent's "estimates" of the state for rendering, internal agent states for rendering
    # obusing the environment for also keeping some internal agent state
    # targtes: agent estimations of all targets latent vectors(position in 2D), different for each agent
    # agents: agent estimations of all agents latent vectors(position in 2D), different for each agent
    # rewards: agent estimations of rewards/utility matrix, different for each agent

    # mbox of normalized vectors

    def make_space_list(self, l, space_spec):
        return Dict({x: space_spec for x in l})

    def make_boxspace_list(self, l, dim=(1,), low=-1, high=1, normalize=True):
        return self.make_space_list(l, NBox(low=low, high=high, shape=dim, dtype=np.float32, normalize=normalize))

    def make_lvspace_list(self, l, low=-1, high=1, normalize=True):
        return self.make_boxspace_list(l, dim=(self.cfg["latent_dim"],), low=low, high=high, normalize=normalize)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        logging.info(f"definiing action space for {agent=}")
        return Dict({"estimates":
            Dict({
                "targets_lv": self.make_lvspace_list(self.possible_targets),
                "agents_lv": self.make_lvspace_list(self.possible_agents),
                "rewards": self.make_space_list(self.possible_agents,
                                                self.make_boxspace_list(self.possible_targets, low=0, high=1,
                                                                        normalize=False))}),
            "selected_target": Discrete(self.cfg["num_targets"])})

    # observation are actually reflection of action additional information
    # real agent observation
    # selected targets:  what targets were selected by each agent
    # observed_rewards: what rewards were abserved by this agent, different for each agent as might have some noise
    # estimates is just agent internal state as provvidied by action

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        logging.info(f"definiing observation space for {agent=}")
        return Dict({"estimates":
            Dict({
                "targets_lv": self.make_lvspace_list(self.possible_targets),
                "agents_lv": self.make_lvspace_list(self.possible_agents),
                "rewards": self.make_space_list(self.possible_agents,
                                                self.make_boxspace_list(self.possible_targets, low=0, high=1))}),
            "selected_targets": self.make_space_list(self.possible_agents, MultiDiscrete(self.cfg["num_targets"])),
            "observed rewards": self.make_boxspace_list(self.possible_agents, dim=(self.cfg["num_targets"],), low=0,
                                                        high=1, normalize=False)})

    def __init__(self, cfg=bde_default_cfg):
        logging.info(f"initialize enviornment {cfg=}")
        self.cfg = copy(cfg)
        self.possible_agents = list(range(self.cfg["num_agents"]))
        self.possible_targets = list(range(self.cfg["num_targets"]))
        self.target_lv = rld(self.possible_targets, self.cfg["latent_dim"])
        self.agent_lv = rld(self.possible_agents, self.cfg["latent_dim"])
        self.prepare_mean_rewards_matrix()

    # mean reward based on configuration, start from dot product
    def reward(self, alv, tlv):
        return (1 + np.dot(alv.T, tlv)) / 2  # scale from [-1,1] to [0:1] range

    # precompute mean reward for all agent-vector pairs
    def prepare_mean_rewards_matrix(self):
        self.mean_rewards = {}
        for a, alv in self.agent_lv.items():
            self.mean_rewards[a] = {}
            for t, tlv in self.target_lv.items():
                self.mean_rewards[a][t] = self.reward(alv, tlv)

    def reset(self, seed=None, options=None):
        logging.info(f"reset enviornment {options=}")
        self.agents = copy(self.possible_agents)
        self.targets = copy(self.possible_targets)
        self.last_infos, self.last_terminations, self.last_truncations = v2d(self.agents), v2d(self.agents), v2d(
            self.agents)

        self.last_rewards = v2d(self.agents, 0)  # initialize last reward, if asked before first action
        self.last_selections = v2d(self.agents, -1)  # initialize last target selection, if asked before first action

        self.last_observations = {
            a: {
                "estimates": {
                    "targets_lv": rld(self.targets, self.cfg["latent_dim"]),
                    "agents_lv": rld(self.agents, self.cfg["latent_dim"]),
                    "rewards": {
                        aa: {t: -1 for t in self.targets}
                        for aa in self.agents
                    }
                },
                "selected_targets": dict(self.last_selections),  # copy dict
                "observed_rewards": dict(self.last_rewards)  # copy dict
            }
            for a in self.agents
        }

        return self.last_observations, self.last_infos

    # add noise to mean reward
    def stochastic_reward(self, a, t):
        return self.mean_rewards[a][t] + np.random.normal(0, self.cfg["reward_noise"])

    # add agent-specifoc noise for the common reward
    def observed_rewards(self):
        return {a: r + np.random.normal(0, self.cfg["observation_noise"]) for a, r in self.last_rewards.items()}

    def step(self, actions):
        for agent in self.agents:
            logging.info(f"compute step {agent=}")
            self.last_observations[agent]["estimates"] = copy(actions[agent][
                                                                  "estimates"])  # internal agent estimates become observations, abuse to keep agent state in teh environment
            self.last_selections[agent] = actions[agent]["selected_target"]
            self.last_rewards[agent] = self.stochastic_reward(agent, self.last_selections[agent])  # compute rewards

        # update observations
        for agent in self.agents:
            self.last_observations[agent]["observed_rewards"] = self.observed_rewards()
            self.last_observations[agent]["selected_targets"] = copy(self.last_selections)

        return self.last()

    def last(self):
        return self.last_observations, self.last_rewards, self.last_terminations, self.last_truncations, self.last_infos

    def render(self):
        gt = Renderer().draw_state(self.agent_lv, self.target_lv, self.last_selections, self.last_rewards).im
        images = {f"gt{agent}": gt for agent in self.possible_agents}

        for agent, observation in self.last_observations.items():
            images[f"est{agent}"] = Renderer().draw_state(observation["estimates"]["agents_lv"],
                                                          observation["estimates"]["targets_lv"],
                                                          observation["selected_targets"],
                                                          observation["observed_rewards"]).im
        Renderer.render_grid(images, columns=self.cfg["num_render_cols"])





