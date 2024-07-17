"""Multiagent Fixedwing Dogfighting Environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.pz_envs.fixedwing_envs.dogfight_utils import compute_combat_state
from PyFlyt.pz_envs.fixedwing_envs.ma_fixedwing_base_env import MAFixedwingBaseEnv


class MAFixedwingTeamDogfightEnv(MAFixedwingBaseEnv):
    """Team Dogfighting Environment for the Acrowing model using the PettingZoo API.

    Args:
        spawn_height (float): how high to spawn the agents at the beginning of the simulation.
        damage_per_hit (float): how much damage per hit per physics step, each agent starts with a health of 1.0.
        lethal_distance (float): how close before weapons become effective.
        lethal_angle_radians (float): the width of the cone of fire.
        assisted_flight (bool): whether to use high level commands (RPYT) instead of full actuator commands.
        sparse_reward (bool): whether to use sparse rewards or not.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | str): can be "human" or None

    """

    metadata = {
        "render_modes": ["human"],
        "name": "ma_fixedwing_team_dogfight",
    }

    def __init__(
        self,
        team_size: int = 2,
        spawn_radius: float = 5.0,
        spawn_height: float = 15.0,
        damage_per_hit: float = 0.02,
        lethal_distance: float = 15.0,
        lethal_angle_radians: float = 0.1,
        assisted_flight: bool = True,
        sparse_reward: bool = False,
        flight_dome_size: float = 150.0,
        max_duration_seconds: float = 60.0,
        agent_hz: int = 30,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            team_size (int): number of planes that comprises a team.
            spawn_radius (float): agents are spawned in a circle pointing outwards, this value is the radius of that circle.
            spawn_height (float): how high to spawn the agents at the beginning of the simulation.
            damage_per_hit (float): how much damage per hit per physics step, each agent starts with a health of 1.0.
            lethal_distance (float): how close before weapons become effective.
            lethal_angle_radians (float): the width of the cone of fire.
            assisted_flight (bool): whether to use high level commands (RPYT) instead of full actuator commands.
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | str): can be "human" or None

        """
        # placeholder starting positions
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 0.0]] * 2 * team_size),
            start_orn=np.array([[0.0, 0.0, 0.0]] * 2 * team_size),
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation="euler",
            agent_hz=agent_hz,
            assisted_flight=assisted_flight,
            render_mode=render_mode,
        )

        self.team_size = team_size
        self.spawn_radius = spawn_radius
        self.sparse_reward = sparse_reward
        self.damage_per_hit = damage_per_hit
        self.spawn_height = spawn_height
        self.lethal_distance = lethal_distance
        self.lethal_angle = lethal_angle_radians
        self.hit_colour = np.array([0.7, 0.7, 0.7, 0.2])
        self.nohit_color = np.array([0.0, 0.0, 0.0, 0.2])

        # observation_space
        self._observation_space = spaces.Dict(
            {
                # base + health
                "self": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.combined_space.shape[0] + 1,),
                ),
                # watch n opponents attitude + health
                "opponents": spaces.Sequence(
                    space=spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(12 + 1,),
                    ),
                    stack=True,
                )
            }
        )

    def observation_space(self, agent: Any = None) -> spaces.Dict:
        """observation_space.

        Args:
            agent (Any): agent

        Returns:
            spaces.Box:

        """
        return self._observation_space

    def _get_start_pos_orn(self) -> tuple[np.ndarray, np.ndarray]:
        """_get_start_pos_orn.

        Args:
            seed (None | int): seed

        Returns:
            tuple[np.ndarray, np.ndarray]:

        """
        start_z_radian = np.pi / self.team_size * np.arange(self.team_size * 2)

        # define the starting positions
        start_pos = np.zeros((self.team_size * 2, 3))
        start_pos[:, 0] = self.spawn_radius * np.cos(start_z_radian)
        start_pos[:, 1] = self.spawn_radius * np.sin(start_z_radian)
        start_pos[:, 2] = self.spawn_height

        # define the starting orientations
        start_orn = np.zeros((self.team_size * 2, 3))
        start_orn[:, 2] = start_z_radian
        start_orn[:, 2] = 0.0

        return start_pos, start_orn

    def reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """reset.

        Args:
            seed (None | int): seed
            options (dict[str, Any]): options

        Returns:
            tuple[dict[str, Any], dict[str, Any]]:

        """
        self.start_pos, self.start_orn = self._get_start_pos_orn()

        # define custom forward velocity
        _, start_vec = self.compute_rotation_forward(self.start_orn)
        start_vec *= 10.0
        drone_options = [dict() for _ in range(self.num_possible_agents)]
        for i in range(len(drone_options)):
            drone_options[i]["starting_velocity"] = start_vec[i]
            drone_options[i]["drone_model"] = (
                "acrowing_blue" if i < self.team_size else "acrowing_red"
            )

        super().begin_reset(seed, options, drone_options=drone_options)

        # reset runtime parameters
        self.opponent_attitudes = np.zeros(
            (self.num_possible_agents, self.num_possible_agents, 4, 3), dtype=np.float64
        )
        self.healths = np.ones(self.num_possible_agents, dtype=np.float64)
        self.in_cone = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=bool
        )
        self.in_range = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=bool
        )
        self.chasing = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=bool
        )
        self.current_hits = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=bool
        )
        self.current_angles = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=np.float64
        )
        self.current_offsets = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=np.float64
        )
        self.current_distances = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=np.float64
        )
        self.previous_hits = np.zeros(
            (self.num_possible_agents, (self.num_possible_agents)), dtype=bool
        )
        self.previous_angles = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=np.float64
        )
        self.previous_offsets = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=np.float64
        )
        self.previous_distances = np.zeros(
            (self.num_possible_agents, self.num_possible_agents), dtype=np.float64
        )
        self.last_obs_time = -1.0
        self.rewards = np.zeros((self.num_possible_agents,), dtype=np.float64)
        self.last_rew_time = -1.0

        super().end_reset(seed, options)

        observations = {
            ag: self.compute_observation_by_id(self.agent_name_mapping[ag])
            for ag in self.agents
        }
        infos = {ag: dict() for ag in self.agents}
        return observations, infos

    def _compute_agent_states(self) -> None:
        """_compute_agent_states.

        Returns:
            None:

        """
        # get the states of both drones
        self.attitudes = np.stack(self.aviary.all_states, axis=0)

        # record some past statistics
        self.previous_hits = self.current_hits.copy()
        self.previous_distances = self.current_distances.copy()
        self.previous_angles = self.current_angles.copy()
        self.previous_offsets = self.current_offsets.copy()

        # compute new states
        (
            self.in_cone,
            self.in_range,
            self.chasing,
            self.current_hits,
            self.current_distances,
            self.current_angles,
            self.current_offsets,
            self.opponent_attitudes,
        ) = compute_combat_state(
            self.attitudes,
            self.lethal_angle,
            self.lethal_distance,
            self.damage_per_hit,
        )

        # compute whether anyone hit anyone
        self.healths -= self.damage_per_hit * self.current_hits.sum(axis=0)

        # flatten things
        flat_attitudes = self.attitudes.reshape(self.num_agents, -1)
        flat_healths = self.healths.reshape(self.num_agents, -1)
        flat_opponent_attitudes = self.opponent_attitudes.reshape(self.num_agents, self.num_agents, -1)

        # start stacking the observations
        self.observations: list[dict[Literal["self", "opponents"], np.ndarray]] = [
            {
                # formulate the observation of self
                "self": np.concatenate(
                    (flat_attitudes[i], self.aviary.aux_state(i), flat_healths[i], self.past_actions[i]),
                    axis=-1,
                ),
                # formulate the opponents status, select perspective, then delete self from that perspective
                "opponents": np.concatenate(
                    (
                        np.delete(flat_opponent_attitudes[i], i, axis=0),
                        np.delete(flat_healths, i, axis=0),
                    ),
                    axis=-1
                )
            } for i in range(self.num_possible_agents)
        ]

    def compute_observation_by_id(self, agent_id: int) -> dict[Literal["self", "opponents"], np.ndarray]:
        """compute_observation_by_id.

        Args:
            agent_id (int): agent_id

        Returns:
            np.ndarray:

        """
        # don't recompute if we've already done it
        if self.last_obs_time != self.aviary.elapsed_time:
            self.last_obs_time = self.aviary.elapsed_time
            self._compute_agent_states()
        return self.observations[agent_id]

    def _compute_engagement_rewards(self) -> None:
        """_compute_engagement_rewards."""
        # reset rewards
        rewards = np.zeros((self.num_possible_agents, self.num_possible_agents), dtype=np.float64)

        # sparse reward computation
        if not self.sparse_reward:
            # reward for closing the distance
            rewards += (
                np.clip(
                    self.previous_distances - self.current_distances,
                    a_min=0.0,
                    a_max=None,
                )
                * (~self.in_range & self.chasing)
                * 1.0
            )

            # reward for progressing to engagement, penalty for losing angles is less
            # WARNING: NaN introduced here
            delta_angles = self.previous_angles - self.current_angles
            delta_angles[delta_angles > 0.0] *= 0.5
            rewards += delta_angles * self.in_range * 10.0

            # reward for engaging the enemy
            # WARNING: NaN introduced here
            rewards += 3.0 / (self.current_angles + 0.1) * self.in_range

        # reward for hits and being hit
        rewards += (30.0 * self.current_hits) + (-20.0 * self.current_hits.T)

        # remove the nans, and sum rewards along axes
        np.fill_diagonal(rewards, 0.0)
        self.rewards = rewards.sum(axis=1)

    def compute_term_trunc_reward_info_by_id(
        self, agent_id: int
    ) -> tuple[bool, bool, float, dict[str, Any]]:
        """Computes the termination, truncation, and reward of the current timestep."""
        # don't recompute if we've already done it
        if self.last_rew_time != self.aviary.elapsed_time:
            self.last_rew_time = self.aviary.elapsed_time
            self._compute_engagement_rewards()

        # initialize
        reward = self.rewards[agent_id]
        term = self.num_agents < 2
        trunc = self.step_count > self.max_steps
        info = dict()

        # collision
        if np.any(self.aviary.contact_array[self.aviary.drones[agent_id].Id]):
            reward -= 3000.0
            info["collision"] = True
            term |= True

        # exceed flight dome
        if np.linalg.norm(self.aviary.state(agent_id)[-1]) > self.flight_dome_size:
            reward -= 3000.0
            info["out_of_bounds"] = True
            term |= True

        # out of health
        if self.healths[agent_id] <= 0.0:
            reward -= 100.0
            info["dead"] = True
            term |= True

        # all the info things
        info["healths"] = self.healths

        return term, trunc, reward, info

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """step.

        Args:
            actions (dict[str, np.ndarray]): actions

        Returns:
            tuple[
                dict[str, Any],
                dict[str, float],
                dict[str, bool],
                dict[str, bool],
                dict[str, dict[str, Any]],
            ]:

        """
        returns = super().step(actions=actions)

        # colour the gunsights conditionally
        if self.render_mode and not np.all(self.previous_hits == self.current_hits):
            self.previous_hits = self.current_hits.copy()
            for i in range(2):
                self.aviary.changeVisualShape(
                    self.aviary.drones[i].Id,
                    7,
                    rgbaColor=(
                        self.hit_colour if self.current_hits[i] else self.nohit_color
                    ),
                )

        return returns
