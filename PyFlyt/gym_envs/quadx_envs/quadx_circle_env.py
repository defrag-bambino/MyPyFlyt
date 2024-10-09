"""QuadX Circle Environment."""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv


class QuadXCircleEnv(QuadXBaseEnv):
    """Circle Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is to circle around (0, 0, 1) with a configurable radius (default=0.5m).

    Args:
    ----
        sparse_reward (bool): whether to use sparse rewards or not.
        flight_mode (int): the flight mode of the UAV
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution.
        circle_radius (float): the radius of the circle to circle around (default=0.5m)

    """

    def __init__(
        self,
        sparse_reward: bool = False,
        flight_mode: int = 0,
        flight_dome_size: float = 3.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 40,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        circle_radius: float = 0.5,
    ):
        """__init__.

        Args:
        ----
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_mode (int): the flight mode of the UAV
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution.
            circle_radius (float): the radius of the circle to circle around (default=0.5m)

        """
        super().__init__(
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        """GYMNASIUM STUFF"""
        self.observation_space = self.combined_space

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward
        self.circle_radius = circle_radius

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset.

        Args:
        ----
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(seed, drone_options=options)
        super().end_reset(seed, options)

        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state of the current timestep.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        - auxiliary information (vector of 4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # combine everything
        if self.angle_representation == 0:
            self.state = np.concatenate(
                [
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.action,
                    #aux_state,
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            self.state = np.concatenate(
                [ang_vel, quaternion, lin_vel, lin_pos, self.action], axis=-1 #aux_state], axis=-1
            )

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        if not self.sparse_reward:
            # distance from 0, 0, 1 hover point
            linear_distance = np.linalg.norm(
                self.env.state(0)[-1] - np.array([0.0, 0.0, 1.0])
            )

            # distance from the circle
            circle_distance = np.linalg.norm(
                self.env.state(0)[-1][:2] - np.array([0.0, 0.0])
            ) - self.circle_radius

            self.reward -= (2 - circle_distance) * np.linalg.norm(self.env.state(0)[2]) * (1.0 - self.env.state(0)[-1, 3])
            self.reward += .1

