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

        # The following dense reward encourages circling about (0, 0, 1) at the desired circle radius.
        # It consists of:
        # 1. A distance penalty that penalizes deviation from the circle: the horizontal error
        #    (i.e. difference between the current horizontal distance and the circle_radius)
        #    and a vertical error (difference between the current altitude and 1.0).
        # 2. A velocity penalty that penalizes deviation from a target tangential speed.
        #    The target speed is configurable (via self.target_speed, defaulting to 0.5 m/s).
        if not self.sparse_reward:
            # Extract the linear velocity and position from the state.
            # The state layout depends on the angle representation.
            if self.angle_representation in [0, "euler"]:
                # state = [ang_vel (3), ang_pos (3), lin_vel (3), lin_pos (3), previous_action (4)]
                lin_vel = self.state[6:9]
                lin_pos = self.state[9:12]
            else:
                # state = [ang_vel (3), quaternion (4), lin_vel (3), lin_pos (3), previous_action (4)]
                lin_vel = self.state[7:10]
                lin_pos = self.state[10:13]

            # Compute the error in position.
            # We want the drone to maintain a horizontal distance equal to self.circle_radius
            # from (0,0) and an altitude of 1.0.
            horizontal_distance = np.linalg.norm(lin_pos[:2])
            horizontal_error = np.abs(horizontal_distance - self.circle_radius)
            vertical_error = np.abs(lin_pos[2] - 1.0)
            distance_reward = - (horizontal_error**2 + vertical_error**2)

            # Compute the tangential speed error.
            # The desired direction of travel is perpendicular to the radial vector in the horizontal plane.
            if horizontal_distance > 1e-6:
                unit_tangent = np.array([-lin_pos[1], lin_pos[0]]) / horizontal_distance
            else:
                unit_tangent = np.array([0.0, 0.0])
            tangential_speed = np.dot(lin_vel[:2], unit_tangent)

            # Get the target tangential speed; this is a configurable parameter.
            target_speed = getattr(self, "target_speed", 0.5)  # default to 0.5 m/s if not defined
            velocity_error = np.abs(tangential_speed - target_speed)
            velocity_reward = - (velocity_error**2)

            # Combine the rewards; the velocity term is weighted by 0.1 (this weight can be tuned).
            self.reward = distance_reward + 0.1 * velocity_reward

        else:
            # Sparse reward version:
            # The agent receives a reward of 1.0 only if both the distance error and the tangential
            # velocity error are below small thresholds; otherwise, the reward is 0.
            if self.angle_representation in [0, "euler"]:
                lin_vel = self.state[6:9]
                lin_pos = self.state[9:12]
            else:
                lin_vel = self.state[7:10]
                lin_pos = self.state[10:13]

            horizontal_distance = np.linalg.norm(lin_pos[:2])
            horizontal_error = np.abs(horizontal_distance - self.circle_radius)
            vertical_error = np.abs(lin_pos[2] - 1.0)

            if horizontal_distance > 1e-6:
                unit_tangent = np.array([-lin_pos[1], lin_pos[0]]) / horizontal_distance
            else:
                unit_tangent = np.array([0.0, 0.0])
            tangential_speed = np.dot(lin_vel[:2], unit_tangent)
            target_speed = getattr(self, "target_speed", 0.5)
            velocity_error = np.abs(tangential_speed - target_speed)

            # Define error thresholds for a successful circle.
            distance_threshold = 0.1  # meters
            velocity_threshold = 0.1  # m/s

            if (horizontal_error < distance_threshold and
                vertical_error < distance_threshold and
                velocity_error < velocity_threshold):
                self.reward = 1.0
            else:
                self.reward = 0.0


