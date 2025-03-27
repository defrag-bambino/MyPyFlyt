"""QuadX Goal Environment."""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pybullet as p
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv


class QuadXGoalEnv(QuadXBaseEnv):
    """QuadX Goal Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The task is to fly to a random goal position in 3D space with low velocity.
    A significant reward is given when the drone reaches the goal with low velocity.

    Args:
    ----
        goal_reach_distance (float): distance to the goal for it to be considered reached.
        max_velocity_threshold (float): maximum velocity magnitude for the reward to be given.
        goal_reward (float): reward given when reaching the goal with low velocity.
        sparse_reward (bool): whether to use sparse rewards or not.
        flight_mode (int): the flight mode of the UAV.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution.
    """

    def __init__(
        self,
        goal_reach_distance: float = 0.2,
        max_velocity_threshold: float = 0.5,
        goal_reward: float = 10.0,
        sparse_reward: bool = False,
        flight_mode: int = 0,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (960, 960),
    ):
        """__init__.

        Args:
        ----
            goal_reach_distance (float): distance to the goal for it to be considered reached.
            max_velocity_threshold (float): maximum velocity magnitude for the reward to be given.
            goal_reward (float): reward given when reaching the goal with low velocity.
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_mode (int): the flight mode of the UAV.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution.
        """
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # Goal-related parameters
        self.goal_reach_distance = goal_reach_distance
        self.max_velocity_threshold = max_velocity_threshold
        self.goal_reward = goal_reward
        self.sparse_reward = sparse_reward
        self.goal_position = None
        self.goal_visual_id = None
        self.goals_reached = 0

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "goal_delta": spaces.Box(
                    low=-2 * flight_dome_size,
                    high=2 * flight_dome_size,
                    shape=(3,),
                    dtype=np.float64,
                ),
            }
        )

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[Literal["attitude", "goal_delta"], np.ndarray], dict]:
        """Resets the environment.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(seed, options)
        
        # Generate a random goal position
        self._generate_random_goal(seed)
        
        # Create a visual marker for the goal
        self._create_goal_visual()
        
        self.info["goal_reached"] = False
        self.info["goals_reached"] = 0
        self.goals_reached = 0
        super().end_reset()

        return self.state, self.info

    def _generate_random_goal(self, seed=None):
        """Generate a random goal position within the flight dome."""
        if seed is not None:
            np.random.seed(seed)
            
        # Generate a random position within the flight dome
        # Keep the goal at least 1.0 unit away from the current position
        min_height = 0.5  # Minimum height from the ground
        max_attempts = 10
        
        # Use current drone position if available, otherwise use default start
        if hasattr(self, 'env') and hasattr(self, 'state') and self.state is not None:
            # Extract position based on angle representation
            if self.angle_representation == 0:  # Euler
                current_pos = self.state["attitude"][9:12]
            else:  # Quaternion
                current_pos = self.state["attitude"][10:13]
        else:
            # Default starting position
            current_pos = np.array([0.0, 0.0, 1.0])
        
        for _ in range(max_attempts):
            # Random point within the flight dome
            r = self.flight_dome_size * np.random.random() ** (1/3)  # Cube root for uniform distribution in sphere
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            # Ensure minimum height
            z = max(z, min_height)
            
            goal_pos = np.array([x, y, z])
            
            # Check if the goal is far enough from the current position
            if np.linalg.norm(goal_pos - current_pos) >= 1.0:
                self.goal_position = goal_pos
                return
        
        # Fallback if we couldn't find a suitable position
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        self.goal_position = current_pos + 1.5 * direction
        self.goal_position[2] = max(self.goal_position[2], min_height)  # Ensure minimum height

    def _create_goal_visual(self):
        """Create a visual marker for the goal."""
        if self.goal_visual_id is not None:
            self.env.removeBody(self.goal_visual_id)
            
        self.goal_visual_id = self.env.createVisualShape(
            self.env.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[0, 1, 0, 0.7],  # Green, semi-transparent
        )
        
        self.env.createMultiBody(
            baseVisualShapeIndex=self.goal_visual_id,
            basePosition=self.goal_position,
        )

    def compute_state(self) -> None:
        """Computes the state of the current timestep.

        This returns the observation as well as the distance to goal.
        - "attitude" (Box)
        ----- ang_vel (vector of 3 values)
        ----- ang_pos (vector of 3/4 values)
        ----- lin_vel (vector of 3 values)
        ----- lin_pos (vector of 3 values)
        ----- previous_action (vector of 4 values)
        - "goal_delta" (Box)
        ----- vector from drone to goal in global frame (vector of 3 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # Compute distance to goal
        goal_delta = self.goal_position - lin_pos

        # Combine everything
        new_state: dict[Literal["attitude", "goal_delta"], np.ndarray] = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.concatenate(
                [
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.action,
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.concatenate(
                [
                    ang_vel,
                    quaternion,
                    lin_vel,
                    lin_pos,
                    self.action,
                ],
                axis=-1,
            )

        new_state["goal_delta"] = goal_delta

        self.state: dict[Literal["attitude", "goal_delta"], np.ndarray] = new_state

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        # Get position and velocity
        if self.angle_representation == 0:
            lin_vel = self.state["attitude"][6:9]
            lin_pos = self.state["attitude"][9:12]
        else:  # quaternion
            lin_vel = self.state["attitude"][7:10]
            lin_pos = self.state["attitude"][10:13]

        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(self.goal_position - lin_pos)
        
        # Calculate velocity magnitude
        velocity_magnitude = np.linalg.norm(lin_vel)
        
        # Check if goal is reached with low velocity
        goal_reached = (distance_to_goal < self.goal_reach_distance)
        low_velocity = (velocity_magnitude < self.max_velocity_threshold)
        
        if goal_reached and low_velocity:
            # Give a big reward for reaching the goal with low velocity
            self.reward += self.goal_reward
            
            # Increment the counter for goals reached
            self.goals_reached += 1
            self.info["goals_reached"] = self.goals_reached
            
            # Generate a new goal
            self._generate_random_goal()
            self._create_goal_visual()
        elif not self.sparse_reward:
            # Shaped reward to guide the drone to the goal
            # Reward inversely proportional to distance
            self.reward += 0.1 / (1.0 + distance_to_goal)
            
            # Small penalty for high velocity when close to the goal
            if distance_to_goal < 1.0:
                self.reward -= 0.05 * velocity_magnitude 