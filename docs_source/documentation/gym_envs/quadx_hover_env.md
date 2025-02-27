# `PyFlyt/QuadX-Hover-v3`

```{figure} https://raw.githubusercontent.com/jjshoots/PyFlyt/master/readme_assets/quadx_hover.gif
    :width: 50%
```

## Task Description

The goal of this environment is to hover a quadrotor drone for as long as possible.

## Usage

```python
import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/QuadX-Hover-v2", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())
```

## Environment Options

```{eval-rst}
.. autoclass:: PyFlyt.gym_envs.quadx_envs.quadx_hover_env.QuadXHoverEnv
```
