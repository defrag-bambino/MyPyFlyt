# PyFlyt - UAV Flight Simulator for Reinforcement Learning Research

[![PyPI - Version](https://img.shields.io/pypi/v/PyFlyt.svg?logo=pypi&logoColor=FFE873)](https://img.shields.io/pypi/v/PyFlyt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/github/jjshoots/PyFlyt/graph/badge.svg?token=4RP08UIPIF)](https://codecov.io/github/jjshoots/PyFlyt)
[![total downloads](https://static.pepy.tech/personalized-badge/pyflyt?period=total&units=international_system&left_color=grey&right_color=green&left_text=total%20downloads)](https://pepy.tech/project/pyflyt)
[![weekly downloads](https://static.pepy.tech/personalized-badge/pyflyt?period=week&units=international_system&left_color=grey&right_color=green&left_text=weekly%20downloads)](https://pepy.tech/project/pyflyt)


Comes with Gymnasium and PettingZoo environments built in!

<p align="center">
    <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/pyflyt_cover_photo.png?raw=true" width="650px"/>
    <p align="center">
        <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/fixedwing_waypoint.gif?raw=true" height="150px"/>
        <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/fixedwing_dogfight.gif?raw=true" height="150px"/>
    </p>
    <p align="center">
        <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/quadx_pole_waypoint.gif?raw=true" height="150px"/>
        <img src="https://github.com/jjshoots/PyFlyt/blob/master/readme_assets/quadx_waypoint.gif?raw=true" height="150px"/>
    </p>
</p>

#### [View the documentation here!](https://jjshoots.github.io/PyFlyt/documentation.html)

This is a library for testing reinforcement learning algorithms on UAVs.
This repo is still under development.
We are also actively looking for users and developers, if this sounds like you, don't hesitate to get in touch!

## Installation

```sh
pip3 install wheel numpy
pip3 install pyflyt
```

> `numpy` and `wheel` must be installed prior to `pyflyt` such that `pybullet` is built with `numpy` support.

## Usage

Usage is similar to any other Gymnasium and PettingZoo environment:

### Gymnasium

```python
import gymnasium
import PyFlyt.gym_envs # noqa

env = gymnasium.make("PyFlyt/QuadX-Hover-v2", render_mode="human")
obs = env.reset()

termination = False
truncation = False

while not termination or truncation:
    observation, reward, termination, truncation, info = env.step(env.action_space.sample())
```

View the official documentation for gymnasium environments [here](https://jjshoots.github.io/PyFlyt/documentation/gym_envs.html).

### PettingZoo

```python
from PyFlyt.pz_envs import MAFixedwingDogfightEnv

env = MAFixedwingDogfightEnv(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

View the official documentation for pettingzoo environments [here](https://jjshoots.github.io/PyFlyt/documentation/pz_envs.html).

## Citation

If you use our work in your research and would like to cite it, please use the following bibtex entry:

```
@article{tai2023pyflyt,
  title={PyFlyt--UAV Simulation Environments for Reinforcement Learning Research},
  author={Tai, Jun Jet and Wong, Jim and Innocente, Mauro and Horri, Nadjim and Brusey, James and Phang, Swee King},
  journal={arXiv preprint arXiv:2304.01305},
  year={2023}
}
```
