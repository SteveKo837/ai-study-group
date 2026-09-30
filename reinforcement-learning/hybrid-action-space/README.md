# Hybrid Action Space

**Author:** JasonLuo  
**Date:** 2026-07-05  
**Domain:** Reinforcement Learning

## Overview

A study group presentation on hybrid (parameterized) action spaces, where an
agent selects a discrete action and its continuous parameters. A battery
control example motivates the choice between discrete, continuous, and hybrid
actions before introducing parameterized action Markov decision processes
(PAMDPs).

The presentation compares P-DQN, H-PPO, and HyAR in terms of architecture,
training, scalability, sample efficiency, and implementation complexity.
It focuses on HyAR's action embeddings, conditional variational autoencoder,
dynamics prediction, latent-space TD3 training, and mechanisms for constraining
latent actions and correcting representation shifts.

## Materials

- [Hybrid (Parameterized) Action Space RL：從電池調控談 P-DQN、H-PPO 與 HyAR](./index.html)
  — HTML presentation in Traditional Chinese, with supporting figures in
  `assets/`.

Open `index.html` in a browser. Formula rendering requires an internet
connection to load MathJax from its CDN; offline, formulas remain as raw LaTeX.

## References

- [Parametrized Deep Q-Networks Learning: Reinforcement Learning with Discrete-Continuous Hybrid Action Space](https://arxiv.org/abs/1810.06394)
- [Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space](https://arxiv.org/abs/1903.01344)
- [HyAR: Addressing Discrete-Continuous Action Reinforcement Learning via Hybrid Action Representation](https://arxiv.org/abs/2109.05490)
- [Deep Reinforcement Learning in Parameterized Action Space](https://arxiv.org/abs/1511.04143)
- [Reinforcement Learning with Parameterized Actions](https://arxiv.org/abs/1509.01644)
- [Multi-Pass Q-Networks for Deep Reinforcement Learning with Parameterised Action Spaces](https://arxiv.org/abs/1905.04388)
- [Deep Multi-Agent Reinforcement Learning with Discrete-Continuous Hybrid Action Spaces](https://www.ijcai.org/proceedings/2019/0323.pdf)
