# PPO Training Tricks

**Author:** JasonLuo  
**Date:** 2026-08-03  
**Domain:** Reinforcement Learning

## Overview

An interactive study group presentation on Proximal Policy Optimization (PPO)
and the implementation choices that affect reinforcement learning training.
It traces policy gradients through importance sampling and KL constraints to
the PPO clipped objective, then reviews the studies *Implementation Matters in
Deep Policy Gradients* and *What Matters in On-Policy Reinforcement Learning?*.

Topics include network architecture and initialization, normalization,
generalized advantage estimation (GAE), value losses, data reuse, optimizers,
and reproducible baselines and ablation experiments. The evidence focuses on
MuJoCo continuous control; the suggested settings are starting points to test,
not universal defaults.

## Materials

- [What Matters in RL training](./index.html) — interactive HTML presentation
  in Traditional Chinese, approximately 45–60 minutes.

## References

- [PPO (v3) lecture slides — Hung-yi Lee](https://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/PPO%20%28v3%29.pdf)
- [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729)
- [What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990)
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
