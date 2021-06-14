# LyDROO

*Lyapunov-guided Deep Reinforcement Learning for Stable Online Computation Offloading in Mobile-Edge Computing Networks*

Python code to reproduce LyDROO algorithm [1], which is an online computation offloading algorithm to maximize the network data processing capability subject to the long-term data queue stability and average power constraints. It applies Lyapunov optimization to
decouple the multi-stage stochastic MINLP into deterministic per-frame MINLP subproblems and solves each subproblem via [DROO](https://github.com/revenol/DROO) algorithm.

## About our works

1. Suzhi Bi, Liang Huang, and Ying-jun Angela Zhang, ''[Lyapunov-guided Deep Reinforcement Learning for Stable Online Computation Offloading in Mobile-Edge Computing Networks]((https://ieeexplore.ieee.org/document/9449944))'', *IEEE Transactions on Wireless Communications*, 2021, doi:10.1109/TWC.2021.3085319.

## About authors

- Suzhi BI, bsz AT szu.edu.cn

- Liang HUANG, lianghuang AT zjut.edu.cn

- Ying Jun (Angela) Zhang, yjzhang AT ie.cuhk.edu.hk


## How the code works

- For LyDROO algorithm, run the file, [main.py](main.py)
