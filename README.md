# Deep Hedging with Market Impact
 
Code for the "Deep Hedging with Market Impact" paper submitted to The 37th Canadian Conference on Artificial Intelligence.

## Abstract
Dynamic hedging is the practice of periodically transacting financial instruments to offset the risk caused by an investment or a liability. Dynamic hedging optimization can be framed as a sequential decision problem; thus, Reinforcement Learning (RL) models were recently proposed to tackle this task. However, existing RL works for hedging do not consider market impact caused by the finite liquidity of traded instruments. Integrating such a feature can be crucial to achieve optimal performance when hedging options on stocks with limited liquidity. In this paper, we propose a novel general market impact dynamic hedging model based on Deep Reinforcement Learning (DRL) that considers several realistic features such as convex market impacts, and impact persistence through time. The optimal policy obtained from the DRL model is analysed using several option hedging simulations and compared to commonly used procedures such as delta hedging. Results show our DRL model behaves better in contexts of low liquidity by, among others: 1) learning the extent to which portfolio rebalancing actions should be dampened or delayed to avoid high costs, 2) factoring in the impact of features not considered by conventional approaches, such as previous hedging errors through the portfolio value, and the underlying asset's drift (i.e. the magnitude of its expected return).

## Contents
All neural network architectures contain the prefix "DeepAgent".  
The performance of the different architectures are compared in "Deep Hedging Experiments.py".  
The effect of different portfolio values is tested in "Effect_of_Market_Impact.py".  
The effect of different levels of market impact and impact persistence is tested in "Effect_of_Market_Impact_Single_Path.py"

