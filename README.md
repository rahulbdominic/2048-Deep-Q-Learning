# 2048-Deep-Q-Learning

For some reason the scores per game doesnt seem to be increasing however the highest scores over 20 games does show a rising trend. I'm investigating the reason for the lack of convergence and hopefully will soon post a better working model. However the best configuration I experimented with is given below:

Implemented 2048 AI based on DeepMind Deep Q Learning.

High score: 13420

Configuration: Fully connected NN:

Layer 1: 16 nodes input vector

Layer 2: 64 nodes

Layer 3: 4 nodes (associated with 4 moves)
