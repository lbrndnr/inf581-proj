# Reinforcement Learning Algorithms applied on the arcade game Snake

Run the demo in the terminal using
```python
python3 demo.py
```

## Source Code
- `demo.py` a small file that showcases the different techniques
- `agent_DQN.py` is the agent that makes use of the DQN (training procedures etc.)
- `DQN.py` an implementation of the Deep-Q-Network used to predict q-values using Keras
- `DQN_tf.py` an implementation of the Deep-Q-Network used to predict q-values using Tensorflow
- `memory.py` an implementation of the experience playback buffer
- `agent_RL.py` incorporates all the Monte Carlo and the Q-Learning algorithms
- `agent_SE.py` is the naive search algorithm used as a reference
- `environment_3d.py` the snake game implemented using 3 directions: STRAIGHT, LEFT and RIGHT
- `geometry_3d.py` includes some utility functions for `environment_3d.py`
- `environment_4d.py` the snake game implemented using 4 directions: NORTH, EAST, SOUTH, WEST
- `geometry_4d.py` includes some utility functions for `environment_4d.py`
- `res/plot.py` utility function to generate the plots seen in the paper

The different implementations of the environment and geometry functions were used to evaluate the difference it made to use 3 directions instead of 4 regarding the training procedure for the various algorithms. While the algorithms using 3 directions learned more quickly, it is assumed that both implementations will be equally good once the learning curve converged.

## Dependencies
To run any RL algorithm developed for this project `numpy` is necessary. Additionally, depending on the implementation, the DQN agent requires `tensorflow`, or `keras` and `theano` respectively.

## Sources
- P.J Kindermans, [Deep Q Networks in Tensorflow](https://github.com/pikinder/DQN), 2017
- D. Grattarola, [Playing Snake with Deep Q-Learning](https://github.com/danielegrattarola/deep-q-snake), 2015
- Y. Gutz, [snake-ai-reinforcement](https://github.com/YuriyGuts/snake-ai-reinforcement), 2017