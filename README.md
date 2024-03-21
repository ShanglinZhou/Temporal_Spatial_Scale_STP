Codes for training RNNs with Short-term plasticity mechanism to control both temporal and spatial scales of neural dynamics and sensorimotor behaviors.
Optimization is implemented in TensorFlow 2.3.

For details please see 'Unified control of temporal and spatial scales of sensorimotor behavior through neuromodulation of short-term synaptic plasticity' By Shanglin Zhou and Dean Buonomano

Start with the main file PARAM.py which will call functions of the RNN model and task-related functions in model.py.
Codes in PARAM.py are meant to incorporate the hyperparameter search. For a specific set of hyperparameters, just run the code after the second 'for' loop with the hyperparameter specified in the 'hp' variable'.

As described in the paper, series tasks were implemented:

readysetgoContext:  ready-set-go task

iafc:  IAFC task

motorTraj_time:  temporal scaling task with input-cued-digit setting

motorTraj_space:  spatial scaling task with input-cued-digit setting

motorTraj: joint control temporal and spatial scales with input-cued-digit setting

motorTraj_digit: joint control temporal and spatial scales with alpha-cued-digit setting

motorTraj_time_inputScale: temporal scaling task  with input-cued-digit setting and input-ampplitude-cued-scaling

motorTraj_space_inputScale: spatial scaling task with input-cued-digit setting and input-ampplitude-cued-scaling
