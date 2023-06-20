# DQN_for_Qubit_Mapping
This repository contains and implementation of the Deep Q-learning algorithm for Qubit Mapping. It uses the TF-Agent library of tensorflow.

To execute the code, run the following command:
'''
python3 mapping_dqn.py
'''
The code is set to train for quantum circuits of 5 qubits and depth 5 and the 5 qubit processor FakeAthensV2 from qiskit. To use a random processor use the function generate_random_non_directional_target() in the environemnt.py file.
