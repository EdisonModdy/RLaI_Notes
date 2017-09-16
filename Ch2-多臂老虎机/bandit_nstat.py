import numpy as np
import matplotlib.pyplot as plt
import random as rd

epsilon = 0.1
ARM_NUM = 10
RUN_NUM = 2000
EPOCH_NUM = 100000

# Store the average rewards in each step
avg_rewards = [0 for i in range(EPOCH_NUM)]
avg_optimals = [0 for i in range(EPOCH_NUM)]

for run in range(RUN_NUM):
    qstars = [rd.gauss(0,1) for i in range(ARM_NUM)]
    qestims = [0 for i in range(ARM_NUM)]
    nums = [0 for i in range(ARM_NUM)]
    for epoch in range(EPOCH_NUM):
        # epsilon greedy
        if rd.random() < epsilon:
            action = rd.choice(qcurs)
        else:
            action = np.argmax(qcurs)
        # Incrementally compute the estimates
        reward = rd.gauss(qstars[action], 1)
        nums[action] += 1
        qcurs[action] += 1.0 / nums[action] * (reward - qcurs[action])
        # Add the statistic variables
        avg_rewards[epoch] += reward
        if action == np.argmax(qstars):
            avg_optimals[epoch] += 1
        # Add the increment
        qstars = [qstar+rd.gauss(0,0.01) for qstar in qstars]

# Plot the figure
