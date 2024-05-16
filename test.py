import numpy as np

num_machines = 10
num_jobs = 5

machine_list = np.tile(np.arange(1, num_machines + 1), num_jobs)
np.random.shuffle(machine_list)
print(machine_list)