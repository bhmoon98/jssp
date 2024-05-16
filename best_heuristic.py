import h5py
import os
import numpy as np
from collections import deque, defaultdict

def load_data():
    media_dir = '/media/NAS/USERS/moonbo/jssp'
    filenames = ['jssp_data_real.h5']
    all_times = []
    all_machines = []

    for filename in filenames:
        with h5py.File(os.path.join(media_dir, filename), 'r') as f:
            times = f['times'][:]
            machines = f['machines'][:]
            all_times.append(times)
            all_machines.append(machines)

    return np.concatenate(all_times), np.concatenate(all_machines)

class MachineSet:
    def __init__(self):
        self.machines = {}
    
    def get_machine(self, id):
        return self.machines[id]

    def add_machine(self, machine):
        self.machines[machine.id] = machine

    def __repr__(self):
        return f'MachineSet with Machines: {list(self.machines.values())}'

class Machine:
    def __init__(self, id):
        self.id = id
        self.operations = deque()

    def add_operation(self, operation):
        self.operations.append(operation)
    
    def fifo(self):
        return self.operations.popleft()

    def __repr__(self):
        return f'Machine {self.id} with Operations: {list(self.operations)}'

class Job:
    def __init__(self, id):
        self.id = id
        self.operations = queue()
    
    def add_operation(self, operation):
        self.operations.append(operation)

class Operation:
    def __init__(self, job, machine, sequence, time):
        self.job = job
        self.machine = machine
        self.sequence = sequence
        self.time = time

    def __repr__(self):
        return f'Operation {self.job}-{self.sequence} on Machine {self.machine} taking {self.time} time'

if __name__ == '__main__':
    times, machines = load_data()
    print(times.shape, machines.shape)

    machine_set_data = list(MachineSet())

    # Create and add operations to machines
    for time, machine in zip(times, machines):
        machine_set = MachineSet()
        machine_set_data.append(machine_set)
        for j in range(1, time.shape[1] + 1):
            machine = Machine(j)
            machine_set.add_machine(machine)
        for i in range(time.shape[0]):
            for j in range(time.shape[1]):
                operation = Operation(i, machine[i, j], j, time[i, j])
                machine_set.get_machine.add_operation(operation)

    print(machine_set)

                

            


