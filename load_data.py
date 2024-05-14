import h5py

with h5py.File('jssp_data.h5', 'r') as f:
    times = f['times'][:]
    machines = f['machines'][:]

print("Data loaded from jssp_data.h5")
print(times.shape)  # (30000, 20, 100)
print(machines.shape)  # (30000, 20, 100)