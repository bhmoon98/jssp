import numpy as np

data = np.load('/media/NAS/USERS/moonbo/jssp/jssp_supervision_data.npz', allow_pickle=True)
x = data['x']
y = data['y']
print(x.shape, y.shape)
print(data)