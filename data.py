import numpy as np

data = np.load('/media/NAS/USERS/moonbo/jssp/jssp_supervision_data.npz', allow_pickle=True)
x = data['x']
y = data['y']
print(x.shape, y.shape) # (50000, 100, 20, 2) (50000,)
a =x[0, :, :, 0]
print(type(a), a.shape, y[0])