import h5py
import os

def load_data():
    media_dir = '/media/NAS/USERS/moonbo/jssp'
    # filenames = ['jssp_data_gpu0_1000.h5', 'jssp_data_gpu1_1000.h5', 'jssp_data_gpu2_1000.h5', 'jssp_data_gpu3_1000.h5']
    filenames = ['jssp_data_real.h5']

    for filename in filenames:
        with h5py.File(os.path.join(media_dir, filename), 'r') as f:
            times = f['times'][:]
            machines = f['machines'][:]