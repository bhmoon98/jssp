import numpy as np
import h5py
import argparse
import os
from tqdm import tqdm

def generate_times(num_instances, num_jobs, num_machines, time_min=1, time_max=100):
    times = np.random.randint(time_min, time_max + 1, size=(num_instances, num_jobs, num_machines), dtype=np.int32)
    return times

def prepare_machine_list(num_instances, num_jobs, num_machines):
    machines = np.zeros((num_instances, num_jobs, num_machines), dtype=np.int32)
    for i in range(num_instances):
        machine_list = np.tile(np.arange(1, num_machines + 1), num_jobs)
        np.random.shuffle(machine_list)
        machines[i, :, :] = machine_list.reshape(num_jobs, num_machines)
    return machines

def save_data(times, machines, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('times', data=times, compression='gzip')
        f.create_dataset('machines', data=machines, compression='gzip')
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSSP data using CPU")
    parser.add_argument('--num_instances', type=int, default=50000, help="Number of instances to generate")
    parser.add_argument('--num_jobs', type=int, default=100, help="Number of jobs per instance")
    parser.add_argument('--num_machines', type=int, default=20, help="Number of machines per instance")
    parser.add_argument('--time_min', type=int, default=1, help="Minimum time for operations")
    parser.add_argument('--time_max', type=int, default=100, help="Maximum time for operations")
    parser.add_argument('--media_dir', type=str, default='/media/NAS/USERS/moonbo/jssp', help="Directory to save generated files")
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_instances} instances with {args.num_jobs} jobs and {args.num_machines} machines...")
    
    # Initialize progress bar
    with tqdm(total=args.num_instances, desc="Generating Instances", unit="instance") as pbar:
        # Generate times and machines
        times = np.zeros((args.num_instances, args.num_jobs, args.num_machines), dtype=np.int32)
        machines = np.zeros((args.num_instances, args.num_jobs, args.num_machines), dtype=np.int32)
        
        for i in range(args.num_instances):
            times[i] = generate_times(1, args.num_jobs, args.num_machines, args.time_min, args.time_max)
            machines[i] = prepare_machine_list(1, args.num_jobs, args.num_machines)
            pbar.update(1)
    
    # Save the generated data
    os.makedirs(args.media_dir, exist_ok=True)
    filename = os.path.join(args.media_dir, 'jssp_data_real.h5')
    save_data(times, machines, filename)
    
    print("Data generation completed.")
