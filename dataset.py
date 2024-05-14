import numpy as np
from numba import cuda
import math
import h5py
import argparse
import multiprocessing
import GPUtil

# CUDA 커널 함수
@cuda.jit
def generate_instances_kernel(times, machines, time_min, time_max, seed):
    pos = cuda.grid(1)  # 현재 스레드의 고유 인덱스
    num_jobs, num_machines = times.shape[1], times.shape[2]
    
    if pos < times.shape[0]:  # 유효한 인덱스 범위 내인지 확인
        # Seed 생성
        np.random.seed(seed + pos)
        
        for i in range(num_jobs):
            times[pos, i, :] = np.random.randint(time_min, time_max + 1, size=num_machines)
        
        # Generate a list of machines, each appearing exactly num_jobs times
        machine_list = np.tile(np.arange(1, num_machines + 1), num_jobs)
        np.random.shuffle(machine_list)
        
        # Reshape and assign to machines array
        machines[pos, :, :] = machine_list.reshape(num_jobs, num_machines)

def generate_instances(num_instances, num_jobs, num_machines, time_min=1, time_max=100, block_size=256, gpu_id=0):
    # 특정 GPU 선택
    cuda.select_device(gpu_id)

    # Output arrays
    times = np.zeros((num_instances, num_jobs, num_machines), dtype=np.int32)
    machines = np.zeros((num_instances, num_jobs, num_machines), dtype=np.int32)

    # Grid and block dimensions
    grid_size = math.ceil(num_instances / block_size)
    
    # Launch CUDA kernel
    generate_instances_kernel[grid_size, block_size](times, machines, time_min, time_max, np.random.randint(1, 100000))

    return times, machines

def save_data(times, machines, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('times', data=times, compression='gzip')
        f.create_dataset('machines', data=machines, compression='gzip')
    print(f"Data saved to {filename}")

def worker(gpu_id, num_instances, num_jobs, num_machines, time_min, time_max, block_size):
    print(f"Starting GPU {gpu_id}")
    times, machines = generate_instances(num_instances, num_jobs, num_machines, time_min, time_max, block_size, gpu_id)
    filename = f'jssp_data_gpu{gpu_id}.h5'
    save_data(times, machines, filename)
    print(f"Finished GPU {gpu_id}")

def get_least_busy_gpu():
    gpus = GPUtil.getGPUs()
    # Get GPU with the lowest memory utilization
    return min(gpus, key=lambda gpu: gpu.memoryUtil).id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSSP data using multiple GPUs")
    parser.add_argument('--num_gpu', type=int, default = 4, help="Number of GPUs to use")
    parser.add_argument('--num_instances', type=int, default=50000, help="Number of instances to generate")
    parser.add_argument('--num_jobs', type=int, default=20, help="Number of jobs per instance")
    parser.add_argument('--num_machines', type=int, default=100, help="Number of machines per instance")
    parser.add_argument('--time_min', type=int, default=1, help="Minimum time for operations")
    parser.add_argument('--time_max', type=int, default=100, help="Maximum time for operations")
    parser.add_argument('--block_size', type=int, default=256, help="Block size for CUDA kernel")
    
    args = parser.parse_args()

    num_instances_per_gpu = args.num_instances // args.num_gpu
    processes = []

    for _ in range(args.num_gpu):
        gpu_id = get_least_busy_gpu()
        p = multiprocessing.Process(
            target=worker,
            args=(gpu_id, num_instances_per_gpu, args.num_jobs, args.num_machines, args.time_min, args.time_max, args.block_size)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All GPUs have finished processing.")


