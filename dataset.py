import numpy as np
from numba import cuda
import math
import h5py
import argparse
import multiprocessing
import os
from tqdm import tqdm
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# CUDA 커널 함수
@cuda.jit
def generate_times_kernel(rng_states, times, time_min, time_max):
    pos = cuda.grid(1)  # 현재 스레드의 고유 인덱스
    num_jobs, num_machines = times.shape[1], times.shape[2]
    
    if pos < times.shape[0]:  # 유효한 인덱스 범위 내인지 확인
        for i in range(num_jobs):
            for j in range(num_machines):
                rand_val = xoroshiro128p_uniform_float32(rng_states, pos)
                times[pos, i, j] = int(rand_val * (time_max - time_min + 1)) + time_min

# CPU 측 코드
def prepare_machine_list(num_instances, num_jobs, num_machines):
    machines = np.zeros((num_instances, num_jobs, num_machines), dtype=np.int32)
    for i in range(num_instances):
        machine_list = np.tile(np.arange(1, num_machines + 1), num_jobs)
        np.random.shuffle(machine_list)
        machines[i, :, :] = machine_list.reshape(num_jobs, num_machines)
    return machines

def generate_instances(num_instances, num_jobs, num_machines, time_min=1, time_max=100, block_size=256, gpu_id=0):
    # 특정 GPU 선택
    cuda.select_device(gpu_id)
    
    # RNG states
    threads_per_block = block_size
    blocks_per_grid = (num_instances + (threads_per_block - 1)) // threads_per_block
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=np.random.randint(1, 100000))

    # Output arrays
    times = np.zeros((num_instances, num_jobs, num_machines), dtype=np.int32)
    times_device = cuda.to_device(times)
    machines = prepare_machine_list(num_instances, num_jobs, num_machines)
    machines_device = cuda.to_device(machines)

    # Launch CUDA kernel
    generate_times_kernel[blocks_per_grid, threads_per_block](rng_states, times_device, time_min, time_max)
    times_device.copy_to_host(times)

    return times, machines

def save_data(times, machines, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('times', data=times, compression='gzip')
        f.create_dataset('machines', data=machines, compression='gzip')
    print(f"Data saved to {filename}")

def worker(gpu_id, task_queue, progress_queue, media_dir):
    while True:
        try:
            num_instances, num_jobs, num_machines, time_min, time_max, block_size = task_queue.get_nowait()
        except:
            return
        
        print(f"Starting GPU {gpu_id} for {num_instances} instances")
        times, machines = generate_instances(num_instances, num_jobs, num_machines, time_min, time_max, block_size, gpu_id)
        filename = os.path.join(media_dir, f'jssp_data_gpu{gpu_id}_{num_instances}.h5')
        save_data(times, machines, filename)
        print(f"Finished GPU {gpu_id} for {num_instances} instances")
        progress_queue.put(num_instances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSSP data using multiple GPUs")
    parser.add_argument('--num_gpu', type=int, default=4, help="Number of GPUs to use")
    parser.add_argument('--num_instances', type=int, default=50000, help="Number of instances to generate")
    parser.add_argument('--num_jobs', type=int, default=20, help="Number of jobs per instance")
    parser.add_argument('--num_machines', type=int, default=100, help="Number of machines per instance")
    parser.add_argument('--time_min', type=int, default=1, help="Minimum time for operations")
    parser.add_argument('--time_max', type=int, default=100, help="Maximum time for operations")
    parser.add_argument('--block_size', type=int, default=256, help="Block size for CUDA kernel")
    parser.add_argument('--media_dir', type=str, default='/media/NAS/USERS/moonbo/jssp', help="Directory to save generated files")
    
    args = parser.parse_args()

    num_instances_per_task = 1000  # 각 작업당 인스턴스 수
    task_queue = multiprocessing.Queue()
    progress_queue = multiprocessing.Queue()

    for i in range(0, args.num_instances, num_instances_per_task):
        task_queue.put((min(num_instances_per_task, args.num_instances - i), args.num_jobs, args.num_machines, args.time_min, args.time_max, args.block_size))

    processes = []
    total_progress = tqdm(total=args.num_instances, desc="Generating Instances")

    for gpu_id in range(args.num_gpu):
        p = multiprocessing.Process(
            target=worker,
            args=(gpu_id, task_queue, progress_queue, args.media_dir)
        )
        processes.append(p)
        p.start()

    # Update progress bar based on completed tasks
    completed_instances = 0
    while completed_instances < args.num_instances:
        completed = progress_queue.get()
        completed_instances += completed
        total_progress.update(completed)

    for p in processes:
        p.join()

    total_progress.close()
    print("All GPUs have finished processing.")
