import numpy as np
import h5py
import os
from tqdm import tqdm

# import matplotlib.pyplot as plt

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

# 스케줄링 함수
def schedule_jobs(times, machines, heuristic):
    num_jobs, num_machines = times.shape

    # 작업 정렬
    if heuristic == "FIFO":
        job_indices = np.arange(num_jobs)
    elif heuristic == "LIFO":
        job_indices = np.arange(num_jobs)[::-1]
    elif heuristic == "LOR":
        job_indices = np.argsort(times[:, 0])[::-1]
    elif heuristic == "LPT":
        job_indices = np.argsort(-times.sum(axis=1))
    elif heuristic == "LTPT":
        job_indices = np.argsort(-times.max(axis=1))
    elif heuristic == "MOR":
        job_indices = np.argsort(times[:, 0])
    elif heuristic == "SPT":
        job_indices = np.argsort(times.sum(axis=1))
    elif heuristic == "STPT":
        job_indices = np.argsort(times.max(axis=1))
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    # times와 machines 배열을 정렬된 인덱스로 재정렬
    times = times[job_indices]
    machines = machines[job_indices]

    # 기계 별로 현재 작업이 완료되는 시간을 저장
    machine_completion_times = np.zeros(machines.max() + 1)
    # 각 작업의 시작 시간과 완료 시간을 저장
    job_start_times = np.zeros_like(times)
    job_completion_times = np.zeros_like(times)

    # 모든 작업을 순차적으로 처리
    for job in range(num_jobs):
        for task_index in range(num_machines):
            machine = machines[job][task_index]
            task_time = times[job][task_index]
            # 현재 기계에서 가장 빨리 끝나는 시간을 확인 후 해당 작업 시작
            if task_index == 0:
                start_time = machine_completion_times[machine]
            else:
                start_time = max(machine_completion_times[machine], job_completion_times[job][task_index - 1])
            job_start_times[job][task_index] = start_time
            job_completion_times[job][task_index] = start_time + task_time
            # 기계의 완료 시간 업데이트
            machine_completion_times[machine] = job_completion_times[job][task_index]

    # 모든 작업의 마지막 작업 완료 시간 중 최대값이 Cmax
    cmax = np.max(job_completion_times[:, -1])
    return cmax, job_start_times, job_completion_times

if __name__ == '__main__':
    times, machines = load_data()
    print(times.shape, machines.shape)


heuristics = ["FIFO", "LIFO", "LOR", "LPT", "LTPT", "MOR", "SPT", "STPT"]
results = []

for instance_idx, (time, machine) in tqdm(enumerate(zip(times, machines)), total=times.shape[0]):
    instance_results = {}
    for heuristic in heuristics:
        cmax, job_start_times, job_completion_times = schedule_jobs(time, machine, heuristic)
        instance_results[heuristic] = cmax
    
    # Cmax가 가장 낮은 heuristic 찾기
    best_heuristic = min(instance_results, key=instance_results.get)
    best_heuristic_idx = heuristics.index(best_heuristic)
    results.append(int(best_heuristic_idx))
    

# 데이터를 x, y로 저장
x = np.stack((times, machines), axis=-1)
y = results
print(results[:3])
# 데이터를 .npz 파일로 저장
np.savez('/media/NAS/USERS/moonbo/jssp/jssp_supervision_data.npz', x=x, y=y)


# # 시각화
# fig, axs = plt.subplots(len(heuristics) + 1, 1, figsize=(12, 3 * (len(heuristics) + 1)))

# def plot_schedule(ax, job_start_times, job_completion_times, title):
#     num_jobs, num_machines = job_start_times.shape
#     colors = plt.get_cmap('tab20').colors
#     for i in range(num_jobs):
#         for j in range(num_machines):
#             start_time = job_start_times[i][j]
#             duration = job_completion_times[i][j] - start_time
#             ax.broken_barh([(start_time, duration)], (j*10+5, 8), facecolors=colors[i % len(colors)])
#     ax.set_ylim(0, num_machines*10+5)
#     ax.set_xlim(0, job_completion_times.max()+10)
#     ax.set_xlabel('Processing time')
#     ax.set_ylabel('Machine')
#     ax.set_yticks([10*i + 9 for i in range(num_machines)])
#     ax.set_yticklabels([f'Machine{i+1}' for i in range(num_machines)])
#     ax.set_title(title)
#     ax.grid(True)

# 각 알고리즘에 대해 스케줄 시각화
# for i, heuristic in enumerate(heuristics):
#     cmax, job_start_times, job_completion_times = results[heuristic]
#     plot_schedule(axs[i], job_start_times, job_completion_times, f'{heuristic} Scheduling (Cmax: {cmax})')

# # 범례 추가
# colors = plt.get_cmap('tab20').colors
# handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i % len(colors)]) for i in range(num_jobs)]
# axs[-1].legend(handles, [f'Job {i+1}' for i in range(num_jobs)], loc='center', ncol=5, fontsize='small', title='Job Legend')
# axs[-1].axis('off')

# plt.tight_layout()
# plt.show()