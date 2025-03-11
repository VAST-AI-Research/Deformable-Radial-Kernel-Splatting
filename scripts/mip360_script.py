import os
import multiprocessing
from queue import Empty
from tensorboard.backend.event_processing import event_accumulator

def divide_chunks(l, n):
    o = []
    for i in range(0, len(l), n): 
        o.append(l[i:i + n])
    return o


data_path = 'PATH_TO_MIPNERF360_DATASET'
output_path = 'output/mipnerf360/'


flag_path = os.path.join(output_path, 'flag')
if not os.path.exists(flag_path):
    os.makedirs(flag_path)


def run_exp(gpu, task_queue):
    while True:
        try:
            data, method, density = task_queue.get_nowait()
        except Empty:
            break

        if not os.path.exists(os.path.join(flag_path)):
            return
        if method == 'GS':
            cmd = f'CUDA_VISIBLE_DEVICES={gpu} python train.py -s {os.path.join(data_path, data)} -m {os.path.join(output_path, data)} --eval --gs_type {method} --metric_mask --is_unbounded --pure_train'
            print('\n\n'+cmd)
            os.system(cmd)
        else:
            cmd = f'CUDA_VISIBLE_DEVICES={gpu} python train.py -s {os.path.join(data_path, data)} -m {os.path.join(output_path, data)}_{density} --eval --gs_type {method} --kernel_density {density} --metric_mask --is_unbounded --pure_train'
            print('\n\n'+cmd)
            os.system(cmd)

        if not os.path.exists(os.path.join(flag_path)):
            return
        if method == 'GS':
            cmd = f'CUDA_VISIBLE_DEVICES={gpu} python train.py -s {os.path.join(data_path, data)} -m {os.path.join(output_path, data)}_{density} --eval --gs_type {method} --metric --metric_mask --is_unbounded --pure_train'
            print('\n\n'+cmd)
            os.system(cmd)
        else:
            cmd = f'CUDA_VISIBLE_DEVICES={gpu} python train.py -s {os.path.join(data_path, data)} -m {os.path.join(output_path, data)}_{density} --eval --gs_type {method} --kernel_density {density} --metric --load_iteration -1 --metric_mask --is_unbounded --pure_train'
            print('\n\n'+cmd)
            os.system(cmd)


datasets = sorted(os.listdir(data_path))
density_list = ['dense', 'middle', 'sparse']
task_list = []
for data in datasets:
    task_list.append([data, 'GS', 'None'])
    for density in density_list:
        task_list.append([data, 'DRK', density])


gpus = [0, 1, 2, 3, 4, 5, 6, 7] * 1


# Create a multiprocessing queue and add all tasks
task_queue = multiprocessing.Queue()
for task in task_list:
    task_queue.put(task)

# Create and start a process for each GPU
processes = []
for gpu in gpus:
    p = multiprocessing.Process(target=run_exp, args=(gpu, task_queue))
    processes.append(p)
    p.start()

# Wait for all processes to finish
for p in processes:
    p.join()