import threading
import os
import sys
import json
import subprocess
from termcolor import colored
import argparse
import time
cap = 1
cmd_idx = 0
gpus = [1,3,4,5,6,7]

DATA_BASE = 'demo_file/zjh'
CMD_BASE = 'python test_video_swapsingle.py --isTrain false --use_mask --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path {} --video_path {} --output_path ./output/{}.mp4 --temp_path ./temp_results/{}'
images = os.listdir(f'{DATA_BASE}/images')
videos = os.listdir(f'{DATA_BASE}/videos')
print(images)
print(videos)
cmds = []
for img in images:
    for vid in videos:
        img_root = os.path.splitext(img)[0]
        vid_root = os.path.splitext(vid)[0]
        out_root = f"{img_root}_{vid_root}"
        cmd = CMD_BASE.format(f'{DATA_BASE}/images/{img}', f'{DATA_BASE}/videos/{vid}', out_root, out_root)
        cmds.append(cmd)

def popen_and_call(on_exit, exit_args, popen_args, popen_kwargs):
    """
    Runs the given args in a subprocess.Popen, and then calls the function
    on_exit when the subprocess completes.
    on_exit is a callable object, and popen_args is a list/tuple of args that
    would give to subprocess.Popen.
    """
    # def run_in_thread(on_exit, exit_args, popen_args, popen_kwargs):
    def run_in_thread():
        proc = subprocess.Popen(*popen_args, **popen_kwargs)
        proc.wait()
        on_exit(*exit_args)
        return
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    # returns immediately after the thread starts
    return thread


def call_next(local_rank, prev_cmd=None, start_time=None):
    global cmd_idx, walking, cmds, logs
    if prev_cmd is not None:
        took = time.time()-start_time
        logs.append((prev_cmd, took))
        print(colored(f"Command: {prev_cmd} took: {took}", "blue"))

    # if cmd_idx < len(cmds):  # call on available gpus
    cmd = cmds[cmd_idx]
    print(f"Running command:{colored(f'#{cmd_idx+1}/#{len(cmds)} on gpu: {local_rank}', 'magenta')}\n{colored(cmd, 'cyan')}")
    popen_and_call(call_next, [local_rank, cmd, time.time()], [cmd], dict(shell=True, env={
        **os.environ,
        "CUDA_VISIBLE_DEVICES": f'{local_rank}'
    }))  # returns immediately after the thread starts
    cmd_idx += 1

    if cmd_idx == len(cmds):
        with open('auto_test.log', 'w') as f:
            for log in logs:
                f.write(f'{log[0]}, {log[1]}')

logs = []

print(colored(f'Running evaluation tasks on GPUS: {gpus}', "yellow"))
for gpu in gpus:
    for _ in range(cap):
        call_next(gpu)
