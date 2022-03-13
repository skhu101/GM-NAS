__all__ = ['run_cmd', 'popen_cmd', 'kill']

import psutil
import subprocess

def run_cmd(command):
    result = subprocess.run(command.split(), stdout = subprocess.PIPE)
    return result.stdout.decode().strip()

def popen_cmd(command):
    return subprocess.Popen(command, shell = True)

def kill(pid):
    process = psutil.Process(pid)
    for proc in process.children(recursive = True):
        proc.kill()
    process.kill()
