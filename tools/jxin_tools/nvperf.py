import os
import logging
import subprocess

from tools.filesystem import *


def generate_nsys_profile_cmd(input_cmd, input_file, result_path):
    profile_cmd = "nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --stop-on-exit=true -f true -o "
    profile_cmd += result_path + " " + input_cmd + " "
    profile_cmd += input_file
    return profile_cmd

def generate_nsys_stats_cmd(input_file, result_path, result_name):
    stats_cmd = "nsys stats --report nvtxppsum,nvtxpptrace,nvtxkernsum,gputrace --force-overwrite true --format csv --timeunit nsec --output "
    result_file = os.path.join(result_path, result_name + "..nsys-rep")
    stats_cmd += result_file  + " " + input_file
    return stats_cmd