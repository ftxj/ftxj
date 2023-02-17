import csv
import logging
import os
import pandas as pd
import math

def get_kernel_time(filepath):
    dir_time = {}
    time = 0
    df = pd.read_csv(filepath, usecols=['Duration (ns)','Name'])
    for _, row in df.iterrows():
        kernel_name = row["Name"]
        if "memcpy" in kernel_name:
            continue
        if "memset" in kernel_name:
            continue
        if kernel_name in dir_time:
            dir_time[kernel_name] += float(row['Duration (ns)'])
        else:
            dir_time[kernel_name] = float(row['Duration (ns)'])
        time += float(row['Duration (ns)'])

    return dir_time, time / math.pow(10, 6)

dir, t = get_kernel_time("agnn_graph_fuse.log_gputrace.csv")
for item, v in dir.items():
    print(item, v)
print(t)