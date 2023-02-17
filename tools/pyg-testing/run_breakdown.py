import csv
import logging
import os
import pandas as pd
import math

items = [
    ["memory", ["[CUDA memcpy DtoH]", "[CUDA memcpy HtoD]", "[CUDA memset]", "[CUDA memcpy DtoD]"]],
    ["spmm&sddmm&spspmm" , ["spmm_kernel", "sddmm", "csrgemm", "cusparseIinclusive"]],
    ["elementwise", ["at::native::vectorized_elementwise_kernel", "at::native::unrolled_elementwise_kernel"]],
    ["gemv", ["gemv2N_kernel", "gemvk1_kernel", "gemv2T_kernel", "gemvx::kernel"]],
    ["gemm", ["cutlass::Kernel", "splitKreduce_kernel", "ampere_sgemm", "gemmk1_kernel"]],
    ["sort", ["bitonicSortKInPlace", "cub::DeviceScan", "sort_postprocess_kernel", "DeviceRadixSort", 
                "cub::DeviceCompactInitKernel", "cub::DeviceReduceKernel", 
                "cub::DeviceSelectSweepKernel", "cub::DeviceReduceSingleTileKernel"]],
    ["index", ["indexSelect", "at::native::index_elementwise", "at::native::indexAdd", "::elementwise_kernel_with_index<", "indexing_backward_kernel"]],
    ["scatter", ["void scatter_kernel", "scatter_arg_kernel"]],
    ["reduce", ["at::native::reduce_kernel"]],
    ["scatter_gather", ["at::native::_scatter_gather_elementwise"]],
    ["spline", ["spline_weighting", "spline_basis"]],
    ["known others", ["Imemset", "CatArrayBatchedCopy", "cunn_SoftMax", "fused_dropout_kernel_vec", "softmax_warp_forward", "nll_loss_forward_reduce", "nll_loss_backward_reduce", "softmax_warp_backward"]],
]




# e2e_file_name = "/home/scratch.jxin_gpu/Project/gnn-perf-analysis/data/result/profile2/e2e2.time.log"
# e2e_file = open(e2e_file_name, "r")
# e2e_file_string = e2e_file.read()
# e2e_list = e2e_file_string.split('\n')
# def get_e2e_time(name):
#     e2e_time = 0
#     for idx, line in enumerate(e2e_list):
#         if(line.find(name) != -1):
#            if(line.find("E2E = ") != -1):
#                 e2e_time = float(line[line.find("E2E = ") + 6:])
#     return e2e_time


def get_kernel_time(filepath):
    kernel_times = [0.0] * len(items)
    df = pd.read_csv(filepath, usecols=['Duration (ns)','Name'])
    for _, row in df.iterrows():
        kernel_name = row["Name"]
        for idx, item in enumerate(items):
            for kernel in item[1]:
                if(kernel_name.find(kernel) != -1):
                    # print(kernel_name)
                    # print(kernel)
                    # print(float(row['Duration (ns)']))
                    kernel_times[idx] += float(row['Duration (ns)'])

    for i in range(0, len(kernel_times)):
        kernel_times[i] = kernel_times[i] / math.pow(10, 6)
    return kernel_times


def get_gpu_time(filepath):
    gpu_time = 0.0
    df = pd.read_csv(filepath, usecols=['Duration (ns)','Name'])
    for index, row in df.iterrows():
        gpu_time += float(row['Duration (ns)'])
    return gpu_time / math.pow(10, 6)

# x = get_kernel_time("/home/scratch.jxin_gpu/Project/gnn-perf-analysis/data/result/profile2/agnn/agnn_gputrace.csv")
# gpu_time = get_gpu_time("/home/scratch.jxin_gpu/Project/gnn-perf-analysis/data/result/profile2/agnn/agnn_gputrace.csv")

# total = 0.0
# for idx, item in enumerate(items):
#     total += x[idx] / gpu_time * 100
#     print(item[0], x[idx] / gpu_time * 100)
# print(total)
# exit()
log_file_dir = "/home/scratch.jxin_gpu/Project/gnn-perf-analysis/data/result/profile_no_jittable"
log_file_name = os.path.join(log_file_dir, "kernel_breakdown.log")
logging.basicConfig(filename=log_file_name, level=logging.DEBUG, filemode = 'w', format='%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
run_files_dir = "/home/scratch.jxin_gpu/Project/gnn-perf-analysis/data/result/profile_no_jittable"

iter = os.walk(run_files_dir)
log = "File, GPU(ms)"
for idx, item in enumerate(items):
    log += ", " + item[0] + "%"
logging.info(log)

for path, dir_list, file_list in iter:
    for dir_name in dir_list:
        try:
            run_dir = os.path.join(run_files_dir, dir_name) 
            gpu_trace_file = os.path.join(run_dir, dir_name + "_gputrace.csv")
            module_file = dir_name + ".py"
            
            print(gpu_trace_file)

            # e2e_time = get_e2e_time(module_file)

            gpu_time = get_gpu_time(gpu_trace_file)

            kernel_time = get_kernel_time(gpu_trace_file)

            total = 0.0
            log = module_file + ", " + str(gpu_time)
            for idx, item in enumerate(items):
                t = kernel_time[idx] / gpu_time * 100
                total += t
                log += ", " + str(t)
            log += ", " + str(total)
            logging.info(log)
        except Exception as e:
            logging.info(str(dir_name + ".py") + str(e)[0:10])


if __name__ == "__main__":
    run_breakdown()