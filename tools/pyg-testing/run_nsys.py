import os

import tools.filesystem as fs
import tools.subprocess as sp
import tools.nvperf as nvperf

import argparse

parser = argparse.ArgumentParser(description="GNN Compiler Project")

parser.add_argument('--warmup', default=5, type=int, 
                    help='Number of warmup iteration')

parser.add_argument('--perf-iter', default=5, type=int, 
                    help='Number of perf iteration')

parser.add_argument('--iter-time', default=5, type=int, 
                    help='Number of perf iteration')

parser.add_argument('--log_dir', type=str, 
                    help='log dir')

parser.add_argument('--run_dir', type=str, 
                    help='run dir')

parser.add_argument('--result_dir', type=str, 
                    help='run dir')

Context = parser.parse_args()


class RunningContext:
    def __init__(self, filename, iter):
        self.running_root_dir = os.path.dirname(os.path.abspath(filename))
        self.running_file_basename = os.path.basename(filename).split('.')[0]

        self.result_root_dir = os.path.join(Context.result_dir, self.running_file_basename)
        fs.create_if_nonexist(self.result_root_dir)
        self.result_dir = os.path.join(self.result_root_dir, self.running_file_basename + str(iter))
        fs.create_if_nonexist(self.result_dir)

        self.nsys_profile_result_file = os.path.join(self.result_dir, self.running_file_basename + '.nsys-rep')

        self.log_dir = os.path.join(Context.log_dir)
        fs.create_if_nonexist(self.log_dir)

        self.logger = logger.getLogger(__file__ + filename + str(iter))
        self.logger.setLevel(logging.DEBUG)
        log_file = os.path.join(self.log_dir, filename + "_e2e.log")
        fh = logger_all_model.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)


    def logging_success(self, result):
        result = result.splitlines()[-2]
        e2e = float(result.split("\n")[-2])
        self.success_time += 1
        self.time_e2e += e2e
        self.logger_any_model.info(self.running_file_basename +  ", [success], " + str(e2e) + "ms")
        self.logging_all_model()

    def logging_error(self, error):
        self.logger_any_model.info(self.running_file_basename +  ", [error], " + self.iter_time )
        self.logging_all_model()

    def logging_unknow(self):
        self.logger_any_model.info(self.running_file_basename +  ", [unknwon error], " + self.iter_time )
        self.logging_all_model()

    def logging_timeout(self):
        self.logger_any_model.info(self.running_file_basename +  ", [timeout], " + self.iter_time )
        self.logging_all_model()



def run_nsys():
    print(Context)
    file_list = fs.get_file_list(Context.run_dir)
    for file_name in file_list:
        for iter in range(0, Context.perf_iter):
            print(file_name)
            running_context = RunningContext(file_name, iter)
            
            nsys_profile_cmd = nvperf.generate_nsys_profile_cmd(
                input_cmd = "python", 
                input_file = file_name,
                result_path = running_context.result_dir
            )
            nsys_profile_cmd = nsys_profile_cmd + 
                '--warmup ' + args.warmup + 
                '--iter-time' + args.iter_time + 
                '--gpu-profile True' 


            nsys_stats_cmd = nvperf.generate_nsys_stats_cmd(
                input_file = running_context.nsys_profile_result_file,
                result_path = running_context.result_dir,
                result_name = running_context.running_file_basename
            )

            sp.run_code(nsys_profile_cmd, running_context)
            sp.run_code(nsys_stats_cmd, running_context)

if __name__ == "__main__":
    run_nsys()
