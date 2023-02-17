
import logging
import argparse
import os
import jxin_tools.filesystem as fs
import jxin_tools.subprocess as sp

parser = argparse.ArgumentParser(description="PyG E2E Model E2E testing")

parser.add_argument('--warmup', default=5, type=int, 
                    help='Number of warmup iteration')

parser.add_argument('--iter-time', default=1, type=int, 
                    help='Number of perf iteration')

parser.add_argument('--perf-time', default=1, type=int, 
                    help='Number of perf iteration')

parser.add_argument('--log_dir', type=str, 
                    help='log dir')

parser.add_argument('--run_dir', type=str, 
                    help='run dir')

parser.add_argument('--result_dir', type=str, 
                    help='run dir')

args = parser.parse_args()

logger_all_model = logging.getLogger(__file__)
logger_all_model.setLevel(logging.DEBUG)
log_dir = args.log_dir
log_file_all_model = os.path.join(log_dir, "1-all_model_e2e" + ".log")
fh = logging.FileHandler(log_file_all_model)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(message)s')
fh.setFormatter(formatter)
logger_all_model.addHandler(fh)

class RunningContext:
    def __init__(self, filename):
        self.running_file_basename = os.path.basename(filename)


        self.time_e2e = 0
        self.success_time = 0
        self.iter_time = 0
        self.logger_any_model = logging.getLogger(__file__ + filename)
        self.logger_any_model.setLevel(logging.DEBUG)
        log_file_any_model = os.path.join(log_dir, self.running_file_basename + "_e2e.log")
        fh = logging.FileHandler(log_file_any_model)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(message)s')
        fh.setFormatter(formatter)

        self.logger_any_model.addHandler(fh)
        self.logger_all_model = logger_all_model

    def update_before_run_code(self):
        self.iter_time = self.iter_time + 1

    def logging_all_model(self):
        if(self.iter_time == args.perf_time):
            succ_str = ", [success], "
            if (self.success_time > 0):
                succ_str += str(self.time_e2e / self.success_time) + "ms"
            else:
                succ_str += "None"
            fail_str = ", [fail], " + str(iter_time - self.success_time)
            self.logger_all_model.info(
                self.running_file_basename + 
                succ_str +
                fail_str
            )

    def logging_success(self, result):
        result = result.splitlines()[-2]
        e2e = float(result.split("\n")[-2])
        self.success_time += 1
        self.time_e2e += e2e
        self.logger_any_model.info(self.running_file_basename +  ", [success], " + str(e2e) + "ms")
        self.logging_all_model()

    def logging_error(self, error):
        self.logger_any_model.info(self.running_file_basename +  ", [error], " + str(self.iter_time) )
        self.logging_all_model()

    def logging_unknow(self):
        self.logger_any_model.info(self.running_file_basename +  ", [unknwon error], " + str(self.iter_time) )
        self.logging_all_model()

    def logging_timeout(self):
        self.logger_any_model.info(self.running_file_basename +  ", [timeout], " + str(self.iter_time) )
        self.logging_all_model()

def run_e2e():
    file_names = fs.get_file_list(args.run_dir)
    for file_name in file_names:
        running_context = RunningContext(file_name)
        for iter in range(args.perf_time):
            cmd = "python " + file_name + \
                " --warmup " + str(args.warmup) + \
                " --iter-time " + str(args.iter_time)
            sp.run_code(cmd, running_context)

if __name__ == "__main__":
    run_e2e()
