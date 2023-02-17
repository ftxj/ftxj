
import os
import jxin_tools.filesystem as fs
import jxin_tools.codetrans as ct

import argparse

parser = argparse.ArgumentParser(description="Dynamo Perf Code Convert")

parser.add_argument('--code-dir', type=str, 
                    help='The models which need to be convert')

parser.add_argument('--result-dir', type=str, 
                    help='result models dir path')

args = parser.parse_args()

class RunningContext:
    def __init__(self, run_file):
        self.running_root_dir = os.path.dirname(os.path.abspath(run_file))
        fs.create_if_nonexist(args.result_dir)
        self.running_file_basename = os.path.basename(run_file)
        self.result_file = os.path.join(args.result_dir, self.running_file_basename)

def run_convert():
    files = fs.get_file_list(args.code_dir, ext="py")
    for file in files:
        print(file)
        running_context = RunningContext(file)
        ct.auto_jit(file, running_context.result_file, "dynamo", "nvfuser")

if __name__ == "__main__":
    run_convert()
