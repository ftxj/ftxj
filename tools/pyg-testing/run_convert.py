
import os
import jxin_tools.filesystem as fs
import jxin_tools.codetrans as ct

import argparse

parser = argparse.ArgumentParser(description="GNN Model Code Convert")

parser.add_argument('--pattern-dir', type=str, 
                    help='Code Transfer Template Dir')

parser.add_argument('--code-dir', type=str, 
                    help='The model which need to be convert')

parser.add_argument('--result-dir', type=str, 
                    help='New model Dir')

args = parser.parse_args()

class RunningContext:
    def __init__(self, run_file):
        self.running_root_dir = os.path.dirname(os.path.abspath(run_file))
        fs.create_if_nonexist(args.result_dir)
        self.running_file_basename = os.path.basename(run_file)
        self.result_file = os.path.join(args.result_dir, self.running_file_basename)

def run_convert():
    patterns = ct.get_codetrans_patterns(args.pattern_dir)
    files = fs.get_file_list(args.code_dir, ext="py")
    print(files)
    for file in files:
        print(file)
        running_context = RunningContext(file)
        fs.line_by_line(file)
        f = open(file, 'r')
        code_line = f.read().split('\n')
        for pattern in patterns:
            code_line = pattern.run_codetrans(code_line)
        fs.write_file(running_context.result_file, '\n'.join(code_line))

if __name__ == "__main__":
    run_convert()
