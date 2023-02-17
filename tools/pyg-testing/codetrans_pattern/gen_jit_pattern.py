import os
import ast
import sys
import re

from tools.codetrans import Pattern, to_json, TransferMethod
import tools.filesystem as fs

# from torch_geometric.nn import AGNNConv
def get_gnn_layer_from_import(line):
    splits = line.split(" ")
    res = []
    for nn in splits[3:]:
        res.append(nn.replace(",",""))
    return res

# class AGNNConv(MessagePassing):
def get_gnn_layer_from_class(line):
    splits = line.split(" ")
    print(splits)
    return splits[1].split("(")[0]

def get_gnn_layer_name_from_import(code_lines):
    for line in code_lines:
        if line.find("from torch_geometric.nn import") != -1:
            return get_gnn_layer_from_import(line)
    return []

def get_gnn_layer_name_from_class(code_lines):
    for line in code_lines:
        if re.match(r"class .*\(MessagePassing\):", line):
            print("Matched")
            return get_gnn_layer_from_class(line)
    return ""

def generate_jit_patterns_impl(nn_names):
    patterns = []
    for nn_name in nn_names:
        pattern = Pattern(
            name = "jit_" + nn_name,
            method = TransferMethod.Rewrite,
            old_str = nn_name + "[(].*?[)]",
            new_str = "~$before_pattern/" + "$pattern/" + ".jittable()/" + "$after_pattern/",
            break_line= "jittable()"
        )
        patterns.append(pattern)
    return patterns


def generate_jit_patterns_from_import(read_filepath):
    fs.line_by_line(read_filepath)
    text_file = open(read_filepath, "r")
    code = text_file.read()
    code_lines = code.split('\n')
    nn_names = get_gnn_layer_name_from_import(code_lines)
    return generate_jit_patterns_impl(nn_names)

def parser_pyg_nn_dir(nn_dir):
    file_list = fs.get_file_list(nn_dir)
    nn_names = []
    for read_filepath in file_list:
        print(read_filepath)
        fs.line_by_line(read_filepath)
        text_file = open(read_filepath, "r")
        code = text_file.read()
        code_lines = code.split('\n')
        nn_name = get_gnn_layer_name_from_class(code_lines)
        if(nn_name != ""):
            nn_names.append(nn_name)
    return generate_jit_patterns_impl(nn_names)
    

pyg_nn_patterns = []

jit_script_pattern = Pattern(
    name = "jit_script_pattern",
    method = TransferMethod.After,
    old_str = r"model(.*)=(.*)to(.*)",
    new_str = '''
# try:
model = torch.jit.script(model)
# except Exception as e:
#     import logging
#     import os
#     result_file_root = "/workspace2/Project/gnn-perf-analysis/data/result/profile_jittable/"
#     log_path = os.path.join(result_file_root, "jittable_result" + ".log")
#     basic_file = os.path.basename(__file__)
#     logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode = 'a', format='%(levelname)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
#     logging.info("[script fail], "  + basic_file + ", " + repr(e))
#     exit()
'''
)

pyg_nn_patterns.append(jit_script_pattern)

pyg_nn_patterns = pyg_nn_patterns + parser_pyg_nn_dir("/workspace/env/pytorch_geometric/torch_geometric/nn")

to_json(pyg_nn_patterns, "/workspace/test/res_dir/jit.json")
