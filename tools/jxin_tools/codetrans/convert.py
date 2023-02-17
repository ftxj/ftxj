
import os
import jxin_tools.filesystem as fs

import argparse
import jxin_tools.codetrans.pattern as pattern
import jxin_tools.codetrans.perf_pattern as perf_pattern


frontend_list = [
    "dynamo"
]

backend_list = [
    "nvfuser"
]



def auto_jit(old_file, new_file, frontend, backend, perf = True):
    if frontend not in frontend_list or backend not in backend_list:
        raise NotImplementedError
    
    pattern_run = []
    if frontend == 'dynamo' and backend == 'nvfuser':
        pattern_run.append(pattern.Pattern(
                    name = "dynamo_nvfuser_decorate",
                    method = pattern.TransferMethod.Before,
                    old_str = "def forward",
                    new_str = '''
@torch._dynamo.optimize("ts_nvfuser")
'''))

    pattern_run = pattern_run + perf_pattern.perf_patterns

    fs.line_by_line(old_file)
    f = open(old_file, 'r')
    code_line = f.read().split('\n')
    for p in pattern_run:
        code_line = p.run_codetrans(code_line)
    fs.write_file(new_file, '\n'.join(code_line))


