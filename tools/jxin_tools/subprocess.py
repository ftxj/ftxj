
import os
import subprocess

def get_error_from_stderr(str):
    strs = str.split('\\n')
    for idx, s in enumerate(strs):
        if(s.find("Error") != -1):
            return '\n'.join(strs[idx:])
    return str

def run_code(cmd, context):
    try:
        print(cmd)
        my_env = os.environ.copy()
        res = subprocess.check_output(cmd.split(" "), timeout = 1000, stderr=subprocess.STDOUT, env=my_env).decode("utf-8")
        context.logging_success(res)
    except subprocess.CalledProcessError as exc:
        str = exc.output.__str__()
        str = get_error_from_stderr(str)
        context.logging_error(str)
    except subprocess.TimeoutExpired as exc:
        str = exc.output.__str__()
        context.logging_timeout(str)
    except:
        context.logging_unknow()