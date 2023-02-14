import torch
from torch import Tensor
import prettytable as pt
import time
from dataclasses import dataclass

@dataclass
class FnWrapper:
    eager_fn = None
    nvfuser_fn = None
    triton_fn = None
    name = ""

class Pattern(object):
    def __init__(self):
        self.profile_fn_list = []
        self.str_data = "Please Give me data name"
        self.name = "Please Give me name"
        self.shape = []
        self.warm_up = 20
        self.test_run = 100
        self.summary = pt.PrettyTable(["Data", "Pattern", "eager", "triton", "nvfuser", "nvfuser vs triton"])

    def benchmark(self, fn, *args):
        
        # torch.cuda.nvtx.range_push(self.name + "warmup")
        for i in range(self.warm_up):
            fn(*args)
        torch.cuda.nvtx.range_pop()

        # torch.cuda.synchronize()

        # torch.cuda.nvtx.range_push(self.name)
        start = time.time()
        for i in range(self.test_run):
            fn(*args)
        torch.cuda.synchronize()
        # torch.cuda.nvtx.range_pop()

        stop = time.time()

        time.sleep(3)
        return stop - start
    
    def init_data():
        pass

    def gen_jit_fn(self):
        pass

    def check(self, f, *args):
        e = f.eager_fn(*args)
        if(f.nvfuser_fn != None):
            o = f.nvfuser_fn(*args)
            assert torch.allclose(o, e, atol=1e-4)
        if(f.triton_fn != None):
            o = f.triton_fn(*args)
            assert torch.allclose(o, e, atol=1e-4)

    def run(self):
        self.gen_fn()
        for test_id in range(len(self.shape)):
            data = self.init_data(test_id)
            for fn in self.profile_fn_list:
                self.check(fn, *data)
                if fn.eager_fn != None:
                    t_eager = self.benchmark(fn.eager_fn, *data)
                    print("eager", t_eager)
                if fn.nvfuser_fn != None:
                    t_nvfuser = self.benchmark(fn.nvfuser_fn, *data)
                    print("t_nvfuser", t_nvfuser)
                if fn.triton_fn != None:
                    t_triton = self.benchmark(fn.triton_fn, *data)
                    print("t_triton", t_triton)
                t1, t2, t3 = self.gen_summary(t_eager, t_nvfuser, t_triton)
                self.summary.add_row([self.str_data, self.name + fn.name, 1, t1, t2, t3])
        print(self.summary)
    

    def gen_summary(self, eager, nvfuser, triton):
        triton_eager, nvfuser_eager, nvfuser_triton = "/", "/", "/"
        if nvfuser != None:
            nvfuser_eager = str(round(eager / nvfuser, 2)) + "x"
        if triton != None:
            triton_eager = str(round(eager / triton, 2)) + "x"
        if triton != None and nvfuser != None:
            nvfuser_triton = str(round(triton / nvfuser, 2)) + "x"
        return triton_eager, nvfuser_eager, nvfuser_triton
