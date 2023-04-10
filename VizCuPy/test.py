import time
import torch
import test2


def fib3(x):
    return x
def fib2(x):
    return fib3(x)
def fib(a):
    if a == 1:
        # time.sleep(0.001)
        return fib2(a)
    if a == 2:
        # time.sleep(0.001)
        return fib2(a)
    return fib(a - 1) + fib(a - 2)


import ftxj.profiler as profiler

tracer = profiler.Tracer()

tracer.start()
tracer.enable_cuda()

x = torch.randn(10000,10000).to('cuda')

x = torch.add(x, x)

x = torch.mul(x, x)


test2.mySad(x)

tracer.stop()
tracer.disable_cuda()





def to_json(x):
    data = x.dump()
    k = []
    for l in data:
        k = k + l
    import json
    data2 = {}
    data2['traceEvents'] = k
    data2 = json.dumps(data2, indent=4)
    print(data2)
    # with open("sample6.json", "w") as outfile:
    #     outfile.write(data2)

to_json(tracer)

