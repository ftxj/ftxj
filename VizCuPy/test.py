import time
import torch
import test2
x = torch.randn(10000,10000).to('cuda')


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

# from viztracer import VizTracer
# tracer = VizTracer()



x = x * 2

tracer.start()

x = torch.add(x, x)

x = torch.zeros(20, 20)

test2.mySad(x)

tracer.stop()

# tracer.print()

# tracer.save()

def to_json(x):
    data = x.data()
    import json
    data2 = {}
    data2['traceEvents'] = data
    data2 = json.dumps(data2, indent=4)
    print(data2)

to_json(tracer)

