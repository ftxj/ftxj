import ftxj.profiler as profiler
tracer = profiler.Tracer()
import time

tracer.start()

def fib(a):
    if a == 1:
        # time.sleep(0.001)
        return 1
    if a == 2:
        # time.sleep(0.001)
        return 2
    return fib(a - 1) + fib(a - 2)

fib(4)

tracer.stop()

