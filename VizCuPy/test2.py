import viztracer

tracer = viztracer.VizTracer()


def fib(a):
    print("fib ", a)
    if a == 1:
        return 1
    if a == 2:
        return 2
    return fib(a - 1) + fib(a - 2)

print("set")
tracer.start()
