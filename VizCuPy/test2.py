
import ftxj.profiler as profiler

tracer = profiler.Tracer()

tracer.start()

tracer.timeline_split()


tracer.stop()


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

to_json(tracer)

