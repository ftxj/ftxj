
import ftxj.profiler as profiler

tracer = profiler.Tracer()

tracer.start()

tracer.timeline_split()


tracer.stop()


def to_json(x):
    data = x.dump()
    import json
    data2 = json.dumps(data, indent=4)
    print(data2)

to_json(tracer)

