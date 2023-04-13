

def to_json(x):
    data = x.data()
    print(data)
    import json
    data2 = json.dumps(data)
    print(data)

    data["displayTimeUnit"] = "ns"
    k["version"] = "VizCuPy 0.0.1"
    data["otherData"] = k


