

def to_json(x):
    data = x.data()
    print(data)
    import json
    data2 = json.dumps(data)
    print(data)


