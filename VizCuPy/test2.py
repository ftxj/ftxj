import torch

def mySad(x):
    print("3")
    x = torch.add(x, x)
    x = x * 2
    return x