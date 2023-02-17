import torch
from torch.nn import Linear

class TestM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(Linear(2, 2))
        self.linears.append(Linear(2, 2))
        self.linears.append(Linear(2, 2))

    def forward(self, input):
        x = self.linears[:-1]
        for i in x:
            out = i(input)

# fn1 = torch.jit.script(TestL())

fn2 = torch.jit.script(TestM())


