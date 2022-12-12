import torch
import inspect
from abc import abstractmethod
import os
import sys
import os.path as osp
import importlib
class Base1(torch.nn.Module):
    @abstractmethod
    def forward(self, x):
        """ABC forward"""

class Base2(torch.nn.Module):
    @abstractmethod
    def forward(self, x):
        """ABC forward"""

class Net1(Base1):
    def __init__(self):
        super(Net1, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self, x):
        return self.linear(x)


class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

class Net(torch.nn.Module):
    def __init__(self, flag):
        super(Net, self).__init__()
        self.layer = torch.nn.ModuleList()

        if(flag):
            self.layer.append(Net1())
            self.layer.append(Net2())
        else:
            self.layer.append(Net1())
            self.layer.append(Net2())
            self.make_ts_happy = [Net1(), Net2]

    def forward(self, x):
        b = x
        for layer in self.layer:
            if isinstance(layer, Base1):
                x = x + layer(x)
            elif isinstance(layer, Base2):
                x = x * layer(x)
        return b

    @torch.jit.unused
    def jittable(self):
        try:
            from jinja2 import Template
        except ImportError:
            raise ModuleNotFoundError(
                "No module named 'jinja2' found on this machine. "
                "Run 'pip install jinja2' to install the library.")

        root = os.path.dirname(osp.realpath(__file__))
        with open(osp.join(root, 'ts.jinja'), 'r') as f:
            template = Template(f.read())

        source = inspect.getsource(self.__class__)
        real_doing = ""
        for idx in range(len(self.layer)):
            if isinstance(self.layer[idx], Base1):
                real_doing = real_doing + "        x = x + self.layer[idx](x)"
            elif isinstance(self.layer[idx], Base2):
                real_doing = real_doing + "        x = x * self.layer[idx](x)"
        cls_name = "NetJit"
        jit_module_repr = template.render(
            real_doing=real_doing,
            cls_name=cls_name
        )
        cls = class_from_module_repr(cls_name, jit_module_repr)
        module = cls.__new__(cls)
        module.__dict__ = self.__dict__.copy()
        module.jittable = None
        return module

def class_from_module_repr(cls_name, module_repr):
    f = open("/workspace/github/examples/jit.py", 'w')
    f.write(module_repr)
    spec = importlib.util.spec_from_file_location(cls_name, "/workspace/github/examples/jit.py")
    mod = importlib.util.module_from_spec(spec)

    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)

    return getattr(mod, cls_name)
    

model = Net(False)
scripted = torch.jit.script(model.jittable())

print(scripted.graph)


x = torch.randn(10,10)
print(scripted(x))