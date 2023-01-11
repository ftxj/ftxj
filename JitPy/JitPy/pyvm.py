import dis
import inspect
import operator
from typing import Any, List, Optional
import sys 
class BuildinFunctions(object):
    def __init__(self, name, bin):
        self.name = name
        self.bin = bin

class Functions(object):
    def __init__(self, name, bin):
        self.name = name
        self.bin = bin

class Frame(object):
    def __init__(self):
        self.block_stack = []
        self.data_stack = []
        self.f_locals = {}
        self.f_globals = {}
        self.f_builtins = {}
        self.f_builtins["print"] = BuildinFunctions("print", print)

    def push_data(self, elm):
        self.data_stack.append(elm)

    def pop_data(self):
        data = self.data_stack[-1]
        self.data_stack.pop()
        return data

class Instruction(object):
    def __init__(self, opcode, opname, argval, argrepr, arg):
        self.opcode = opcode
        self.opname = opname
        self.argval = argval
        self.argrepr = argrepr
        self.arg = arg

def convert_instruction(i: dis.Instruction):
    return Instruction(
        i.opcode,
        i.opname,
        i.argval,
        i.argrepr,
        i.arg
    )

binaryOperationList = {
    'POWER':    pow,
    'MULTIPLY': operator.mul,
    'DIVIDE':   getattr(operator, 'div', lambda x, y: None),
    'FLOOR_DIVIDE': operator.floordiv,
    'TRUE_DIVIDE':  operator.truediv,
    'MODULO':   operator.mod,
    'ADD':      operator.add,
    '+':      operator.add,
    'SUBTRACT': operator.sub,
    'SUBSCR':   operator.getitem,
    'LSHIFT':   operator.lshift,
    'RSHIFT':   operator.rshift,
    'AND':      operator.and_,
    'XOR':      operator.xor,
    'OR':       operator.or_,
}

class VirtualMachine(object):
    def __init__(self):
        self.frame = []
        self.frame.append(Frame())
        self.instructions = []
        self.pc = 0

    def push(self, elm):
        self.frame[0].push_data(elm)

    def pop(self):
        return self.frame[0].pop_data()

    def binaryOp(self, op):
        lhs = self.pop()
        rhs = self.pop()
        self.push(binaryOperationList[op](lhs, rhs))
    
    def LOAD_CONST(self, inst):
        name = inst.argval
        self.push(name)

    def LOAD_NAME(self, inst):
        frame = self.frame[0]
        name = inst.argval
        if name in frame.f_locals:
            val = frame.f_locals[name]
        elif name in frame.f_globals:
            val = frame.f_globals[name]
        elif name in frame.f_builtins:
            val = frame.f_builtins[name]
        else:
            raise NameError("name '%s' is not defined" % name)
        self.push(val)

    def STORE_NAME(self, inst):
        name = inst.argval
        self.frame[-1].f_locals[name] = self.pop()

    def BINARY_OP(self, inst):
        self.binaryOp(inst.argrepr)
    
    def BINARY_ADD(self, inst):
        self.binaryOp('ADD')
    
    def POP_TOP(self, inst):
        self.pop()

    def PUSH_NULL(self, inst):
        self.push(None)

    def CALL(self, inst):
        num_params = inst.arg
        params = []
        for i in range(num_params):
            params.append(self.pop())
        params.reverse()
        fn = self.pop()
        if fn.name in self.frame[-1].f_builtins:
            fn.bin(tuple(params))
    
    def CALL_FUNCTION(self, inst):
        num_params = inst.arg
        params = []
        for i in range(num_params):
            params.append(self.pop())
        params.reverse()
        fn = self.pop()
        self.push(fn.bin(tuple(params)))

    def RESUME(self, inst):
        pass
    def PRECALL(self, inst):
        pass

    def RETURN_VALUE(self, inst):
        self.pop()

    def MAKE_FUNCTION(self, inst):
        fn = self.pop()

    def IMPORT_NAME(self, inst):
        fromlist = self.pop()
        level = self.pop()
        frame = self.frame[-1]
        self.push(
            __import__(inst.argval, frame.f_globals, frame.f_locals, fromlist, level)
        )


    def LOAD_METHOD(self, inst):
        frame = self.frame[-1]
        name = inst.argval
        namespace_num = inst.arg
        namespace = []
        for i in range(namespace_num):
            namespace.append(self.pop())
        namespace.reverse()
        namespace.append(name)
        method = getattr(namespace[0], name)
        self.push(Functions(name, method))

    def CALL_METHOD(self, inst):
        num_params = inst.arg
        params = []
        for i in range(num_params):
            params.append(self.pop())
        params.reverse()
        fn = self.pop()
        self.push(fn.bin(tuple(params)))

    def step(self):
        inst = self.instructions[self.pc]
        getattr(self, inst.opname)(inst)

    def run(self, code):
        print(dis.dis(code))
        self.instructions = list(map(convert_instruction, dis.get_instructions(code)))
        self.pc = 0
        while(self.pc < len(self.instructions)):
            self.step()
            self.pc = self.pc + 1

def test():
    code = '''
import torch
a = torch.randn(3,3)
b = torch.randn(3,3)
print(a + b)
'''
    vm = VirtualMachine()
    vm.run(code)

test()