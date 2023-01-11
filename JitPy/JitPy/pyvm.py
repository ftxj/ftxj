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

class ByteCodeFn(object):
    def __init__(self, name, code):
        self.name = name
        self.bin = dis.Bytecode(code)
        
class Frame(object):
    def __init__(self, locals={}, globals={}, fast=[]):
        self.block_stack = []
        self.data_stack = []
        self.fast = fast
        self.f_locals = locals
        self.f_globals = globals
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
        self.frames = []
        self.frames.append(Frame())
        self.instructions = []
        self.pc = []

    def push(self, elm):
        self.frames[-1].push_data(elm)

    def pop(self):
        return self.frames[-1].pop_data()

    def binaryOp(self, op):
        lhs = self.pop()
        rhs = self.pop()
        self.push(binaryOperationList[op](lhs, rhs))
    
    def LOAD_CONST(self, inst):
        name = inst.argval
        self.push(name)

    def LOAD_NAME(self, inst):
        frame = self.frames[-1]
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

    def LOAD_FAST(self, inst):
        val = self.frames[-1].fast[-1]
        self.frames[-1].fast.pop()
        self.push(val)

    def STORE_NAME(self, inst):
        name = inst.argval
        self.frames[-1].f_locals[name] = self.pop()

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
        fn.bin(tuple(params))
    
    def CALL_FUNCTION(self, inst):
        num_params = inst.arg
        params = []
        for i in range(num_params):
            params.append(self.pop())
        fn = self.pop()

        if(isinstance(fn, ByteCodeFn)):
            frame = Frame(globals=self.frames[-1].f_globals, fast=params)
            self.frames.append(frame)
            self.push(self.run(fn.bin))
        else:
            params.reverse()
            self.push(fn.bin(tuple(params)))

    def RESUME(self, inst):
        pass
    def PRECALL(self, inst):
        pass

    def RETURN_VALUE(self, inst):
        return self.pop()

    def MAKE_FUNCTION(self, inst):
        name = self.pop()
        code = self.pop()
        self.push(ByteCodeFn(name, code))

    def IMPORT_NAME(self, inst):
        fromlist = self.pop()
        level = self.pop()
        frame = self.frames[-1]
        self.push(
            __import__(inst.argval, frame.f_globals, frame.f_locals, fromlist, level)
        )

    def LOAD_METHOD(self, inst):
        frame = self.frames[-1]
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
        inst = self.instructions[-1][self.pc[-1]]
        # print(inst.opname)
        return getattr(self, inst.opname)(inst)

    def run(self, code):
        self.pc.append(0)
        if(isinstance(code, dis.Bytecode)):
            self.instructions.append(list(map(convert_instruction, code)))
        else:
            print(dis.dis(code))
            self.instructions.append(list(map(convert_instruction, dis.get_instructions(code))))
        ret = None
        while(self.pc[-1] < len(self.instructions[-1])):
            ret = self.step()
            self.pc[-1] = self.pc[-1] + 1
        self.pc.pop()
        self.instructions.pop()
        self.frames.pop()
        return ret

def test():
    code = '''
import torch
a = torch.randn(3,3)
b = torch.randn(3,3)
def foo(a):
    return a
print(a + b)
print(foo(23) + foo(33))
'''
    vm = VirtualMachine()
    vm.run(code)

test()