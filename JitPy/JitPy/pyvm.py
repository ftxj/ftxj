import dis
import inspect
import operator
from typing import Any, List, Optional

class Frame(object):
    def __init__(self):
        self.block_stack = []
        self.data_stack = []
    
    def push_data(self, elm):
        self.data_stack.append(elm)

    def pop_data(self):
        data = self.data_stack[-1]
        self.data_stack.pop()
        return data

class Instruction(object):
    def __init__(self, opcode, argval):
        self.opcode = opcode
        self.argval = argval

binaryOperationList = {
    'POWER':    pow,
    'MULTIPLY': operator.mul,
    'DIVIDE':   getattr(operator, 'div', lambda x, y: None),
    'FLOOR_DIVIDE': operator.floordiv,
    'TRUE_DIVIDE':  operator.truediv,
    'MODULO':   operator.mod,
    'ADD':      operator.add,
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

    def run(self, code):
        pass

def test():
    a = 5
    b = 6
    vm = VirtualMachine()
    parm1 = Instruction(100, 3)
    parm2 = Instruction(100, 7)
    vm.LOAD_CONST(parm1)
    vm.LOAD_CONST(parm2)
    vm.binaryOp("ADD")
    print(vm.pop())    

test()