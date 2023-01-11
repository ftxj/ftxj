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
    def __init__(self, opcode, opname, argval):
        self.opcode = opcode
        self.opname = opname
        self.argval = argval

def convert_instruction(i: dis.Instruction):
    return Instruction(
        i.opcode,
        i.opname,
        i.argval
    )

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

    def step(self):
        inst = self.instructions[self.pc]
        getattr(self, inst.opname)(inst)

    def run(self, code):
        self.instructions = list(map(convert_instruction, dis.get_instructions(code)))
        self.pc = 1
        while(self.pc < len(self.instructions)):
            self.step()
            self.pc = self.pc + 1
            return


def test_add_fn():
    return 3 + 4

def test():
    vm = VirtualMachine()
    vm.run(test_add_fn)
    print(vm.pop())    

test()