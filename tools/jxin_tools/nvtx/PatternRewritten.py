

import ast
import inspect
import util
from enum import Enum

class State(Enum):
    Init = 0
    Begin = 1
    Matching = 2
    Stop = 3
    Fail = 4


class PatternAstNode():
    


class PatternMatching():

    def __init__(self, pattern) -> None:
        self.pattern = pattern
        self.pattern_node = pattern
        self.begin_matching = []
        self.state = State.Init
        
    def update_state(self, matching_result):
        if(self.state == State.Init):
            if(matching_result):
                self.state = State.Matching
        elif(self.state == State.Matching):
            if(matching_result == False):
                self.state = State.Stop
                        
    def run(self, obj):
        if(isinstance(obj, str)):
            ast_root = ast.parse(obj)
        else:
            source_code = inspect.getsource(obj)
            ast_root = ast.parse(source_code)

        self.visit(ast_root)
        return self.matched
    
    def visit(self, code_node):
        return self.visit(code_node, self.pattern_node)
    
    def visit(self, code_node, pattern_node):
        if self.maybe_wildcard(pattern_node):
            step = True
        elif isinstance(code_node, ast.Call):
            step = self.visit_Call(code_node, pattern_node)
        elif isinstance(code_node, ast.Name):
            step = self.visit_Name(code_node, pattern_node)
        elif isinstance(code_node, ast.For):
            step = self.visit_For(code_node, pattern_node)
        else:
            step = self.generic_visit(code_node, pattern_node)
        self.update_state(step)
    
    def match(self, str1, str2):
        return str1==str2 or str2.contains("__rewrite_wildcard")

    def is_wildcard(self, node_str):
        return node_str.contain("__rewrite_wildcard")

    def maybe_wildcard(self, node):
        if isinstance(node, ast.Name) and self.is_wildcard(node.id):
            return True
        return False

    def match_list(self, node, pattern_node):
        for idx in range(0, node.size()):
            if pattern_node.size() < idx:
                return False
            if self.maybe_wildcard(pattern_node[idx]):
                return True
            is_matching = is_matching & self.visit(node[idx], pattern_node[idx])
        return is_matching
    

    def generic_visit(self, node, pattern_node):
        return True

    def visit_Name(self, node, pattern_node):
        if self.maybe_wildcard(pattern_node):
            return True
        else:
            if(node.id == pattern_node.id and pattern_node.ctx == node.ctx):
                return True
        return False
    
    def visit_Call(self, node, pattern_node):
        ismatching = isinstance(pattern_node, ast.Call)
        ismatching = ismatching & self.visit(node.func, pattern_node.func)
        ismatching = ismatching & self.match_list(node.args, pattern_node.args)
        ismatching = ismatching & self.match_list(node.body, pattern_node.body)
        return ismatching

    
    def visit_For(self, node, pattern_node):
        if(not isinstance(pattern_node, ast.For)):
            self.matching_pattern = self.pattern
            return False
        else:
            ismatching = self.visit(node.target.id, self.pattern)
            ismatching = ismatching & self.match(node.target.id, self.pattern)


code = '''
best_val_acc = test_acc = 0
for epoch in range(1, 3):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
          f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
'''


src_pattern = '''
for epoch in range(__rewrite_wildcard_1, __rewrite_wildcard_2):
    __rewrite_wildcard_3
'''

dst_pattern = '''
for epoch in range(__rewrite_wildcard_1, __rewrite_wildcard_1 + 2):
    __rewrite_wildcard_3
'''

rewrite(code, src_pattern, dst_pattern)