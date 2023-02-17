import ast
import inspect
import util



def load_python_to_ast(filepath):
    text_file = open(filepath, "r")
    data = text_file.read()
    py_ast = ast.parse(data)
    return py_ast

py_ast = load_python_to_ast("../pyg-examples/gcn2-cora.py")

print(ast.dump(py_ast, indent=4))