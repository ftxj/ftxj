import ast
import inspect
import util

nvtx_start_template = '''
variable_name = prof.nvtx.start("stage_name", variable_name, require_grad)
'''
nvtx_stop_template = '''
variable_name = prof.nvtx.stop("stage_name", variable_name, require_grad)
'''


nvtx_pass_functions = [
    "forward",
    "train"
]

nvtx_break_assign = [
    "nvtx"
]

nvtx_node_bypass = [
    ast.Pass,
    ast.Import,
    ast.ImportFrom,
    ast.Break,
    ast.Continue
]

class nvtxPass():

    def __init__(self) -> None:
        self.current_scope = []
        self.parent_node_stack = []
        self.nvtx_range_stack = []
        self.parent_node = None
        self.variable_index = 0
        self.enable_return = False

        self.nvtx_start = False

    def visit(self, node):
        if isinstance(node, ast.Module):
            return self.visit_Module(node)
        elif isinstance(node, ast.Import):
            return self.visit_Others(node)
        elif isinstance(node, ast.ClassDef):
            return self.visit_ClassDef(node)
        elif isinstance(node, ast.Assign):
            return self.visit_Assign(node)
        elif isinstance(node, ast.Call):
            return self.visit_Call(node)
        elif isinstance(node, ast.FunctionDef):
            return self.visit_FunctionDef(node)
        elif isinstance(node, ast.Name):
            return self.visit_Name(node)
        elif isinstance(node, ast.Return):
            return self.visit_Name(node)
        elif node.__class__ in nvtx_node_bypass:
            return self.visit_Bypass(node)
        else:
            raise NameError('unhandle type: ' + node.__class__.__name__) 
    
    def visit_Bypass(self, node):
        return node

    def visit_Module(self, node : ast.Module):
        # Module(stmt* body, type_ignore* type_ignores)
        if hasattr(node, 'body'):
            for idx, stmt in enumerate(node.body):
                self.node_range_start(node)
                self.visit(stmt)
                self.node_range_stop()
        return node

    def visit_Name(self, node : ast.Name):
        return node

    def visit_ClassDef(self, node : ast.ClassDef):
        # ClassDef(name, bases, keywords, starargs, kwargs, body, decorator_list)
        for func in node.body:
            self.node_range_start(node)
            self.visit(func)
            self.node_range_stop()
        return node

    def visit_FunctionDef(self, node : ast.FunctionDef):
        # ast.FunctionDef(name, args, body, decorator_list, returns, type_comment)
        if(nvtx_pass_functions.count(node.name) > 0):
            self.nvtx_start = True
        else:
            return node
        stage_name = util.get_identifier(node.name)
        for arg in node.args.args:
            if util.is_differentiable(arg):
                variable_name = util.get_identifier(arg)
                require_grad = "require_grad"
                break
        require_grad_node = util.make_arg_node(require_grad)
        node.args.args.insert(1, require_grad_node)
        nvtx_start_node = self.construct_nvtx_start(stage_name, variable_name, require_grad)
        self.current_scope.append(nvtx_start_node)

        self.node_range_start(node)
        for idx, stmt in enumerate(node.body):
            self.current_scope.append(self.visit(stmt))
        self.node_range_stop()
        
        nvtx_stop_node = self.construct_nvtx_stop(variable_name, require_grad)
        if(self.enable_return):
            self.current_scope.insert(-2, nvtx_stop_node)
        else:
            self.current_scope.insert(-1, nvtx_stop_node)
        node.body = self.current_scope
        self.current_scope = []
        self.nvtx_start = False
        

    def visit_Assign(self, node : ast.Assign):
        # ast.Assign(targets, value, type_comment)
        if(not self.nvtx_start):
            return node
        
        self.node_range_start(node)
        node.value = self.visit(node.value)
        self.node_range_stop()

        for target in node.targets:
            if util.is_differentiable(target):
                nvtx_stop_node = self.construct_nvtx_stop(util.get_identifier(target), "require_grad")
                self.current_scope.append(node)
                return nvtx_stop_node

    def visit_Call(self, node : ast.Call):
        # ast.Call(func, args, keywords, starargs, kwargs)
        if(util.is_expr(self.parent_node)):
            tmp_assign_target = ast.Name(
                id="tmp_" + str(self.variable_index),
                ctx=ast.Store()
                )
            self.variable_index += 1
            tmp_assign_node = ast.Assign(targets=[tmp_assign_target], value=node)

            self.node_range_start(node)
            self.visit(tmp_assign_node)
            self.node_range_stop()

            return tmp_assign_node

        for idx, arg in enumerate(node.args):
            self.node_range_start(node)
            node.args[idx] = util.lvalue(self.visit(arg))
            self.node_range_stop()

        stage_name = util.get_identifier(node)
        for idx, arg in enumerate(node.args):
            if util.is_differentiable(arg):
                variable_name = util.get_identifier(arg)
                require_grad = "require_grad"
                break
        if variable_name == '':
            variable_name = util.get_identifier(node.args[0])
            require_grad = "False"
        
        nvtx_start_node = self.construct_nvtx_start(stage_name, variable_name, require_grad)
        self.current_scope.append(nvtx_start_node)
        return node

    
    def visit_Return(self, node : ast.Return):
        # ast.Return(value)
        self.enable_return = True
        self.node_range_start(node)
        node.value = util.lvalue(self.visit(node.value))
        self.node_range_stop()
        return node

    def run(self, obj):
        if(isinstance(obj, str)):
            ast_root = ast.parse(obj)
        else:
            source_code = inspect.getsource(obj)
            ast_root = ast.parse(source_code)

        ast_root = self.visit(ast_root)
        ast.fix_missing_locations(ast_root)
        # print(ast.dump(ast_module, indent=4))
        module_code = ast.unparse(ast_root)
        return module_code
    
    def node_range_start(self, node):
        self.parent_node_stack.append(self.parent_node)
        self.parent_node = node

    def node_range_stop(self):
        self.parent_node = self.parent_node_stack.pop()

    def construct_nvtx_start(self, stage_name, variable_name, require_grad):
        self.nvtx_range_stack.append(stage_name)
        nvtx_start_node_template = ast.parse(nvtx_start_template).body[0]
        return self.construct_nvtx_impl(nvtx_start_node_template, stage_name, variable_name, require_grad)
    
    def construct_nvtx_stop(self, variable_name, require_grad):
        nvtx_start_node_template = ast.parse(nvtx_stop_template).body[0]
        return self.construct_nvtx_impl(nvtx_start_node_template, self.nvtx_range_stack.pop(), variable_name, require_grad)
    
    def construct_nvtx_impl(self, nvtx_start_node_template, stage_name, variable_name, require_grad):
        nvtx_start_node_template.targets[0] = ast.Name(id = variable_name, ctx = ast.Store())
        nvtx_start_node_template.value.args[0] = ast.Constant(value = stage_name)
        nvtx_start_node_template.value.args[1] = ast.Name(id = variable_name, ctx = ast.Load())
        nvtx_start_node_template.value.args[2] = ast.Name(id = require_grad, ctx = ast.Load())
        return nvtx_start_node_template


module = '''
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
'''


def main():
    nvtx_pass = nvtxPass()
    code = nvtx_pass.run(module)
    print(code)

if __name__ == "__main__":
    main()
