
class Frame(object):
    def __init__(self):
        self.block_stack = []
        self.data_stack = []
    
    def push_data(self, elm):
        self.data_stack.push(elm)

    def pop_data(self):
        return self.data_stack.pop()