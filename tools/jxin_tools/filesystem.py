import os
import ast


def get_dir_list(dirname):
    return list(filter(os.path.isdir,
                       map(lambda filename: os.path.join(dirname, filename),     
                           os.listdir(dirname) )      
                      ))
 
def get_file_list(dirname, ext ='.py'):
    filenames = []
    for root,dir_list,file_list in os.walk(dirname):  
        for file_name in file_list:  
            if(file_name.endswith(ext)):
                filenames.append(os.path.join(root, file_name))
    return filenames

def get_file_base_name(filename):
    return filename.split("/")[-1]

def basename(filename):
    return os.path.basename(file_name)

def create_if_nonexist(filepath):
    if os.path.exists(filepath) == False:
        os.mkdir(filepath)


def line_by_line(filepath):
    text_file = open(filepath, "r")
    ast_root = ast.parse(text_file.read())
    ast.fix_missing_locations(ast_root)
    code = ast.unparse(ast_root)
    text_file.close()
    text_file = open(filepath, "w")
    text_file.write(code)

def write_file(filepath, data):
    f = open(filepath, 'w')
    f.write(data)