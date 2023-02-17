from enum import Enum
import json
import re
import os
import jxin_tools.filesystem as fs

class TransferMethod(str, Enum):
    Before = "Before"
    After = "After"
    Last = "Last"
    Begin = "Begin"
    Rewrite = "Rewrite"
    Nothing = "Nothing"
    AfterBlock = "AfterBlock"
    
def get_space_suffix(s):
    num = 0
    for c in s:
        if c == ' ':
            num = num + 1
        else: 
            break
    return ' ' * num

class Pattern:

    def __init__(self, name, method, old_str, new_str, break_line = None) -> None:
        self.name = name
        self.method = method
        self.old_str = old_str
        self.new_str = new_str
        self.break_line = break_line

    def __str__(self):
        str = "name = " + self.name + "\n"
        str += "method = " + self.method + "\n"
        str += "old_str = " + self.old_str + "\n"
        str += "new_str = " + self.new_str + "\n"
        if (self.break_line):
            str += "break_line = " + self.break_line + "\n"
        return str

    def get_new_str(self, old_line):
        if(self.new_str[0] != "~"):
            return self.new_str.split("\n")
        else:
            res = ""
            finded = re.search(self.old_str, old_line).group(0)
            pre = old_line[0:old_line.find(finded)]
            last = old_line[old_line.find(finded) + len(finded):]
            patterns = self.new_str[1:].split("/")
            for pattern in patterns:
                if(pattern == "$before_pattern"):
                    res += pre
                elif(pattern == "$pattern"):
                    res += finded
                elif(pattern == "$after_pattern"):
                    res += last
                else:
                    res += pattern
            return [res]
    
    def run_begin_pattern(self, code_lines):
        code_lines = self.get_new_str("") + code_lines
        return code_lines

    def run_last_pattern(self, code_lines):
        code_lines += self.get_new_str("")
        return code_lines

    def string_space_number(self, line):
        number = 0
        if line == "":
            return 100000
        for c in line:
            if(c == ' '):
                number += 1
            else:
                return number
        return number

    def run_afterblock_pattern(self, code_lines):
        new_code_lines = []
        breaking_space = 0
        breaking = False
        success = False
        for idx, line in enumerate(code_lines):
            current_space = self.string_space_number(line)
            if(line.find(self.old_str) != -1):
                breaking_space = self.string_space_number(line)
                breaking = True
            elif(breaking and current_space <= breaking_space):
                new_code_lines += self.get_new_str(line)
                breaking = False
                success = True
                breaking_space = 0
            new_code_lines.append(line)
        if(success == False):
            new_code_lines += self.get_new_str(line)
        return new_code_lines

    def run_before_pattern(self, code_lines):
        new_code_lines = []
        for idx, line in enumerate(code_lines):
            if(re.search(self.old_str, line)):
                space = get_space_suffix(line)
                new_strs = self.get_new_str(line)
                merged = []
                for l in new_strs:
                    merged.append(space + l)
                new_code_lines = new_code_lines + merged
            new_code_lines.append(line)
        return new_code_lines

    def run_after_pattern(self, code_lines):
        new_code_lines = []
        for idx, line in enumerate(code_lines):
            if(line.isspace()):
                continue
            new_code_lines.append(line)
            if(re.search(self.old_str, line)):
                new_code_lines += self.get_new_str(line)
        return new_code_lines

    def run_rewrite_pattern(self, code_lines):
        new_code_lines = []
        for idx, line in enumerate(code_lines):
            if((self.break_line == None or line.find(self.break_line) == -1) and re.search(self.old_str, line)):
                new_code_lines += self.get_new_str(line)
            else:
                new_code_lines.append(line)
        return new_code_lines

    def run_codetrans(self, code_lines):
        if self.method == TransferMethod.Before:
            code_lines = self.run_before_pattern(code_lines)
        elif self.method == TransferMethod.Last:
            code_lines = self.run_last_pattern(code_lines)
        elif self.method == TransferMethod.After:
            code_lines = self.run_after_pattern(code_lines)
        elif self.method == TransferMethod.Rewrite:
            code_lines = self.run_rewrite_pattern(code_lines)
        elif self.method == TransferMethod.Begin:
            code_lines = self.run_begin_pattern(code_lines)
        elif self.method == TransferMethod.AfterBlock:
            code_lines = self.run_afterblock_pattern(code_lines)
        else:
            raise NameError("Pattern Format Error : \n" + self.__str__())
        return code_lines

    def to_json(self):
        return {self.name : {'old': self.old, 'new': self.new}}


def from_json(filepath):
    json_file = open(filepath)
    patterns = json.load(json_file)
    all_patterns = []
    for pattern in patterns:
        p = Pattern(
            name = pattern['name'],
            method = TransferMethod[pattern['method']],
            old_str= pattern["old_str"],
            new_str = pattern["new_str"],
            break_line=pattern["break_line"]
        )
        all_patterns.append(p)
    return all_patterns

def to_json(patterns, filename=""):
    json_str = json.dumps([pattern.__dict__ for pattern in patterns], indent=4, separators=(',', ': '))
    if(filename != ""):
        text_file = open(filename, "w")
        text_file.write(json_str)
    return json_str

def get_codetrans_patterns(pattern_dir):
    if os.path.isfile(pattern_dir):
        patterns = from_json(pattern_dir)
        return patterns
    json_files = fs.get_file_list(pattern_dir, ext="json")
    patterns = []
    for json_file in json_files:
        print(json_file)
        patterns = patterns + from_json(json_file)
    return patterns