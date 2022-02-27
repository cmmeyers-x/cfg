import sys
from collections import deque


# A class representing a Context Free Grammar
# Assume all fields in Constructor are present. You should just have to interface with this
class CFG:
    def __init__(self, cfg: dict, rules: list, terminals: set, start_symbol: str):
        self.cfg = cfg  # A dictionary with key : non-terminal, value : list of production results
        self.rules = rules  # a list of tuple (non-terminal, production result)
        self.terminals = terminals  # non_terminals are just keys in cfg
        self.start_symbol = start_symbol

    def derives_to_lambda(self, L: str, T: deque) -> bool:
        if L == "S": return False
        if L == "A": return False
        if L == "B": return True
        if L == "C": return False

    def first_set(self, Xb: list, T: set = None) -> (set(), set()):
        if Xb[0] == "S": return set("a"), None
        if Xb[0] == "A": return set("a"), None
        if Xb[0] == "B": return set("b", "c", "q", "a"), None
        if Xb[0] == "C": return set("c", "q"), None


    def follow_set(self, A: str, T: set=None) -> (set(), set()):
        pass


# file_name : a file in the cwd of the script
# returns : a list of lines with newlines extra spaces removed
def parse_file(file_name: str) -> list:
    result_list = []
    # open file and put it in an array of lines
    with open(file_name) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            items = line.split(' ')
            items[-1] = items[-1].replace("\n", "")  # remove newline if exists
            for i in range(len(items) - 1, -1, -1):  # trim empty spaces
                if items[i] == '':
                    items.pop(i)
            result_list.append(' '.join(items))

    return result_list


# lines : a list of lines that describe production rules
# returns : a CFG object
def generate_cfg(lines: list) -> dict:
    cfg = {}
    non_terminal = ""
    for line in lines:
        # production grab the non-terminal and add rules for it
        if '->' in line:
            split = line.index('->')
            non_terminal = line[:split].strip()
            rhs = line[2 + split:].strip()
            rules = rhs.split('|')
            for rule in rules:
                if non_terminal in cfg:
                    cfg[non_terminal].append(rule.strip())
                else:
                    cfg[non_terminal] = [rule.strip()]
        elif non_terminal != "":  # | on its own line with a previously specified non-terminal
            if "|" in line:
                rules = line.split('|')
                for rule in rules:
                    if rule != "":
                        cfg[non_terminal].append(rule.strip())

    return cfg


def main(file):
    lines = parse_file(file)
    cfg = generate_cfg(lines)

    # Print all rules
    rules = []  # tuple of NonTerminal -> Result
    print("Rules:")
    rule_num = 0
    for key, val in cfg.items():
        for result in val:
            rules.append((key, result))
            print(rule_num, "\t", key, "->", result)
            rule_num += 1

    # Print start symbol
    start_symbol = ""
    for key, val in cfg.items():
        for result in val:
            if "$" in result:
                print("Start symbol:", key)
                start_symbol = key
                break

    # Print non-terminals:
    print("\nNon Terminals:")
    for key in cfg.keys():
        print(key, end=" ")
    print("")

    # Print terminals
    terminals = set()
    for key, val in cfg.items():
        for v in val:
            for value in v.split(' '):
                if value not in cfg:  # if not a non terminal
                    terminals.add(value)

    print("\nTerminals")
    for term in terminals:
        print(term, end=" ")
    print("")

    grammer = CFG(cfg, rules, terminals, start_symbol)


if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("please pass file")
        exit(1)
    main(sys.argv[1])
