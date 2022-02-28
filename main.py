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
        if Xb[0] == "S": return set(("a")), None
        if Xb[0] == "A": return set(("a")), None
        if Xb[0] == "B": return set(("b", "c", "q", "a")), None
        if Xb[0] == "C": return set(("c", "q")), None

        return set(Xb[0]), None

    # Return list of rules with rhs occurrences of non_terminal
    def find_rhs_occurrences(self, non_terminal: str) -> list:
        occurrences = []
        for (lhs, rhs) in self.rules:
            if non_terminal in rhs:
                occurrences.append((lhs, rhs))
        return occurrences

    # returns true a char c in character sequence is a part of non-terminals or terminals
    def pi_and_sigma_intersection(self, char_sequence: list) -> bool:
        for c in char_sequence:
            if c in self.terminals or c in self.cfg[c]:
                return True
        return False

    # derives to lambda true for all
    def derives_to_lambda_forall(self, char_sequence: list) -> bool:
        for c in char_sequence:
            if c == ' ': continue
            if not self.derives_to_lambda(c, deque()):
                return False
        return True


    def follow_set(self, A: str, T: set = None) -> (set(), set()):
        if T is None:
            T = set()
        if A in T:
            return set(), T

        follow_set = set()
        for lhs, rhs in self.find_rhs_occurrences(A):
            non_t_in_rhs = rhs.find(A, 0)
            pi = []
            while non_t_in_rhs != -1:
                pi.append(rhs[non_t_in_rhs + 1:].strip())
                non_t_in_rhs = rhs.find(A, non_t_in_rhs + 1)

            for p in pi:
                if len(p) > 0:
                    G, _ = self.first_set(p)
                    follow_set = follow_set | G
                elif len(p) == 0 or (not self.pi_and_sigma_intersection(p) and self.derives_to_lambda_forall(p)):
                    G, _ = self.follow_set(lhs)
                    follow_set = follow_set | G

        return follow_set, T

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
# returns : a cfg dict
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

    grammar = CFG(cfg, rules, terminals, start_symbol)

    print(grammar.follow_set("B"))


if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("please pass file")
        exit(1)
    main(sys.argv[1])
