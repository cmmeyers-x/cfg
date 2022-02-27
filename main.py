import sys

# parse the cfg passed in, trims newlines and empty spaces to make processing easier
def parse_file(file_name):
    result_list = []
    # open file and put it in an array of lines
    with open(file_name) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            items = line.split(' ')
            items[-1] = items[-1].replace("\n", "")  # remove newline if exists
            for i in range (len(items)-1, -1, -1):  # trim empty spaces
                if items[i] == '':
                    items.pop(i)
            result_list.append(' '.join(items))

    return result_list

def main(file):
    lines = parse_file(file)
    cfg = {} # dictionary key : non-terminal, value : list of productions
    nonTerminal = ""
    # parse
    for line in lines:
        if '->' in line: # production
            split = line.index('->')
            lhs = line[:split].strip()
            rhs = line[2 + split:].strip()
            while '|' in rhs:
                pipe = rhs.find("|")
                rhs1 = rhs[:pipe].strip()
                rhs2 = rhs[1+pipe:].strip()
                if lhs in cfg:
                    cfg[lhs].append(rhs1)
                else:
                    cfg[lhs] = [rhs1]
                rhs = rhs2
            else:
                if lhs in cfg:
                    cfg[lhs].append(rhs)
                else:
                    cfg[lhs] = [rhs]
            nonTerminal = lhs
        elif nonTerminal != "":  # | on its own line
            if "|" in line:
                rules = line.split('|')
                for rule in rules:
                    if rule != "":
                        cfg[nonTerminal].append(rule.strip())

    # rules as read
    i = 0
    for key, val in cfg.items():
        for result in val:
            print(i, key, "->", result)
            i = i + 1
    i = i + 1

    # start symbol
    for key, val in cfg.items():
        for result in val:
            if "$" in result:
                print("Start symbol:", key)

    # non terminals:
    nonTerminal = {}
    for key in cfg.keys():
        nonTerminal[key] = -1
    print("\nNon Terminals")
    for key in nonTerminal.keys():
        print(key)

    # terminals
    terminals = {}
    for key, values in cfg.items():
        for v in values:
            for value in v.split(' '):
                if value not in nonTerminal:
                    terminals[value] = -1

    print("\nTerminals")
    for term in terminals.keys():
        print(term)

if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("please pass file")
        exit(1)
    main(sys.argv[1])
