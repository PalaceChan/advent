## Problem 1
import os
import re

special = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
with open(f"{os.getcwd()}/input.txt", "r") as f:
    res = 0
    lis = []
    pat = "(?=(" + '|'.join(special) + "|[1-9]))"
    for l in f:
        l = l.strip()
        m = re.findall(pat, l)
        a = m[0]
        b = m[-1]
        a = str(special.index(a) + 1) if a in special else a
        b = str(special.index(b) + 1) if b in special else b
        s = a+b
        print(f"{l} {m} {a} {b} {s}")
        lis.append(s)
        res += int(s)
    print(res)

## Problem 2
# Part I
import os
import re

puz = {'red': 12, 'green': 13, 'blue': 14}

def is_possible(t):
    for rv in t.split(";"):
        for x in re.findall(r'[0-9]+ red|[0-9]+ blue|[0-9]+ green', rv):
            k, col = x.split()
            k = int(k)
            if puz[col] < k:
                return False
    return True

with open(f"{os.getcwd()}/input.txt", "r") as f:
    lis = []
    for l in f:
        h, t = l.split(": ")
        id = int(h.split()[1])
        if is_possible(t):
            lis.append(id)

# Part II
import os
import re
import math

def calc_mins(t):
    res = {'red': 0, 'green': 0, 'blue': 0}
    for rv in t.split(";"):
        for x in re.findall(r'[0-9]+ red|[0-9]+ blue|[0-9]+ green', rv):
            k, col = x.split()
            res[col] = max(res[col], int(k))
    return res

with open(f"{os.getcwd()}/input.txt", "r") as f:
    g = {}
    for l in f:
        h, t = l.split(": ")
        id = int(h.split()[1])
        g[id] = calc_mins(t)
    print(sum([math.prod(v.values()) for v in g.values()]))

## Problem 3
import os
import re
import itertools as it
from collections import defaultdict

with open(f"{os.getcwd()}/input.txt", "r") as f:
    rows = [l.strip() for l in f.readlines()]
    pos_nums = defaultdict(list)
    pos_syms = []
    for rn, row in enumerate(rows):
        for m in re.finditer("[0-9]+", row):
            pos_nums[rn].append((m.start(), m.end() - 1, m.group()))
        for m in re.finditer("[^0-9.]", row):
            assert m.start() == (m.end() - 1)
            pos_syms.append((m.start(), rn, m.group()))

# Part I
lis = set()
for x,y,_ in pos_syms:
    for dx, dy in it.product([-1, 1, 0], [-1, 1, 0]):
        if (y + dy) in pos_nums:
            nums_in_search_row = pos_nums[y+dy]
            for ns, ne, n in nums_in_search_row:
                if ns <= x+dx <= ne:
                    lis.add((y+dy, ns, ne, n))
print(sum([int(t[-1]) for t in lis]))

# Part II
lis = []
stars = [t for t in pos_syms if t[-1] == '*']

for x,y,_ in stars:
    nums = set()
    for dx, dy in it.product([-1, 1, 0], [-1, 1, 0]):
        if (y + dy) in pos_nums:
            nums_in_search_row = pos_nums[y+dy]
            for ns, ne, n in nums_in_search_row:
                if ns <= x+dx <= ne:
                    nums.add((y+dy, ns, ne, n))
    if len(nums) == 2:
        gr = 1
        for num in nums:
            gr *= int(num[-1])
        lis.append(gr)
print(sum(lis))
