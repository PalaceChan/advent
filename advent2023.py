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

## Problem 4
import os

with open(f"{os.getcwd()}/input.txt", "r") as f:
    points = []
    matches = []
    for l in f:
        l = l.strip()
        win, haz = re.search(r"Card\s+[0-9]+: ([^|]+) \| (.*)",l).groups()
        win = set(win.split())
        haz = set(haz.split())
        com = haz & win
        value = 2**(len(com)-1) if len(com) > 0 else 0
        points.append(value)
        matches.append(len(com))
    print(f"part I: {sum(points)}")
    cards = [1] * len(matches)
    for i, p in enumerate(matches):
        for j in range(p):
            cards[i+j+1] += cards[i]
    print(f"part II: {sum(cards)}")

## Problem 5
import os
import re
import math

def get_val_from_maps(k, maps):
    for m in maps:
        drs, srs, rl = m
        if k >= srs and k < srs+rl:
            off = srs - drs
            res = k - off
            return res
    return k

def get_final_val(v):
    v = get_val_from_maps(v, seed2soil)
    v = get_val_from_maps(v, soil2fertilizer)
    v = get_val_from_maps(v, fertilizer2water)
    v = get_val_from_maps(v, water2light)
    v = get_val_from_maps(v, light2temperature)
    v = get_val_from_maps(v, temperature2humidity)
    v = get_val_from_maps(v, humidity2location)
    return v

def get_ivals_from_ival_and_maps(ival, maps):
    res = []
    leftl = []
    pending = [ival]
    while pending:
        leftl = pending
        pending = []
        for left in leftl:
            for m in maps:
                drs, srs, rl = m
                off = srs - drs
                a, b = left
                # ival falls to left of srs (b < srs) -> continue
                if b < srs:
                    continue
                # ival falls to right of srs+rl-1 (a >= srs+rl) -> continue
                elif a > srs+rl-1:
                    continue
                # ival is sub-ival of [srs, srs+rl-1] -> fully remap
                elif a >= srs and b <= srs+rl-1:
                    res.append((a - off, b - off))
                    break
                # m chops ival to the left -> two new ivals
                elif srs <= a and (srs+rl-1) < b:
                    res.append((a - off, srs+rl-1-off))
                    pending.append((srs+rl, b))
                    break
                # m chops ival to the right -> two new ivals
                elif srs > a and (srs+rl-1) >= b:
                    res.append((srs - off, b - off))
                    pending.append((a, srs-1))
                    break
                # m is strict sub-ival of ival -> three new ivals
                elif a < srs and (srs+rl-1) < b:
                    res.append((srs-off, srs+rl-1-off))
                    pending.extend([(a, srs-1), (srs+rl, b)])
                    break
                else:
                    assert(False)
            else:
                res.append(left)
    return res

def get_ivals_from_ivals_and_maps(ivals, maps):
    res = []
    for ival in ivals:
        nvals = get_ivals_from_ival_and_maps(ival, maps)
        res.extend(nvals)
    return res

def get_final_ivals(ivals):
    ivals = get_ivals_from_ivals_and_maps(ivals, seed2soil)
    ivals = get_ivals_from_ivals_and_maps(ivals, soil2fertilizer)
    ivals = get_ivals_from_ivals_and_maps(ivals, fertilizer2water)
    ivals = get_ivals_from_ivals_and_maps(ivals, water2light)
    ivals = get_ivals_from_ivals_and_maps(ivals, light2temperature)
    ivals = get_ivals_from_ivals_and_maps(ivals, temperature2humidity)
    ivals = get_ivals_from_ivals_and_maps(ivals, humidity2location)
    return ivals

with open(f"{os.getcwd()}/input.txt", "r") as f:
    seeds = None
    seed2soil = []
    soil2fertilizer = []
    fertilizer2water = []
    water2light = []
    light2temperature = []
    temperature2humidity = []
    humidity2location = []
    for l in f:
        l = l.strip()
        if m := re.search(r"seeds: (.*)", l):
            seeds = [int(v) for v in m.groups()[0].split()]
        elif m := re.search(r"([a-z]+)-to-([a-z]+) map:", l):
            a,b = m.groups()
            lis = eval(f"{a}2{b}")
        elif m := re.search(r"([0-9]+) ([0-9]+) ([0-9]+)", l):
            drs, srs, rl = m.groups()
            lis.append((int(drs), int(srs), int(rl)))
        else:
            assert len(l) == 0

# part i
min_locs = math.inf
for v in seeds:
    v = get_final_val(v)
    min_locs = min(v, min_locs)
print(f"part i: {min_locs}")

# part ii
ivals = []
for i in range(0, len(seeds), 2):
    a, b = seeds[i:i+2]
    ivals.append((a, a+b-1))
fivals = get_final_ivals(ivals)
min_locs = min([a for a,_ in fivals])
print(f"part i: {min_locs}")
