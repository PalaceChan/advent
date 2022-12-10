## Problem 1
import os

with open(f"{os.getcwd()}/input.txt", "r") as f:
    s = f.read()
    max_cals = [0, 0, 0]
    tot_cal = 0
    for k in s.split('\n'):
        if k:
            tot_cal += int(k)
        else:
            if tot_cal > max_cals[0]:
                old_max = max_cals[0]
                max_cals[0] = tot_cal
                if old_max > max_cals[1]:
                    oold_max = max_cals[2]
                    max_cals[1] = old_max
                    if oold_max > max_cals[2]:
                        max_cals[2] = oold_max
            elif tot_cal > max_cals[1]:
                old_max = max_cals[1]
                max_cals[1] = tot_cal
                if old_max > max_cals[2]:
                    max_cals[2] = old_max
            elif tot_cal > max_cals[2]:
                max_cals[2] = tot_cal
            tot_cal = 0

    print(sum(max_cals))

## Problem 2
import os

move_d = {'A': 'r', 'B': 'p', 'C': 's', 'X': 'r', 'Y': 'p', 'Z': 's'}
shape_d = {'r': 1, 'p': 2, 's': 3}

def score(elf, you):
    if you == elf:
        s_outcome = 3
    elif you == 'r':
        s_outcome = 0 if elf == 'p' else 6
    elif you == 'p':
        s_outcome = 0 if elf == 's' else 6
    elif you == 's':
        s_outcome = 0 if elf == 'r' else 6

    s = shape_d[you] + s_outcome
    return s

def move_from_res(elf, res):
    if res == 'Y':
        return elf
    elif elf == 'r':
        you = 's' if res == 'X' else 'p'
    elif elf == 'p':
        you = 'r' if res == 'X' else 's'
    elif elf == 's':
        you = 'p' if res == 'X' else 'r'

    return you

with open(f"{os.getcwd()}/input.txt", "r") as f:
    s = 0
    for l in f:
        # s += score(*map(move_d.get, l.rstrip().split(' ')))
        elf, res = l.rstrip().split(' ')
        elf = move_d[elf]
        you = move_from_res(elf, res)
        s += score(elf, you)
    print(s)

## Problem 3
import os

def getp(c):
    s = ord(c) - 96
    if s < 0:
        s = ord(c) - 38
    return s

def p1(f):
    p = 0
    for l in f:
        l = l.rstrip()
        c = (set(l[0:(len(l)//2)]) & set(l[len(l)//2:])).pop()
        p += getp(c)
    print(p)

def p2(f):
    p = 0
    ls = f.readlines()
    for e1, e2, e3 in zip(*(iter(ls),) * 3):
        c = (set(e1.rstrip()) & set(e2.rstrip()) & set(e3.rstrip())).pop()
        p += getp(c)
    print(p)

with open(f"{os.getcwd()}/input.txt", "r") as f:
    p2(f)

## Problem 4
import os

def overlap_full(a, b, x, y):
    return (a <= x and y <= b) or (x <= a and b <= y)

def overlap_any(a, b, x, y):
    disjoint = (x > b) or (a > y)
    return not disjoint

def p(f, fun):
    ans = 0
    for l in f:
        e1, e2 = l.rstrip().split(',')
        a, b = [int(i) for i in e1.split('-')]
        x, y = [int(i) for i in e2.split('-')]
        if fun(a, b, x, y):
            ans += 1
    print(ans)

with open(f"{os.getcwd()}/input.txt", "r") as f:
    # p(f, overlap_full)
    p(f, overlap_any)

## Problem 5
import os
import re
from collections import deque

def get_staks(f):
    ls = []
    for l in f:
        if l.startswith(' 1'):
            break
        else:
            ls.append(l.rstrip())

    nstaks = max([int(i) for i in l.split()])
    staks = [deque() for i in range(nstaks)]
    for i in range(nstaks):
        pos = l.find(str(i+1))
        for bl in ls:
            if pos < len(bl) and bl[pos] != ' ':
                staks[i].appendleft(bl[pos])
    return staks

def get_moves(f):
    moves = []
    for l in f:
        if m := re.match(r'move ([0-9]+) from ([0-9]+) to ([0-9]+)', l):
            move = tuple(int(i) for i in m.groups())
            moves.append(move)
    return moves

def p1(staks, moves):
    for n, s, d in moves:
        src = staks[s-1]
        dst = staks[d-1]
        for _ in range(n):
            c = src.pop()
            dst.append(c)
    print(''.join([s.pop() for s in staks if len(s)]))

def p2(staks, moves):
    hel = deque()
    for n, s, d in moves:
        src = staks[s-1]
        dst = staks[d-1]
        for _ in range(n):
            c = src.pop()
            hel.append(c)
        for _ in range(n):
            c = hel.pop()
            dst.append(c)
        hel.clear()
    print(''.join([s.pop() for s in staks if len(s)]))

with open(f"{os.getcwd()}/input.txt", "r") as f:
    staks = get_staks(f)
    moves = get_moves(f)
    p2(staks, moves)

## Problem 6
import os

def detect(l, k):
    w = deque()
    w.extend(l[0:k])
    for i in range(k, len(l)):
        if len(set(w)) == k:
            print(i)
            break
        else:
            w.popleft()
            w.append(l[i])

with open(f"{os.getcwd()}/input.txt", "r") as f:
    for l in f:
        detect(l, 14)

## Problem 7
import os

with open(f"{os.getcwd()}/test.txt", "r") as f:
    pass
