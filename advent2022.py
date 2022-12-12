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

class File:
    def __init__(self, name, ftype, size):
        self.name = name
        self.ftype = ftype
        self.size = size
        self.parent = None
        self.children = []

    @classmethod
    def create(cls, name, ftype, size, parent):
        f = cls(name, ftype, size)
        f.parent = parent
        return f

    def propagate(self, sz):
        self.size += sz
        if self.name != '/':
            self.parent.propagate(sz)

    def dump(self, depth = 0):
        sfx = f'(dir, size={self.size})' if self.ftype == 'd' else f'(file, size={self.size})'
        print(' ' * depth + f'- {self.name} {sfx}')
        for c in self.children:
            c.dump(depth + 1)

def p1(f):
    def _helper(f):
        tot = 0
        ok = f.ftype == 'd' and f.size <= 100000
        if ok:
            tot += f.size
        for c in f.children:
            tot += _helper(c)
        return tot

    tot = _helper(f)
    print(tot)

def p2(f):
    have = 7e7 - fs.size
    need = 3e7 - have
    cands = []
    def _helper(f, cands, need):
        ok = f.ftype == 'd' and f.size >= need and f.size
        if ok:
            cands.append(f)
        for c in f.children:
            _helper(c, cands, need)

    _helper(f, cands, need)
    cands.sort(key=lambda x: x.size)
    print(f'name={cands[0].name} size={cands[0].size}')

def build_fs(f):
    root = File.create('/', 'd', 0, None)
    root.parent = root

    next(f)
    wd = root
    for l in f:
        if l.startswith('$ cd '):
            tgt = l.removeprefix('$ cd ').rstrip()
            if tgt == '..':
                wd = wd.parent
            else:
                wd = next(d for d in wd.children if d.name == tgt)
        elif l.startswith('$ ls'):
            pass
        else:
            x, fn = l.rstrip().split(' ')
            ft = 'd' if x == 'dir' else 'f'
            sz = 0 if x == 'dir' else int(x)
            if fn not in wd.children:
                wd.children.append(File.create(fn, ft, sz, wd))
                wd.propagate(sz)

    return root

with open(f"{os.getcwd()}/input.txt", "r") as f:
    fs = build_fs(f)
    # p1(fs)
    p2(fs)

## Problem 8
import os
import numpy as np

def p1(f):
    rows = []
    for l in f:
        rows.append([int(c) for c in l.rstrip()])
    m = np.array(rows)
    v = np.zeros_like(m)
    nr, nc = m.shape
    exterior = 2 * (nr + nc) - 4 # border visibile

    for r in range(1, nr):
        top_down, down_top = r, nr - r -1
        v[top_down,:] += np.all(m[top_down,:] > m[0:top_down,:], axis=0)
        v[down_top,:] += np.all(m[down_top,:] > m[(down_top+1):,:], axis=0)
    for c in range(1, nc):
        left_right, right_left = c, nc - c -1
        v[:,left_right] += np.all(m[:,left_right] > m[:,0:left_right].T, axis=0)
        v[:,right_left] += np.all(m[:,right_left] > m[:,(right_left+1):].T, axis=0)

    interior = np.sum(v[1:(nr-1),1:(nc-1)] > 0)
    print(exterior + interior)

with open(f"{os.getcwd()}/input.txt", "r") as f:
    p1(f)
