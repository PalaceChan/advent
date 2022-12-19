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
import itertools as it
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

def p2(f):
    rows = []
    for l in f:
        rows.append([int(c) for c in l.rstrip()])
    m = np.array(rows)
    nr, nc = m.shape

    def ls(i, j):
        s = 0
        v = m[i,j]
        while j > 0:
            j -= 1
            s += 1
            if m[i,j] >= v:
                break
        return s

    def rs(i, j):
        s = 0
        v = m[i,j]
        while j < nc-1:
            j += 1
            s += 1
            if m[i,j] >= v:
                break
        return s

    def ts(i, j):
        s = 0
        v = m[i,j]
        while i > 0:
            i -= 1
            s += 1
            if m[i,j] >= v:
                break
        return s

    def bs(i, j):
        s = 0
        v = m[i,j]
        while i < nr-1:
            i += 1
            s += 1
            if m[i,j] >= v:
                break
        return s

    ms = 0
    for i, j in it.product(range(nr), range(nc)):
        s = ls(i,j) * rs(i,j) * ts(i,j) * bs(i,j)
        if s > ms:
            ms = s
    print(ms)

with open(f"{os.getcwd()}/input.txt", "r") as f:
    p2(f)

## Problem 9
import os
import numpy as np

def p1(f):
    def calc_tails(h, t):
        d = (h[0] - t[0], h[1] - t[1])
        a = None
        if abs(d[0]) == abs(d[1]):
            a = t # d is (0,0) or one away diagonally
        elif abs(d[0]) + abs(d[1]) == 1:
            a = t # d is one along an axis and same the other axis
        elif abs(d[0]) + abs(d[1]) == 2:
            a = (t[0] + np.sign(d[0]), t[1] + np.sign(d[1]))
        elif abs(d[0]) + abs(d[1]) == 3:
            a = (t[0] + np.sign(d[0]), t[1] + np.sign(d[1]))
        else:
            import pdb; pdb.set_trace()
            assert(False)
        # print(f"h={h[0],h[1]}, t={t[0],t[1]}, a={a[0],a[1]}")
        return a

    h = (4,0)
    t = (4,0)
    positions = set()
    positions.add((4,0))
    for l in f:
        d, s = l.rstrip().split(' ')
        for i in range(int(s)):
            # m = np.full([5,6], '.')
            # m[t[0], t[1]] = 'T'
            # print(d)
            if d == 'R':
                h = (h[0], h[1] + 1)
                t = calc_tails(h, t)
            elif d == 'L':
                h = (h[0], h[1]-1)
                t = calc_tails(h, t)
            elif d == 'U':
                h = (h[0]-1, h[1])
                t = calc_tails(h, t)
            elif d == 'D':
                h = (h[0]+1, h[1])
                t = calc_tails(h, t)
            # m[h[0], h[1]] = 'H'
            # m[t[0], t[1]] = 'X'
            # print(m)
            positions.add(t)
        positions.add(t)
        # m = np.full([5,6], '.')
        # m[t[0], t[1]] = 'T'
        # m[h[0], h[1]] = 'H'
        # print(m)

    # n = np.full([5,6], '.')
    # for pos in positions:
    #     n[pos[0], pos[1]] = 'T'
    # print(n)
    print(len(positions))

with open(f"{os.getcwd()}/input.txt", "r") as f:
    def calc_tail(h, t):
        d = (h[0] - t[0], h[1] - t[1])
        a = None
        if abs(d[0]) == abs(d[1]) and abs(d[0]) <= 1:
            a = t # d is (0,0) or one away diagonally
        elif abs(d[0]) == abs(d[1]) and abs(d[0]) == 2:
            # d is two away diagonally
            a = (t[0] + np.sign(d[0]), t[1] + np.sign(d[1]))
        elif abs(d[0]) + abs(d[1]) == 1:
            a = t # d is one along an axis and same the other axis
        elif abs(d[0]) + abs(d[1]) == 2:
            a = (t[0] + np.sign(d[0]), t[1] + np.sign(d[1]))
        elif abs(d[0]) + abs(d[1]) == 3:
            a = (t[0] + np.sign(d[0]), t[1] + np.sign(d[1]))
        else:
            import pdb; pdb.set_trace()
            assert(False)
        return a
    
    def calc_tails(h, t):
        a = [None] * len(t)
        a[0] = calc_tail(h, t[0])
        # print(f"CALC{1} h={h[0],h[1]}, t{1}={t[0][0],t[0][1]}, a={a[0][0],a[0][1]}")
        for i in range(1, len(t)):
            a[i] = calc_tail(a[i-1], t[i])
            # print(f"CALC{i+1} h={h[0],h[1]}, t{i}={a[i-1][0],a[i-1][1]}, a={a[i][0],a[i][1]}")
        return a

    def paint(h, t):
        m = np.full([21,26], '.')
        m[h[0], h[1]] = 'H'
        for i, p in reversed(list(enumerate(t))):
            m[p[0], p[1]] = str(i+1)
        print(m)

    h = (15,11)
    t = [(15,11) for _ in range(9)]
    positions = set()
    positions.add((15,11))
    for l in f:
        d, s = l.rstrip().split(' ')
        # print(l)
        # paint(h, t)
        for i in range(int(s)):
            if d == 'R':
                h = (h[0], h[1] + 1)
                t = calc_tails(h, t)
            elif d == 'L':
                h = (h[0], h[1]-1)
                t = calc_tails(h, t)
            elif d == 'U':
                h = (h[0]-1, h[1])
                t = calc_tails(h, t)
            elif d == 'D':
                h = (h[0]+1, h[1])
                t = calc_tails(h, t)
            positions.add(t[-1])
        # paint(h, t)
        positions.add(t[-1])
    print(len(positions))

## Problem 10
import os
import numpy as np

def p1(f):
    curr = 0
    xval = 1
    pdic = {}
    def is_probe_cycle(i):
            ans = (i == 20 or (i - 20) % 40 == 0)
            return ans

    for l in f:
        if l.rstrip() == 'noop':
            curr += 1
            if is_probe_cycle(curr):
                pdic[curr] = xval
        else:
            ins, val = l.rstrip().split(' ')
            assert(ins) == 'addx'
            for _ in range(2):
                curr += 1
                if is_probe_cycle(curr):
                    pdic[curr] = xval
            xval += int(val)
    s = 0
    for c, v in pdic.items():
        # print(f"{c}th v={v} s={c*v}")
        s += c*v
    print(s)

with open(f"{os.getcwd()}/input.txt", "r") as f:
    curr = 0
    xval = 1
    nr, nc = 6, 40
    i, j = 0, 0
    m = np.full([nr,nc], '.')

    def get_next_pix(i, j):
        if j+1 < nc:
            return i, j+1
        else:
            return i+1,0

    def touches_sprite(i, j, xval):
        return xval - 1 <= j and j <= xval + 1

    for l in f:
        if l.rstrip() == 'noop':
            curr += 1
            if touches_sprite(i, j, xval):
                m[i, j] = '#'
            i, j = get_next_pix(i, j)
        else:
            ins, val = l.rstrip().split(' ')
            assert(ins) == 'addx'
            for _ in range(2):
                curr += 1
                if touches_sprite(i, j, xval):
                    m[i, j] = '#'
                i, j = get_next_pix(i, j)
            xval += int(val)

    print(m)

## Problem 11
import os
import re
import time
import pprint
from collections import deque
from dataclasses import dataclass

def p1(monkeys):
    t0 = time.time()
    for round in range(20):
        for i, k in enumerate(monkeys):
            while len(k.items) > 0:
                w = k.items.popleft()
                nw = eval(k.op.replace('old', str(w))) // 3
                tgt = k.tgt_true if (nw % k.div == 0) else k.tgt_false
                monkeys[tgt].items.append(nw)
                k.business += 1
    # pprint.pp(monkeys)
    b = sorted([k.business for k in monkeys], reverse=True)
    t1 = time.time()
    print(f"{b[0] * b[1]} (took {t1 - t0})")

def p2(monkeys):
    n = 1
    for k in monkeys:
        n *= k.div
    t0 = time.time()
    for round in range(10000):
        for i, k in enumerate(monkeys):
            while len(k.items) > 0:
                w = k.items.popleft()
                nw = eval(k.op.replace('old', str(w))) % n
                tgt = k.tgt_true if (nw % k.div == 0) else k.tgt_false
                monkeys[tgt].items.append(nw)
                k.business += 1
    # pprint.pp(monkeys)
    b = sorted([k.business for k in monkeys], reverse=True)
    t1 = time.time()
    print(f"{b[0] * b[1]} (took {t1 - t0})")

with open(f"{os.getcwd()}/input.txt", "r") as f:
    @dataclass
    class Monkey():
        items: deque
        op: str = ''
        div: int = -69
        tgt_true: int = -1
        tgt_false: int = -1
        business: int = 0

    monkeys = []
    for l in f:
        if l.startswith('Monkey'):
            monkeys.append(Monkey(deque()))
        else:
            k = monkeys[-1]
            if m := re.search(r"\s+Starting items: (.*)", l):
                k.items = deque([int(x) for x in m.groups()[0].split(',')])
            elif m := re.search(r"\s+Operation: new = (.*)", l):
                k.op = m.groups()[0]
            elif m := re.search(r"\s+Test: divisible by ([0-9]+)", l):
                k.div = int(m.groups()[0])
            elif m := re.search(r"\s+If true: throw to monkey ([0-9]+)", l):
                k.tgt_true = int(m.groups()[0])
            elif m := re.search(r"\s+If false: throw to monkey ([0-9]+)", l):
                k.tgt_false = int(m.groups()[0])
    # pprint.pp(monkeys)
    # p1(monkeys)
    p2(monkeys)
