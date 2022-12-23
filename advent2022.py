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

## Problem 12
import os
import heapq
import time
import itertools as it
import numpy as np

class Vertex():
    def __init__(self, letr, elev, coor):
        self.letr = letr
        self.elev = elev
        self.coor = coor
        self.dist = np.inf
        self.nbor = []

    def __lt__(self, other):
        return self.dist < other.dist

def solve(beg, end, m):
    Q = [v for r in m for v in r]
    heapq.heapify(Q)
    while len(Q) > 0:
        u = heapq.heappop(Q)
        d = False
        for v in u.nbor:
            alt = u.dist + 1
            if alt < v.dist:
                v.dist = alt
                d = True
        if d:
            heapq.heapify(Q)

    return m[end[0]][end[1]].dist

def p1(beg, end, m):
    t0 = time.time()
    sol = solve(beg, end, m)
    t1 = time.time()
    print(f"{sol} (t = {t1-t0})")

def p2(end, m):
    t0 = time.time()
    begd = {}
    nr, nc = len(m), len(m[0])
    for i,j in it.product(range(nr), range(nc)):
        if m[i][j].elev == 0:
            begd[(i,j)] = np.inf

    for beg in begd.keys():
        for i,j in it.product(range(nr), range(nc)):
            if m[i][j].coor == beg:
                m[i][j].dist = 0
            else:
                m[i][j].dist = np.inf
        begd[beg] = sol = solve(beg, end, m)

    bbeg = None
    bsol = np.inf
    for beg, sol in begd.items():
        if sol < bsol:
            bbeg = beg
            bsol = sol

    t1 = time.time()
    print(f"solved {len(begd)} cases, best was {bsol} from {bbeg} (t={t1-t0})")

with open(f"{os.getcwd()}/input.txt", "r") as f:
    m = []
    beg = None
    end = None
    for l in f:
        i = len(m)
        row = list(l.rstrip())
        vrow = []
        for j, x in enumerate(row):
            e = ord(x) - ord('a')
            if x == 'S':
                e = ord('a') - ord('a')
                beg = (i, j)
            if x == 'E':
                e = ord('z') - ord('a')
                end = (i, j)

            v = Vertex(x, e, (i, j))
            if x == 'S':
                v.dist = 0
            vrow.append(v)
        m.append(vrow)

    nr, nc = len(m), len(m[0])
    for i,j in it.product(range(nr), range(nc)):
        v = m[i][j]
        if i-1 >= 0 and m[i-1][j].elev <= v.elev+1:
            v.nbor.append(m[i-1][j])
        if i+1 < nr and m[i+1][j].elev <= v.elev+1:
            v.nbor.append(m[i+1][j])
        if j-1 >= 0 and m[i][j-1].elev <= v.elev+1:
            v.nbor.append(m[i][j-1])
        if j+1 < nc and m[i][j+1].elev <= v.elev+1:
            v.nbor.append(m[i][j+1])

    # p1(beg, end, m)
    p2(end, m)

## Problem 13
import os
import numpy as np

def in_order(la, lb):
    for a, b in zip(la, lb):
        if isinstance(a, int) and isinstance(b, int):
            s = np.sign(a - b)
            if s != 0:
                return s
        elif isinstance(a, list) and isinstance(b, list):
            s = in_order(a, b)
            if s != 0:
                return s
        else:
            na = a if isinstance(a, list) else [a]
            nb = b if isinstance(b, list) else [b]
            s = in_order(na, nb)
            if s != 0:
                return s

    # ran through list and could not determine
    s = np.sign(len(la) - len(lb))
    return s

def p1(lines):
    pairs = []
    for i in range(0, len(lines), 3):
        pairs.append((eval(lines[i]), eval(lines[i+1])))

    order = []
    for i, (la, lb) in enumerate(pairs):
        s = in_order(la, lb)
        order.append(s)

    idx = [i+1 for i,s in enumerate(order) if s < 0]
    print(sum(idx))

def p2(lines):
    class Packet():
        def __init__(self, l):
            self.l = l

        def __eq__(self, other):
            return self.l == other.l

        def __lt__(self, other):
            s = in_order(self.l, other.l)
            return s < 0

        def __repr__(self):
            return str(self.l)

    packets = [ Packet([[2]]), Packet([[6]]) ]
    for i in range(0, len(lines), 3):
        packets.append(Packet(eval(lines[i])))
        packets.append(Packet(eval(lines[i+1])))

    packets.sort()
    # pprint.pp(packets)
    i2 = packets.index(Packet([[2]]))
    i6 = packets.index(Packet([[6]]))
    print(f"i2={i2+1}, i6={i6+1} sol={(i2+1)*(i6+1)}")

with open(f"{os.getcwd()}/input.txt", "r") as f:
    lines = f.readlines()
    # p1(lines)
    p2(lines)

## Problem 14
import os
import numpy as np

def p1(lbnd, rbnd, dbnd, arrs):
    narrs = []
    for arr in arrs:
        narrs.append([(x-lbnd,y) for x,y in arr])
    arrs = narrs
    m = np.full([dbnd+1+1,rbnd-lbnd+1+2], '.')
    m[0,500-lbnd+1] = '+'
    for arr in arrs:
        for i in range(1, len(arr)):
            x0,y0 = arr[i-1]
            x1,y1 = arr[i]
            cslc = slice(x0+1, x1+2) if x0 <= x1 else slice(x1+1, x0+2)
            dslc = slice(y0, y1+1) if y0 <= y1 else slice(y1, y0+1)
            m[dslc, cslc] = '#'

    def sim(m):
        nr, nc = m.shape
        i,j = 0, 500-lbnd+1
        for _ in range(1000):
            ni, nj = i+1, j
            if ni < nr and m[ni, nj] == '.':
                i,j = ni, nj
                continue
            ni, nj = i+1, j-1
            if ni < nr and m[ni, nj] == '.':
                i,j = ni,nj
                continue
            ni,nj = i+1, j+1
            if ni < nr and m[ni,nj] == '.':
                i,j = ni,nj
                continue

            # nowhere to go or hit the abyss row
            if ni == nr:
                return True
            else:
                m[i,j] = 'o'
                return False
        else:
            assert(False)

    u = 0
    for i in range(1000):
        done = sim(m)
        if done:
            break
        else:
            u += 1
    else:
        assert False
    print(u)

def p2(lbnd, rbnd, dbnd, arrs):
    def paint(m):
        for r in m:
            print(''.join(list(r)))

    npads = 1000
    narrs = []
    for arr in arrs:
        narrs.append([(x-lbnd,y) for x,y in arr])
    arrs = narrs
    m = np.full([dbnd+3,rbnd-lbnd+1+2*npads], '.')
    m[0,500-lbnd+npads] = '+'
    for arr in arrs:
        for i in range(1, len(arr)):
            x0,y0 = arr[i-1]
            x1,y1 = arr[i]
            cslc = slice(x0+npads, x1+npads+1) if x0 <= x1 else slice(x1+npads, x0+npads+1)
            dslc = slice(y0, y1+1) if y0 <= y1 else slice(y1, y0+1)
            m[dslc, cslc] = '#'
    m[-1,:] = '#'
    # paint(m)

    def sim(m):
        nr, nc = m.shape
        i,j = 0, 500-lbnd+npads
        if m[i,j] == 'o':
            return True
        for _ in range(1000):
            ni, nj = i+1, j
            if ni < nr and m[ni, nj] == '.':
                i,j = ni, nj
                continue
            ni, nj = i+1, j-1
            if ni < nr and m[ni, nj] == '.':
                i,j = ni,nj
                continue
            ni,nj = i+1, j+1
            if ni < nr and m[ni,nj] == '.':
                i,j = ni,nj
                continue

            # sand flow stopped
            m[i,j] = 'o'
            return False
        else:
            assert(False)

    u = 0
    for i in range(100000):
        done = sim(m)
        if done:
            break
        else:
            u += 1
    else:
        assert False
    print(u)

with open(f"{os.getcwd()}/input.txt", "r") as f:
    lbnd = np.inf
    rbnd = 0
    dbnd = 0
    arrs = []
    for l in f:
        arr = []
        for s in l.rstrip().split(' -> '):
            x, y = s.split(',')
            arr.append((int(x), int(y)))
            lbnd = min(int(x), lbnd)
            rbnd = max(int(x), rbnd)
            dbnd = max(int(y), dbnd)
        arrs.append(arr)

    # p1(lbnd, rbnd, dbnd, arrs)
    p2(lbnd, rbnd, dbnd, arrs)

## Problem 15
import os
import re
import time
import pprint

class Sb():
    @staticmethod
    def dist(x0,y0,x1,y1):
        return abs(x0-x1) + abs(y0-y1)

    def __init__(self, sx, sy, bx, by):
        self.sx = int(sx)
        self.sy = int(sy)
        self.bx = int(bx)
        self.by = int(by)
        self.dis = Sb.dist(self.sx, self.sy, self.bx, self.by)

    def __repr__(self):
        return f"S{(self.sx, self.sy)} B{(self.bx, self.by)} d={self.dis}"

def p1(sensors, y):
    off = set()
    ivals = []
    for s in sensors:
        if s.by == y:
            off.add(s.bx)
        dy = abs(s.sy - y)
        if dy <= s.dis:
            lb = s.sx - (s.dis - dy)
            ub = s.sx + (s.dis - dy)
            ivals.append([lb, ub])
            if dy == 0:
                off.add(s.sx)
    ivals.sort(key = lambda x: x[0])
    # print(ivals)
    merged = [ivals.pop(0)]
    for ival in ivals:
        if merged[-1][1] < ival[0]:
            merged.append(ival)
        elif merged[-1][1] < ival[1]:
            merged[-1][1] = ival[1]
    # print(merged)
    print(sum([ival[1]-ival[0]+1 for ival in merged]) - len(off))

def p2(sensors, lim):
    t0 = time.time()
    hole = None
    for y in range(lim):
        ivals = []
        for s in sensors:
            if s.by == y:
                ivals.append([s.bx, s.bx])
            dy = abs(s.sy - y)
            if dy <= s.dis:
                lb = s.sx - (s.dis - dy)
                ub = s.sx + (s.dis - dy)
                ivals.append([lb, ub])
        ivals.sort(key = lambda x: x[0])
        merged = [ivals.pop(0)]
        for ival in ivals:
            if merged[-1][1] < ival[0]:
                merged.append(ival)
            elif merged[-1][1] < ival[1]:
                merged[-1][1] = ival[1]

        if len(merged) == 1:
            assert merged[0][0] <= 0 and merged[0][1] >= lim
        else:
            for i in range(1, len(merged)):
                pre = merged[i-1]
                cur = merged[i]
                dx = cur[0] - pre[1]
                if dx > 1:
                    assert dx == 2
                    hole = (pre[1] + 1, y)
                    # print(merged)
        if hole is not None:
            break
    t1 = time.time()
    print(f"solution {hole[0]*lim + hole[1]} (t={t1-t0})")

with open(f"{os.getcwd()}/input.txt", "r") as f:
    sensors = []
    for l in f:
        if m := re.search("x=([0-9-]+), y=([0-9-]+): closest beacon is at x=([0-9-]+), y=([0-9-]+)", l):
            sx, sy, bx, by = m.groups()
            sb = Sb(sx, sy, bx, by)
            sensors.append(sb)

    # pprint.pp(sensors)
    # p1(sensors, 2000000)
    p2(sensors, 4000000)

## Problem 16
import os
import re
import time

mscore = 0

def p1(v, init_pend):
    t0 = time.time()

    def sim(src, pos, score, tleft, pend):
        if len(pend) == 0:
            return # all valves open
        if tleft <= 1:
            return # nothing you can do with one minute left

        global mscore
        flo, cands = v[pos]
        can_open = flo > 0 and pos in pend

        # in both cant open or choose not to open scenarios doubling back is wasteful
        for dst in cands:
            if src != dst:
                sim(src=pos, pos=dst, score=score, tleft=tleft-1, pend=pend.copy())

        # open this valve
        if can_open:
            npend = pend - {pos}
            ntleft = tleft - 1
            nscore = score + ntleft * flo
            mscore = max(mscore, nscore)
            for dst in cands:
                sim(src=pos, pos=dst, score=nscore, tleft=ntleft-1, pend=npend.copy())

    sim(src=None, pos='AA', score=0, tleft=30, pend=init_pend)
    t1 = time.time()
    print(f"{mscore} (t={t1-t0})")

def p2(v, init_pend):
    t0 = time.time()
    memo = {}

    def solve(src_a, src_b, pos_a, pos_b, tleft, pend):
        sim_key = ''.join(sorted([pos_a] + [pos_b]) + sorted(pend))
        if sim_key in memo:
            sim_tleft, sim_val = memo[sim_key]
            if tleft <= sim_tleft: # cached is an upper bound value and we just care about a max
                return sim_val

        if len(pend) == 0:
            return 0 # all valves open
        if tleft <= 1:
            return 0 # nothing you can do with one minute left

        max_flo = 0
        max_flo2 = 0
        for key in pend:
            flo = v[key][0]
            if flo > max_flo:
                max_flo2 = max_flo
                max_flo = flo
            elif max_flo2 < flo < max_flo:
                max_flo2 = flo

        flo_a, cands_a = v[pos_a]
        flo_b, cands_b = v[pos_b]
        can_open_a = flo_a > 0 and pos_a in pend
        can_open_b = flo_b > 0 and pos_b in pend

        # if two minutes left and cant open any valves here game is also over
        if tleft == 2 and not (can_open_a or can_open_b):
            return 0

        possible_scores = [0]

        # you cannot ignore a valve if:
        # 1. there is only one left
        # 2. it is the max flo valve
        # 3. it is the second largest flo valve and the other valve at play is the max flo one
        cond_3_a = (flo_a == max_flo2 and flo_b == max_flo)
        cond_3_b = (flo_b == max_flo2 and flo_a == max_flo)
        must_open_a = can_open_a and (len(pend) == 1 or flo_a == max_flo or cond_3_a)
        must_open_b = can_open_b and (len(pend) == 1 or flo_b == max_flo or cond_3_b)

        # both A and B cant open or choose not to open
        if not (must_open_a or must_open_b):
            for dst_a in cands_a:
                for dst_b in cands_b:
                    if src_a != dst_a and src_b != dst_b and dst_a != dst_b:
                        s = solve(src_a=pos_a, src_b=pos_b, pos_a=dst_a, pos_b=dst_b, tleft=tleft-1, pend=pend.copy())
                        possible_scores.append(s)

        # A opens valve, B cant open or chooses not to open (force A to move to same spot to penalize his minute)
        if can_open_a and not must_open_b:
            npend = pend - {pos_a}
            nscore = (tleft - 1) * flo_a
            dst_a = pos_a
            for dst_b in cands_b:
                if src_b != dst_b and dst_a != dst_b:
                    s = nscore + solve(src_a=pos_a, src_b=pos_b, pos_a=dst_a, pos_b=dst_b, tleft=tleft-1, pend=npend.copy())
                    possible_scores.append(s)

        # B opens valve, A cant open or chooses not to open (force B to move to same spot to penalize his minute)
        if can_open_b and not must_open_a:
            npend = pend - {pos_b}
            nscore = (tleft - 1) * flo_b
            dst_b = pos_b
            for dst_a in cands_a:
                if src_a != dst_a and dst_a != dst_b:
                    s = nscore + solve(src_a=pos_a, src_b=pos_b, pos_a=dst_a, pos_b=dst_b, tleft=tleft-1, pend=npend.copy())
                    possible_scores.append(s)

        # both A and B open valves (force their move to self loop to penalize the minute)
        if can_open_a and can_open_b and (pos_a != pos_b):
            npend = pend - {pos_a, pos_b}
            nscore = (tleft - 1) * (flo_a + flo_b)
            dst_a = pos_a
            dst_b = pos_b
            if dst_a != dst_b:
                s = nscore + solve(src_a=pos_a, src_b=pos_b, pos_a=dst_a, pos_b=dst_b, tleft=tleft-1, pend=npend.copy())
                possible_scores.append(s)

        best_score = max(possible_scores)
        memo[sim_key] = (tleft, best_score)
        return best_score

    mscore = solve(src_a=None, src_b=None, pos_a='AA', pos_b='AA', tleft=26, pend=init_pend)
    t1 = time.time()
    print(f"mscore={mscore} (t={t1-t0})")

with open(f"{os.getcwd()}/input.txt", "r") as f:
    v = {}
    init_pend = set()
    for l in f:
        if m := re.search("Valve ([A-Z]+) has flow rate=([0-9]+); tunnels? leads? to valves? (.*)", l):
            src, fstr, dsts = m.groups()
            val = (int(fstr), tuple(dsts.split(', ')))
            v[src] = val
            if val[0] > 0:
                init_pend.add(src)
        else:
            raise ValueError(l)

    # pprint.pp(v)
    # p1(v, init_pend)
    p2(v, init_pend)

## Problem 17
import os
import time
import numpy as np
from collections import defaultdict

def paint(m, frok):
    rmin = np.min(frok[:,0])
    nr, nc = m.shape
    mm = m[rmin:nr,:].copy()
    srok = frok - [rmin, 0]
    for row in srok:
        mm[row[0], row[1]] = '@'
    for row in mm:
        print(''.join(row))
    print('')

def collides(m, rok):
    for row in rok:
        if m[row[0], row[1]] != '.':
            return True
    return False

def solve(roks, jets, nbox):
    t0 = time.time()
    nr, nc = 10**4, 7
    m = np.full([nr, nc], '.')
    fast_solved = False

    ridx = 0
    jidx = 0
    patt = defaultdict(list)
    rpos = nr - 1
    for s in range(nbox):
        key = (ridx, jidx)
        if len(patt[key]) > 1:
            fst = patt[key][0]
            sec = patt[key][1]
            sd = sec[0] - fst[0]
            hd = sec[1] - fst[1]
            if (nbox - fst[0]) % sd == 0:
                mults = (nbox - sec[0]) // sd
                soln = sec[1] + mults*hd
                t1 = time.time()
                print(f"fast_solved[{(s, ridx, jidx, fst, sec, sd, hd, mults)}] height = {soln} (t={t1-t0})")
                fast_solved = True
                break

        patt[key].append((s, nr - rpos - 1))
        rok = roks[ridx]
        ridx = (ridx + 1) % len(roks)
        rmax = np.max(rok[:,0])
        rok = rok + [rpos - rmax - 3, 2]
        # paint(m, rok)

        rbnd = np.array([np.min(rok[:,0]), np.max(rok[:,0])])
        cbnd = np.array([np.min(rok[:,1]), np.max(rok[:,1])])
        for _ in range(1000):
            jet = jets[jidx]
            jidx = (jidx + 1) % len(jets)
            if jet == '<' and cbnd[0] - 1 >= 0:
                nrok = rok - [0,1]
                if not collides(m, nrok):
                    rok = nrok
                    cbnd -= 1
            elif jet == '>' and cbnd[1] + 1 < nc:
                nrok = rok + [0,1]
                if not collides(m, nrok):
                    rok = nrok
                    cbnd += 1

            moved_down = False
            if rbnd[1] + 1 < nr:
                nrok = rok + [1, 0]
                if not collides(m, nrok):
                    rok = nrok
                    rbnd += 1
                    moved_down = True
            if not moved_down:
                rpos = min(rpos, rbnd[0]-1)
                for row in rok:
                    m[row[0], row[1]] = '#'
                break
        else:
            assert False

    if not fast_solved:
        t1 = time.time()
        print(f"height = {nr - rpos - 1} (t={t1-t0})")

with open(f"{os.getcwd()}/input.txt", "r") as f:
    jets = f.read().rstrip()

    r0 = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    r1 = np.array([[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]])
    r2 = np.array([[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]])
    r3 = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    r4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    roks = [r0, r1, r2, r3, r4]
    # solve(roks, jets, 2022)
    solve(roks, jets, 1000000000000)

## Problem 18
import os
import numpy as np
import itertools as it
from collections import deque, defaultdict

def get_cubes(f):
    pts = []
    for l in f:
        x, y, z = l.rstrip().split(',')
        pts.append((int(x)+1, int(y)+1, int(z)+1))

    m = np.zeros((max([p[0] for p in pts])+2, max([p[1] for p in pts])+2, max([p[2] for p in pts])+2))
    for x,y,z in pts:
        m[x,y,z] = 1

    return pts, m

def p1(pts, m):
    ts = 0
    nx, ny, nz = m.shape
    for x,y,z in pts:
        s = 6

        if x + 1 < nx:
            s -= m[x + 1, y, z]
        if y + 1 < ny:
            s -= m[x, y + 1, z]
        if z + 1 < nz:
            s -= m[x, y, z + 1]

        if x - 1 >= 0:
            s -= m[x - 1, y, z]
        if y - 1 >= 0:
            s -= m[x, y - 1, z]
        if z - 1 >= 0:
            s -= m[x, y, z - 1]

        ts += s
    print(ts)

def p2(pts, m):
    # memo caches result of scanning starting from coord (not in m) toward d along ax, result can be None or a face
    # coord --> { (d, ax) --> face }
    memo = defaultdict(dict)
    faces = set() # face: ((x,y,z), (d, ax)) belongs to cube at x,y,z (in m) that got hit by steam moving along d on ax
    # face 0 is sentinel value of nothing
    done_cubes = set() # (x,y,z) here once scans have emitted from him already, (these are only for 0s in m)

    # these outer layers are all 0 in m
    pend = set()
    nx, ny, nz = m.shape
    for x,y,z in it.product([0, nx-1], range(ny), range(nz)):
        pend.add((x,y,z))
    for x,y,z in it.product(range(nx), [0, ny-1],  range(nz)):
        pend.add((x,y,z))
    for x,y,z in it.product(range(nx), range(ny),  [0, nz-1]):
        pend.add((x,y,z))
    pend = deque(pend)

    for _ in range(100000):
        cb = pend.popleft()
        cd = memo.get(cb, {})

        # increasing along x-axis
        dax = (1, 0)
        if cd.get(dax, None) is None:
            x,y,z = cb
            cbs = [cb]
            fc = 0 # sentinel
            for i in range(x+1, nx):
                if m[i, y, z] == 1:
                    fc = ((i, y, z), dax)
                    faces.add(fc)
                    break
                else:
                    cbs.append((i,y,z))
            # cache that all cubes in 'cbs' strike face 'fc' (which could be sentinel)
            # add them to pending cubes for search
            for i, new_cb in enumerate(cbs):
                memo[new_cb][dax] = fc
                if i > 0 and new_cb not in done_cubes: pend.append(new_cb)

        # increasing along y-axis
        dax = (1,1)
        if cd.get(dax, None) is None:
            x,y,z = cb
            cbs = [cb]
            fc = 0
            for i in range(y+1, ny):
                if m[x, i, z] == 1:
                    fc = ((x, i, z), dax)
                    faces.add(fc)
                    break
                else:
                    cbs.append((x,i,z))
            for i, new_cb in enumerate(cbs):
                memo[new_cb][dax] = fc
                if i > 0 and new_cb not in done_cubes: pend.append(new_cb)

        # increasing along z-axis
        dax = (1,2)
        if cd.get(dax, None) is None:
            x,y,z = cb
            cbs = [cb]
            fc = 0
            for i in range(z+1, nz):
                if m[x, y, i] == 1:
                    fc = ((x, y, i), dax)
                    faces.add(fc)
                    break
                else:
                    cbs.append((x,y,i))
            for i, new_cb in enumerate(cbs):
                memo[new_cb][dax] = fc
                if i > 0 and new_cb not in done_cubes: pend.append(new_cb)

        # decreasing along x-axis
        dax = (-1, 0)
        if cd.get(dax, None) is None:
            x,y,z = cb
            cbs = [cb]
            fc = 0
            for i in range(x-1, -1, -1):
                if m[i, y, z] == 1:
                    fc = ((i, y, z), dax)
                    faces.add(fc)
                    break
                else:
                    cbs.append((i,y,z))
            for i, new_cb in enumerate(cbs):
                memo[new_cb][dax] = fc
                if i > 0 and new_cb not in done_cubes: pend.append(new_cb)

        # decreasing along y-axis
        dax = (-1, 1)
        if cd.get(dax, None) is None:
            x,y,z = cb
            cbs = [cb]
            fc = 0
            for i in range(y-1, -1, -1):
                if m[x, i, z] == 1:
                    fc = ((x, i, z), dax)
                    faces.add(fc)
                    break
                else:
                    cbs.append((x,i,z))
            for i, new_cb in enumerate(cbs):
                memo[new_cb][dax] = fc
                if i > 0 and new_cb not in done_cubes: pend.append(new_cb)

        # decreasing along z-axis
        dax = (-1, 2)
        if cd.get(dax, None) is None:
            x,y,z = cb
            cbs = [cb]
            fc = 0
            for i in range(z-1, -1, -1):
                if m[x, y, i] == 1:
                    fc = ((x, y, i), dax)
                    faces.add(fc)
                    break
                else:
                    cbs.append((x,y,i))
            for i, new_cb in enumerate(cbs):
                memo[new_cb][dax] = fc
                if i > 0 and new_cb not in done_cubes: pend.append(new_cb)

        done_cubes.add(cb)
        if len(pend) == 0:
            break
    else:
        assert False

    print(len(faces))

with open(f"{os.getcwd()}/input.txt", "r") as f:
    pts, m = get_cubes(f)
    # p1(pts, m)
    p2(pts, m)

## Problem 19
import os
import re
import time
import pprint
from copy import copy
from dataclasses import dataclass
from collections import namedtuple

@dataclass
class Blueprint:
    id: int
    ore_bot_ore_cost: int
    clay_bot_ore_cost: int
    obsidian_bot_ore_cost: int
    obsidian_bot_clay_cost: int
    geode_bot_ore_cost: int
    geode_bot_obsidian_cost: int

    def __post_init__(self):
        self.id = int(self.id)
        self.ore_bot_ore_cost = int(self.ore_bot_ore_cost)
        self.clay_bot_ore_cost = int(self.clay_bot_ore_cost)
        self.obsidian_bot_ore_cost = int(self.obsidian_bot_ore_cost)
        self.obsidian_bot_clay_cost = int(self.obsidian_bot_clay_cost)
        self.geode_bot_ore_cost = int(self.geode_bot_ore_cost)
        self.geode_bot_obsidian_cost = int(self.geode_bot_obsidian_cost)

Bots = namedtuple("Bots", "ore clay obsidian geode")

def solve(bp, ore, clay, obsidian, bots, tleft):
    key = hash((bots, ore, clay, obsidian, tleft))
    if key in memo:
        return memo[key]

    if tleft == 1:
        return bots.geode

    can_build_geode = (ore >= bp.geode_bot_ore_cost and obsidian >= bp.geode_bot_obsidian_cost)
    if tleft == 2:
        if can_build_geode:
            return 2*bots.geode + 1
        else:
            return 2*bots.geode

    scores = []

    # if you can build a geode bot, always build it
    if can_build_geode:
        nore = ore - bp.geode_bot_ore_cost + bots.ore
        nobsidian = obsidian - bp.geode_bot_obsidian_cost + bots.obsidian
        nbots = Bots(bots.ore, bots.clay, bots.obsidian, bots.geode+1)
        s = bots.geode + solve(bp, nore, clay+bots.clay, nobsidian, nbots, tleft-1)
        scores.append(s)
    else:
        # just accrue minerals
        s = bots.geode + solve(bp, ore+bots.ore, clay+bots.clay, obsidian+bots.obsidian, bots, tleft-1)
        scores.append(s)

        # build an ore bot (below 4 rounds left makes no sense)
        if ore >= bp.ore_bot_ore_cost and tleft > 4:
            nore = ore - bp.ore_bot_ore_cost + bots.ore
            nbots = Bots(bots.ore+1, bots.clay, bots.obsidian, bots.geode)
            s = bots.geode + solve(bp, nore, clay+bots.clay, obsidian+bots.obsidian, nbots, tleft-1)
            scores.append(s)

        # build a clay bot (below 7 rounds left makes no sense)
        if ore >= bp.clay_bot_ore_cost and tleft > 7:
            nore = ore - bp.clay_bot_ore_cost + bots.ore
            nbots = Bots(bots.ore, bots.clay+1, bots.obsidian, bots.geode)
            s = bots.geode + solve(bp, nore, clay+bots.clay, obsidian+bots.obsidian, nbots, tleft-1)
            scores.append(s)

        # build an obsidian bot (below 4 rounds left makes no sense)
        if ore >= bp.obsidian_bot_ore_cost and clay >= bp.obsidian_bot_clay_cost and tleft > 4:
            nore = ore - bp.obsidian_bot_ore_cost + bots.ore
            nclay = clay - bp.obsidian_bot_clay_cost + bots.clay
            nbots = Bots(bots.ore, bots.clay, bots.obsidian+1, bots.geode)
            s = bots.geode + solve(bp, nore, nclay, obsidian+bots.obsidian, nbots, tleft-1)
            scores.append(s)

    best_score = max(scores)
    if tleft > 4: # use less storage for the leaves
        memo[key] = best_score
    return best_score

def p1(bps, init_bots, tleft):
    scores = {}
    t0 = time.time()
    for bp in bps:
        memo = {}
        st0 = time.time()
        mscore = solve(bp, ore=0, clay=0, obsidian=0, bots=init_bots, tleft=tleft)
        st1 = time.time()
        scores[bp.id] = (mscore, st1-st0)
        print(f"bp={bp.id} mscore={mscore} (t={st1-st0})")
    qsum = sum([bid * s[0] for bid, s in scores.items()])
    t1 = time.time()
    print(f"qsum={qsum} (t={t1-t0})")

def p2(bps, init_bots, tleft):
    scores = {}
    t0 = time.time()
    for bp in bps[0:3]:
        memo = {}
        st0 = time.time()
        mscore = solve(bp, ore=0, clay=0, obsidian=0, bots=init_bots, tleft=tleft)
        st1 = time.time()
        scores[bp.id] = (mscore, st1-st0)
        print(f"bp={bp.id} mscore={mscore} (t={st1-st0})")
    mul = np.prod([s[0] for _, s in scores.items()])
    t1 = time.time()
    print(f"mul={mul} (t={t1-t0})")

with open(f"{os.getcwd()}/input.txt", "r") as f:
    bps = []
    for l in f:
        if m := re.search("Blueprint (\d+): Each ore robot costs (\d+) ore. Each clay robot costs (\d+) ore. Each obsidian robot costs (\d+) ore and (\d+) clay. Each geode robot costs (\d+) ore and (\d+) obsidian.", l):
            bp = Blueprint(*m.groups())
            bps.append(bp)
        else:
            assert False
    # pprint.pp(bps)
    init_bots = Bots(1, 0, 0, 0)
    # p1(bps, init_bots, 24)
    p2(bps, init_bots, 32)
