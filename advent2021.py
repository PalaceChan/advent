## Problem 1
with open(f"{os.getcwd()}/input.txt", "r") as f:
    lines = [int(l) for l in f]

    # Part 1
    incs = 0
    for i in range(1, len(lines)):
        if lines[i] > lines[i-1]:
            incs += 1
    print(incs)

    # Part 2
    incs2 = 0
    for i in range(3, len(lines)):
        if sum(lines[i:(i-3):-1]) > sum(lines[(i-1):(i-4):-1]):
            incs2 += 1
    print(incs2)

## Problem 2 Part I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    hor, dep = 0, 0
    for l in f:
        i, v = l.strip().split(" ")
        v = int(v)
        if i == "forward":
            hor += v
        elif i == "up":
            dep -= v
        elif i == "down":
            dep += v
        else:
            assert(False)

    print(f'hor={hor} dep={dep} ans={hor*dep}')

## Problem 2 Part II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    hor, dep, aim = 0, 0, 0
    for l in f:
        i, v = l.strip().split(" ")
        v = int(v)
        if i == "forward":
            hor += v
            dep += aim * v
        elif i == "up":
            aim -= v
        elif i == "down":
            aim += v
        else:
            assert(False)

    print(f'hor={hor} dep={dep} aim={aim} ans={hor*dep}')

## Problem 3
from collections import defaultdict

with open(f"{os.getcwd()}/input.txt", "r") as f:
    lines = [l.rstrip() for l in f.readlines()]

    def compute_dicts_with_mask(mask):
        cnt0 = defaultdict(int)
        cnt1 = defaultdict(int)
        maxi = 0

        for k, l in enumerate(lines):
            if mask[k] == 0:
                continue

            for i, c in enumerate(l):
                if c == '0':
                    cnt0[i] += 1
                elif c == '1':
                    cnt1[i] += 1
                else:
                    assert(False)
            maxi = max(maxi, i)

        return cnt0, cnt1, maxi

    # Part I
    cnt0, cnt1, maxi = compute_dicts_with_mask([1] * len(lines))
    gamma = [0] * (maxi+1)
    for i in range(maxi+1):
        if cnt0[i] > cnt1[i]:
            gamma[i] = '0'
        elif cnt0[i] < cnt1[i]:
            gamma[i] = '1'
        else:
            assert(False)

    g = int(''.join(gamma), 2)
    e = (1 << (maxi+1)) - 1 - g
    print(f'{g} {e} ans = {g*e}')

    # Part II
    nox, oxy = len(lines), [1] * len(lines)
    ox = None
    for i in range(maxi+1):
        cnt0, cnt1, _ = compute_dicts_with_mask(oxy)
        if cnt0[i] > cnt1[i]:
            # print(f'bit {i+1} zero more common')
            for j, l in enumerate(lines):
                if oxy[j] > 0 and l[i] != '0':
                    nox -= 1
                    oxy[j] = 0
                    if nox == 1:
                        ox = lines[oxy.index(1)]

        elif cnt0[i] <= cnt1[i]:
            # print(f'bit {i+1} one more common or same')
            for j, l in enumerate(lines):
                if oxy[j] > 0 and l[i] != '1':
                    nox -= 1
                    oxy[j] = 0
                    if nox == 1:
                        ox = lines[oxy.index(1)]

        # print('left')
        # for k, l2 in enumerate(lines):
        #     if oxy[k] == 1:
        #         print(l2)

    noc, co2 = len(lines), [1] * len(lines)
    c2 = None
    for i in range(maxi+1):
        cnt0, cnt1, _ = compute_dicts_with_mask(co2)
        if cnt0[i] > cnt1[i]:
            for j, l in enumerate(lines):
                if co2[j] > 0 and l[i] != '1':
                    noc -= 1
                    co2[j] = 0
                    if noc == 1:
                        c2 = lines[co2.index(1)]

        elif cnt0[i] <= cnt1[i]:
            for j, l in enumerate(lines):
                if co2[j] > 0 and l[i] != '0':
                    noc -= 1
                    co2[j] = 0
                    if noc == 1:
                        c2 = lines[co2.index(1)]

    print(f'ox = {ox} c2 = {c2} ans2 = {int(ox,2) * int(c2,2)}')

## Problem 4
import numpy as np

with open(f"{os.getcwd()}/input.txt", "r") as f:
    lines = f.readlines()
    moves = [int(x) for x in lines[0].rstrip().split(',')]

    def get_board(rows):
        b = np.zeros((5,5))
        for i, row in enumerate(rows):
            for j, c in enumerate([x for x in row.rstrip().split(' ') if x != '']):
                b[i,j] = int(c)

        return b

    boards = []
    rows = []
    for i in range(1, len(lines)):
        if lines[i] == '\n':
            continue
        else:
            rows.append(lines[i])

        if len(rows) == 5:
            b = get_board(rows)
            boards.append(b)
            rows.clear()

    def mask_wins(m):
        return np.any(np.sum(m, 0) == 5) or np.any(np.sum(m, 1) == 5)

    def score_board(b, m):
        return np.sum(b[m == 0])

    masks = [np.zeros((5,5)) for _ in range(len(boards))]
    score = 0
    done = [False for _ in range(len(boards))]
    for mov in moves:
        for i, (b, m) in enumerate(zip(boards, masks)):
           m[b == mov] = 1
           if not done[i] and mask_wins(m):
               score = mov * score_board(b, m)
               done[i] = True
               if sum(done) == 1:
                   print(f'board {i} wins first, score = {score}')
               elif all(done):
                   print(f'board {i} wins last, score = {score}')

## Problem 5
import re
import numpy as np

with open(f"{os.getcwd()}/input.txt", "r") as f:
    segs = []
    maxv = 0
    for l in f:
        m = re.match(r'(\d+),(\d+) -> (\d+),(\d+)', l)
        seg = [int(i) for i in m.groups()]
        segs.append(seg)
        maxv = max(maxv, max(seg))

    # Part I
    g = np.zeros((maxv+1, maxv+1))
    for seg in segs:
        if seg[0] == seg[2] and seg[1] != seg[3]: # vertical
            a, b = min(seg[1], seg[3]), max(seg[1], seg[3])
            g[a:(b+1), seg[0]] += 1
        elif seg[1] == seg[3] and seg[0] != seg[2]: # horizontal
            a, b = min(seg[0], seg[2]), max(seg[0], seg[2])
            g[seg[1], a:(b+1)] += 1

    min_two_overlap = np.sum(g >= 2)

    # Part II
    g = np.zeros((maxv+1, maxv+1))
    for seg in segs:
        if seg[0] == seg[2] and seg[1] != seg[3]: # vertical
            a, b = min(seg[1], seg[3]), max(seg[1], seg[3])
            g[a:(b+1), seg[0]] += 1
        elif seg[1] == seg[3] and seg[0] != seg[2]: # horizontal
            a, b = min(seg[0], seg[2]), max(seg[0], seg[2])
            g[seg[1], a:(b+1)] += 1
        elif abs(seg[0] - seg[2]) == abs(seg[1] - seg[3]): # diagonal
            # print(seg)
            step = abs(seg[0] - seg[2])
            xsgn = np.sign(seg[2] - seg[0])
            ysgn = np.sign(seg[3] - seg[1])
            for i in range(step+1):
                # print(f'marking {seg[0] +i*xsgn}, {seg[1] + i*ysgn}')
                g[seg[1] + i*ysgn, seg[0] +i*xsgn] += 1
        else:
            print(seg)
            assert(False)

    min_two_overlap = np.sum(g >= 2)

## Problem 6
import numpy as np
from collections import Counter

with open(f"{os.getcwd()}/input.txt", "r") as f:
    l = f.readlines()[0].rstrip()

    # Part I ndays = 80
    def solve_naive(l, ndays):
        x = np.array([int(c) for c in l.split(',')])
        # print(f'start: {x}')
        for i in range(1, ndays + 1):
            x -= 1
            spawn = np.sum(x == -1)
            x[x == -1] = 6
            x = np.append(x, np.repeat(8, spawn))
            # print(f'After {i} days {x}')

        return x.size

    def solve_less_naive(l, ndays):
        cnt = Counter([int(c) for c in l.split(',')])
        # print(f'start: {cnt}')
        while ndays > 0:
            min_key = min(cnt.keys())
            his_val = cnt[min_key]
            days_to_jump = min(ndays, min_key + 1)

            ndays -= days_to_jump
            alive_cnt = {k - days_to_jump: v for k,v in cnt.items() if k >= days_to_jump}
            reset_cnt = {6: v for k,v in cnt.items() if k < days_to_jump}
            birth_cnt = {8: reset_cnt.get(6, 0)}
            cnt = {i: alive_cnt.get(i, 0) + reset_cnt.get(i, 0) + birth_cnt.get(i, 0) for i in range(9)}
            # print(f'After {days_to_jump} days {cnt}')

        return np.sum([v for v in cnt.values()])

    print(solve_naive(l, 80))
    print(solve_less_naive(l, 256))

## Problem 7
import numpy as np

with open(f"{os.getcwd()}/input.txt", "r") as f:
    pos = np.array([int(c) for c in f.readlines()[0].rstrip().split(',')])

    def solve_constant():
        i, j = np.min(pos), np.max(pos)
        res = np.zeros(j+1-i)
        for k in range(i,j+1):
            res[k-i] = np.sum(np.abs(pos - k))

        best_pos = np.argmin(res) + i
        best_res = res[best_pos]
        print(f'align at {best_pos} for fuel of {best_res}')

    def solve_arithmetic():
        def calc_fuel_cost(n):
            return n/2 * (2 + (n - 1))

        i, j = np.min(pos), np.max(pos)
        res = np.zeros(j+1-i)
        for k in range(i,j+1):
            res[k-i] = sum([calc_fuel_cost(n) for n in np.abs(pos - k)])

        best_pos = np.argmin(res) + i
        best_res = res[best_pos]
        print(f'align at {best_pos} for fuel of {best_res}')

    solve_constant()
    solve_arithmetic()

## Problem 8
from collections import Counter
import itertools as it

with open(f"{os.getcwd()}/input.txt", "r") as f:
    xs = []
    ys = []
    for l in f:
        x, y = l.rstrip().split('|')
        xs.append(x.rstrip().split(' '))
        ys.append(y.lstrip().split(' '))

    dfwd = {'0': 'abcefg',
            '1': 'cf', #''.join(sorted('cf'))
            '2': 'acdeg', #''.join(sorted('acdeg'))
            '3': 'acdfg', #''.join(sorted('acdfg'))
            '4': 'bcdf', #''.join(sorted('bcdf'))
            '5': 'abdfg', #''.join(sorted('abdfg'))
            '6': 'abdefg', #''.join(sorted('abdefg'))
            '7': 'acf', #''.join(sorted('acf'))
            '8': 'abcdefg',
            '9': 'abcdfg', #''.join(sorted('abcdfg'))
            }
    drev = {v:k for k,v in dfwd.items()}

    alp = 'abcdefg'
    def brute(p, x, y):
        p_to_alp = {k:v for k,v in zip(p, alp)}
        used = set()
        res = {}
        for i, word in enumerate(x):
            word2 = ''.join(sorted([p_to_alp[c] for c in word]))
            if word2 not in used and word2 in drev:
                used.add(word2)
                res[''.join(sorted(word))] = drev[word2]
            else:
                return None
        return res

    def solve(x, y):
        for p in it.permutations(alp):
            res = brute(p, x, y)
            if res is None:
                continue
            num = int(''.join([res[''.join(sorted(w))] for w in y]))
            return num
        else:
            assert(False)

    def part2():
        sols = []
        for x, y in zip(xs, ys):
            sol = solve(x, y)
            sols.append(sol)
        return sum(sols)

    def part1():
        segs = {
            1 : 2,
            7 : 3,
            4 : 4,
            2 : 5,
            3 : 5,
            5 : 5,
            0 : 6,
            6 : 6,
            9 : 6,
            8 : 7,
        }
        cnts = Counter(segs.values())

        n = 0
        for y in ys:
            for s in y:
                if cnts[len(s)] == 1:
                    n += 1
        return n

## Problem 9
import numpy as np
import itertools as it

with open(f"{os.getcwd()}/input.txt", "r") as f:
    m = np.stack([np.array([int(c) for c in l.rstrip()]) for l in f])

    lows = []
    for i, j in it.product(range(m.shape[0]), range(m.shape[1])):
        x = m[i,j]
        if i-1 >= 0 and m[i-1, j] <= x:
            continue
        if j-1 >= 0 and m[i, j-1] <= x:
            continue
        if i+1 < m.shape[0] and m[i+1, j] <= x:
            continue
        if j+1 < m.shape[1] and m[i, j+1] <= x:
            continue
        lows.append((i, j))

    # print(f'part I {sum([m[i,j]+1 for i,j in lows])}')
    cnts = {l: 1 for l in lows}
    for i, j in it.product(range(m.shape[0]), range(m.shape[1])):
        x = m[i,j]
        if x == 9 or (i,j) in cnts:
            continue
        a, b, y = i, j, x
        while (a,b) not in cnts:
            if a-1 >= 0 and m[a-1, b] < y:
                a, b, y = a-1, b, m[a-1, b]
            elif b-1 >= 0 and m[a, b-1] < y:
                a, b, y = a, b-1, m[a, b-1]
            elif a+1 < m.shape[0] and m[a+1, b] < y:
                a, b, y = a+1, b, m[a+1, b]
            elif b+1 < m.shape[1] and m[a, b+1] < y:
                a, b, y = a, b+1, m[a, b+1]
            else:
                assert(False)
        cnts[(a,b)] += 1

    print(np.prod(sorted(cnts.values())[-3:]))

## Problem 10
from collections import deque

with open(f"{os.getcwd()}/input.txt", "r") as f:
    ls = [l.rstrip() for l in f.readlines()]

    otc = {k:v for k,v in zip('([{<', ')]}>')}
    cto = {v:k for k,v in otc.items()}
    its = {k:v for k,v in zip(')]}>', [3, 57, 1197, 25137])}
    def p1_score(l):
        stack = deque()
        for x in l:
            if x in otc.keys():
                stack.append(x)
            elif x in cto.keys():
                o = cto[x]
                if len(stack) == 0 or stack.pop() != o:
                    return its[x]
        else:
            return 0

    scores1 = [p1_score(l) for l in ls]
    # print(f'part i score {sum(scores1)}')

    its = {k:v for k,v in zip(')]}>', [1, 2, 3, 4])}
    def p2_solve(l):
        stack = deque()
        for x in l:
            if x in otc.keys():
                stack.append(x)
            else:
                stack.pop()
        fin = [otc[x] for x in list(stack)[::-1]]
        scr = 0
        for x in fin:
            scr = 5*scr + its[x]
        return (fin, scr)

    scores2 = [p2_solve(ls[i]) for i in range(len(scores1)) if scores1[i] == 0]
    # print(f'part ii {np.median([v for _,v in scores2])}')

## Problem 11
import numpy as np

with open(f"{os.getcwd()}/input.txt", "r") as f:
    m = np.stack([np.array([int(c) for c in l.rstrip()]) for l in f])

    n = 1000
    fps = [0] * (n+1)
    for k in range(1, n+1):
        m += 1
        will_flash = set()
        while True:
            fij = {(i,j) for i,j in np.argwhere(m > 9)} - will_flash
            will_flash.update(fij)
            if len(fij) == 0:
                break
            else:
                for i, j in fij:
                    if i-1 >= 0:
                        m[i-1, j] += 1
                    if i-1 >= 0 and j-1 >= 0:
                        m[i-1, j-1] += 1
                    if j-1 >= 0:
                        m[i, j-1] += 1
                    if j-1 >= 0 and i+1 < m.shape[0]:
                        m[i+1, j-1] += 1
                    if i+1 < m.shape[0]:
                        m[i+1, j] += 1
                    if i+1 < m.shape[0] and j+1 < m.shape[1]:
                        m[i+1, j+1] += 1
                    if j+1 < m.shape[1]:
                        m[i, j+1] += 1
                    if j+1 < m.shape[1] and i-1 >= 0:
                        m[i-1, j+1] += 1

        m[m > 9] = 0
        fps[k] = len(will_flash)

    # print(f'part i = {sum(fps[1:101])}')
    # print(f'part ii = {fps.index(100)}')

## Problem 12
from collections import defaultdict

with open(f"{os.getcwd()}/input.txt", "r") as f:
    opts = defaultdict(list)
    smol = set()
    for l in f:
        a, b = l.rstrip().split('-')
        opts[a].append(b)
        opts[b].append(a)
        if a == a.lower():
            smol.add(a)
        if b == b.lower():
            smol.add(b)

    sols = []

    def solve1(path, smalls):
        global sols
        for opt in opts[path[-1]]:
            if opt == 'end':
                sol = path + [opt]
                sols.append(sol)
            elif opt not in smalls:
                npath = path + [opt]
                nsmlls = smalls.union({opt}) if opt in smol else smalls
                solve1(npath, nsmlls)

    # solve1(['start'], {'start'})
    # print(sols)
    # print(len(sols))

    def solve2(path, smalls, can_violate):
        global sols
        # if path == ['start', 'A', 'b', 'A']:
        #     import pdb; pdb.set_trace()
        for opt in opts[path[-1]]:
            if opt == 'end':
                sol = path + [opt]
                sols.append(sol)
            elif opt == 'start':
                continue
            elif opt not in smol:
                npath = path + [opt]
                solve2(npath, smalls, can_violate)
            elif opt not in smalls or can_violate:
                assert(opt in smol)
                npath = path + [opt]
                nsmlls = smalls.union({opt})
                ncan_violate = opt not in smalls if can_violate else False
                solve2(npath, nsmlls, ncan_violate)

    solve2(['start'], {'start'}, True)
    # print(sols)
    print(len(sols))

## Problem 13
import numpy as np

with open(f"{os.getcwd()}/input.txt", "r") as f:
    points = []
    folds = []
    xdim, ydim = 0, 0
    for l in f:
        if l == '\n':
            break
        x, y = l.rstrip().split(',')
        points.append([int(x), int(y)])
        xdim = max(xdim, points[-1][0])
        ydim = max(ydim, points[-1][1])
    for l in f:
        k, v = l.rstrip().replace('fold along ', '').split('=')
        folds.append([k, int(v)])

    g = np.zeros((ydim+1, xdim+1))
    for x, y in points:
        g[y,x] = 1

    def debug_with_set(n):
        pset = {(x,y) for x,y in points}

        def _fold(pset, ax, v):
            if ax == 'y':
                res = set()
                for x, y in pset:
                    if y > v:
                        res.add((x, 2*v - y))
                    else:
                        res.add((x, y))
            else:
                res = set()
                for x, y in pset:
                    if x > v:
                        res.add((2*v - x, y ))
                    else:
                        res.add((x, y))
            return res

        for ax, v in folds[0:n]:
            pset = _fold(pset, ax, v)

        xdim, ydim = 0, 0
        for x, y in pset:
            xdim = max(xdim, x)
            ydim = max(ydim, y)

        g = np.zeros((ydim+1, xdim+1))
        for x, y in pset:
            g[y,x] = 1

        return g

    def fold_up(g, v):
        for i in range(v+1, g.shape[0]):
            j = 2*v - i
            g[j, :] = np.maximum(g[i, :], g[j, :])

        return g[0:v, :]

    def fold_left(g, v):
        for i in range(v+1, g.shape[1]):
            j = 2*v - i
            g[:, j] = np.maximum(g[:, i], g[:, j])

        return g[: ,0:v]

    for i, (ax, v) in enumerate(folds[0:12]):
        g = fold_up(g, v) if ax == 'y' else fold_left(g, v)
        if i == 0:
            print(f'part i {g.sum()}')

    def pretty_print(g, x):
        rows = [''.join([{0: ' ', 1: x}[int(c)] for c in g[i,:]]) for i in range(g.shape[0])]
        return '\n'.join(rows)

    # g2 = debug_with_set(12)
    # print(np.array_equal(g, g2))

    print('part ii')
    print(pretty_print(g, 'O'))

## Problem 14
import math
from collections import defaultdict

with open(f"{os.getcwd()}/input.txt", "r") as f:
    for l in f:
        if l == '\n':
            break
        x = l.rstrip()

    count = {} # final answers by letter
    bylet = {} # a memo table for each letter
    rules = {}
    chars = set()
    for l in f:
        k, v = l.rstrip().replace(' -> ', ',').split(',')
        chars.add(v)
        chars.add(k[0])
        chars.add(k[1])
        rules[k] = v

    count = {k:0 for k in chars}
    bylet = {k : {} for k in chars}
    for k in x:
        count[k] += 1

    def expand(k, pair, n):
        global count
        global bylet
        if n == 0:
            return 0
        elif (pair, n) in bylet[k]:
            return bylet[k][(pair,n)]
        elif pair in rules:
            kn = rules[pair]
            vv = (k == kn)
            pl = pair[0] + kn
            pr = kn + pair[1]
            cnt = vv + expand(k, pl, n-1) + expand(k, pr, n-1)
            bylet[k][(pair, n)] = cnt
            return cnt


    N = 40
    for i in range(1, len(x)):
        pair = x[(i-1):(i+1)]
        for k in chars:
            cnt = expand(k, pair, N)
            count[k] += cnt

    mn, mx = math.inf, 0
    for _, v in count.items():
        if v < mn:
            mn = v
        if v > mx:
            mx = v

    print(f'N={N} sol is {mx} - {mn} = {mx - mn}')

## Problem 15
import numpy as np
import itertools as it

with open(f"{os.getcwd()}/input.txt", "r") as f:
    rows = []
    for l in f:
        rows.append([int(c) for c in l.rstrip()])
    m = np.array(rows)

    part = 2
    if part == 2:
        m2 = np.zeros((m.shape[0]*5, m.shape[1]*5))
        m2[0:m.shape[0], 0:m.shape[1]] = m
        for c in range(1,5):
            src = m2[0:m.shape[0], ((c-1)*m.shape[1]):(c*m.shape[1])]
            dst = src + 1
            dst[dst > 9] = 1
            m2[0:m.shape[0], (c*m.shape[1]):((c+1)*m.shape[1])] = dst

        for r in range(1, 5):
            for c in range(5):
                src = m2[((r-1)*m.shape[0]):(r*m.shape[0]), (c*m.shape[1]):((c+1)*m.shape[1])]
                dst = src + 1
                dst[dst > 9] = 1
                m2[(r*m.shape[0]):((r+1)*m.shape[0]), (c*m.shape[1]):((c+1)*m.shape[1])] = dst

        m = m2

    Q = {(i,j) for i, j in it.product(range(m.shape[0]), range(m.shape[1]))}
    prev = [None] * m.size
    dist = [np.inf] * m.size
    dist[0] = 0

    print(f'starting, Q len {len(Q)}')
    while len(Q) > 0:
        mi, mj, md = -1, -1, np.inf
        for i,j in Q:
            idx = i*m.shape[1] + j
            d = dist[idx]
            if d < md:
                mi, mj, md = i, j, d

        if (mi, mj) == (m.shape[0]-1, m.shape[1]-1):
            print(f'finished early, Q still len {len(Q)}')
            break

        Q.remove((mi,mj))
        for ni, nj in [(mi-1, mj), (mi, mj-1), (mi+1, mj), (mi, mj+1)]:
            if (ni,nj) in Q:
                idxu = mi*m.shape[1] + mj
                idxv = ni*m.shape[1] + nj
                alt = dist[idxu] + m[ni, nj]
                if alt < dist[idxv]:
                    dist[idxv] = alt
                    prev[idxv] = idxu

    print(f'part {part} min risk {dist[-1]}')

## Problem 16
with open(f"{os.getcwd()}/input.txt", "r") as f:
    m = f.readline().rstrip()
    b = ''
    p = 0
    for c in m:
        x = bin(int(c, 16)).replace('0b', '')
        prefix = '0' * (4 - len(x))
        x = prefix + x
        b += x

    class Node:
        def __init__(self):
            self.version = None
            self.ptype = None
            self.value = None
            self.ltype = None
            self.lvalue = None
            self.children = []

        def decode(self, b, p):
            pstart = p
            self.version = int(b[p:(p+3)], 2)
            p += 3
            self.ptype = int(b[p:(p+3)], 2)
            p += 3

            if self.ptype == 4:
                pend = self.decode_literal(b, p)
            else:
                pend = self.decode_operator(b, p)

            return pend - pstart

        def decode_literal(self, b, p):
            last_group = False
            literal = ''
            while not last_group:
                last_group = (b[p] == '0')
                p += 1
                literal += b[p:(p+4)]
                p += 4

            self.value = int(literal, 2)
            return p

        def decode_operator(self, b, p):
            pstart = p
            self.ltype = b[p]
            p += 1
            if self.ltype == '0':
                self.lvalue = int(b[p:(p+15)], 2)
                p += 15

                bits_left_to_parse = self.lvalue
                while bits_left_to_parse > 0:
                    child = Node()
                    bits = child.decode(b, p)
                    self.children.append(child)
                    p += bits
                    bits_left_to_parse -= bits

            else:
                self.lvalue = int(b[p:(p+11)], 2)
                p += 11

                for _ in range(self.lvalue):
                    child = Node()
                    bits = child.decode(b, p)
                    self.children.append(child)
                    p += bits

            return p

        def vsum(self):
            res = self.version
            for c in self.children:
                res += c.vsum()

            return res

        def calc(self):
            if self.ptype == 0:
                self.value = 0
                for c in self.children:
                    self.value += c.value
            elif self.ptype == 1:
                self.value = 1
                for c in self.children:
                    self.value *= c.value
            elif self.ptype == 2:
                self.value = min([c.value for c in self.children])
            elif self.ptype == 3:
                self.value = max([c.value for c in self.children])
            elif self.ptype == 4:
                assert(self.value is not None)
            elif self.ptype == 5:
                assert(len(self.children) == 2)
                v0 = self.children[0].value
                v1 = self.children[1].value
                self.value = int(v0 > v1)
            elif self.ptype == 6:
                v0 = self.children[0].value
                v1 = self.children[1].value
                self.value = int(v0 < v1)
            elif self.ptype == 7:
                v0 = self.children[0].value
                v1 = self.children[1].value
                self.value = int(v0 == v1)
            else:
                assert(False, f'ptype of {ptype} invalid')

        def visit(self):
            for c in self.children:
                c.visit()
            self.calc()


    root = Node()
    bits = root.decode(b, p)
    root.visit()

## Problem 17
import os
import re

with open(f"{os.getcwd()}/input.txt", "r") as f:
    l = f.readline().rstrip()
    m = re.match(r'target area: x=(-?\d+)..(-?\d+), y=(-?\d+)..(-?\d+)', l)
    x0, x1, y0, y1 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))

    def solve_x(x0):
        for n in range(2**32):
            if n * (n+1) / 2 >= x0:
                break
        return n

    def solve_y(y0, n):
        return -1*y0 - 1

    def trip(i):
        return i * (i+1) / 2

    x = solve_x(x0)
    y = solve_y(y0, x)

    print(f'part i ({x},{y}) with trip of {trip(y)}')

    def solve_p2(x, y, x0, x1, y0, y1):
        one_shots = (x1-x0+1)*(y1-y0+1)

        def _works(a, b):
            px, py = 0, 0
            while True:
                px += a
                py += b
                a = max(0, a - 1)
                b -= 1
                hit = (px >= x0 and px <= x1 and py >= y0 and py <= y1)
                if hit:
                    return True
                out = (px > x1 or py < y0)
                if out:
                    return False

            others = 0
            for a in range(x, x0):
                for b in range(y0, y+1)
                if _works(a, b):
                    others += 1

            return one_shots + others

    p2 = solve_p2(x, y, x0, x1, y0, y1)
    print(f'part ii {p2}')
