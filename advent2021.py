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

## Problem 18
import os
import functools

with open(f"{os.getcwd()}/input.txt", "r") as f:
    lines = f.readlines()

    class Node:
        def __init__(self, depth):
            self.depth = depth
            self.value = None
            self.left = None
            self.right = None
            self.parent = None

        def __add__(self, oth):
            root = Node(0)
            root.left = self
            root.right = oth
            root.left.parent = root
            root.right.parent = root

            root.left.bury()
            root.right.bury()
            return root

        def bury(self):
            self.depth += 1
            if self.left is not None:
                self.left.bury()
            if self.right is not None:
                self.right.bury()

        def to_str(self):
            if self.value is not None:
                return str(self.value)
            else:
                return f'[{self.left.to_str(), {self.right.to_str()}]'

        def __repr__(self):
            return self.to_str()

        def shed(self, v, where):
            prv = self
            cur = self.parent
            while cur is not None:
                n = cur.left if where == 'left' else cur.right
                if n is not prv:
                    if where == 'left':
                        while n.right is not None and n.value is None:
                            n = n.right
                    else:
                        while n.left is not None and n.value is None:
                            n = n.left

                    if n.value is not None:
                        n.value += v
                        return True

                prv = cur
                cur = cur.parent

        def explode(self):
            self.shed(self.left.value, 'left')
            self.shed(self.right.value, 'right')

            self.value = 0
            self.left = None
            self.right = None

        def split(self):
            d = self.depth + 1
            self.left = Node(d)
            self.right = Node(d)
            self.left.parent = self
            self.right.parent = self
            self.left.value = int(self.value / 2)
            self.right.value = int(self.value / 2 + 0.5)
            self.value = None

        def try_explode(self):
            l = self.left
            r = self.right

            if l is not None and l.try_explode():
                return True

            if l is not None and l.value is not None and r is not None and r.value is not None and self.depth == 4:
                self.explode()
                return True

            if r is not None and r.try_explode():
                return True

            return False

        def try_split():
            l = self.left
            r = self.right

            if l is not None and l.try_split():
                return True

            if self.value is not None and self.value >= 10:
                self.split()
                return True

            if r is not None and r.try_split():
                return True

            return False

        def reduce(self):
            while self.try_explode() or self.try_split():
                pass
            return self

        def magnitude(self):
            if self.value is not None:
                return self.value
            else:
                return 3 * self.left.magnitude() + 2 * self.right.magnitude()

    def to_tree(line):
        d = 0
        root = Node(d)
        curr = root
        for c in line:
            if c == '[':
                d += 1
                curr.left = Node(d)
                curr.right = Node(d)
                curr.left.parent = curr
                curr.right.parent = curr
                curr = curr.left
            elif c == ']':
                d -= 1
                curr = curr.parent
            elif c == ',':
                curr = curr.parent.right
            else:
                curr.value = int(c)

        return root

    def solve_p2(lines):
        smax = None
        mmax = -1
        for i in range(len(lines)):
            for j in range(len(lines)):
                a = to_tree(lines[i])
                b = to_tree(lines[i])
                c = (a + b).reduce()
                m = c.magnitude()
                if m > mmax:
                    smax = c
                    mmax = m

        return mmax, smax

    trees = [to_tree(l.rstrip()) for l in lines]
    acc = functools.reduce(lambda x,y: (x+y).reduce(), trees)
    print(f'part i sum is {acc} magnitude of {acc.magnitude()}')

    mmax, smax = solve_p2([l.rstrip() for l in lines])
    print(f'part ii mmax is {mmax}')

## Problem 19
import os
import numpy as np
import itertools as it
from dataclasses import dataclass
from typing import Any

with open(f"{os.getcwd()}/input.txt", "r") as f:
    @dataclass
    class Scanner:
        idx: int
        pos: Any
        orient: Any
        points: Any

    linno = 0
    lines = [l.rstrip() for l in f]
    scanners = []
    while linno < len(lines):
        if lines[linno].startswith('---'):
            s = Scanner(len(scanners), None, None, None)
            p = []
            if s.idx = 0:
                s.pos = np.array([0, 0, 0])
                s.orient = np.eye(3, dtype=int)
            linno += 1
            while linno < len(lines):
                if lines[linno] == '':
                    break
                x, y, z = lines[linno].split(',')
                xyz = np.array([int(x), int(y), int(z)])
                p.append(xyz)
                linno += 1
            s.points = np.vstack(p)
            scanners.append(s)

        linno += 1

    def symmetries24():
        res = []
        for o in it.product([1, -1], repeat=3):
            oa = np.array(o)
            for p in it.permutations([0,1,2]):
                pa = np.array(p)
                m = oa * np.eye(3, dtype=int)[:, pa]
                if np.linalg.det(m) > 0:
                    res.append(m)

        return res

    def intersect(x, y):
        assert(x.orient is not None and x.pos is not None)
        assert(y.orient is not None and y.pos is not None)
        xpoints = x.pos + x.points @ x.orient
        ypoints = y.pos + y.points @ y.orient
        rx = set([tuple(r) for r in xpoints])
        ry = set([tuple(r) for r in ypoints])
        rb = ry & rx
        return rb

    def resolve(x, y, s24):
        assert(x.orient is not None and x.pos is not None)
        xpoints = x.pos + x.points @ x.orient
        rx = set([tuple(r) for r in xpoints])
        for sym in s24:
            for px in xpoints:
                # if px a common point and sym the y orientation..
                ysym = y.points @ sym
                ypos = px - ysym
                for yp in ypos:
                    py = yp + ysym
                    ry = set([tuple(r) for r in py])
                    rb = ry & rx
                    assert(len(rb) <= 12)
                    if len(rb) == 12:
                        return yp, sym, rb
        return None

    def solve_p1():
        s24 = symmetries24()
        done = [0]
        pend = set(range(1, len(scanners)))

        # resolve scanners
        while len(pend) > 0:
            for x, y in it.product(done[::-1], pend):
                res = resolve(scanners[x], scanners[y], s24)
                if res is not None:
                    print(f'solved {y} using {x}')
                    ypos, yorient, _ = res
                    scanners[y].pos = ypos
                    scanners[y].orient = yorient
                    pend.remove(y)
                    done.append(y)
                    break
            else:
                assert(False, f'could not solve {pend} from {done}')

        # intersect them
        all_beacons = set()
        for x, y in it.product(done, repeat=2):
            if x <= y:
                rb = intersect(scanners[x], scanners[y])
                all_beacons.update(rb)

        return all_beacons

    def solve_p2():
        maxd = -1
        for x, y in it.product(range(len(scanners)), repeat=2):
            if x <= y:
                md = np.sum(np.abs(scanners[x].pos - scanners[y].pos))
                if md > maxd:
                    maxd = md

        return maxd

    print(f'part i ans = {len(solve_p1())}')
    print(f'part ii ans = {solve_p2}')

## Problem 20
import os
import numpy as np

with open(f"{os.getcwd()}/input.txt", "r") as f:
    pmap = {'#':1, '.':0}
    algo = [pmap[c] for c in f.readline().rstrip()]

    f.readline()
    rows = []
    for l in f:
        row = [pmap[c] for c in l.rstrip()]
        rows.append(np.array(row))

    s = len(rows)
    m = np.zeros((s+4, s+4), dtype=int)
    m[2:(s+2),2:(s+2)] = np.vstack(rows)

    def pprint(m):
        ipmap = {1:'#', 0:'.'}
        print('\n'.join([''.join([ipmap[c] for c in r]) for r in m]))

    def enhance(m):
        m2 = m.copy()

        # border
        bval = algo[0] if m[0,0] == 0 else algo[-1]
        m2[0, :] = bval
        m2[-1, :] = bval
        m2[:, 0] = bval
        m2[:, -1] = bval

        # inside
        for i, j in it.product(range(1, m.shape[0]-1), repeat=2):
            sm = m[(i-1):(i+2), (j-1):(j+2)]
            bs = ''.join([str(x) for x in sm.reshape(9)])
            idx = int(bs, 2)
            m2[i, j] = algo[idx]

        s = m2.shape[0]
        m3 = np.ones((s+4, s+4), dtype=int) * bval
        m3[2:(s+2),2:(s+2)] = m2

        return m3

    def solve_p1():
        m3 = enhance(enhance(m))
        return m3.sum()

    def solve_p2():
        def compose(f, n):
            def fn(x):
                for i in range(n):
                    x = f(x)
                return x
            return fn

        e50 = compose(enhance, 50)
        m50 = e50(m)
        return m50.sum()

    print(f'part i ans = {solve_p1()}}')
    print(f'part ii ans = {solve_p2()}}')

## Problem 21
import os
import re
import numpy as np
import itertools as it

with open(f"{os.getcwd()}/input.txt", "r") as f:
    p1 = int(re.match('Player \d starting position: (\d+)\n', f.readline()).group(1))
    p2 = int(re.match('Player \d starting position: (\d+)\n', f.readline()).group(1))

    def solve_p1():
        scr = [0, 0]
        pos = [p1, p2]
        who = 0
        die = it.cycle(range(1,101))
        num = 0
        while True:
            roll = next(die) + next(die) + next(die)
            num += 3
            pos[who] = (pos[who] + roll) % 10
            if pos[who] == 0:
                pos[who] = 10
            scr[who] += pos[who]
            if scr[who] >= 1000:
                break
            who = 1 if who == 0 else 0

        oth = 1 if who == 0 else 0
        return num * scr[oth]

    def solve_p2():
        #Counter([sum(list(x)) for x in it.product([1,2,3], repeat=3)])
        to_steps = np.array([3, 4, 5, 6, 7, 8, 9])
        to_univs = np.array([1, 3, 6, 7, 6, 3, 1])

        def step(ply, who, pos, opos, scr, oscr, idx):
            end = False
            ucnt = to_univs[idx]
            roll = to_steps[idx]
            pos = (pos + roll) % 10
            if pos == 0:
                pos = 10
            scr += pos
            if scr >= 21:
                end = True
                if who != ply:
                    ucnt = 0
            who = 1 if who == 0 else 0
            return end, who, opos, pos, oscr, scr, ucnt

        def solve(ply, who, pos, opos, scr, oscr, ucnt, idx):
            end, nwho, npos, nopos, nscr, noscr, nucnt = step(ply, who, pos, opos, scr, oscr, idx)
            if end:
                return ucnt * nucnt
            else:
                univs = 0
                for i in range(7):
                    univs += solve(ply, nwho, npos, nopos, nscr, noscr, ucnt * nucnt, i)
                return univs

            p1_univs = sum([solve(0, 0, p1, p2, 0, 0, 1, i) for i in range(7)])
            p2_univs = sum([solve(1, 0, p1, p2, 0, 0, 1, i) for i in range(7)])
            return max(p1_univs, p2_univs)

        print(f'part i ans = {solve_p1()}')
        print(f'part ii ans = {solve_p2()}')

## Problem 22
import os
import re
from typing import Tuple
from dataclasses import dataclass

with open(f"{os.getcwd()}/input.txt", "r") as f:
    @dataclass
    class Step():
        bit: int
        xb: Tuple
        yb: Tuple
        zb: Tuple

    steps = []
    for l in f:
        m = re.match(r'(\w+) x=(-?\d+)..(-?\d+),y=(-?\d+)..(-?\d+),z=(-?\d+)..(-?\d+)', l)
        bit = 1 if m.group(1) == 'on' else 0
        xb = (int(m.group(2)), int(m.group(3)))
        yb = (int(m.group(4)), int(m.group(5)))
        zb = (int(m.group(6)), int(m.group(7)))
        s = Step(bit, xb, yb, zb)
        steps.append(s)

    def solve_p1():
        m = np.zeros((102, 102, 102), dtype=int)
        for s in steps:
            xb = (max(s.xb[0] + 50, 0), min(s.xb[1] + 51, 101))
            yb = (max(s.yb[0] + 50, 0), min(s.yb[1] + 51, 101))
            zb = (max(s.zb[0] + 50, 0), min(s.zb[1] + 51, 101))
            xo = xb[0] > 100 or xb[1] < 1
            yo = yb[0] > 100 or yb[1] < 1
            zo = zb[0] > 100 or zb[1] < 1
            if xo or yo or zo:
                continue
            m[xb[0]:xb[1], yb[0]:yb[1], zb[0]:zb[1]] = s.bit

        return m.sum()

    @dataclass
    class Box:
        x0: int
        x1: int
        y0: int
        y1: int
        z0: int
        z1: int

        def points(self):
            return (self.x1 - self.x0 + 1) * (self.y1 - self.y0 + 1) * (self.z1 - self.z0 + 1)

        def __hash__(self):
            return hash((self.x0, self.x1, self.y0, self.y1, self.z0, self.z1))

    def solve_p2():
        bs = set()

        def subsumes(b, o):
            sx = (b.x0 <= o.x0 and o.x1 <= b.x1)
            sy = (b.y0 <= o.y0 and o.y1 <= b.y1)
            sz = (b.z0 <= o.z0 and o.z1 <= b.z1)

            return (sx and sy and sz)

        def overalps(b, o):
            dx = (b.x1 < o.x0 or o.x1 < b.x0)
            dy = (b.y1 < o.y0 or o.y1 < b.y0)
            dz = (b.z1 < o.z0 or o.z1 < b.z0)

            # not disjoint
            return not (dx or dy or dz)

        def remove_all_subsumed(b):
            dead = []
            for o in bs:
                if subsumes(b, o):
                    dead.append(o)
            for o in dead:
                bs.remove(o)

        def calc_diff(b, o):
            # for each face of o that falls between faces of b, yield a box
            diffs = []
            if b.x0 < o.x0:
                #+x
                diffs.append(Box(b.x0, o.x0 - 1, b.y0, b.y1, b.z0, b.z1))
            if b.y0 < o.y0:
                #+y
                diffs.append(Box(b.x0, b.x1, b.y0, o.y0 - 1, b.z0, b.z1))
            if b.z0 < o.z0:
                #+z
                diffs.append(Box(b.x0, b.x1, b.y0, b.y1, b.z0, o.z0 - 1))
            if o.x1 < b.x1:
                #-x
                diffs.append(Box(o.x1 + 1, b.x1, b.y0, b.y1, b.z0, b.z1))
            if o.y1 < b.y1:
                #-y
                diffs.append(Box(b.x0, b.x1, o.y1 + 1, b.y1, b.z0, b.z1))
            if o.z1 < b.z1:
                #-z
                diffs.append(Box(b.x0, b.x1, b.y0, b.y1, o.z1 + 1, b.z1))

            return diffs

        def add_on_box(b):
            # if subsumed by another box ignore
            for o in bs:
                if subsumes(o, b):
                    return

            # remove any boxes b subsumes
            remove_all_subsumed()

            # if overlaps with a box, calc diff boxes and add those instead
            diffs = []
            for o in bs:
                if overlaps(b, o):
                    diffs = calc_diff(b, o)
                    break

            if not diffs:
                bs.add(b)
            else:
                for db in diffs:
                    add_on_box(db)

        def add_off_box(b):
            # rmeove any boxes b subsumes
            remove_all_subsumed(b)

            # while overlap with a box, calc diff boxes, remove box and replace with diff boxes
            while True:
                olapb = None
                for o in bs:
                    if overlaps(b, o):
                        olapb = o
                        diffs = calc_diff(o, b)
                        break

                if olapb is not None:
                    bs.remove(olapb)
                    assert(len(diffs) > 0)
                    for db in diffs:
                        add_on_box(db)
                else:
                    break

        for i, s in enumerate(steps):
            b = Box(s.xb[0], s.xb[1], s.yb[0], s.yb[1], s.zb[0], s.zb[1])
            if s.bit == 1:
                add_on_box(b)
            else:
                add_off_box(b)

        n = 0
        for b in bs:
            n += b.points()
        return n

    print(f'part i ans = {solve_p1()}')
    print(f'part ii ans = {solve_p2()}')

## Problem 23
import os
import copy
import numpy as np
np.set_printoptions(linewidth=100000)
from dataclasses import dataclass

with open(f"{os.getcwd()}/input.txt", "r") as f:
    _, _   = f.readline(), f.readline()
    r0, r1 = f.readline(), f.readline()

    @dataclass
    class Pod:
        code: str
        cost: int
        x: int
        y: int
        goal_y: int

    cost = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
    goal = {'A': 3, 'B': 5, 'C': 7, 'D': 9}
    bady = [3, 5, 7, 9]

    opt_cost = 2**32

    def solve_p1():
        o = np.zeros((3, 13), dtype=str)
        o[:] = 'X'
        o[0, 1:12] = ' '

        pods = []
        for i, r in enumerate([r0, r1]):
            for j, c in enumerate(r):
                if c in cost.keys():
                    o[i+1, j] = c
                    pods.append(Pod(c, cost[c], i+1, j, goal[c]))

        def calc_to_hall(p, o):
            moves = []
            for d in [-1, 1]:
                c = p.x * p.cost
                x, y = 0, p.y

                while True:
                    y += d
                    c += p.cost
                    if not o[x,y] == ' ':
                        break
                    elif y not in bady:
                        dw = +1 if (p.y == p.goal_y) else 0
                        moves.append((c, x, y, dw))

            return moves

        def try_move_to_dst_room(p, o):
            c = p.x * p.cost
            d = np.sign(p.goal_y - p.y)
            x, y = 0, p.y

            # try to get to outside your dst room
            while y != p.goal_y:
                y += d
                c += p.cost
                if o[x,y] != ' ':
                    return []

            # go in as deep as can go
            while True:
                if (x+1) <= 2 and o[x+1,y] == ' ':
                    x += 1
                    c += p.cost
                else:
                    break

            if x == 0 or (x == 1 and o[2, y] != p.code):
                return []
            else:
                dw = -1
                return [(c, x, y, dw)]

        def calc_moves(idx, p, o):
            # if in own room:
            #    if at x=2, do nothing
            #    else assert x=1 and if behind u not ur bro, to hall
            # if in oth room:
            #    if at x=2 and blocked do nothing
            #    else assert x=1 try to go to own room (implies x=2)
            #         if fail to reach own room go to hall
            # if in hall:
            #    try to go to own room (implies x=2)
            #    if fail do nothing

            if p.y == p.goal_y:
                assert(p.x == 1 or p.x == 2)
                if p.x == 2:
                    return []
                elif o[2, p.y] == p.code:
                    return []
                else:
                    return calc_to_hall(p, o)

            if p.y != p.goal_y and p.x > 0:
                if p.x == 2 and o[1, p.y] != ' ':
                    return []
                else:
                    # at 2 unblocked or at 1 so also unblocked
                    move_to_dst_room = try_move_to_dst_room(p, o)
                    if move_to_dst_room:
                        return move_to_dst_room
                    else:
                        return calc_to_hall(p, o)

            if p.x == 0:
                move_to_dst_room = try_move_to_dst_room(p, o)
                if move_to_dst_room:
                    return move_to_dst_room
                else:
                    return []

        def solve(idx, e, w, pods, o, path):
            global opt_cost

            # if cost is higher than best cost so far, abort
            if e >= opt_cost:
                return

            # if this move wins (w=0) update opt_cost and return
            if w == 0:
                opt_cost = e
                return

            # for every pod calculate valid moves
            # for every move play game with that move
            for i, p in enumerate(pods):
                moves = calc_moves(idx, p, o)
                for c, x, y, dw in moves:
                    nleg = (p.code, p.x, p.y, x, y, w+dw)
                    npath = [*path, nleg]
                    npods = copy.deepcopy(pods)
                    no = o.copy()
                    no[p.x, p.y] = ' '
                    no[x, y] = p.code
                    npods[i].x = x
                    npods[i].y = y
                    solve(idx + 1, e + c, w + dw, npods, no, npath)

            w = np.sum([p.y != p.goal_y for p in pods])
            solve(0, 0, w, pods, o, [])
            return opt_cost

    def solve_p2():
        r0_1 = '  #D#C#B#A#'
        r0_1 = '  #D#B#A#C#'

        o = np.zeros((5, 13), dtype=str)
        o[:] = 'X'
        o[0, 1:12] = ' '

        pods = []
        for i, r in enumerate([r0, r0_1, r0_2, r1]):
            for j, c in enumerate(r):
                if c in cost.keys():
                    o[i+1, j] = c
                    pods.append(Pod(c, cost[c], i+1, j, goal[c]))

        def calc_to_hall(p, o):
            moves = []
            for d in [-1, 1]:
                c = p.x * p.cost
                x, y = 0, p.y

                while True:
                    y += d
                    c += p.cost
                    if not o[x,y] == ' ':
                        break
                    elif y not in bady:
                        dw = +1 if (p.y == p.goal_y) else 0
                        moves.append((c, x, y, dw))

            return moves

        def try_move_to_dst_room(p, o):
            c = p.x * p.cost
            d = np.sign(p.goal_y - p.y)
            x, y = 0, p.y

            # try to get to outside your dst room
            while y != p.goal_y:
                y += d
                c += p.cost
                if o[x,y] != ' ':
                    return []

            # go in as deep as can go
            while True:
                if (x+1) <= 4 and o[x+1,y] == ' ':
                    x += 1
                    c += p.cost
                else:
                    break

            # could not enter room
            if x == 0:
                return []

            # return if detect non-bros
            non_bros = False
            for i in range(x+1, 5):
                if o[i, y] != p.code:
                    return []

            # move ok
            dw = -1
            return [(c, x, y, dw)]

        def calc_moves(idx, p, o):
            # if in own room:
            #    if someone in front do nothing
            #    else if anyone behind u not ur bro, to hall
            # if in oth room:
            #    if someone in front do nothing
            #    else try to go to own room
            #         if fail to reach own room go to hall
            # if in hall:
            #    try to go to own room
            #    if fail do nothing

            if p.y == p.goal_y:
                if o[p.x - 1, p.y] != ' ':
                    return []
                else:
                    for i in range(p.x+1, 5):
                        if o[i, p.y] != p.code:
                            return calc_to_hall(p, o)
                    return []

            if p.y != p.goal_y and p.x > 0:
                if o[p.x - 1, p.y] != ' ':
                    return []
                else:
                    move_to_dst_room = try_move_to_dst_room(p, o)
                    if move_to_dst_room:
                        return move_to_dst_room
                    else:
                        return calc_to_hall(p, o)

            if p.x == 0:
                move_to_dst_room = try_move_to_dst_room(p, o)
                if move_to_dst_room:
                    return move_to_dst_room
                else:
                    return []

        def dump():
            path = []
            print(f'{o}')
            for leg in path:
                print(f'{leg}')
                c, x, y, x2, y2, _ = leg
                for p in pods:
                    if p.code == c and p.x == x and p.y == y:
                        o[p.x, p.y] = ' '
                        o[x2, y2] = p.code
                        p.x = x2
                        p.y = y2
                        print(f'{o}')
                        print('\n')
                        break
                else:
                    raise AssertionError(f'{pods}')

        def solve(idx, e, w, pods, o, path):
            global opt_cost

            # if too many moves deep, log
            if idx == 200:
                raise AssertionError(f'idx={idx} is extreme, path is {path}')

            # if cost is no better than best cost so far, abort
            if e >= opt_cost:
                return

            # if this move wins (w=0) update opt_cost and return
            if w == 0:
                opt_cost = e
                return

            for i, p in enumerate(pods):
                moves = calc_moves(idx, p, o)
                for c, x, y, dw in moves:
                    nleg = (p.code, p.x, p.y, x, y, w+dw)
                    npath = [*path, nleg]
                    npods = copy.deepcopy(pods)
                    no = o.copy()
                    no[p.x, p.y] = ' '
                    no[x, y] = p.code
                    npods[i].x = x
                    npods[i].y = y
                    solve(idx + 1, e + c, w + dw, npods, no, npath)

            w = np.sum([p.y != p.goal_y for p in pods])
            solve(0, 0, w, pods, o, [])
            return opt_cost

    print(f'part i ans = {solve_p1()}')
    print(f'part ii ans = {solve_p2()}')
