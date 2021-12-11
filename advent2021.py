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
