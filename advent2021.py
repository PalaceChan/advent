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
