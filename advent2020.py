## Imports
import os, re, math, copy, functools, itertools as it, numpy as np
from tqdm import tqdm
from collections import defaultdict, OrderedDict, deque

## Problem 1
def get_prod(n):
    with open(f"{os.getcwd()}/input.txt", "r") as f:
        x = [int(l) for l in f.readlines()]
        for y in it.product(range(len(x)), repeat=n):
            if sum([x[i] for i in y]) == 2020:
                print(f"entries {y} work, product is {math.prod([x[i] for i in y])}")
                break
        else:
            print("failed")


get_prod(2)
get_prod(3)

## Problem 2
def is_valid_one(line):
    range_str, letter, pwd = line.rstrip().split(' ')
    letter = letter.replace(':', '')
    min_str, max_str = range_str.split('-')
    min_val = int(min_str)
    max_val = int(max_str)
    seen = 0
    for c in pwd:
        if c == letter:
            seen += 1
    return seen >= min_val and seen <= max_val

def is_valid_two(line):
    pos_str, letter, pwd = line.rstrip().split(' ')
    letter = letter.replace(':', '')
    i_str, j_str = pos_str.split('-')
    i = int(i_str) - 1
    j = int(j_str) - 1
    return (pwd[i] == letter) ^ (pwd[j] == letter)
    
i = 0
with open(f"{os.getcwd()}/input.txt", "r") as f:
    for l in f:
        i += is_valid_two(l)
print(i)

## Problem 3
res = 1
slopes = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]
with open(f"{os.getcwd()}/input.txt", "r") as f:
    lines = f.readlines()
    for right, skip in slopes:
        pos = 0        
        trees = 0        
        for i in range(skip, len(lines), skip):
            l = lines[i]
            x = list(l.rstrip())
            pos = (pos + right) % len(x)
            trees += (x[pos] == '#')
        res *= trees
    print(f"answer is {res}")

## Problem 4
with open(f"{os.getcwd()}/input.txt", "r") as f:
    req = set(['byr','iyr','eyr','hgt','hcl','ecl','pid'])
    s = f.read()
    x = [p.replace("\n", " ") for p in s.split("\n\n")]
    num_valid = 0
    for y in x:
        z = dict([(e.split(':')[0], e.split(':')[1]) for e in y.split(' ') if e])
        missing = req - set(z.keys())
        valid_one = not missing or missing == {'cid',}
        valid_two = True
        for k,v in z.items():
            if k == 'byr':
                is_valid = len(v) == 4 and int(v) >= 1920 and int(v) <= 2002
                #print(f"{k} of {v} valid={is_valid}")
                valid_two = valid_two and is_valid
            elif k == 'iyr':
                is_valid = len(v) == 4 and int(v) >= 2010 and int(v) <= 2020
                #print(f"{k} of {v} valid={is_valid}")
                valid_two = valid_two and is_valid
            elif k == 'eyr':
                is_valid = len(v) == 4 and int(v) >= 2020 and int(v) <= 2030
                #print(f"{k} of {v} valid={is_valid}")
                valid_two = valid_two and is_valid
            elif k == 'hgt':
                is_valid = False
                if re.match(r"\d+in", v) is not None:
                    is_valid = int(v.replace("in", "")) >= 59 and int(v.replace("in", "")) <= 76
                elif re.match(r"\d+cm", v) is not None:
                    is_valid = int(v.replace("cm", "")) >= 150 and int(v.replace("cm", "")) <= 193
                #print(f"{k} of {v} valid={is_valid}")
                valid_two = valid_two and is_valid
            elif k == 'hcl':
                is_valid = re.match(r"#[0-9a-z]{6}$", v) is not None
                #print(f"{k} of {v} valid={is_valid}")
                valid_two = valid_two and is_valid
            elif k == 'ecl':
                is_valid = v in ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']
                #print(f"{k} of {v} valid={is_valid}")
                valid_two = valid_two and is_valid
            elif k == 'pid':
                is_valid = len(v) == 9 and v.isdigit()
                #print(f"{k} of {v} valid={is_valid}")
                valid_two = valid_two and is_valid
        num_valid += valid_one and valid_two
    print(f"answer is {num_valid}")
        
## Problem 5
def cut_one(s = "FBFBBFF"):
    l,u = 0, 127
    for i, c in enumerate(s):
        if c == "F":
            u = (l + u) // 2
        else:
            assert(c == "B")
            l = (l + u) // 2 + 1
    return min(l, u)

def cut_two(s = "RLR"):
    l,u = 0, 7
    for i, c in enumerate(s):
        if c == "L":
            u = (l + u) // 2
        else:
            assert(c == "R")
            l = (l + u) // 2 + 1
    return min(l, u)

with open(f"{os.getcwd()}/input.txt", "r") as f:
    max_id = -1
    all_seats = set(range(0, 1024))
    for l in f:
        x = l.rstrip()
        r, c = cut_one(x[0:7]), cut_two(x[7:])
        res = r * 8 + c
        if res > max_id:
            max_id = res
        all_seats.discard(res)
    for s in all_seats:
        if s - 1 not in all_seats and s+1 not in all_seats:
            print(s)

## Problem 6
with open(f"{os.getcwd()}/input.txt", "r") as f:
    s = f.read()
    #x = [set(p.replace("\n", "")) for p in s.split("\n\n")]
    #ans = sum([len(y) for y in x])
    x = [p.split("\n") for p in s.split("\n\n")]
    ans = 0
    for g in x:
        ans += len(set.intersection(*[set(e) for e in g if e]))
    print(ans)

## Problem 7 I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    d = defaultdict(set)
    for l in f:
        l = l.rstrip()
        if "no other" in l:
            continue
        l = re.sub(" bags contain [0-9]+ ", ",", l)
        l = re.sub("[0-9]+ ", "", l)
        l = re.sub(r" bags?\.?", "", l)
        l = re.sub(r", ", ",", l)
        bags = l.split(",")
        for b in bags[1:]:
            d[b].add(bags[0])
    ans = d["shiny gold"]
    prev_len = len(ans)
    for i in range(100):
    # while True:
        for x in ans:
            ans = ans.union(d[x])
        new_len = len(ans)
        if new_len == prev_len:
            break
        else:
            prev_len = new_len
    print(ans)
    print(len(ans))
        
## Problem 7 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    d = defaultdict(list)
    for l in f:
        l = l.rstrip()
        if "no other" in l:
            continue
        l = re.sub(" bags contain ", ",", l)
        l = re.sub(r"(bags?)|(\.)", "", l)
        l = re.sub(r"(bags?)|(\.)", "", l)
        bags = l.split(",")
        for b in bags[1:]:
            m = re.match(r"\s*([0-9]+) ([a-z]+ [a-z]+)\s*", b)
            d[bags[0]].append(m.groups())

    def count_bags(b):
        num = 0
        if d[b]:
            for i, j in d[b]:
                num += int(i) + int(i) * count_bags(j)
            return num
        else:
            return num

    print(count_bags("shiny gold"))
        
## Problem 8 I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    acc = 0
    ins = f.readlines()
    ran = set()
    i = 0
    for _ in range(10000):
        if i in ran:
            print(acc)
            break

        ran.add(i)
        op, arg = ins[i].rstrip().split(" ")
        if op == "nop":
            pass
        elif op == "acc":
            acc += int(arg)
        elif op == "jmp":
            i = i + int(arg)
            assert(i >= 0)
            continue
        else:
            assert(False)
        i += 1

## Problem 8 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    def check_if_swap(j, ins):
        acc = 0
        ran = set()
        i = 0
        for _ in range(10000):
            if i in ran:
                return False, acc
            elif i == len(ins):
                return True, acc

            ran.add(i)
            op, arg = ins[i].rstrip().split(" ")
            if i == j and op == "nop":
                op = "jmp"
            elif i == j and op == "jmp":
                op = "nop"
            
            if op == "nop":
                pass
            elif op == "acc":
                acc += int(arg)
            elif op == "jmp":
                i = i + int(arg)
                assert(i >= 0)
                continue
            else:
                assert(False)
            i += 1
            
    ins = f.readlines()
    for j in range(len(ins)):
        op, _ = ins[j].split(" ")
        if op == "nop" or op == "jmp":
            worked, acc = check_if_swap(j, ins)
            if worked:
                print(acc)
                break

## Problem 9 I
n = 25
with open(f"{os.getcwd()}/input.txt", "r") as f:
    nums = [int(l) for l in f.readlines()]
    for i in range(n, len(nums)):
        k = nums[i]
        for a, b in it.combinations(range(1,  n+1), 2):
            if nums[i-a] + nums[i-b] == k:
                break
        else:
            print(k)
            break

## Problem 9 II
bad = 731031916 #127
with open(f"{os.getcwd()}/input.txt", "r") as f:
    done = False
    nums = [int(l) for l in f.readlines()]
    for i in range(1, len(nums)):
        for j in range(len(nums)):
            if j+i < len(nums) and not done:
                s = sum(nums[j:(j+i+1)])
                if s == bad:
                    done = True
                    m, x = min(nums[j:(j+i+1)]), max(nums[j:(j+i+1)])
                    print(f"{m+x}")

## Problem 10 I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    nums = sorted([int(l) for l in f.readlines()])
    nums.append(max(nums)+3)
    curr = 0
    ass = []
    jds = []
    for a in nums:
        jd = a - curr
        if jd > 0 and jd <= 3:
           ass.append(a)
           jds.append(jd)
           curr = a
    print(jds.count(1) * jds.count(3))

## Problem 10 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    all_nums = sorted([int(l) for l in f.readlines()])
    all_nums.append(max(all_nums)+3)

    @functools.lru_cache(maxsize = 1000)
    def sols_from(nums, curr):
        if not nums:
            return 1
        else:
            sols = 0
            for i in range(min(3, len(nums))):
                if nums[i] - curr > 0 and nums[i] - curr <= 3:
                    sols += sols_from(nums[(i+1):], nums[i])
            return sols

    print(sols_from(tuple(all_nums), 0))

## Problem 11 I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    l0 = [list(l.rstrip()) for l in f.readlines()]
    l1 = copy.deepcopy(l0)
    N = len(l0)
    M = len(l0[0])

    def count_occupied(ll):
        occ = 0
        for i, j in it.product(range(N), range(M)):
            occ += ll[i][j] == "#"
        return occ
    
    for k in range(1000):
        for i, j in it.product(range(N), range(M)):
            if l0[i][j] == ".":
                continue
            
            occ = 0
            occ += (j > 0 and l0[i][j-1] == "#")
            occ += (j > 0 and i > 0 and l0[i-1][j-1] == "#")
            occ += (i > 0 and l0[i-1][j] == "#")
            occ += (i > 0 and j < M-1 and l0[i-1][j+1] == "#")
            occ += (j < M-1 and l0[i][j+1] == "#")
            occ += (i < N-1 and j < M-1 and l0[i+1][j+1] == "#")
            occ += (i < N-1 and l0[i+1][j] == "#")
            occ += (i < N-1 and j > 0 and l0[i+1][j-1] == "#")
                
            if l1[i][j] == "L" and occ == 0:
                l1[i][j] = "#"
            elif l1[i][j] == "#" and occ >= 4:
                l1[i][j] = "L"
                
        if l0 == l1:
            print(count_occupied(l0))
            break
        else:
            l0 = copy.deepcopy(l1)
    else:
        assert(False)
    
## Problem 11 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    l0 = [list(l.rstrip()) for l in f.readlines()]
    l1 = copy.deepcopy(l0)
    l2 = copy.deepcopy(l0)    
    N = len(l0)
    M = len(l0[0])

    def print_mat(ll):
        for l in ll:
            print(''.join([str(x) for x in l]))
        print("---")

    def count_occupied(ll):
        occ = 0
        for i, j in it.product(range(N), range(M)):
            occ += ll[i][j] == "#"
        return occ
    
    for k in range(1000):
        for i, j in it.product(range(N), range(M)):
            l2[i][j] = 0

        for i, j in it.product(range(N), range(M)):
            if l0[i][j] != "#":
                continue
            
            for jj in range(j+1, M): #>
                if l0[i][jj] != ".":
                    l2[i][jj] += 1
                    break

            for jj in range(j-1, -1, -1): #<
                if l0[i][jj] != ".":
                    l2[i][jj] += 1
                    break

            for ii in range(i+1, N): #v
                if l0[ii][j] != ".":
                    l2[ii][j] += 1
                    break

            for ii in range(i-1, -1, -1): #^
                if l0[ii][j] != ".":
                    l2[ii][j] += 1
                    break

            for ii, jj in zip(range(i-1, -1, -1), range(j-1, -1, -1)): #^<
                if l0[ii][jj] != ".":
                    l2[ii][jj] += 1
                    break
            
            for ii, jj in zip(range(i-1, -1, -1), range(j+1, M)): #^>
                if l0[ii][jj] != ".":
                    l2[ii][jj] += 1
                    break
            
            for ii, jj in zip(range(i+1, N), range(j+1, M)): #v>
                if l0[ii][jj] != ".":
                    l2[ii][jj] += 1
                    break
            
            for ii, jj in zip(range(i+1, N), range(j-1, -1, -1)): #v<
                if l0[ii][jj] != ".":
                    l2[ii][jj] += 1
                    break

        for i, j in it.product(range(N), range(M)):
            if l0[i][j] == "L" and l2[i][j] == 0:
                l1[i][j] = "#"
            elif l0[i][j] == "#" and l2[i][j] >= 5:
                l1[i][j] = "L"

        if l0 == l1:
            print(count_occupied(l0))
            break
        else:
            l0 = copy.deepcopy(l1)
    else:
        assert(False)
    
## Problem 12 I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    p = [0, 0]
    d = 'E'
    dirs = {'N': [0, 1, 270], 'E': [1, 0, 0], 'S': [0, -1, 90], 'W': [-1, 0, 180]}
    angs = {270: 'N', 0: 'E', 90: 'S', 180: 'W'}
    for l in f:
        ins, val = re.match(r"([A-Z])(\d+)", l).groups()
        val = int(val)
        if ins == "F":
            p[0] += val * dirs[d][0]
            p[1] += val * dirs[d][1]
        elif ins == "R" or ins == "L":
            m = 1 if ins == "R" else -1
            ang = (dirs[d][2] + m * val) %  360
            d = angs[ang]
        else:
            p[0] += val * dirs[ins][0]
            p[1] += val * dirs[ins][1]
        #print(f"{l.rstrip()} --> ({p[0]}, {p[1]}, {d})")
    print(f"{abs(p[0]) + abs(p[1])}")

## Problem 12 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    ship = [0, 0]
    wayp = [10, 1]
    d = 'E'
    dirs = {'N': [0, 1, 270], 'E': [1, 0, 0], 'S': [0, -1, 90], 'W': [-1, 0, 180]}
    angs = {270: 'N', 0: 'E', 90: 'S', 180: 'W'}
    for l in f:
        ins, val = re.match(r"([A-Z])(\d+)", l).groups()
        val = int(val)
        if ins == "F":
            ship[0] += val * wayp[0]
            ship[1] += val * wayp[1]
        elif ins == "R" or ins == "L":
            m = 1 if ins == "R" else -1
            rot = (m * val) % 360
            ang = (dirs[d][2] + rot) %  360
            nd = angs[ang]
            while rot > 0:
                wayp[0], wayp[1] = wayp[1], -wayp[0]
                rot -= 90
            d = nd
        else:
            wayp[0] += val * dirs[ins][0]
            wayp[1] += val * dirs[ins][1]
        print(f"{l.rstrip()} --> ship=({ship[0]}, {ship[1]}) way=({wayp[0]}, {wayp[1]}, {d})")
    print(f"{abs(ship[0]) + abs(ship[1])}")

## Problem 13 I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    l0,l1 = f.readlines()
    t = int(l0)
    oids = [int(b) for b in l1.split(",") if b != "x"]
    min_id = -1
    min_rt = math.inf
    for b in oids:
        m = t // b
        rt = t if (b * m == t) else b * (m+1)
        if rt < min_rt:
            min_id = b
            min_rt = rt
    print(f"t={t}, bus={min_id} ride at {min_rt} ans={min_id * (min_rt - t)}")

## Problem 13 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    def ext_euclidean(a, b):
        s, old_s = 0, 1
        r, old_r = b, a

        for _ in range(100):
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_s, s = s, old_s - q * s
            if r == 0:
                break
        else:
            assert(False)
        bt = (old_r - old_s * a) // b
        return old_s, bt, old_r
    
    _,l1 = f.readlines()
    oids = [int(b) if b != 'x' else 0 for b in l1.split(",")]
    idxs = list(range(len(oids)))
    tups = [(x,y) for x,y in zip(oids, idxs) if x > 0]
    cops = [(-y % x, x) for x,y in tups]
    
    sol = 0
    N = math.prod([t[1] for t in cops])
    for a, n in cops:
        Nn = N // n
        r, s, z = ext_euclidean(n, Nn)
        assert(z == 1)
        assert(r * n + s * Nn == 1)
        sol += a * s * Nn
    print(sol % N)

## Problem 14 I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    lines = f.readlines()
    mask = lines[0].rstrip().replace(" ", "").split("=")[1]

    writes = []
    for l in lines[1:]:
        if l.startswith("mask"):
            mask = l.rstrip().replace(" ", "").split("=")[1]
        else:
            addr, val = re.match(r"mem\[(\d+)\] = (\d+)", l).groups()
            writes.append((mask, int(addr), int(val)))

    mem = {}
    for mask, addr, val in writes:
        bval = f"{val:036b}"
        mval = [b if m == "X" else m for m, b in zip(mask, bval)]
        # print(f"addr {addr} <-- {int(''.join(mval), 2)}")
        mem[addr] = int(''.join(mval), 2)
    print(sum(mem.values()))

## Problem 14 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    lines = f.readlines()
    mask = lines[0].rstrip().replace(" ", "").split("=")[1]

    writes = []
    for l in lines[1:]:
        if l.startswith("mask"):
            mask = l.rstrip().replace(" ", "").split("=")[1]
        else:
            addr, val = re.match(r"mem\[(\d+)\] = (\d+)", l).groups()
            writes.append((mask, int(addr), int(val)))

    mem = {}
    for mask, addr, val in writes:
        baddr = f"{addr:036b}"
        maddr = [m if m == "X" else "1" if m == "1" else b for m, b in zip(mask, baddr)]
        idxs = [i for i, x in enumerate(maddr) if x == "X"]
        for perm in it.product(["0","1"], repeat=maddr.count("X")):
            for i, b in zip(idxs, perm):
                maddr[i] = b
            mem[int(''.join(maddr), 2)] = val
    print(sum(mem.values()))

##Problem 15
nums = [19, 0, 5, 1, 10, 13]
d = defaultdict(list)
last = None
for i, n in enumerate(nums):
    l = d[n]
    l.append(i)
    last = n
for i in range(len(nums), 30000000):
    l = d[last]
    # print(f"turn {i+1} found {l}...")
    assert(l)
    if len(l) > 2:
        l.pop(0)
        
    if len(l) == 1:
        last = 0
        d[last].append(i)
    else:
        assert(len(l) == 2)
        last = l[1] - l[0]
        d[last].append(i)
print(f"speak {last}")
    
##Problem 16
with open(f"{os.getcwd()}/input.txt", "r") as f:
    rules = OrderedDict()
    tickets, valid_tickets = [], []
    for l in f:
        if not l.strip():
            continue
        elif m := re.match(r"^([^:]+): ([0-9]+)-([0-9]+) or ([0-9]+)-([0-9]+)", l):
            field, r0, r1, r2, r3 = m.groups()
            rules[field] = [int(r0), int(r1), int(r2), int(r3)]
            continue
        elif re.match(r".*tickets?:$", l):
            continue
        t = [int(i) for i in l.rstrip().split(",")]
        tickets.append(t)

    s = 0
    valid_tickets.append(tickets[0])
    for t in tickets[1:]:
        valid_ticket = True
        for v in t:
            num_valid = 0
            for rs in rules.values():
                a = v >= rs[0] and v <= rs[1]
                b = v >= rs[2] and v <= rs[3]
                if a or b:
                    num_valid += 1
            if num_valid == 0:
                s += v
                valid_ticket = False
                
        if valid_ticket:
           valid_tickets.append(t)

    field_map = defaultdict(set)
    for field, rs in rules.items():
        for pos in range(len(tickets[0])):
            for t in valid_tickets:
                v = t[pos]
                a = v >= rs[0] and v <= rs[1]
                b = v >= rs[2] and v <= rs[3]
                if not (a or b):
                    break
            else:
                field_map[field].add(pos)

    done_singletons = set()
    for _ in range(100):
        singletons = set()
        for field, candidates in field_map.items():
            assert(candidates)
            if len(candidates) == 1 and next(iter(candidates)) not in done_singletons:
                # print(f"{field} is unambiguous!")
                singletons.add(next(iter(candidates)))

        if not singletons:
            break
                
        for field in field_map.keys():
            candidates = field_map[field]
            if len(candidates) > 1:
                # print(f"cleaning up {field} by removing {singletons}")
                field_map[field] = candidates.difference(singletons)
                assert(field_map[field])

        done_singletons.update(singletons)
    else:
        assert(False)

    p = 1
    for field, pos in field_map.items():
        if field.startswith("departure"):
            p *= tickets[0][next(iter(pos))]
    print(p)
    
##Problem 17 I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    def gen_inactive_cube():
        return [".", 0]
    
    coords = defaultdict(gen_inactive_cube)
    y = 0
    for line in f:
        l = list(line.rstrip())
        for x in range(len(l)):
            coords[(x, y, 0)] = [l[x], 0]
        y += 1

    dirs = set(it.product((-1, 0, 1), repeat=3)).difference({(0,0,0),})
    xr = [-1, y]
    yr = [-1, y]
    zr = [-1, 1]

    for _ in range(6):
        for p in it.product(range(xr[0], xr[1]+1), range(yr[0], yr[1]+1), range(zr[0], zr[1]+1)):
            for d in dirs:
                c = tuple(i+j for i,j in zip(p, d))
                n = coords[c]
                v = (n[0] == "#")
                coords[p][1] += v

        actives = 0
        for k in coords.keys():
            if coords[k][0] == "#" and (coords[k][1] < 2 or coords[k][1] > 3):
                coords[k][0] = "."
            elif coords[k][0] == "." and coords[k][1] == 3:
                coords[k][0] = "#"
            coords[k][1] = 0
            actives += coords[k][0] == "#"

        print(f"ans = {actives}")

        xr[0], xr[1] = xr[0] - 1, xr[1] + 1
        yr[0], yr[1] = yr[0] - 1, yr[1] + 1
        zr[0], zr[1] = zr[0] - 1, zr[1] + 1

##Problem 17 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    def gen_inactive_cube():
        return [".", 0]
    
    coords = defaultdict(gen_inactive_cube)
    y = 0
    for line in f:
        l = list(line.rstrip())
        for x in range(len(l)):
            coords[(x, y, 0, 0)] = [l[x], 0]
        y += 1

    dirs = set(it.product((-1, 0, 1), repeat=4)).difference({(0,0,0,0),})
    xr = [-1, y]
    yr = [-1, y]
    zr = [-1, 1]
    wr = [-1, 1]

    for _ in range(6):
        rxr = range(xr[0], xr[1]+1)
        ryr = range(yr[0], yr[1]+1)
        rzr = range(zr[0], zr[1]+1)
        rwr = range(wr[0], wr[1]+1)
        
        for p in it.product(rxr, ryr, rzr, rwr):
            for d in dirs:
                c = tuple(i+j for i,j in zip(p, d))
                n = coords[c]
                v = (n[0] == "#")
                coords[p][1] += v

        actives = 0
        for k in coords.keys():
            if coords[k][0] == "#" and (coords[k][1] < 2 or coords[k][1] > 3):
                coords[k][0] = "."
            elif coords[k][0] == "." and coords[k][1] == 3:
                coords[k][0] = "#"
            coords[k][1] = 0
            actives += coords[k][0] == "#"

        print(f"ans = {actives}")

        xr[0], xr[1] = xr[0] - 1, xr[1] + 1
        yr[0], yr[1] = yr[0] - 1, yr[1] + 1
        zr[0], zr[1] = zr[0] - 1, zr[1] + 1
        wr[0], wr[1] = wr[0] - 1, wr[1] + 1

##Problem 18
with open(f"{os.getcwd()}/input.txt", "r") as f:
    ops = {"*": 0, "+": 1}
    answers = []
    
    for line in f:
        # if not line.startswith("8 + 4 + (5 * 5) + (6 + 8 * (7"):
        #     continue
        l = line.rstrip().replace("(", "( ").replace(")", " )").split(" ")
        
        stack = deque()
        op_stack = []
        # print(f"evaluating {line.rstrip()}")
        for e in l:
            # print(f"look at {e}, op_stack is {op_stack} and stack is {stack}")
            if e in ops:
                if op_stack and ops[e] < ops[op_stack[-1]]:
                    while len(stack) >= 3:
                        y = stack.pop()
                        o = stack.pop()
                        x = stack.pop()
                        z = stack.pop() if len(stack) > 0 else None
                        if z is not None:
                            stack.append(z)
                            
                        if o == "*":
                            stack.append(int(x) * int(y))
                            op_stack.pop()
                        elif o == "+":
                            stack.append(int(x) + int(y))
                            op_stack.pop()
                        else:
                            stack.append(x)
                            stack.append(o)
                            stack.append(y)
                            break

                        if z is not None and z == "(":
                            break

                stack.append(e)
                op_stack.append(e)
            elif e == ")":
                for i in range(1000):
                    y = stack.pop()
                    o = stack.pop()
                    x = stack.pop()
                    z = stack.pop()
                    if z != "(":
                        stack.append(z)
                        
                    if o == "*":
                        stack.append(int(x) * int(y))
                        op_stack.pop()                        
                    elif o == "+":
                        stack.append(int(x) + int(y))
                        op_stack.pop()
                    else:
                        assert(False)

                    if z == "(":
                        break
                else:
                     assert(False)   
            else:
                stack.append(e)

        while len(stack) >= 3:
           y = stack.pop()
           o = stack.pop()
           x = stack.pop()
           if o == "*":
               stack.append(int(x) * int(y))
               op_stack.pop()               
           elif o == "+":
               stack.append(int(x) + int(y))
               op_stack.pop()
           else:
               assert(False)
        assert(len(stack) == 1)
        print(stack)
        answers.append(stack.pop())
    print(answers)
    print(sum([int(a) for a in answers]))
    
##Problem 19 I
with open(f"{os.getcwd()}/test.txt", "r") as f:
    rules = {}
    msgs = []
    for line in f:
        if line.strip() == "":
            continue
        
        m = re.match(r"([0-9]+): (.+)", line)
        if m is not None:
            i, l = m.groups()
            l = l.rstrip().replace('"','').split("|")
            rules[i] = [ll.strip().split(" ") for ll in l]
        else:
            msgs.append(line.rstrip())

    rules['8'] = [['42'], ['42', '8']]
    rules['11'] = [['42', '31'], ['42', '11', '31']]

    def satisfies(s, l):
        print(f"called with s={s} and l={l}")
        if type(l) == str and l.isdigit():
            ok, ss = satisfies(s, rules[l])
            return ok, ss
        elif type(l) == str:
            assert(l == 'a' or l == 'b')
            ok = len(s) >= 1 and s[0] == l
            ss = s[1:] if ok else s
            return ok, ss            
        elif type(l[0]) == list:
            for ll in l:
                ok, ss = satisfies(s, ll)
                if ok:
                    return True, ss
            return False, s
        elif type(l) == list:
            ss = s
            for ll in l:
                ok, ss = satisfies(ss, ll)
                if not ok:
                    return False, s
            return True, ss
            
        else:
            assert(False)
            

    n = 0
    for m in msgs:
        if m != 'babbbbaabbbbbabbbbbbaabaaabaaa':
            continue
        print(f"checking {m}")
        ok, ss = satisfies(m, rules['0'])
        if ss == "" and ok:
            n += 1
            print(f"{m} matches and {ss}")
        else:
            print(f"{m} fails and {ss}")
    print(n)

##Problem 19 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    rules = {}
    msgs = []
    for line in f:
        if line.strip() == "":
            continue
        
        m = re.match(r"([0-9]+): (.+)", line)
        if m is not None:
            i, l = m.groups()
            l = l.rstrip().replace('"','').split("|")
            rules[i] = [ll.strip().split(" ") for ll in l]
        else:
            msgs.append(line.rstrip())

    def get_possible(l):
        if type(l) == str and l.isdigit():
            return get_possible(rules[l])
        elif type(l) == str:
            return [l]
        elif type(l[0]) == list:
            possible = []
            for ll in l:
                pos = get_possible(ll)
                possible.extend(pos)
            return possible
        elif type(l) == list:
            possible = []
            for ll in l:
                pos = get_possible(ll)
                possible.append(pos)
            possible = [''.join(p) for p in it.product(*possible)]
            return possible
        else:
            assert(False)

    possible = set(get_possible(rules['0']))
    possible31 = set(get_possible(rules['31']))
    possible42 = set(get_possible(rules['42']))

    
    lens31 = {len(p) for p in possible31}
    lens42 = {len(p) for p in possible42}
    assert(len(lens31) == 1 and len(lens42) == 1)
    l31 = next(iter(lens31))
    l42 = next(iter(lens42))

    def is_new_rules_8_11(m):
        # rules['8'] = [['42'], ['42', '8']]
        # rules['11'] = [['42', '31'], ['42', '11', '31']]
        # 42 42...42 42 42 31 31 31
        
        n = 0
        lm = len(m)
        assert(lm % l42 == 0)
        for i in range(lm // l42 + 1):
            j = (lm - i * l42) // l31
            if i < 2 or j < 1 or j >= i:
                #at least 2 42s and 1 31 required, more 42s than 31s
                continue

            still_ok = True
            mm = m
            for k in range(i):
                if not still_ok:
                    break
                
                for p42 in possible42:
                    if mm.startswith(p42):
                        mm = mm[l42:]
                        break
                else:
                    still_ok = False

            if not still_ok:
                continue

            for k in range(j):
                if not still_ok:
                    break
                
                for p31 in possible31:
                    if mm.startswith(p31):
                        mm = mm[l31:]
                        break
                else:
                    still_ok = False

            if still_ok:
                assert(len(mm) == 0)
                return True

        return False

    n = 0
    for m in msgs:
        if is_new_rules_8_11(m):
            n += 1
    print(n)

##Problem 20
with open(f"{os.getcwd()}/input.txt", "r") as f:
    tiles = {}
    curr_tile = []
    sea_pattern = [".#...#.###...#.##.O#",
                   "O.##.OO#.#.OO.##.OOO",
                   "#O.#O#.O##O..O.#O##."]

    tid = None
    for line in f:
        if line.strip() == "":
            tiles[tid] = curr_tile
            curr_tile = []
        elif m := re.match(r"Tile ([0-9]+):", line):
            tid = int(m.group(1))
        else:
            curr_tile.append(line.rstrip())
    tiles[tid] = curr_tile

    def border_to_hash(s):
        bs = s.replace(".", "1").replace("#", "0")
        return int(bs, 2)

    def matrix_flip(m, s):
        n = len(m)
        m2 = copy.deepcopy(m)
        if s == 0: #R0
            pass
        elif s == 1: #R1
            #align to (0,0) then (i,j) -> (-j, i)
            #so (i - (n-1) / 2, j - (n-1) / 2) ->
            #((n-1) / 2 - j, i - (n-1) / 2) ->
            #(n - 1 - j, i)
            for i, j in it.product(range(n), repeat=2):
                m2[n - 1 - j][i] = m[i][j]
        elif s == 2: #R2
            #align to (0,0) then (i,j) -> (-i, -j)
            #so (i - (n-1) / 2, j - (n-1) / 2) ->
            #((n-1) / 2 - i, (n-1) / 2 - j) ->
            #(n - 1 - i, n - 1 - j)
            for i, j in it.product(range(n), repeat=2):
                m2[n - 1 - i][n - 1 - j] = m[i][j]
        elif s == 3: #R3
            #align to (0,0) then (i,j) -> (j, -i)
            #so (i - (n-1) / 2, j - (n-1) / 2) ->
            #(j - (n-1) / 2, (n-1) / 2 - i) -> 
            #(j, n - 1 - i)
            for i, j in it.product(range(n), repeat=2):
                m2[j][n - 1 - i] = m[i][j]
        elif s == 4: #M1
            for i, j in it.product(range(n), repeat=2):
                m2[i][n - j - 1] = m[i][j]
        elif s == 5: #M2
            for i, j in it.product(range(n), repeat=2):
                m2[n - i - 1][j] = m[i][j]
        elif s == 6: #D1
            for i, j in it.product(range(n), repeat=2):
                m2[j][i] = m[i][j]
        elif s == 7: #D2
            for i, j in it.product(range(n), repeat=2):
                m2[n - j - 1][n - i - 1] = m[i][j]

        return m2

    class Tile:
        def __init__(self, i, t, b, l, r, img):
            tblr = []
            tblr.append([t, b, l, r]) #R0
            tblr.append([r, l, t[::-1], b[::-1]]) #R1
            tblr.append([b[::-1], t[::-1], r[::-1], l[::-1]]) #R2
            tblr.append([l[::-1], r[::-1], b, t]) #R3
            tblr.append([t[::-1], b[::-1], r, l]) #M1
            tblr.append([b, t, l[::-1], r[::-1]]) #M2
            tblr.append([l, r, t, b]) #D1
            tblr.append([r[::-1], l[::-1], b[::-1], t[::-1]]) #D2

            itblr = []
            for x in tblr:
                y = [border_to_hash(s) for s in x]
                itblr.append(y)

            self.i = i
            self.img = img
            self.tblr = tblr
            self.itblr = itblr

        def all(self):
            hs = set()
            for x in self.itblr:
                hs.update(set(x))
            return hs

        def top(self, s):
            return self.itblr[s][0]

        def bot(self, s):
            return self.itblr[s][1]

        def lft(self, s):
            return self.itblr[s][2]

        def rgt(self, s):
            return self.itblr[s][3]

        def image(self, s):
            imglo = [list(r) for r in self.img]
            imglf = matrix_flip(imglo, s)
            imgf = ["".join(r) for r in imglf]
            return imgf
        
        def __repr__(self):
            img_str = ""
            for img_row in self.img:
                img_str = f"{img_str}{img_row}\n"
            img_str = img_str.rstrip()
            
            rep = f"id={self.i}"
            for i in range(len(self.tblr)):
                rep = f"{rep}\n {i}: {self.itblr[i]}  {self.tblr[i]}"
            rep = f"{rep}\n image:\n{img_str}"
            return rep
        
    ts = []
    edge_map = defaultdict(set)
    for i, k in zip(range(len(tiles)), tiles.keys()):
        t = tiles[k]        
        l, r = [], []
        for j in range(len(t)):
            l.append(t[j][0])
            r.append(t[j][-1])

        top = t[0]
        bot = t[-1]
        lft = ''.join(l)
        rgt = ''.join(r)

        img = []
        for j in range(1, len(t) - 1):
            img.append(t[j][1:-1])
            
        tile = Tile(k, top, bot, lft, rgt, img)
        ts.append(tile)
        for h in tile.all():
            edge_map[h].add(i)

    nt = len(ts)
    n = int(math.sqrt(nt))
    assert(n*n == nt)

    def solve(pos, il, sl, remain):
        t = il[-1]
        s = sl[-1]
        has_lft = (pos % n != 0)
        has_top = (pos > n - 1)
        has_rgt = (pos % n != (n-1))

        ok = True
        viable = remain
        if has_lft:
            lft_t = il[-2]
            lft_s = sl[-2]
            ok = (ts[lft_t].rgt(lft_s) == ts[t].lft(s))
            
        if has_top:
            top_t = il[pos - n]
            top_s = sl[pos - n]
            ok = (ts[top_t].bot(top_s) == ts[t].top(s))

        if not remain:
            if ok:
                return zip(il, sl)
            else:
                return None
        
        if has_rgt:
            viable = remain.intersection(edge_map[ts[t].rgt(s)])

        if not ok or not viable:
            return None

        for vi, s in it.product(viable, range(8)):
            pil = il + [vi]
            psl = sl + [s]
            rem = remain.difference({vi})            
            sol = solve(pos + 1, pil, psl, rem)
            if sol is not None:
                return sol
        else:
            return None

    def sol_to_bordered_image_debug(solution):
        for grp in [list(range(n * i, n * (i+1))) for i in range(n)]:
            sub_params = [(solution[k][0], solution[k][1]) for k in grp]
            ln = len(ts[0].tblr[0][0])
            for j in range(ln):
                row = ""
                for t, s in sub_params:
                    if j == 0:
                        row = row + ts[t].tblr[s][0] + " "
                    elif j == ln - 1:
                        row = row + ts[t].tblr[s][1] + " "
                    else:
                        row = row + ts[t].tblr[s][2][j] + (" " * (ln - 2)) + ts[t].tblr[s][3][j] + " "
                print(row)
            print("")

    def sol_to_image(solution):
        full_image = []
        for grp in [list(range(n * i, n * (i+1))) for i in range(n)]:
            sub_images = [ts[solution[k][0]].image(solution[k][1]) for k in grp]
            for j in range(len(sub_images[0][0])):
                row = "".join([img[j] for img in sub_images])
                full_image.append(row)
        return full_image

    def cnt_sea_monsters(full_image):                
        def _find_monsters(i, cand):
            sl = len(sea_pattern[0])
            cl = len(cand[0])
            monsters = []

            j = 0
            ok = True
            for _ in range(1000):
                if j >= cl - sl:
                    break
                
                ok = True
                for k in range(sl):
                    sc = sea_pattern[0][k]
                    cc = cand[i][k+j]
                    match = sc != "O" or (cc == "#")
                    if not match:
                        ok = False
                        break

                if not ok:
                    j = j + 1
                    continue
                
                for l in range(1, len(sea_pattern)):
                    if ok:
                        for k in range(sl):
                            sc = sea_pattern[l][k]
                            cc = cand[i+l][k + j]
                            match = sc != "O" or (cc == "#")                            
                            if not match:
                                ok = False
                                break
                if ok:
                    monsters.append((i, j))
                    j = j + sl
                else:
                    j = j + 1

                
            return monsters
                
        def _compute_sea_and_roughness(cand):            
            i = 0
            monsters = []
            for _ in range(1000):
                new_monsters = _find_monsters(i, cand)
                monsters.extend(new_monsters)
                if new_monsters:
                    print(f"monsters {new_monsters} between {i} and {i+2}")
                i = i + 1
                if i+2 >= len(cand):
                    break

            for i, j in monsters:
                for k, sr in enumerate(sea_pattern):
                    for l, sc in enumerate(sr):
                        if sc == "O":
                            cand[i+k][l+j] = sc

            num_clean_hash = 0
            for img_row in cand:
                for ir in img_row:
                    num_clean_hash += (ir == "#")

            roughness = num_clean_hash
            return len(monsters), roughness

        full_image_mat = [list(r) for r in full_image]
        for s in range(8):
            full_image_matf = matrix_flip(full_image_mat, s)
            cnt_monsters, roughness = _compute_sea_and_roughness(full_image_matf)
            if cnt_monsters > 0:
                print(f"symm {s} has {cnt_monsters} monsters, roughness {roughness}")

    num_solutions = 0
    for i, s in it.product(range(nt), range(8)):
        remain = set(range(nt)).difference({i})
        solution = solve(0, [i], [s], remain)
        if solution:
            num_solutions += 1
            
        if solution and num_solutions > 0:
            solution = list(solution)
            # sol_to_bordered_image_debug(solution)
            solstr = ""
            for j in range(nt):
                solstr = f"{solstr} {ts[solution[j][0]].i}"
                if j % n == n -1:
                    solstr += "\n"
            # print(solstr)
            tl, bl = ts[solution[0][0]].i, ts[solution[n-1][0]].i
            tr, br = ts[solution[nt-n][0]].i, ts[solution[nt-1][0]].i
            print(f"{tl} * {tr} * {bl} * {br} = {tl*tr*bl*br}")
            full_image = sol_to_image(solution)
            full_image_str = "\n".join(full_image)
            # print(f"{full_image_str}")
            cnt_sea_monsters(full_image)
            break

## Problem 21
with open(f"{os.getcwd()}/input.txt", "r") as f:
    a_to_i = {}
    all_i = set()
    foods = []
    
    for l in f:
        x, y = re.match(r"(.+) \(contains (.+)\)$", l).groups()
        ingredients = [i.strip() for i in x.split(" ")]
        allergens = [a.strip() for a in y.split(",")]

        assert(len(set(ingredients)) == len(ingredients))
        foods.append(set(ingredients))
        all_i.update(set(ingredients))
        for a in allergens:
            if a in a_to_i:
                a_to_i[a] = a_to_i[a].intersection(set(ingredients))
            else:
                a_to_i[a] = set(ingredients)

    # print(all_i)
    # print(a_to_i)

    for _ in range(1000):
        pruned = False
        for k in a_to_i.keys():
            v = a_to_i[k]
            if len(v) == 1:
                for k2 in a_to_i.keys():
                    if k != k2:
                        v2 = a_to_i[k2]
                        v2 = v2.difference(v)
                        if v != v2:
                            pruned = True
                            a_to_i[k2] = v2
        if not pruned:
            break

    # print(a_to_i)
    screwed = set()
    for k, v in a_to_i.items():
        if len(v) == 1:
            screwed.add(next(iter(v)))

    safe = all_i.difference(screwed)
    n = 0
    for i in safe:
        for food in foods:
            n += (i in food)
    # print(n)

    print(a_to_i)
    print(screwed)
    print(safe)
    
    i_to_a = {}
    for i in screwed:
        found = False
        for k, v in a_to_i.items():
            if i in v:
                assert(found == False)
                found = True
                i_to_a[i] = k
    print(i_to_a)
    can = ",".join(dict(sorted(i_to_a.items(), key=lambda x: x[1])).keys())
    print(can)

## Problem 22 I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    tr = 1
    p1 = deque()
    p2 = deque()
    
    for l in f:
        if l.startswith("P"):
            continue
        elif l == "\n":
            tr = 2
            continue
        
        p = p1 if tr == 1 else p2
        p.append(int(l))

    # print(p1)
    # print(p2)
    
    for _ in range(2000):
        c1 = p1.popleft()
        c2 = p2.popleft()

        if c1 > c2:
            p1.append(c1)
            p1.append(c2)
        elif c2 > c1:
            p2.append(c2)
            p2.append(c1)
        else:
            assert(False)

        if not p1 or not p2:
            break
    else:
        assert(False)

    # print(p1)
    # print(p2)
    s = 0
    pw = p1 if p1 else p2
    for i, c in enumerate(pw):
        s += (len(pw) - i) * c
    print(s)

## Problem 22 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    tr = 1
    p1 = deque()
    p2 = deque()
    
    for l in f:
        if l.startswith("P"):
            continue
        elif l == "\n":
            tr = 2
            continue
        
        p = p1 if tr == 1 else p2
        p.append(int(l))

    def play(p1, p2):
        rounds = set()
        for _ in range(20000):
            p1h = ".".join([str(p) for p in p1])
            p2h = ".".join([str(p) for p in p2])
            rh = p1h+p2h
            
            if rh in rounds:
                return (1, p1)
            else:
                rounds.add(rh)
                
            c1 = p1.popleft()
            c2 = p2.popleft()

            winner = None
            if c1 <= len(p1) and c2 <= len(p2):
                p1c = deque([p1[i] for i in range(c1)])
                p2c = deque([p2[i] for i in range(c2)])
                winner, _ = play(p1c, p2c)
            else:
                winner = 1 if c1 > c2 else 2
                
            if winner == 1:
                p1.append(c1)
                p1.append(c2)
            elif winner == 2:
                p2.append(c2)
                p2.append(c1)
            else:
                assert(False)

            if not p2:
                return (1, p1)
            elif not p1:
                return (2, p2)
        else:
            assert(False)
        
    _, fpw = play(p1, p2)
    s = 0
    for i, c in enumerate(fpw):
        s += (len(fpw) - i) * int(c)
    print(s)    

## Problem 23 I
#"389125467":
if input := "327465189":
    l = [int(c) for c in input]
    m = min(l)
    x = max(l)
    n = len(l)
    l = [l, copy.copy(l)]
    i, j = 0, 0

    for move in tqdm(range(100)):
        k = None
        s = l[i][j] - 1
        s = x if s < m else s
        oi = (i+1) % 2
        pu = ((j+1) % n, (j+2) % n, (j+3) % n)        

        #find destination, k
        while k is None:
            try:
                k = l[i].index(s)
            except ValueError:
                s = x if (s-1) < m else (s-1)
                continue
            if k in pu:
                s = x if (s-1) < m else (s-1)
                k = None
                continue

        # print(f"-- move {move+1} --")
        # print(f"cups: {l[i]}, curr={l[i][j]}")
        # print(f"pick up: {l[i][pu[0]]}, {l[i][pu[1]]}, {l[i][pu[2]]}")
        # print(f"destination: {l[i][k]}\n")

        #curr cup goes to same position
        jj = j
        l[oi][jj] = l[i][jj]
        layed = 1
        
        #skip pu and lay cups, break after laying down cup k
        jjo = (jj + 1) % n
        jji = (jj + 4) % n
        for _ in range(n+1):
            l[oi][jjo] = l[i][jji]
            layed = layed + 1
            if jji == k:
                jji, jjo = (jji + 1) % n, (jjo + 1) % n
                break                
            jji, jjo = (jji + 1) % n, (jjo + 1) % n
        else:
            assert(False)

        #lay the pu cups
        for u in pu:
            l[oi][jjo] = l[i][u]
            layed = layed + 1            
            jjo = (jjo + 1) % n

        #lay any remaining cups
        while layed < n:
            l[oi][jjo] = l[i][jji]
            layed += 1
            jji, jjo = (jji + 1) % n, (jjo + 1) % n            

        #toggle lists and move current 
        i = (i+1) % 2
        j = (j+1) % n


    j1 = l[i].index(1)
    fl = [str(l[i][(j1 + off) % n]) for off in range(1,n)]
    
    print(f"-- final --")
    print(f"cups: {l[i]}")
    print(f"{''.join(fl)}")
        
## Problem 23 II
# time ./a.out 327465189
# 749102 * 633559 = 474600314018
# real 960m15.471s
if input := "327465189":
    l = [int(c) for c in input]
    x = max(l)
    
    tail = [x+i-len(l)+1 for i in range(len(l), 1000*1000)]
    l.extend(tail)
    m = min(l)
    x = max(l)    
    n = len(l)
    ll = [[(i+1) % n, v] for i,v in enumerate(l)]

    def find_first(ll, s):
        d = 0
        for _ in range(len(ll)):
            nxt, v = ll[d]
            if v == s:
                return d
            d = nxt
        else:
            return None

    def dump_list(ll):
        oss = ""
        cur = 0
        for i in range(len(ll)):
            oss = oss + str(ll[cur][1]) + " "
            cur = ll[cur][1]
        print(f"{oss}")

    cur = 0        
    for move in tqdm(range(10*1000*1000), file=sys.stdout):
    # for move in range(1000):
        # dump_list(ll)
        s = ll[cur][1] - 1
        s = x if s < m else s
        nxt1 = ll[cur][0]
        nxt2 = ll[nxt1][0]
        nxt3 = ll[nxt2][0]
        nxt4 = ll[nxt3][0]
        pu = (nxt1, nxt2, nxt3)

        #find destination, dst
        dst = None
        while dst is None:
            dst = find_first(ll, s)
            if dst is None:
                s = x if (s-1) < m else (s-1)
                continue
            elif dst in pu:
                s = x if (s-1) < m else (s-1)
                dst = None
                continue

        #curr cup next points to nxt4 and nxt4 prev to curr
        ll[cur][0] = nxt4

        #link pu triplet dst -> nxt1 -> ... -> nxt3 -> ndst
        ndst = ll[dst][0]
        ll[dst][0] = nxt1
        ll[nxt3][0] = ndst
        
        #move cur ahead
        cur = nxt4
        

    j1 = find_first(ll, 1)
    j1nxt1 = ll[j1][0]
    j1nxt2 = ll[j1nxt1][0]
    fl1, fl2 = ll[j1nxt1][1], ll[j1nxt2][1]
    print(f"{fl1} * {fl2} = {fl1 * fl2}")

## Problem 24
