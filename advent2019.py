## Imports
import os, re, math, copy, functools, itertools as it, numpy as np
# from tqdm import tqdm
from collections import defaultdict, OrderedDict, deque

##Problem 1
with open(f"{os.getcwd()}/input.txt", "r") as f:    
    mass = [int(l) for l in f]
    fuel = [m // 3 - 2 for m in mass]
    print(sum(fuel))

    def get_fuel2(m):
        fu = m // 3 - 2
        if fu > 0:
            return fu + get_fuel2(fu)
        else:
            return 0

    fuel2 = [get_fuel2(m) for m in mass]
    print(sum(fuel2))

##Problem 2
with open(f"{os.getcwd()}/input.txt", "r") as f:
    ins = [int(i) for i in f.readlines()[0].split(",")]

    def run_prgm(ins, n, v):
        ins[1] = n
        ins[2] = v
        ip = 0
        for _ in range(100):
            op, l, r, d = ins[ip:(ip+4)]
            if op == 1:
                ins[d] = ins[l] + ins[r]
            elif op == 2:
                ins[d] = ins[l] * ins[r]
            elif op == 99:
                break
            else:
                assert(False)

            ip += 4
        else:
            assert(False)
        return ins[0]
    print(run_prgm(copy.copy(ins), 12, 2))

    for n, v in it.product(range(100), repeat=2):
        if run_prgm(copy.copy(ins), n, v) == 19690720:
            print(f"100 * {n} + {v} = {100*n + v}")
            break
        
##Problem 3
with open(f"{os.getcwd()}/input.txt", "r") as f:
    a, b = f.readlines()
    a = [(i[0], int(i[1:])) for i in a.split(",")]
    b = [(i[0], int(i[1:])) for i in b.split(",")]

    ap = [0,0]
    ac = set()
    for d, u in a:
        if d == "R":
            for i in range(u):
                ap[0] += 1
                ac.add(tuple(ap))
        elif d == "U":
            for i in range(u):            
                ap[1] += 1
                ac.add(tuple(ap))
        elif d == "L":
            for i in range(u):            
                ap[0] -= 1
                ac.add(tuple(ap))
        elif d == "D":
            for i in range(u):            
                ap[1] -= 1
                ac.add(tuple(ap))
        else:
            assert(False)
        
    bp = [0,0]
    bc = set()
    for d, u in b:
        if d == "R":
            for i in range(u):
                bp[0] += 1
                bc.add(tuple(bp))
        elif d == "U":
            for i in range(u):            
                bp[1] += 1
                bc.add(tuple(bp))
        elif d == "L":
            for i in range(u):            
                bp[0] -= 1
                bc.add(tuple(bp))
        elif d == "D":
            for i in range(u):            
                bp[1] -= 1
                bc.add(tuple(bp))
        else:
            assert(False)

    cc = ac.intersection(bc).difference({(0,0),})
    md = [abs(t[0]) + abs(t[1]) for t in cc]
    # print(min(md))

    ccda = dict.fromkeys(cc, 0)
    ccdb = dict.fromkeys(cc, 0)
    def run_cable(cable, ccd):
        s = 0
        ap = [0,0]
        for d, u in cable:
            for i in range(u):            
                if d == "R":
                    ap[0] += 1
                elif d == "U":
                    ap[1] += 1
                elif d == "L":
                    ap[0] -= 1
                elif d == "D":
                    ap[1] -= 1
                else:
                    assert(False)
                s += 1
                tap = tuple(ap)
                if tap in ccd and ccd[tap] == 0:
                    ccd[tap] = s

    run_cable(a, ccda)
    run_cable(b, ccdb)
    ms = math.inf
    for k in ccda.keys():
        va = ccda[k]
        vb = ccdb[k]
        if va+vb < ms:
            ms = va+vb
    print(ms)
    
##Problem 4
if input := "136760-595730":
    m, x = input.split("-")
    m, x = int(m), int(x)

    n = 0
    for p in range(m, x+1):
        ds = [int(d) for d in str(p)]
        ok = True
        saw_adj_eq = False
        for i in range(1, len(ds)):
            if ds[i-1] == ds[i]:
                saw_adj_eq = True
            elif ds[i-1] > ds[i]:
                ok = False
                break
        if saw_adj_eq and ok:
            n += 1
    print(n)

    n = 0
    for p in range(m, x+1):
        ds = [int(d) for d in str(p)]
        ok = True
        ls = ds[0]
        st = 1
        saw_adj_eq = False
        for i in range(1, len(ds)):
            if ds[i-1] > ds[i]:
                ok = False
                break
            
            if ls == ds[i]:
                st += 1
                if st == 2 and i == (len(ds)-1):
                    saw_adj_eq = True
            else:
                saw_adj_eq = saw_adj_eq or (st == 2)
                ls = ds[i]
                st = 1                
        if saw_adj_eq and ok:
            n += 1
    print(n)

##Problem 5
with open(f"{os.getcwd()}/input.txt", "r") as f:
    ins = [i for i in f.readlines()[0].split(",")]

    def get_op(x):
        op = str(x)[-2:]
        if len(op) == 1:
            return "0" + op
        else:
            assert(len(op) == 2)
            return op

    def get_modes(x, nparams):
        ds = list(str(x))
        modes = ds[::-1][2:]
        for i in range(nparams - len(modes)):
            modes.append('0')
        assert(len(modes) == nparams)
        return modes

    def run_prgm(ins, i):                    
        ip = 0
        for _ in range(1000):
            op = ins[ip]
            de = get_op(op)
            if de == '01':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)
                assert(dm == '0')
                ins[int(d)] = lv + rv
                ip += 4
            elif de == '02':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)                    
                assert(dm == '0')
                ins[int(d)] = lv * rv
                ip += 4
            elif de == '03':
                d = ins[(ip+1)]
                ins[int(d)] = i
                ip += 2
            elif de == '04':
                d = ins[(ip+1)]
                print(f"out: {ins[int(d)]}")
                ip += 2
            elif de == '05':
                a, b = ins[(ip+1):(ip+3)]
                am, bm = get_modes(op, 2)
                av = int(ins[int(a)]) if am == '0' else int(a)
                bv = int(ins[int(b)]) if bm == '0' else int(b)
                if av != 0:
                    ip = bv
                else:
                    ip += 3
            elif de == '06':
                a, b = ins[(ip+1):(ip+3)]
                am, bm = get_modes(op, 2)
                av = int(ins[int(a)]) if am == '0' else int(a)
                bv = int(ins[int(b)]) if bm == '0' else int(b)
                if av == 0:
                    ip = bv
                else:
                    ip += 3
            elif de == '07':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)                    
                assert(dm == '0')
                ins[int(d)] = 1 if lv < rv else 0
                ip += 4
            elif de == '08':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)                    
                assert(dm == '0')
                ins[int(d)] = 1 if lv == rv else 0
                ip += 4
            elif de == '99':
                break
            else:
                assert(False)
        else:
            assert(False)

    run_prgm(copy.copy(ins), 5)

##Problem 6
with open(f"{os.getcwd()}/input.txt", "r") as f:
    om = defaultdict(list)
    for l in f:
        orbitee, orbiter = l.rstrip().split(")")
        om[orbiter].append(orbitee)

    def orbits_from(k, om):
        if k in om:
            vs = om[k]
            for v in vs:
                return 1 + orbits_from(v, om)
        else:
            return 0

    # n = 0
    # for k in om.keys():
    #     n += orbits_from(k, om)
    # print(n)

    def get_path_for(k):
        path = [k]
        for _ in range(1000):
            assert(k in om)
            k = om[k]
            assert(len(k) == 1)
            path.append(k[0])
            if k[0] == 'COM':
                break
            k = k[0]
        else:
            assert(False)
        return path

    san_path = get_path_for('SAN')
    you_path = get_path_for('YOU')
    common = set(san_path).intersection(set(you_path))
    m = math.inf
    for c in common:
        mm = san_path.index(c) + you_path.index(c) - 2
        if mm < m:
            m = mm
    print(m)

##Problem 7 I
with open(f"{os.getcwd()}/input.txt", "r") as f:
    ins = [i for i in f.readlines()[0].split(",")]

    def get_op(x):
        op = str(x)[-2:]
        if len(op) == 1:
            return "0" + op
        else:
            assert(len(op) == 2)
            return op

    def get_modes(x, nparams):
        ds = list(str(x))
        modes = ds[::-1][2:]
        for i in range(nparams - len(modes)):
            modes.append('0')
        assert(len(modes) == nparams)
        return modes

    def run_thruster_prgm(ins, phase, in_signal):
        i = [in_signal, phase]
        o = None
        ip = 0
        for _ in range(1000):
            op = ins[ip]
            de = get_op(op)
            if de == '01':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)
                assert(dm == '0')
                ins[int(d)] = lv + rv
                ip += 4
            elif de == '02':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)                    
                assert(dm == '0')
                ins[int(d)] = lv * rv
                ip += 4
            elif de == '03':
                d = ins[(ip+1)]
                ins[int(d)] = i.pop()
                ip += 2
            elif de == '04':
                d = ins[(ip+1)]
                o = ins[int(d)]
                ip += 2
            elif de == '05':
                a, b = ins[(ip+1):(ip+3)]
                am, bm = get_modes(op, 2)
                av = int(ins[int(a)]) if am == '0' else int(a)
                bv = int(ins[int(b)]) if bm == '0' else int(b)
                if av != 0:
                    ip = bv
                else:
                    ip += 3
            elif de == '06':
                a, b = ins[(ip+1):(ip+3)]
                am, bm = get_modes(op, 2)
                av = int(ins[int(a)]) if am == '0' else int(a)
                bv = int(ins[int(b)]) if bm == '0' else int(b)
                if av == 0:
                    ip = bv
                else:
                    ip += 3
            elif de == '07':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)                    
                assert(dm == '0')
                ins[int(d)] = 1 if lv < rv else 0
                ip += 4
            elif de == '08':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)                    
                assert(dm == '0')
                ins[int(d)] = 1 if lv == rv else 0
                ip += 4
            elif de == '99':
                break
            else:
                assert(False)
        else:
            assert(False)
            
        return o

    mo = 0
    mp = []
    for phases in it.permutations(range(5)):
        i = 0
        for p in phases:
            i = run_thruster_prgm(copy.copy(ins), p, i)
        if i > mo:
            mo = i
            mp = phases
    print(f"max signal {mo} from {mp}")

##Problem 7 II
with open(f"{os.getcwd()}/input.txt", "r") as f:
    ins = [i for i in f.readlines()[0].rstrip().split(",")]

    def get_op(x):
        op = str(x)[-2:]
        if len(op) == 1:
            return "0" + op
        else:
            assert(len(op) == 2)
            return op

    def get_modes(x, nparams):
        ds = list(str(x))
        modes = ds[::-1][2:]
        for i in range(nparams - len(modes)):
            modes.append('0')
        assert(len(modes) == nparams)
        return modes

    def run_thruster_prgm2(ins, in_ip, phase, signal):
        o = None
        ip = in_ip
        for _ in range(1000):
            op = ins[ip]
            de = get_op(op)
            if de == '01':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)
                assert(dm == '0')
                ins[int(d)] = lv + rv
                ip += 4
            elif de == '02':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)                    
                assert(dm == '0')
                ins[int(d)] = lv * rv
                ip += 4
            elif de == '03':
                d = ins[(ip+1)]
                ins[int(d)] = phase if ip == 0 else signal
                ip += 2
            elif de == '04':
                d = ins[(ip+1)]
                o = ins[int(d)]
                ip += 2
                return o, ip, False
            elif de == '05':
                a, b = ins[(ip+1):(ip+3)]
                am, bm = get_modes(op, 2)
                av = int(ins[int(a)]) if am == '0' else int(a)
                bv = int(ins[int(b)]) if bm == '0' else int(b)
                if av != 0:
                    ip = bv
                else:
                    ip += 3
            elif de == '06':
                a, b = ins[(ip+1):(ip+3)]
                am, bm = get_modes(op, 2)
                av = int(ins[int(a)]) if am == '0' else int(a)
                bv = int(ins[int(b)]) if bm == '0' else int(b)
                if av == 0:
                    ip = bv
                else:
                    ip += 3
            elif de == '07':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)                    
                assert(dm == '0')
                ins[int(d)] = 1 if lv < rv else 0
                ip += 4
            elif de == '08':
                l, r, d = ins[(ip+1):(ip+4)]
                lm, rm, dm = get_modes(op, 3)
                lv = int(ins[int(l)]) if lm == '0' else int(l)
                rv = int(ins[int(r)]) if rm == '0' else int(r)                    
                assert(dm == '0')
                ins[int(d)] = 1 if lv == rv else 0
                ip += 4
            elif de == '99':
                break
            else:
                assert(False)
        else:
            assert(False)

        return o, ip, True

    mo = 0
    mp = []
    for phases in it.permutations(range(5,10)):
        sgs = [0 for p in phases]
        ips = [0 for p in phases]
        inss = [copy.copy(ins) for p in phases]
        for x in range(10000):
            nh = 0
            for np, p in enumerate(phases):
                sg, ip, h = run_thruster_prgm2(inss[np], ips[np], p, sgs[(np - 1) % len(sgs)])
                if sg is not None:
                    sgs[np] = sg
                ips[np] = ip
                nh += h
            if nh == len(phases):
                break
        else:
            assert(False)
        if sgs[-1] > mo:
            mo = sgs[-1]
            mp = phases
    print(f"max signal {mo} from {mp}")

##Problem 8
with open(f"{os.getcwd()}/input.txt", "r") as f:
    input = f.readlines()[0].rstrip()
    w, h = 25, 6

    row = []
    layer = []
    layers = []
    for i, p in enumerate(input):
        row.append(int(p))
        if (i+1) % w == 0:
            layer.append(row)
            row = []
        if (i+1) % (w * h) == 0:
            layers.append(layer)
            layer = []

    min_zero = math.inf
    min_zero_idx = None
    for i, l in enumerate(layers):
        num_zeros = 0
        for r in l:
            num_zeros += sum([p == 0 for p in r])
        if num_zeros < min_zero:
            min_zero = num_zeros
            min_zero_idx = i

    num_wans = 0
    num_twos = 0
    for r in layers[min_zero_idx]:
        num_wans += sum([p == 1 for p in r])
        num_twos += sum([p == 2 for p in r])
    print(f"{num_wans} * {num_twos} = {num_wans * num_twos}")

    layer = [[-1] * w for _ in range(h)]
    for i, j in it.product(range(h), range(w)):
        p = None
        for k, l in enumerate(layers):
            if l[i][j] != 2:
                # print(f"layer {k} at pos (h={i},w={j}) has pixel {l[i][j]} so setting that boy and breaking")
                layer[i][j] = l[i][j]
                break
    print(layer)
