import itertools
from functools import lru_cache
import jinja2

LMAX = 4

@lru_cache(100)
def xyz2idx(l):
    cache = {}
    ij = 0
    for t in range(l+1):
        for u in range(l+1-t):
            for v in range(l+1-t-u):
                cache[t,u,v] = ij
                ij += 1
    return cache

class _RT:
    def __init__(self, l):
        self.l = l

    def __getitem__(self, key):
        n, t, u, v = key
        if t < 0: return 0
        if u < 0: return 0
        if v < 0: return 0
        return xyz2idx(self.l-n)[t, u, v]

class ToC:
    def __init__(self):
        self.result = []

    def __call__(self, R0, mul, R1, x, R2):
        if mul == 0:
            self.result.append(f'out[{R0}] = {x} * Rt[{R2}]')
        elif mul == 1:
            self.result.append(f'out[{R0}] = {x} * Rt[{R2}] + Rt[{R1}]')
        else:
            self.result.append(f'out[{R0}] = {x} * Rt[{R2}] + {mul} * Rt[{R1}]')

class CacheIdx:
    def __init__(self):
        self.result = [[], [], []]

    def __call__(self, R0, mul, R1, x, R2):
        R0_idx, R1_idx, R2_idx = self.result
        R0_idx.append(R0)
        R1_idx.append(R1)
        R2_idx.append(R2)

def unroll_R_tensor(l, proc):
    Rt = _RT(l)
    n = 0
    for t in range(l+1):
        for u in range(l+1-t):
            for v in range(l+1-t-u):
                if t >= 1:
                    proc(Rt[n,t,u,v], t-1, Rt[n+1,t-2,u,v], 'rx', Rt[n+1,t-1,u,v])
                elif u >= 1:
                    proc(Rt[n,t,u,v], u-1, Rt[n+1,t,u-2,v], 'ry', Rt[n+1,t,u-1,v])
                elif v >= 1:
                    proc(Rt[n,t,u,v], v-1, Rt[n+1,t,u,v-2], 'rz', Rt[n+1,t,u,v-1])
    return proc.result

def generate_R_index(lmax):
    Ridx = CacheIdx()
    offsets = [0]
    for l in range(lmax+1):
        unroll_R_tensor(l, Ridx)
        offsets.append(len(Ridx.result[0]))
    return Ridx.result, offsets

if __name__ == '__main__':
    LMAX = 12
    r_idx, offsets = generate_R_index(LMAX)
    for ix, idx in enumerate(r_idx):
        print(f'static int Rt{ix}_idx[] = {{')
        for l, (k0, k1) in enumerate(itertools.pairwise(offsets)):
            if k0 == k1: continue
            print(f'// l = {l}')
            if k1 - k0 < 20:
                print(','.join([str(x) for x in idx[k0:k1]]) + ',')
            else:
                for i in range(k0, k1, 20):
                    i1 = min(i+20, k1)
                    print(','.join([str(x) for x in idx[i:i1]]) + ',')
        print('};\n')
    print('int Rt_idx_offsets[] = {')
    print(','.join([str(x) for x in offsets]))
    print('};')

    t = jinja2.Template('''static void iter_Rt_{{l}}(double *out, double *Rt, double *rpq)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
{%- for x in code %}
        {{ x }};
{%- endfor %}
}
''')
    for l in range(1, 7):
        code = unroll_R_tensor(l, ToC())
        print(t.render(l=l, code=code))
        print()
