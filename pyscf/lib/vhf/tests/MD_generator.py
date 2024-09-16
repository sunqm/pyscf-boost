import itertools
from functools import lru_cache
import jinja2

LMAX = 4

@lru_cache(100)
def xyz2idx(l):
    cache = {
        (t, u, v): ij for ij, (t, u, v) in enumerate(reduced_cart_iter(l))
    }
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
    def __init__(self, simd=False):
        self.result = []
        self.simd = simd

    def __call__(self, R0, mul, R1, x, R2):
        if self.simd:
            if mul == 0:
                self.result.append(f'MM_STORE(out+{R0}*SIMDD, {x} * MM_LOAD(Rt+{R2}*SIMDD))')
            elif mul == 1:
                self.result.append(f'MM_STORE(out+{R0}*SIMDD, {x} * MM_LOAD(Rt+{R2}*SIMDD) + MM_LOAD(Rt+{R1}*SIMDD))')
            else:
                self.result.append(f'MM_STORE(out+{R0}*SIMDD, {x} * MM_LOAD(Rt+{R2}*SIMDD) + MM_SET1({mul}) * MM_LOAD(Rt+{R1}*SIMDD))')
        else:
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

def unroll_Rt_to_Rt2(l1, l2):
    idx = xyz2idx(l1+l2)
    n = 0
    for kl, (t, u, v) in enumerate(reduced_cart_iter(l1)):
        phase = (-1)**(t+u+v)
        for ij, (e, f, g) in enumerate(reduced_cart_iter(l2)):
            if phase > 0:
                print(f'        Rt2[{n}] = Rt[{idx[t+e,u+f,v+g]}];')
            else:
                print(f'        Rt2[{n}] = -Rt[{idx[t+e,u+f,v+g]}];')
            n += 1

def unroll_Rt_to_Rt2_simd(l1, l2):
    idx = xyz2idx(l1+l2)
    len_ij = len(list(reduced_cart_iter(l2)))
    for kl, (t, u, v) in enumerate(reduced_cart_iter(l1)):
        phase = (-1)**(t+u+v)
        for ij, (e, f, g) in enumerate(reduced_cart_iter(l2)):
            if phase > 0:
                print(f'        MM_SCATTER(Rt2+{kl}*{len_ij}*SIMDD+{ij}, idx_Rt2, Rt[{idx[t+e,u+f,v+g]}], sizeof(double));')
            else:
                print(f'        MM_SCATTER(Rt2+{kl}*{len_ij}*SIMDD+{ij}, idx_Rt2, -Rt[{idx[t+e,u+f,v+g]}], sizeof(double));')

def unroll_rho_dot_Rt2(l1, l2):
    idx = xyz2idx(l1+l2)
    print('        double jvec_kl_val, rho_kl_val;')
    for kl, (t, u, v) in enumerate(reduced_cart_iter(l1)):
        phase = (-1)**(t+u+v)
        print(f'        rho_kl_val = rho_kl[{kl}];')
        print(f'        jvec_kl_val = 0.;')
        for ij, (e, f, g) in enumerate(reduced_cart_iter(l2)):
            if phase > 0:
                print(f'        jvec_kl_val += Rt[{idx[t+e,u+f,v+g]}] * rho_ij[{ij}];')
                print(f'        jvec_ij[{ij}] += Rt[{idx[t+e,u+f,v+g]}] * rho_kl_val;')
            else:
                print(f'        jvec_kl_val -= Rt[{idx[t+e,u+f,v+g]}] * rho_ij[{ij}];')
                print(f'        jvec_ij[{ij}] -= Rt[{idx[t+e,u+f,v+g]}] * rho_kl_val;')
        print(f'        jvec_kl[{kl}] += jvec_kl_val;')

def unroll_rho_dot_Rt2_simd(l1, l2):
    idx = xyz2idx(l1+l2)
    print('        __MD jvec_kl_val, rho_kl_val, s;')
    for kl, (t, u, v) in enumerate(reduced_cart_iter(l1)):
        phase = (-1)**(t+u+v)
        print(f'        rho_kl_val = MM_LOAD(rho_kl+{kl}*SIMDD);')
        print(f'        jvec_kl_val = MM_SET0();')
        for ij, (e, f, g) in enumerate(reduced_cart_iter(l2)):
            print(f'        s = MM_LOAD(Rt+{idx[t+e,u+f,v+g]}*SIMDD);')
            if phase > 0:
                print(f'        jvec_kl_val += s * MM_LOAD(rho_ij+{ij}*SIMDD);')
                print(f'        MM_STORE(jvec_ij+{ij}*SIMDD, MM_LOAD(jvec_ij+{ij}*SIMDD) + s * rho_kl_val);')
            else:
                print(f'        jvec_kl_val -= s * MM_LOAD(rho_ij+{ij}*SIMDD);')
                print(f'        MM_STORE(jvec_ij+{ij}*SIMDD, MM_LOAD(jvec_ij+{ij}*SIMDD) - s * rho_kl_val);')
        print(f'        MM_STORE(jvec_kl+{kl}*SIMDD, MM_LOAD(jvec_kl+{kl}*SIMDD) + jvec_kl_val);')

def generate_Rt2jvec(lmax=3, simd=False):
    for li in range(lmax+1):
        for lj in range(lmax+1):
            print(f'''static void Rt2jvec_{li}_{lj}(double *Rt, double *rho_ij, double *rho_kl, double *jvec_ij, double *jvec_kl)''')
            print('{')
            if simd:
                unroll_rho_dot_Rt2_simd(li, lj)
            else:
                unroll_rho_dot_Rt2(li, lj)
            print('}')

def generate_Rt2(lmax=3):
    for li in range(lmax+1):
        for lj in range(lmax+1):
            print(f'''static void Rt2_{li}_{lj}(double *Rt2, double a, double fac, double *rpq, double *Rt)''')
            print('{')
            print(f'        get_R_tensor(Rt, {li+lj}, a, fac, rpq, Rt2);')
            unroll_Rt_to_Rt2(li, lj)
            print('}')

def generate_Rt2_simd(lmax=3):
    for li in range(lmax+1):
        for lj in range(lmax+1):
            print(f'''static void Rt2_{li}_{lj}_simd(double *Rt2, __MD a, __MD fac, __MD *rpq, __MD *Rt, __m128i idx_Rt2)''')
            print('{')
            print(f'        get_R_tensor_simd(Rt, {li+lj}, a, fac, rpq, (__MD *)Rt2);')
            unroll_Rt_to_Rt2_simd(li, lj)
            print('}')

def reduced_cart_iter(n):
    '''Nested loops for Cartesian components, subject to x+y+z <= n'''
    for x in range(n+1):
        for y in range(n+1-x):
            for z in range(n+1-x-y):
                yield x, y, z

def generate_Rt_idx(lmax=12):
    r_idx, offsets = generate_R_index(lmax+1)
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


def generate_iter_Rt(lmax=6):
    t = jinja2.Template('''static void iter_Rt_{{l}}(double *out, double *Rt, double rx, double ry, double rz)
{
{%- for x in code %}
        {{ x }};
{%- endfor %}
}
''')
    for l in range(1, lmax+1):
        code = unroll_R_tensor(l, ToC())
        print(t.render(l=l, code=code))
        print()

def generate_iter_Rt_simd(lmax=6):
    t = jinja2.Template('''static void iter_Rt_{{l}}(double *out, double *Rt, __MD rx, __MD ry, __MD rz)
{
{%- for x in code %}
        {{ x }};
{%- endfor %}
}
''')
    for l in range(1, lmax+1):
        code = unroll_R_tensor(l, ToC(simd=True))
        print(t.render(l=l, code=code))
        print()

if __name__ == '__main__':
    #generate_Rt_idx(12)
    #generate_iter_Rt(6)
    #generate_Rt2jvec(3)
    #generate_Rt2(3)
    generate_Rt2_simd(3)
    #generate_iter_Rt_simd(6)
