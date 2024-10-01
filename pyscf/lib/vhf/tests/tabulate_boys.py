from os import path
import tempfile
import pickle
import numpy as np
import scipy.special
import mpmath as mp

DECIMALS = 30
mp.mp.dps = DECIMALS
mp.mp.pretty = True

def boys(m, t):
    #             _ 1           2
    #            /     2 m  -t u
    # F (t)  =   |    u    e      du,
    #  m        _/  0
    #
    assert m >= 0
    assert t >= 0
    # downward is alaways more accurate than upward, but ~3x slower
    if (t < m + 1.5):
        return downward(m, t)
    else:
        return upward(m, t)

def downward(m, t, prec=.1**DECIMALS):
    #
    # F[m] = int u^{2m} e^{-t u^2} du
    #      = 1/(2m+1) int e^{-t u^2} d u^{2m+1}
    #      = 1/(2m+1) [e^{-t u^2} u^{2m+1}]_0^1 + (2t)/(2m+1) int u^{2m+2} e^{-t u^2} du
    #      = 1/(2m+1) e^{-t} + (2t)/(2m+1) F[m+1]
    #      = 1/(2m+1) e^{-t} + (2t)/(2m+1)(2m+3) e^{-t} + (2t)^2/(2m+1)(2m+3) F[m+2]
    #
    e = mp.mpf('.5') * mp.exp(-t)
    x = e
    s = e
    b = m + mp.mpf('1.5')
    while x > prec:
        x *= t / b
        s += x
        b += 1

    b = m + mp.mpf('.5')
    f = s / b
    out = [f]
    for i in range(m):
        b -= 1
        f = (e + t * f) / b
        out.append(f)
    return np.array(out[::-1])

def upward(m, t):
    #
    # F[m] = int u^{2m} e^{-t u^2} du
    #      = -1/2t int u^{2m-1} d e^{-t u^2}
    #      = -1/2t [e^{-t u^2} * u^{2m-1}]_0^1 + (2m-1)/2t int u^{2m-2} e^{-t u^2} du
    #      = 1/2t (-e^{-t} + (2m-1) F[m-1])
    #
    tt = mp.sqrt(t)
    f = mp.sqrt(mp.pi)/2 / tt * mp.erf(tt)
    e = mp.exp(-t)
    b = mp.mpf('.5') / t
    out = [f]
    for i in range(m):
        f = b * ((2*i+1) * f - e)
        out.append(f)
    return np.array(out)

MAX_ORDER = 12 # (ff|ff)
INTERVAL = 2.
T_MAX = 60.
DEGREE = 12

def tabulate_chebfit():
    n = DEGREE + 1
    cheb_nodes = np.cos(np.pi / n * (np.arange(n) + .5))

    intervals = [(a, a+INTERVAL) for a in np.arange(0., T_MAX, INTERVAL)]
    cs = []
    for m in range(0, MAX_ORDER+1):
        cs_i = []
        for a, b in intervals:
            # Map chebyshev nodes to the sample points between interval (a, b)
            xs = cheb_nodes*(b-a)/2 + (b+a)/2
            ints_m = np.array([boys(m, x)[m] for x in xs], dtype=np.float64)
            cs_i.append(np.polynomial.chebyshev.chebfit(cheb_nodes, ints_m, DEGREE))
        cs.append(np.array(cs_i))
    return cs

db_file = 'chebfit_tab.pkl'
#with open(db_file, 'rb') as f:
#    chebfit_tab = pickle.load(f)
#with open(db_file, 'wb') as f:
#    pickle.dump(tabulate_chebfit(), f)

def polynomial_approx(l, t):
    #chebfit_tab = tabulate_chebfit()
    return _cheb_eval(l, t, chebfit_tab[l])

def _cheb_eval(l, t, chebfit_tab):
    interval_id = int(t // INTERVAL)
    c = chebfit_tab[interval_id]
    a = interval_id * INTERVAL
    b = a + INTERVAL
    x = (t - (a+b)/2) / ((b-a)/2)

    x2 = 2*x
    c0 = c[DEGREE-1]
    c1 = c[DEGREE]
    for i in range(2, DEGREE + 1):
        tmp = c0
        c0 = c[DEGREE-i] - c1
        c1 = tmp + c1*x2
    ints_m = c0 + c1*x
    return ints_m

def dump_chebfit_tab(chebfit_tab):
    print(f'#define MAX_ORDER       {MAX_ORDER}')
    print(f'#define INTERVAL        {INTERVAL}')
    print(f'#define T_MAX           {T_MAX}')
    print(f'#define INTERVALS       {int(T_MAX//INTERVAL)}')
    print(f'#define DEGREE          {DEGREE}')
    print('static double BOYS_DATA[] = {')
    for l, tab in enumerate(chebfit_tab):
        print(f'// order {l}')
        print('\n'.join([f'{c:24.17e},' for c in tab.ravel()]))
    print('};')

if __name__ == '__main__':
    with open(db_file, 'rb') as f:
        chebfit_tab = pickle.load(f)
    dump_chebfit_tab(chebfit_tab)
