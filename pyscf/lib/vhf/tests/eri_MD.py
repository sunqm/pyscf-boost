import math
import ctypes
import numpy as np
from pyscf import lib, gto, scf
from pyscf.gto import ANG_OF, ATOM_OF, PTR_EXP, PTR_COEFF, PTR_COORD

PTR_BAS_COORD = 7

libvhf = lib.load_library('libvhf_boost')

def gto_norm(l, expnt):
    '''Radial part normalization'''
    assert l >= 0
    # Radial part normalization
    norm = (gto.gaussian_int(l*2+2, 2*expnt)) ** -.5
    # Racah normalization, assuming angular part is normalized to unity
    if l < 2:
        norm *= ((2*l+1)/(4*np.pi))**.5
    return norm

def boys(m, t):
    #             _ 1           2
    #            /     2 m  -t u
    # F (t)  =   |    u    e      du,
    #  m        _/  0
    #
    assert m >= 0
    assert t >= 0
    if (t < m + 1.5):
        return downward(m, t)
    else:
        return upward(m, t)

def downward(m, t, prec=1e-15):
    half = .5
    b = m + half
    e = half * np.exp(-t)
    x = e
    f = e
    while x > prec * e:
        b += 1
        x *= t / b
        f += x
    b = m + half
    f /= b
    out = [f]
    for i in range(m):
        b -= 1
        f = (e + t * f) / b
        out.append(f)
    return np.array(out)[::-1]

def upward(m, t):
    half = .5
    tt = np.sqrt(t)
    f = np.sqrt(np.pi)/2 / tt * math.erf(tt)
    e = np.exp(-t)
    b = half / t
    out = [f]
    for i in range(m):
        f = b * ((2*i+1) * f - e)
        out.append(f)
    return np.array(out)

def get_E_cart_components(li, lj, ai, aj, Ra, Rb):
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra
    Rpb = Rp - Rb
    theta_ij = ai * aj / aij
    Kab = np.exp(-theta_ij * Rab.dot(Rab))

    lij = li + lj
    E_cart = np.empty((li+1, lj+1, lij+1, 3))
    E_cart[0,0,0] = 1.
    E_cart[0,0,0,2] = Kab
    E_cart[0,0,1:] = 0.

    for i in range(1, li+1):
        E_cart[i,0,0] = Rpa * E_cart[i-1,0,0] + E_cart[i-1,0,1]
        for t in range(1, lij+1):
            E_cart[i,0,t] = i*E_cart[i-1,0,t-1] / (2*aij*t)
    for j in range(1, lj+1):
        E_cart[0,j,0] = Rpb * E_cart[0,j-1,0] + E_cart[0,j-1,1]
        for t in range(1, lij+1):
            E_cart[0,j,t] = j*E_cart[0,j-1,t-1] / (2*aij*t)
        for i in range(1, li+1):
            E_cart[i,j,0] = Rpb * E_cart[i,j-1,0] + E_cart[i,j-1,1]
            for t in range(1, lij+1):
                E_cart[i,j,t] = (i*E_cart[i-1,j,t-1] + j*E_cart[i,j-1,t-1]) / (2*aij*t)
    return E_cart.transpose(3,0,1,2)

def get_E_tensor(li, lj, ai, aj, Ra, Rb):
    Ex, Ey, Ez = get_E_cart_components(li, lj, ai, aj, Ra, Rb)

    lij = li + lj
    nfi = (li+1)*(li+2)//2
    nfj = (lj+1)*(lj+2)//2
    nf_ij = (lij+1)*(lij+2)*(lij+3)//6
    Et = np.empty((nfi, nfj, nf_ij))
    for i, (ix, iy, iz) in enumerate(iter_cart_xyz(li)):
        for j, (jx, jy, jz) in enumerate(iter_cart_xyz(lj)):
            # products subject to t+u+v <= li+lj
            for n, (t, u, v) in enumerate(reduced_cart_iter(lij)):
                Et[i,j,n] = Ex[ix,jx,t] * Ey[iy,jy,u] * Ez[iz,jz,v]
    return Et

def get_E_tensor_1(li, lj, ai, aj, Ra, Rb):
    lij = li + lj
    nfi = (li+1)*(li+2)//2
    nfj = (lj+1)*(lj+2)//2
    nfij = (lij+1)*(lij+2)*(lij+3)//6
    Et = np.empty((nfij,nfi,nfj))
    buf = np.empty((3,li+1,lj+1,lij+1))
    libvhf.get_E_tensor(Et.ctypes, ctypes.c_int(li), ctypes.c_int(lj),
                        ctypes.c_double(ai), ctypes.c_double(aj),
                        Ra.ctypes, Rb.ctypes, buf.ctypes)
    return Et.transpose(1,2,0)

def iter_cart_xyz(n):
    '''Produces all (ix,iy,iz) subject to 0 <= ix/iy/iz <= n and ix + iy + iz == n'''
    for i in reversed(range(n+1)):
        for j in reversed(range(n+1-i)):
            yield i, j, n-i-j

def reduced_cart_iter(n):
    '''Nested loops for Cartesian components, subject to x+y+z <= n'''
    for x in range(n+1):
        for y in range(n+1-x):
            for z in range(n+1-x-y):
                yield x, y, z

def get_R_tensor(l, a, rpq):
    rx, ry, rz = rpq
    Rt = np.zeros((l+1, l+1, l+1, l+1))
    r2 = rx*rx + ry*ry + rz*rz

    Rt[:,0,0,0] = (-2*a)**np.arange(l+1) * boys(l, a*r2)
    if l == 0:
        return Rt[:1,0,0,0]

    for n in reversed(range(l+1)):
        for t in range(l+1-n):
            for u in range(l+1-t-n):
                for v in range(l+1-t-u-n):
                    if t >= 1:
                        Rt[n,t,u,v] = (t-1) * Rt[n+1,t-2,u,v] + rx * Rt[n+1,t-1,u,v]
                    elif u >= 1:
                        Rt[n,t,u,v] = (u-1) * Rt[n+1,t,u-2,v] + ry * Rt[n+1,t,u-1,v]
                    elif v >= 1:
                        Rt[n,t,u,v] = (v-1) * Rt[n+1,t,u,v-2] + rz * Rt[n+1,t,u,v-1]
                    else: # t == u == v == 0
                        pass

    nf = (l+1)*(l+2)*(l+3)//6
    R2 = np.zeros(nf)
    ij = 0
    for t in range(l+1):
        for u in range(l+1-t):
            for v in range(l+1-t-u):
                R2[ij] = Rt[0,t,u,v]
                ij += 1
    return R2

def get_R_tensor_1(l, a, rpq, fac=1.):
    nf = (l+1)*(l+2)*(l+3)//6
    Rt = np.empty((2,nf))
    libvhf.get_R_tensor(Rt.ctypes, ctypes.c_int(l), ctypes.c_double(a),
                        ctypes.c_double(fac), rpq.ctypes, Rt[1].ctypes)
    return Rt[0]

def get_Rt2(l1, l2, a, rpq, fac=1.):
    nf1 = (l1+1)*(l1+2)*(l1+3)//6
    nf2 = (l2+1)*(l2+2)*(l2+3)//6
    Rt2, buf = np.empty((2, nf2, nf1))
    libvhf.get_Rt2(Rt2.ctypes, ctypes.c_int(l2), ctypes.c_int(l1),
                   ctypes.c_double(a), ctypes.c_double(fac),
                   rpq.ctypes, buf.ctypes)
    return Rt2.T

def cache_E_tensor(mol):
    ao_loc = mol.ao_loc
    nbas = mol.nbas
    _bas = mol._bas
    coords = mol.atom_coords()
    exps = mol.bas_exps()
    cs = [mol.bas_ctr_coeff(i) for i in range(nbas)]
    norm_cs = []
    for i in range(nbas):
        l = mol.bas_angular(i)
        norm_cs.append(gto_norm(l, exps[i])[:,None] * cs[i])

    Et_cache = {}
    for i in range(nbas):
        i0, i1 = ao_loc[i], ao_loc[i+1]
        di = i1 - i0
        for j in range(nbas):
            j0, j1 = ao_loc[j], ao_loc[j+1]
            dj = j1 - j0
            li = _bas[i, ANG_OF]
            lj = _bas[j, ANG_OF]
            norm_ci = norm_cs[i]
            norm_cj = norm_cs[j]
            Ra = coords[_bas[i, ATOM_OF]]
            Rb = coords[_bas[j, ATOM_OF]]
            Et_ij = []
            for ai, ci in zip(exps[i], norm_ci):
                for aj, cj in zip(exps[j], norm_cj):
                    Et = get_E_tensor(li, lj, ai, aj, Ra, Rb)
                    Et = np.einsum('i,j,pqt->tipjq', ci, cj, Et)
                    Et_ij.append(Et.reshape(Et.shape[0], di, dj))
            Et_cache[i,j] = np.stack(Et_ij)
    return Et_cache

def cache_E_tensor_1(mol):
    mol._bas[:,PTR_BAS_COORD] = mol._atm[mol._bas[:,ATOM_OF],PTR_COORD]
    nbas = mol.nbas
    ls = mol._bas[:,gto.ANG_OF]
    nprim = mol._bas[:,gto.NPRIM_OF]
    lij = ls[:,None] + ls
    nfij = (lij+1)*(lij+2)*(lij+3)//6
    ao_loc = mol.ao_loc
    dims = ao_loc[1:] - ao_loc[:-1]
    Et_size = (dims[:,None]*dims * nfij*nprim[:,None]*nprim).sum()
    Et_cache = np.empty(Et_size)
    Et_locs = np.empty(nbas**2+1, dtype=np.int32)

    exps = mol.bas_exps()
    cs = [mol.bas_ctr_coeff(i) for i in range(nbas)]
    norm_cs = []
    for i in range(nbas):
        l = mol.bas_angular(i)
        c = gto_norm(l, exps[i])[:,None] * cs[i]
        norm_cs.append(c.ravel())
    norm_cs = np.hstack(norm_cs)

    if mol.cart:
        fn = libvhf.cache_Et_cart
    else:
        fn = libvhf.cache_Et_sph
    shls_slice = (ctypes.c_int*4)(0, nbas, 0, nbas)
    fn(Et_cache.ctypes, Et_locs.ctypes, shls_slice,
       ao_loc.ctypes, norm_cs.ctypes,
       mol._atm.ctypes, mol.natm, mol._bas.ctypes, mol.nbas, mol._env.ctypes)
    return Et_cache, Et_locs

def primitive_ERI(li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd, Etab, Etcd):
    aij = ai + aj
    Rp = (ai * Ra + aj * Rb) / aij
    akl = ak + al
    Rq = (ak * Rc + al * Rd) / akl
    Rpq = Rp - Rq
    theta = aij * akl / (aij + akl)
    lij = li + lj
    lkl = lk + ll

    #l4 = lij + lkl
    #Rt = get_R_tensor(l4, theta, Rpq)
    #Rt *= 2*np.pi**2.5/(aij*akl*(aij+akl)**.5)
    #nf_ij = (lij+1)*(lij+2)*(lij+3)//6
    #nf_kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    #Rt2 = np.empty((nf_kl, nf_ij))
    #for kl, (t, u, v) in enumerate(reduced_cart_iter(lkl)):
    #    phase = (-1)**(t+u+v)
    #    for ij, (e, f, g) in enumerate(reduced_cart_iter(lij)):
    #        Rt2[kl,ij] = phase * Rt[t+e,u+f,v+g]

    nf_ij = Etab.shape[0]
    nf_kl = Etcd.shape[0]
    Rt2, buf = np.empty((2, nf_kl, nf_ij))
    fac = 2*np.pi**2.5/(aij*akl*(aij+akl)**.5)
    libvhf.get_Rt2(Rt2.ctypes, ctypes.c_int(lkl), ctypes.c_int(lij),
                   ctypes.c_double(theta), ctypes.c_double(fac),
                   Rpq.ctypes, buf.ctypes)
    Rt2 = Rt2.T

    # Basis contraction can be applied to the gcd array before proceeding to the
    # second np.dot
    gcd = np.dot(Rt2, Etcd.reshape(nf_kl,-1))
    eri = np.dot(Etab.reshape(nf_ij,-1).T, gcd)
    return eri

def contracted_ERI(V, i, j, k, l, bas, coords, exps, Etab, Etcd):
    li = bas[i, ANG_OF]
    lj = bas[j, ANG_OF]
    lk = bas[k, ANG_OF]
    ll = bas[l, ANG_OF]
    Ra = coords[bas[i, ATOM_OF]]
    Rb = coords[bas[j, ATOM_OF]]
    Rc = coords[bas[k, ATOM_OF]]
    Rd = coords[bas[l, ATOM_OF]]
    ij = 0
    for ai in exps[i]:
        for aj in exps[j]:
            kl = 0
            for ak in exps[k]:
                for al in exps[l]:
                    V += primitive_ERI(
                        li, lj, lk, ll, ai, aj, ak, al, Ra, Rb, Rc, Rd,
                        Etab[ij], Etcd[kl]).reshape(V.shape)
                    kl += 1
            ij += 1
    return V

def get_tensor(mol):
    ao_loc = mol.ao_loc
    nao = mol.nao
    coords = mol.atom_coords()
    exps = mol.bas_exps()
    Et_cache = cache_E_tensor(mol)
    V = np.zeros((nao, nao, nao, nao))

    for i, (i0, i1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
        for j, (j0, j1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
            for k, (k0, k1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
                for l, (l0, l1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
                    contracted_ERI(
                        V[i0:i1,j0:j1,k0:k1,l0:l1], i, j, k, l, mol._bas, coords, exps,
                        Et_cache[i,j], Et_cache[k,l])
    return V

def contract_Rt_dm(shls, ls, coords, exps, Et_dm):
    shls = list(shls)
    i, j, k, l = shls
    li, lj, lk, ll = ls[shls]
    Ra, Rb, Rc, Rd = coords[shls]
    out = []
    for ai in exps[i]:
        for aj in exps[j]:
            lij = li + lj
            aij = ai + aj
            Rp = (ai * Ra + aj * Rb) / aij
            Rt_dm = 0.
            kl = 0
            for ak in exps[k]:
                for al in exps[l]:
                    lkl = lk + ll
                    akl = ak + al
                    Rq = (ak * Rc + al * Rd) / akl
                    Rpq = Rp - Rq
                    theta = aij * akl / (aij + akl)
                    l4 = lij + lkl
                    fac = 2*np.pi**2.5/(aij*akl*(aij+akl)**.5)
                    Rt = get_Rt2(lij, lkl, theta, Rpq, fac)
                    Rt_dm += np.einsum('pq,q->p', Rt, Et_dm[kl])
                    kl += 1
            out.append(Rt_dm)
    return np.array(out)

def j_engine(mol, dm):
    ao_loc = mol.ao_loc
    nao = mol.nao
    nbas = mol.nbas
    ls = mol._bas[:,ANG_OF]
    coords = mol.atom_coords()[mol._bas[:,ATOM_OF]]
    exps = mol.bas_exps()
    Et_cache = cache_E_tensor(mol)

    Et_dm = []
    for i, (i0, i1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
        for j, (j0, j1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
            Et_dm.append(np.einsum('mtij,ji->mt', Et_cache[i,j], dm[j0:j1,i0:i1]))

    jmat = np.empty((nao,nao))
    for i, (i0, i1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
        for j, (j0, j1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
            Rt_dm = 0.
            kl = 0
            for k in range(nbas):
                for l in range(nbas):
                    Rt_dm += contract_Rt_dm([i,j,k,l], ls, coords, exps, Et_dm[kl])
                    kl += 1
            jmat[i0:i1,j0:j1] = np.einsum('mt,mtij->ij', Rt_dm, Et_cache[i,j])
    return jmat

def get_matrix(mol, Rc) -> np.ndarray:
    ao_loc = mol.ao_loc
    nao = mol.nao
    coords = mol.atom_coords()
    exps = mol.bas_exps()
    Et_cache = cache_E_tensor(mol)
    V = np.zeros((nao, nao))
    for i, (i0, i1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
        for j, (j0, j1) in enumerate(zip(ao_loc[:-1], ao_loc[1:])):
            contracted_coulomb_1e(
                V[i0:i1,j0:j1], i, j, Rc, mol._bas, coords, exps, Et_cache[i,j])
    return V

def contracted_coulomb_1e(V, i, j, Rc, bas, coords, exps, Et):
    li = bas[i, ANG_OF]
    lj = bas[j, ANG_OF]
    Ra = coords[bas[i, ATOM_OF]]
    Rb = coords[bas[j, ATOM_OF]]
    k = 0
    for ai in exps[i]:
        for aj in exps[j]:
            aij = ai + aj
            Rp = (ai * Ra + aj * Rb) / aij
            fac = 2*np.pi/aij
            Rt2 = get_R_tensor_1(li+lj, aij, Rp-Rc, fac)
            V += np.einsum('tab,t->ab', Et[k], Rt2)
            k += 1
    return V

def new_basis():
    mol = gto.M(
        atom = '''
    C  0.1, 0.5, 0.8
    H2 0.3, 0.0, 1.0
    H3 0.2, 1.9, 0.0
    H4 0.5, 0.0, 0.5''',
        basis = {
            'C': 'ccpvdz',
            'H2': [[3, [.75, 1]]],
            'H3': [[1, [.32, 1]]],
            'H4': [[2, [.51, 1]]],},
        charge=1,
        unit='B',
        cart=True)
    return mol

def test_coulomb_1e_MD():
    mol = new_basis()
    Rc = np.ones(3)
    V = get_matrix(mol, Rc)
    with mol.with_rinv_orig(Rc):
        ref = mol.intor('int1e_rinv')
    assert abs(V - ref).max() < 1e-12

def test_R_tensor():
    a = 1.4
    rpq = np.array([1.2, .15, -.71])
    def check(l, ref, prec=1e-12):
        R0 = get_R_tensor(l, a, rpq)
        R1 = get_R_tensor_1(l, a, rpq)
        assert abs(lib.fp(R1) - ref) < prec, abs(R0 - R1).max() < prec
    check(0, 0.5239812616643992)
    check(1, 0.9064067081830955)
    check(2, 0.0600667085347522)
    check(3, 0.9224532580821307)
    check(4, 3.9528789441715286)
    check(5, 6.502397147343664 )
    check(6, 11.857315024580116)
    check(7, -21.46354515425636)
    check(8, 132.95818058079507)
    check(9, 616.3486716058869 )
    check(13,-602233.4709187684, 1e-9)

    rpq = np.zeros(3)
    assert abs(lib.fp(get_R_tensor_1(0, a, rpq)) - 1.) < 1e-12
    assert abs(lib.fp(get_R_tensor_1(1, a, rpq)) - 1.) < 1e-12
    assert abs(lib.fp(get_R_tensor_1(2, a, rpq)) - 1.9740405854373533) < 1e-12
    assert abs(lib.fp(get_R_tensor_1(3, a, rpq)) - 1.5785771250254077) < 1e-12
    assert abs(lib.fp(get_R_tensor_1(4, a, rpq)) - -5.318800840774767) < 1e-12
    assert abs(lib.fp(get_R_tensor_1(5, a, rpq)) - 4.559237093779861 ) < 1e-12
    assert abs(lib.fp(get_R_tensor_1(6, a, rpq)) - -48.09038027758597) < 1e-12

def test_E_tensor():
    ai, aj = 1.1, .4
    Ra, Rb = np.array([[1.2, .15, -.71],
                       [0.3, 1.1, 0.]])
    def check(li, lj):
        ref = get_E_tensor(li, lj, ai, aj, Ra, Rb)
        dat = get_E_tensor_1(li, lj, ai, aj, Ra, Rb)
        assert abs(ref - dat).max() < 1e-12
    check(0, 1)
    check(1, 1)
    check(2, 1)
    check(0, 2)
    check(2, 3)
    check(3, 0)

    mol = new_basis()
    nbas = mol.nbas
    ref = cache_E_tensor(mol)
    Et_cache, Et_locs = cache_E_tensor_1(mol)
    for i in range(nbas):
        for j in range(nbas):
            p0 = Et_locs[i*nbas+j]
            size = ref[i,j].size
            assert abs(ref[i,j].ravel() - Et_cache[p0:p0+size]).max() < 1e-10

def test_coulomb_ERI():
    mol = new_basis()
    V = get_tensor(mol)
    ref = mol.intor('int2e')
    assert abs(V - ref).max() < 1e-12

def test_j_engine():
    np.random.seed(3)
    mol = new_basis()
    nao = mol.nao
    dm = np.random.rand(nao, nao)
    dm = dm + dm.T
    vj = j_engine(mol, dm)
    ref = scf.hf.get_jk(mol, dm)[0]
    assert abs(vj - ref).max() < 1e-12

def test_Rt2():
    a = 1.4
    rpq = np.array([1.2, .15, -.71])
    def fR(l1, l2, a, rpq, fac=1.):
        nf1 = (l1+1)*(l1+2)*(l1+3)//6
        nf2 = (l2+1)*(l2+2)*(l2+3)//6
        Rt2, buf = np.empty((2, nf1, nf2))
        libvhf.get_Rt2(Rt2.ctypes, ctypes.c_int(l1), ctypes.c_int(l2),
                       ctypes.c_double(a), ctypes.c_double(fac),
                       rpq.ctypes, buf.ctypes)
        return Rt2
    assert abs(lib.fp(fR(0, 0, a, rpq)) - 0.5239812616643992) < 1e-14
    assert abs(lib.fp(fR(0, 1, a, rpq)) - 0.9064067081830955) < 1e-14
    assert abs(lib.fp(fR(0, 2, a, rpq)) - 0.0600667085347519) < 1e-14
    assert abs(lib.fp(fR(0, 3, a, rpq)) - 0.9224532580821302) < 1e-14
    assert abs(lib.fp(fR(1, 0, a, rpq)) - 0.1415558151457027) < 1e-14
    assert abs(lib.fp(fR(1, 1, a, rpq)) - 1.5689430888887017) < 1e-14
    assert abs(lib.fp(fR(1, 2, a, rpq)) - 0.3928928043850768) < 1e-14
    assert abs(lib.fp(fR(1, 3, a, rpq)) - -1.015115561425564) < 1e-14
    assert abs(lib.fp(fR(2, 0, a, rpq)) - 0.3503170138953471) < 1e-14
    assert abs(lib.fp(fR(2, 1, a, rpq)) - 2.1953028473818677) < 1e-14
    assert abs(lib.fp(fR(2, 2, a, rpq)) - -1.759916588322134) < 1e-14
    assert abs(lib.fp(fR(2, 3, a, rpq)) - 1.0938080586476235) < 1e-14
    assert abs(lib.fp(fR(3, 0, a, rpq)) - -0.352583163166223) < 1e-14
    assert abs(lib.fp(fR(3, 1, a, rpq)) - 1.1151091940721036) < 1e-14
    assert abs(lib.fp(fR(3, 2, a, rpq)) - -2.830911458071544) < 1e-14
    assert abs(lib.fp(fR(3, 3, a, rpq)) - -3.409176320263147) < 1e-14
    assert abs(lib.fp(fR(4, 4, a, rpq)) - 159.00680637443983) < 1e-10
    assert abs(lib.fp(fR(5, 5, a, rpq)) - 3881.959704852806 ) < 1e-10
    assert abs(lib.fp(fR(5, 2, a, rpq)) - 110.86694980606967) < 1e-10


if __name__ == '__main__':
    test_Rt2()
    test_R_tensor()
    test_coulomb_1e_MD()
    test_coulomb_ERI()
    test_E_tensor()
    test_j_engine()
