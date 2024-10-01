import ctypes
import numpy as np
from pyscf.lib import logger
from pyscf.gto import ANG_OF, ATOM_OF, NPRIM_OF, NCTR_OF, PTR_COORD
from pyscf import lib
from pyscf.scf._vhf import libcvhf

__all__ = [
    'get_jk',
]

PTR_BAS_COORD = 7
GROUP_NAO = 800
ET_CACHE_MAXMEM = 2000 # 2GB

libvhf_boost = lib.load_library('libvhf_boost')

def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None,
           verbose=None):
    '''Compute J, K matrices with CPU-GPU hybrid algorithm
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if hermi != 1:
        raise NotImplementedError('JK-builder only supports hermitian density matrix')
    if omega is None:
        omega = 0.0

    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    mol = vhfopt.mol
    nbas = mol.nbas
    nao = mol.nao
    ao_loc = mol.ao_loc

    dm = np.asarray(dm, order='C')
    dms = dm.reshape(-1,nao,nao)
    n_dm = dms.shape[0]

    # reorder dm
    ao_idx = vhfopt.ao_idx
    dms = dms[:,ao_idx[:,None],ao_idx]

    vj = vk = None
    vj_ptr = vk_ptr = lib.c_null_ptr()

    if with_k:
        vj = np.zeros(dms.shape)
        vk = np.zeros(dms.shape)
        vj_ptr = vj.ctypes
        vk_ptr = vk.ctypes
        # assign them to an arbitrary object to avoid invalid pointer
        jengine_loc = ao_loc
        Et_dm = dm
    elif with_j: # call J-engine
        jengine_loc = _make_j_engine_pair_locs(mol)
        jvec = np.zeros((n_dm, jengine_loc[nbas*nbas]))
        vj_ptr = jvec.ctypes
        ctr_coef = _MD_ctr_coeffs(mol)
        Et_dm = np.zeros_like(jvec)
        if not mol.cart:
            c2s = mol.cart2sph_coeff()
            dm_cart = lib.einsum('xij,pi,qj->xpq', dms, c2s, c2s)
            dm_cart = np.asarray(dm_cart, order='C')
        else:
            dm_cart = dms
        ao_loc_cart = mol.ao_loc_nr(cart=True)
        libvhf_boost.cache_Et_dot_dm(
            Et_dm.ctypes, dm_cart.ctypes,
            jengine_loc.ctypes, ao_loc_cart.ctypes, ctr_coef.ctypes,
            mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
    else:
        return vj, vk

    dm_cond = [lib.condense('absmax', x, ao_loc) for x in dms]
    dm_cond = np.log(np.max(dm_cond, axis=0) + 1e-300)
    q_cond = vhfopt.q_cond

    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    load_Et_cache, Et_offsets = _cache_E_tensor(mol, l_ctr_bas_loc)

    log_cutoff = np.log(vhfopt.direct_scf_tol)
    l_symb = [lib.param.ANGULAR[i] for i in vhfopt.uniq_l_ctr[:,0]]
    n_groups = len(vhfopt.uniq_l_ctr)
    fn = libvhf_boost.build_jk_simd

    t1 = cput0
    for i in range(n_groups):
        for j in range(i+1):
            ij_shls = (l_ctr_bas_loc[i], l_ctr_bas_loc[i+1],
                       l_ctr_bas_loc[j], l_ctr_bas_loc[j+1])
            Et_ij_cache = load_Et_cache(i, j)
            for k in range(i+1):
                for l in range(k+1):
                    kl_shls = (l_ctr_bas_loc[k], l_ctr_bas_loc[k+1],
                               l_ctr_bas_loc[l], l_ctr_bas_loc[l+1])
                    Et_kl_cache = load_Et_cache(k, l)

                    fn(vj_ptr, vk_ptr, dms.ctypes, Et_dm.ctypes, ctypes.c_int(n_dm),
                       (ctypes.c_int*8)(*ij_shls, *kl_shls),
                       ao_loc.ctypes, jengine_loc.ctypes,
                       Et_ij_cache.ctypes, Et_kl_cache.ctypes, Et_offsets.ctypes,
                       q_cond.ctypes, dm_cond.ctypes, ctypes.c_double(log_cutoff),
                       mol._atm.ctypes, ctypes.c_int(mol.natm),
                       mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                    t1 = log.timer_debug1(
                            f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})', *t1)

    if with_k:
        vk[:,ao_idx[:,None],ao_idx] = vk + vk.transpose(0,2,1)
        vk = vk.reshape(dm.shape)
        vj[:,ao_idx[:,None],ao_idx] = vj + vj.transpose(0,2,1)
        vj *= 2.
        vj = vj.reshape(dm.shape)
    else:
        # The J-engine
        nao_cart = ao_loc_cart[-1]
        vj = np.zeros((n_dm, nao_cart, nao_cart))
        libvhf_boost.jengine_dot_Et(
            vj.ctypes, jvec.ctypes,
            jengine_loc.ctypes, ao_loc_cart.ctypes, ctr_coef.ctypes,
            mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
        if not mol.cart:
            vj = lib.einsum('xpq,pi,qj->xij', vj, c2s, c2s)
        # reorder vj and vk
        vj[:,ao_idx[:,None],ao_idx] = vj + vj.transpose(0,2,1)
        vj *= 2.
        vj = vj.reshape(dm.shape)
    log.timer('vj and vk', *cput0)
    return vj, vk

def _get_jk(mf, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
            omega=None):
    if omega is not None:
        raise NotImplementedError
    vhfopt = mf.opt.get(omega)
    if vhfopt is None:
        vhfopt = mf.opt[omega] = _VHFOpt(mol, mf.direct_scf_tol)
        vhfopt.build()

    vj, vk = get_jk(mol, dm, hermi, vhfopt, with_j, with_k, omega, verbose=log)
    return vj, vk

# TODO: patch to hf.SCF.get_jk
#from pyscf import scf
#scf.hf.SCF.get_jk = _get_jk

class _VHFOpt:
    def __init__(self, mol, cutoff=1e-13):
        self.mol = mol.copy()
        self.direct_scf_tol = 1e-13

        self.uniq_l_ctr = None
        self.l_ctr_offsets = None
        self.ao_idx = None
        self.q_cond = None

    def build(self, group_size=GROUP_NAO, verbose=None):
        mol = self.mol
        log = logger.new_logger(mol, verbose)
        cput0 = (logger.process_clock(), logger.perf_counter())
        # Sort basis according to angular momentum and contraction patterns so
        # as to group the basis functions to blocks in GPU kernel.
        l_ctrs = mol._bas[:,[ANG_OF, NPRIM_OF, NCTR_OF]]
        uniq_l_ctr, _, inv_idx, l_ctr_counts = np.unique(
            l_ctrs, return_index=True, return_inverse=True, return_counts=True, axis=0)

        # Limit the number of AOs in each group
        uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(
            uniq_l_ctr, l_ctr_counts, group_size)
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))

        if mol.verbose >= logger.DEBUG:
            log.debug('Number of shells for each [l, nprim, nctr] group')
            for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
                log.debug('    %s : %s', l_ctr, n)

        sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)
        # Sort contraction coefficients before updating self.mol
        ao_loc = mol.ao_loc
        nao = ao_loc[-1]
        ao_idx = np.array_split(np.arange(nao), ao_loc[1:-1])
        self.ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])

        # Sort basis inplace
        mol._bas = mol._bas[sorted_idx]

        # PTR_BAS_COORD is required by nr_contract_jk.c
        mol._bas[:,PTR_BAS_COORD] = mol._atm[mol._bas[:,ATOM_OF],PTR_COORD]

        nbas = mol.nbas
        ao_loc = mol.ao_loc
        q_cond = np.empty((nbas,nbas))
        intor = mol._add_suffix('int2e')
        libcvhf.CVHFnr_int2e_q_cond(
            getattr(libcvhf, intor), lib.c_null_ptr(),
            q_cond.ctypes, ao_loc.ctypes,
            mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)

        self.q_cond = np.log(q_cond + 1e-300)
        log.timer('Initialize q_cond', *cput0)
        return self

def _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size):
    '''Splits l_ctr patterns into small groups with group_size the maximum
    number of AOs in each group
    '''
    l = uniq_l_ctr[:,0]
    nf = l * (l + 1) // 2
    _l_ctrs = []
    _l_ctr_counts = []
    for l_ctr, counts in zip(uniq_l_ctr, l_ctr_counts):
        l = l_ctr[0]
        n_ctr = l_ctr[2]
        nf = (l + 1) * (l + 2) // 2 * n_ctr
        max_shells = max(group_size // nf, 2)
        if counts <= max_shells:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(counts)
            continue

        nsubs, remaining = counts.__divmod__(max_shells)
        _l_ctrs.extend([l_ctr] * nsubs)
        _l_ctr_counts.extend([max_shells] * nsubs)
        if remaining > 0:
            _l_ctrs.append(l_ctr)
            _l_ctr_counts.append(remaining)
    uniq_l_ctr = np.vstack(_l_ctrs)
    l_ctr_counts = np.hstack(_l_ctr_counts)
    return uniq_l_ctr, l_ctr_counts

def _optimize_basis_contractions():
    pass

def _make_j_engine_pair_locs(mol):
    ls = mol._bas[:,ANG_OF]
    nprim = mol._bas[:,NPRIM_OF]
    lij = ls[:,None] + ls
    nfij = (lij+1)*(lij+2)*(lij+3)//6
    prim_pair_locs = np.append(0, np.cumsum((nfij*nprim[:,None]*nprim).ravel()))
    return prim_pair_locs.astype(np.int32)

def _MD_ctr_coeffs(mol):
    nbas = mol.nbas
    cs = [mol._libcint_ctr_coeff(i).ravel() for i in range(nbas)]
    for i in range(nbas):
        l = mol.bas_angular(i)
        # Racah normalization, assuming angular part is normalized to unity
        # Apply to s and p functions only, to make cartesian functions compatible
        # with libcint results.
        if l < 2:
            cs[i] *= ((2*l+1)/(4*np.pi))**.5
    return np.hstack(cs)

def _cache_E_tensor(mol, bas_loc, max_memory=ET_CACHE_MAXMEM):
    nbas = mol.nbas
    norm_cs = _MD_ctr_coeffs(mol)

    if mol.cart:
        fn = libvhf_boost.cache_Et_cart
    else:
        fn = libvhf_boost.cache_Et_sph

    # Et size for each pair
    ls = mol._bas[:,ANG_OF]
    nprim = mol._bas[:,NPRIM_OF]
    lij = ls[:,None] + ls
    nfij = (lij+1)*(lij+2)*(lij+3)//6
    ao_loc = mol.ao_loc
    dims = ao_loc[1:] - ao_loc[:-1]
    Et_size_table = dims[:,None]*dims * nfij*nprim[:,None]*nprim
    sizes_for_cp = lib.condense('sum', Et_size_table, bas_loc)
    cache_size = max(sizes_for_cp.max()*2, int(max_memory*1e6/8))

    Et_offsets = np.empty((nbas,nbas), dtype=np.int32)
    _cache = {}
    tot_size = 0

    def load_Et_cache(groupi, groupj):
        nonlocal tot_size
        if (groupi, groupj) in _cache:
            return _cache[groupi, groupj]

        tot_size += sizes_for_cp[groupi,groupj]
        while tot_size > cache_size:
            # drop the most recently created Et
            key, val = _cache.popitem()
            tot_size -= val.size
            val = None

        Et_cache = _cache[groupi, groupj] = np.empty(sizes_for_cp[groupi,groupj])
        shls_slice = (bas_loc[groupi], bas_loc[groupi+1],
                      bas_loc[groupj], bas_loc[groupj+1])
        fn(Et_cache.ctypes, Et_offsets.ctypes,
           (ctypes.c_int*4)(*shls_slice), ao_loc.ctypes, norm_cs.ctypes,
           mol._atm.ctypes, mol.natm, mol._bas.ctypes, nbas, mol._env.ctypes)
        return Et_cache

    return load_Et_cache, Et_offsets
