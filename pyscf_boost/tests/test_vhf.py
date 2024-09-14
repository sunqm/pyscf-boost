import numpy as np
from pyscf import gto, scf
from pyscf_boost.vhf import get_jk

def test_get_jk():
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
        unit='B')

    np.random.seed(3)
    nao = mol.nao
    dm = np.random.rand(nao,nao)
    dm = dm + dm.T
    vj, vk = get_jk(mol, dm)
    ref = scf.hf.get_jk(mol, dm)
    assert abs(vj - ref[0]).max() < 1e-12
    assert abs(vk - ref[1]).max() < 1e-12
    vj = get_jk(mol, dm, with_k=False)[0]
    assert abs(vj - ref[0]).max() < 1e-12

    mol.cart = True
    mol.build()
    nao = mol.nao
    dm = np.random.rand(nao,nao)
    dm = dm + dm.T
    vj, vk = get_jk(mol, dm)
    ref = scf.hf.get_jk(mol, dm)
    assert abs(vj - ref[0]).max() < 1e-12
    assert abs(vk - ref[1]).max() < 1e-12
    vj = get_jk(mol, dm, with_k=False)[0]
    assert abs(vj - ref[0]).max() < 1e-12

if __name__ == '__main__':
    test_get_jk()
