#include <stdint.h>
#include <stdlib.h>
#include "vhf.h"

#define AO_BLOCKSIZE    64

int build_jk(double *vj, double *vk, double *dm, double *Et_dm, int n_dm,
             int *shls_slice, int *ao_loc, int *jengine_loc,
             double *Et_ij_cache, double *Et_kl_cache, int *Et_offsets,
             double *q_cond, double *dm_cond, double cutoff,
             int *atm, int natm, int *bas, int nbas, double *env)
{
        MDIntEnvVars envs = {
                natm, nbas, atm, bas, env, Et_ij_cache, Et_kl_cache, Et_offsets,
                ao_loc, jengine_loc,
        };
        int nao = ao_loc[nbas];
        JKMatrix jk = {nao, n_dm, vj, vk, dm, Et_dm};

        // the first shell-quartet in the bin
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        int lsh1 = shls_slice[7];
        int li = bas[ish0*BAS_SLOTS+ANG_OF];
        int lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        int lk = bas[ksh0*BAS_SLOTS+ANG_OF];
        int ll = bas[lsh0*BAS_SLOTS+ANG_OF];
        int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        int lprim = bas[lsh0*BAS_SLOTS+NPRIM_OF];
        int lij = li + lj;
        int lkl = lk + ll;
        int l4 = lij + lkl;
        int di = ao_loc[ish0+1] - ao_loc[ish0];
        int dj = ao_loc[jsh0+1] - ao_loc[jsh0];
        int dk = ao_loc[ksh0+1] - ao_loc[ksh0];
        int dl = ao_loc[lsh0+1] - ao_loc[lsh0];
        int didj = di * dj;
        int dkdl = dk * dl;
        int Et_ij_len = (lij + 1) * (lij + 2) * (lij + 3) / 6;
        int Et_kl_len = (lkl + 1) * (lkl + 2) * (lkl + 3) / 6;
        int Et_ij_lenp = Et_ij_len * iprim * jprim;
        int Et_kl_lenp = Et_kl_len * kprim * lprim;
        //         eri         Rt2_buf
        int size = didj*dkdl + Et_ij_len*Et_kl_len + (l4+1)*(l4+1)*(l4+1) +
        //         RdotE~Rt2*nprim_kl     RdotE*nprim_ij
                   Et_ij_len*Et_kl_lenp + Et_ij_lenp*dkdl;
        double *buf = malloc(sizeof(double) * size);

        int basblk_i = MAX(AO_BLOCKSIZE/di, 4);
        int basblk_j = MAX(AO_BLOCKSIZE/dj, 4);
        int basblk_k = MAX(AO_BLOCKSIZE/dk, 4);
        int basblk_l = MAX(AO_BLOCKSIZE/dl, 4);
        int _ish0, _jsh0, _ksh0, _lsh0, _ish1, _jsh1, _ksh1, _lsh1;

        for (_ish0 = ish0; _ish0 < ish1; _ish0+=basblk_i) {
                _ish1 = MIN(_ish0+basblk_i, ish1);
                jsh1 = MIN(shls_slice[3], _ish1);
                ksh1 = MIN(shls_slice[5], _ish1);
        for (_jsh0 = jsh0; _jsh0 < jsh1; _jsh0+=basblk_j) {
                _jsh1 = MIN(_jsh0+basblk_j, jsh1);
        for (_ksh0 = ksh0; _ksh0 < ksh1; _ksh0+=basblk_k) {
                _ksh1 = MIN(_ksh0+basblk_k, ksh1);
                lsh1 = MIN(shls_slice[7], _ksh1);
        for (_lsh0 = lsh0; _lsh0 < lsh1; _lsh0+=basblk_l) {
                _lsh1 = MIN(_lsh0+basblk_l, lsh1);
                for (int ish = _ish0; ish < _ish1; ish++) {
                for (int jsh = _jsh0; jsh < MIN(ish+1, _jsh1); jsh++) {
                        int bas_ij = ish * nbas + jsh;
                        double q_ij = q_cond[bas_ij];
                        for (int ksh = _ksh0; ksh < MIN(ish+1, _ksh1); ksh++) {
                        for (int lsh = _lsh0; lsh < MIN(ksh+1, _lsh1); lsh++) {
                                int bas_kl = ksh * nbas + lsh;
                                if (bas_ij < bas_kl) continue;

                                double q_kl = q_cond[bas_kl];
                                double q_ijkl = q_ij + q_kl;
                                if (q_ijkl < cutoff) continue;
                                double d_cutoff = cutoff - q_ijkl;
                                if ((dm_cond[jsh*nbas+ish] < d_cutoff) &&
                                    (dm_cond[lsh*nbas+ksh] < d_cutoff) &&
                                    (dm_cond[jsh*nbas+ksh] < d_cutoff) &&
                                    (dm_cond[jsh*nbas+lsh] < d_cutoff) &&
                                    (dm_cond[ish*nbas+ksh] < d_cutoff) &&
                                    (dm_cond[ish*nbas+lsh] < d_cutoff)) continue;

                                MD_jk_kernel(&envs, &jk, ish, jsh, ksh, lsh, buf);
                        } }
                } }
        } } } }

        free(buf);
        return 0;
}
