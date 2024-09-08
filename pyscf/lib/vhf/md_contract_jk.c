#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "vhf.h"

// 2*pi**2.5
#define PI_FAC  34.98683665524972497

void _contract_jk_s4(JKMatrix *jk, double *eri,
                     int ish, int jsh, int ksh, int lsh, int *ao_loc)
{
        int i0 = ao_loc[ish  ];
        int i1 = ao_loc[ish+1];
        int j0 = ao_loc[jsh  ];
        int j1 = ao_loc[jsh+1];
        int k0 = ao_loc[ksh  ];
        int k1 = ao_loc[ksh+1];
        int l0 = ao_loc[lsh  ];
        int l1 = ao_loc[lsh+1];
        int nao = jk->nao;
        int n_dm = jk->n_dm;
        double *vj = jk->vj;
        double *vk = jk->vk;
        double *dm = jk->dm;
        int i, j, k, l, n, i_dm;

        for (i_dm = 0; i_dm < n_dm; i_dm++) {
                n = 0;
                // unlike libcint, here eri is stored in C-order!
                for (i = i0; i < i1; i++) {
                for (j = j0; j < j1; j++) {
                        double dm_ji = dm[j*nao+i];
                        double v_ij = 0.;
                        for (k = k0; k < k1; k++) {
                                double v_ik = 0.;
                                double v_jk = 0.;
                                for (l = l0; l < l1; l++) {
                                        double s = eri[n];
                                        v_ij += s * dm[l*nao+k];
                                        vj[k*nao+l] += s * dm_ji;

                                        vk[i*nao+l] += s * dm[j*nao+k];
                                        vk[j*nao+l] += s * dm[i*nao+k];
                                        v_ik += s * dm[j*nao+l];
                                        v_jk += s * dm[i*nao+l];
                                        n++;
                                }
                                vk[i*nao+k] += v_ik;
                                vk[j*nao+k] += v_jk;
                        }
                        vj[i*nao+j] += v_ij;
                } }
                dm += nao * nao;
                vj += nao * nao;
                vk += nao * nao;
        }
}

static int _jvec_wo_Rt(double *Rt2, int l1, int l2, int len2,
                       double a, double fac, double *rpq, double *buf,
                       double *rho_ij, double *rho_kl, double *jvec_ij, double *jvec_kl)
{
        int l = l1 + l2;
        int info = get_R_tensor(Rt2, l, a, fac, rpq, buf);
        if (l1 == 0) {
                double rho_kl_val = rho_kl[0];
                double jvec_kl_val = 0.;
                for (int i = 0; i < len2; i++) {
                        jvec_kl_val += Rt2[i] * rho_ij[i];
                        jvec_ij[i]  += Rt2[i] * rho_kl_val;
                }
                jvec_kl[0] += jvec_kl_val;
                return info;
        }

        int e, f, g, t, u, v, n, i, k;
        double s;
        if (l2 == 0) {
                for (n = 0, e = 0; e <= l1; e++) {
                for (f = 0; f <= l1-e; f++) {
                for (g = 0; g <= l1-e-f; g++, n++) {
                        if ((e + f + g) % 2 == 0) {
                                jvec_kl[n] += Rt2[n] * rho_ij[0];
                                jvec_ij[0] += Rt2[n] * rho_kl[n];
                        } else {
                                jvec_kl[n] -= Rt2[n] * rho_ij[0];
                                jvec_ij[0] -= Rt2[n] * rho_kl[n];
                        }
                } } }
                return info;
        }

        int stride_l = (l+1);
        int stride_ll = stride_l * (l+1);
        double *Rsub;
        for (n = 0, t = 0; t <= l; t++) {
                Rsub = buf + t*stride_ll;
                for (u = 0; u <= l-t; u++) {
#pragma GCC ivdep
                for (v = 0; v <= l-t-u; v++, n++) {
                        Rsub[u*stride_l+v] = Rt2[n];
                } }
        }

        for (k = 0, e = 0; e <= l1; e++) {
        for (f = 0; f <= l1-e; f++) {
        for (g = 0; g <= l1-e-f; g++, k++) {
                double rho_kl_val = rho_kl[k];
                double jvec_kl_val = 0.;
                if ((e + f + g) % 2 == 0) {
                        for (i = 0, t = 0; t <= l2; t++) {
                                Rsub = buf + (e+t)*stride_ll + f*stride_l + g;
                                for (u = 0; u <= l2-t; u++) {
#pragma GCC ivdep
                                for (v = 0; v <= l2-t-u; v++, i++) {
                                        s = Rsub[u*stride_l+v];
                                        jvec_kl_val += s * rho_ij[i];
                                        jvec_ij[i]  += s * rho_kl_val;
                                } }
                        }
                } else {
                        for (i = 0, t = 0; t <= l2; t++) {
                                Rsub = buf + (e+t)*stride_ll + f*stride_l + g;
                                for (u = 0; u <= l2-t; u++) {
#pragma GCC ivdep
                                for (v = 0; v <= l2-t-u; v++, i++) {
                                        s = Rsub[u*stride_l+v];
                                        jvec_kl_val -= s * rho_ij[i];
                                        jvec_ij[i]  -= s * rho_kl_val;
                                } }
                        }
                }
                jvec_kl[k] += jvec_kl_val;
        } } }
        return info;
}

void MD_jk_kernel(MDIntEnvVars *envs, JKMatrix *jk,
                  int ish, int jsh, int ksh, int lsh, double *buf)
{
        int nbas = envs->nbas;
        int *bas = envs->bas;
        int *ao_loc = envs->ao_loc;
        int *jengine_loc = envs->jengine_loc;
        double *env = envs->env;
        double *Et_ij_cache = envs->Et_ij_cache;
        double *Et_kl_cache = envs->Et_kl_cache;
        int *Et_offsets = envs->Et_offsets;
        double *vj = jk->vj;
        double *vk = jk->vk;
        double *Et_dm = jk->Et_dm;
        char N = 'N';
        char T = 'T';
        double D0 = 0.;
        double D1 = 1.;
        int li = bas[ish*BAS_SLOTS+ANG_OF];
        int lj = bas[jsh*BAS_SLOTS+ANG_OF];
        int lk = bas[ksh*BAS_SLOTS+ANG_OF];
        int ll = bas[lsh*BAS_SLOTS+ANG_OF];
        int iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
        int jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
        int kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        int lprim = bas[lsh*BAS_SLOTS+NPRIM_OF];
        int lij = li + lj;
        int lkl = lk + ll;
        int l4 = lij + lkl;
        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dk = ao_loc[ksh+1] - ao_loc[ksh];
        int dl = ao_loc[lsh+1] - ao_loc[lsh];
        int didj = di * dj;
        int dkdl = dk * dl;
        int Et_ij_len = (lij + 1) * (lij + 2) * (lij + 3) / 6;
        int Et_kl_len = (lkl + 1) * (lkl + 2) * (lkl + 3) / 6;
        int Et_kl_lenp = Et_kl_len * kprim * lprim;
        double *Rt2 = buf + Et_ij_len*Et_kl_len + (l4+1)*(l4+1)*(l4+1);
        double *eri = Rt2 + Et_ij_len * Et_kl_lenp;
        double *RdotE = eri + didj * dkdl;
        double Rpq[3];

        int bas_ij = ish * nbas + jsh;
        int bas_kl = ksh * nbas + lsh;
        // FIXME: the non-symmetric density matrices
        double fac_sym = 1.;
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (bas_ij == bas_kl) fac_sym *= .5;

        double *Et_ij = Et_ij_cache + Et_offsets[bas_ij];
        double *Et_kl = Et_kl_cache + Et_offsets[bas_kl];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        double xixj = ri[0] - rj[0];
        double yiyj = ri[1] - rj[1];
        double zizj = ri[2] - rj[2];
        double xkxl = rk[0] - rl[0];
        double ykyl = rk[1] - rl[1];
        double zkzl = rk[2] - rl[2];
        double *rho_ij = Et_dm + jengine_loc[bas_ij];
        double *rho_kl = Et_dm + jengine_loc[bas_kl];
        double *jvec_ij = vj + jengine_loc[bas_ij];
        double *jvec_kl = vj + jengine_loc[bas_kl];

        for (int ij = 0, ip = 0; ip < iprim; ip++) {
        for (int jp = 0; jp < jprim; jp++, ij++) {
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xij = ri[0] - xixj * aj_aij;
                double yij = ri[1] - yiyj * aj_aij;
                double zij = ri[2] - zizj * aj_aij;
                double *pRt2 = Rt2;
                double *rhop_ij = rho_ij + ij * Et_ij_len;
                double *jvecp_ij = jvec_ij + ij * Et_ij_len;
                for (int kl = 0, kp = 0; kp < kprim; kp++) {
                for (int lp = 0; lp < lprim; lp++, kl++) {
                        double ak = expk[kp];
                        double al = expl[lp];
                        double akl = ak + al;
                        double al_akl = al / akl;
                        double xkl = rk[0] - xkxl * al_akl;
                        double ykl = rk[1] - ykyl * al_akl;
                        double zkl = rk[2] - zkzl * al_akl;
                        double theta = aij * akl / (aij + akl);
                        double fac = PI_FAC/(aij*akl*sqrt(aij+akl)) * fac_sym;
                        Rpq[0] = xij - xkl;
                        Rpq[1] = yij - ykl;
                        Rpq[2] = zij - zkl;

                        if (vk == NULL) {
                                _jvec_wo_Rt(Rt2, lkl, lij, Et_ij_len, theta, fac, Rpq, buf,
                                            rhop_ij, rho_kl+kl*Et_kl_len,
                                            jvecp_ij, jvec_kl+kl*Et_kl_len);
                        } else {
                                get_Rt2(pRt2, lkl, lij, theta, fac, Rpq, buf);
                                pRt2 += Et_ij_len * Et_kl_len;
                        }
                } }

                if (vk == NULL) continue;
                // RdotE = Rt2.T.dot(Et_kl)
                dgemm_(&N, &T, &dkdl, &Et_ij_len, &Et_kl_lenp, &D1, Et_kl, &dkdl,
                       Rt2, &Et_ij_len, &D0, RdotE, &dkdl);
                // Et_kl.T.dot(RdotE)
                if (ij == 0) {
                        dgemm_(&N, &T, &dkdl, &didj, &Et_ij_len, &D1, RdotE, &dkdl,
                               Et_ij+ij*didj*Et_ij_len, &didj, &D0, eri, &dkdl);
                } else {
                        dgemm_(&N, &T, &dkdl, &didj, &Et_ij_len, &D1, RdotE, &dkdl,
                               Et_ij+ij*didj*Et_ij_len, &didj, &D1, eri, &dkdl);
                }
        } }
        if (vk != NULL) {
                _contract_jk_s4(jk, eri, ish, jsh, ksh, lsh, ao_loc);
        }
}
