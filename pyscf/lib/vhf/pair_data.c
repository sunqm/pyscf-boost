#include <stdlib.h>
#include "vhf.h"

double *CINTc2s_bra_sph(double *gsph, int nket, double *gcart, int l);
double *CINTc2s_ket_sph1(double *sph, double *cart, int lds, int ldc, int l);
void get_E_cart_components(double *Ecart, int li, int lj, double ai, double aj,
                           double *Ra, double *Rb);
void get_E_tensor(double *Et, int li, int lj, double ai, double aj,
                  double *Ra, double *Rb, double *buf);

#define Ex_at(i,j,t)    Ex[(i)*stride1+(j)*stride2+t]
#define Ey_at(i,j,t)    Ey[(i)*stride1+(j)*stride2+t]
#define Ez_at(i,j,t)    Ez[(i)*stride1+(j)*stride2+t]

// Shape of E tensor is [:li,:li+lj,:lj]
static void _get_E_itj(double *Et, int li, int lj, double ai, double aj,
                       double *Ra, double *Rb, double *buf)
{
        get_E_cart_components(buf, li, lj, ai, aj, Ra, Rb);
        int lij = li + lj;
        int stride2 = lij+1;
        int stride1 = (lj+1) * stride2;
        int Ex_size = (li+1) * stride1;
        double *Ex = buf;
        double *Ey = Ex + Ex_size;
        double *Ez = Ey + Ex_size;
        int t, u, v, n;
        int ix, iy, iz;
        int jx, jy, jz;

        n = 0;
        for (ix = li; ix >= 0; ix--) {
        for (iy = li-ix; iy >= 0; iy--) {
                iz = li - ix - iy;
                for (t = 0; t <= lij; t++) {
                for (u = 0; u <= lij-t; u++) {
                for (v = 0; v <= lij-t-u; v++) {
                        for (jx = lj; jx >= 0; jx--) {
                        for (jy = lj-jx; jy >= 0; jy--) {
                                jz = lj - jx - jy;
                                Et[n] = Ex_at(ix,jx,t) * Ey_at(iy,jy,u) * Ez_at(iz,jz,v);
                                n++;
                        } }
                } }
        } } }
}

static void _make_ctr_coeff_offsets(int *offsets, int *bas, int nbas)
{
        for (int size = 0, ish = 0; ish < nbas; ish++) {
                int iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
                int ictr = bas[ish*BAS_SLOTS+NCTR_OF];
                offsets[ish] = size;
                size += iprim*ictr;
        }
}

void cache_Et_cart(double *Et_cache, int *Et_offsets, int *shls_slice,
                   int *ao_loc, double *ctr_coef,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        int l2 = 2*LMAX;
        int Et_size = (l2+1)*(l2+2)*(l2+3)/6*NCART_MAX*NCART_MAX;
        int Ex_size = (2*LMAX+1)*(LMAX+1)*(LMAX+1);
        double *Et_raw = malloc(sizeof(double) * (Et_size+3*Ex_size));
        double *buf = Et_raw + Et_size;

        int *ptr_coef = malloc(sizeof(int) * nbas);
        _make_ctr_coeff_offsets(ptr_coef, bas, nbas);

        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int off = 0;
        for (int ish = ish0; ish < ish1; ish++) {
                int li = bas[ish*BAS_SLOTS+ANG_OF];
                int iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
                int ictr = bas[ish*BAS_SLOTS+NCTR_OF];
                double *ai = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *ci = ctr_coef + ptr_coef[ish];
                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                for (int jsh = jsh0; jsh < jsh1; jsh++) {
                        int lj = bas[jsh*BAS_SLOTS+ANG_OF];
                        int jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
                        int jctr = bas[jsh*BAS_SLOTS+NCTR_OF];
                        double *aj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                        double *cj = ctr_coef + ptr_coef[jsh];
                        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];

                        int lij = li + lj;
                        int nfi = (li + 1) * (li + 2) / 2;
                        int nfj = (lj + 1) * (lj + 2) / 2;
                        int nfinfj = nfi * nfj;
                        int di = ao_loc[ish+1] - ao_loc[ish];
                        int dj = ao_loc[jsh+1] - ao_loc[jsh];
                        int didj = di * dj;
                        int Et_len = (lij + 1) * (lij + 2) * (lij + 3) / 6;
                        Et_offsets[ish*nbas+jsh] = off;
                        for (int ip = 0; ip < iprim; ip++) {
                        for (int jp = 0; jp < jprim; jp++) {
                                get_E_tensor(Et_raw, li, lj, ai[ip], aj[jp], ri, rj, buf);
                                // Et = einsum('i,j,pqt->tipjq', ci, cj, Et)
                                double *Et = Et_cache + off;
                                for (int n = 0, t = 0; t < Et_len; t++) {
                                for (int ic = 0; ic < ictr; ic++) {
                                for (int i = 0; i < nfi; i++) {
                                for (int jc = 0; jc < jctr; jc++) {
                                        double cc = ci[ip*ictr+ic] * cj[jp*jctr+jc];
                                        for (int j = 0; j < nfj; j++, n++) {
                                                Et[n] = cc * Et_raw[t*nfinfj+i*nfj+j];
                                        }
                                } } } }
                                off += didj * Et_len;
                        } }
                }
        }
        free(Et_raw);
        free(ptr_coef);
}

void cache_Et_sph(double *Et_cache, int *Et_offsets, int *shls_slice,
                  int *ao_loc, double *ctr_coef,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        int l2 = 2*LMAX;
        int Et_size = (l2+1)*(l2+2)*(l2+3)/6*NCART_MAX*NCART_MAX;
        double *Et_raw = malloc(sizeof(double) * 2*Et_size);
        double *buf = Et_raw + Et_size;
        double *Et_sph;

        int *ptr_coef = malloc(sizeof(int) * nbas);
        _make_ctr_coeff_offsets(ptr_coef, bas, nbas);

        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int off = 0;
        for (int ish = ish0; ish < ish1; ish++) {
                int li = bas[ish*BAS_SLOTS+ANG_OF];
                int iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
                int ictr = bas[ish*BAS_SLOTS+NCTR_OF];
                double *ai = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *ci = ctr_coef + ptr_coef[ish];
                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                for (int jsh = jsh0; jsh < jsh1; jsh++) {
                        int lj = bas[jsh*BAS_SLOTS+ANG_OF];
                        int jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
                        int jctr = bas[jsh*BAS_SLOTS+NCTR_OF];
                        double *aj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                        double *cj = ctr_coef + ptr_coef[jsh];
                        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];

                        int lij = li + lj;
                        int nfj = (lj + 1) * (lj + 2) / 2;
                        int nsph_i = 2 * li + 1;
                        int nsph_j = 2 * lj + 1;
                        int di = ao_loc[ish+1] - ao_loc[ish];
                        int dj = ao_loc[jsh+1] - ao_loc[jsh];
                        int didj = di * dj;
                        int Et_len = (lij + 1) * (lij + 2) * (lij + 3) / 6;
                        int stride_tj = Et_len * nsph_j;
                        Et_offsets[ish*nbas+jsh] = off;
                        for (int ip = 0; ip < iprim; ip++) {
                        for (int jp = 0; jp < jprim; jp++) {
                                _get_E_itj(Et_raw, li, lj, ai[ip], aj[jp], ri, rj, buf);
                                CINTc2s_ket_sph1(buf, Et_raw, Et_len*nfj, Et_len*nfj, li);
                                Et_sph = CINTc2s_bra_sph(Et_raw, Et_len*nsph_i, buf, lj);

                                double *Et = Et_cache + off;
                                for (int n = 0, t = 0; t < Et_len; t++) {
                                for (int ic = 0; ic < ictr; ic++) {
                                for (int i = 0; i < nsph_i; i++) {
                                for (int jc = 0; jc < jctr; jc++) {
                                        double cc = ci[ip*ictr+ic] * cj[jp*jctr+jc];
                                        for (int j = 0; j < nsph_j; j++, n++) {
                                                Et[n] = cc * Et_sph[i*stride_tj+t*nsph_j+j];
                                        }
                                } } } }
                                off += didj * Et_len;
                        } }
                }
        }
        free(Et_raw);
        free(ptr_coef);
}

void cache_Et_dot_dm(double *Et_dm, double *dm,
                     int *jengine_loc, int *ao_loc, double *ctr_coef,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int l2 = 2*LMAX;
        int Et_size = (l2+1)*(l2+2)*(l2+3)/6*NCART_MAX*NCART_MAX;
        int Ex_size = (2*LMAX+1)*(LMAX+1)*(LMAX+1);
        double *Et = malloc(sizeof(double) * (Et_size+3*Ex_size));
        double *buf = Et + Et_size;

        int *ptr_coef = malloc(sizeof(int) * nbas);
        _make_ctr_coeff_offsets(ptr_coef, bas, nbas);

        size_t nao = ao_loc[nbas];
        for (int ish = 0; ish < nbas; ish++) {
                int li = bas[ish*BAS_SLOTS+ANG_OF];
                int iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
                int ictr = bas[ish*BAS_SLOTS+NCTR_OF];
                int i0 = ao_loc[ish];
                double *ai = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *ci = ctr_coef + ptr_coef[ish];
                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                // TODO: consider the non-symmetry density matrix
                for (int jsh = 0; jsh <= ish; jsh++) {
                        int lj = bas[jsh*BAS_SLOTS+ANG_OF];
                        int jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
                        int jctr = bas[jsh*BAS_SLOTS+NCTR_OF];
                        int j0 = ao_loc[jsh];
                        double *aj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                        double *cj = ctr_coef + ptr_coef[jsh];
                        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
                        double *rho = Et_dm + jengine_loc[ish*nbas+jsh];
                        int lij = li + lj;
                        int nfi = (li + 1) * (li + 2) / 2;
                        int nfj = (lj + 1) * (lj + 2) / 2;
                        int Et_len = (lij + 1) * (lij + 2) * (lij + 3) / 6;
                        for (int ip = 0; ip < iprim; ip++) {
                        for (int jp = 0; jp < jprim; jp++) {
                                get_E_tensor(Et, li, lj, ai[ip], aj[jp], ri, rj, buf);
                                for (int ic = 0; ic < ictr; ic++) {
                                for (int jc = 0; jc < jctr; jc++) {
                                        double cc = ci[ip*ictr+ic] * cj[jp*jctr+jc];
                                        double *pdm = dm + (j0+jc*nfj)*nao + i0+ic*nfi;
                                        for (int n = 0, t = 0; t < Et_len; t++) {
                                                double rho_t = 0.;
                                                for (int i = 0; i < nfi; i++) {
                                                for (int j = 0; j < nfj; j++, n++) {
                                                        rho_t += Et[n] * cc * pdm[j*nao+i];
                                                } }
                                                rho[t] += rho_t;
                                        }
                                } }
                                rho += Et_len;
                        } }
                }
        }
        free(Et);
        free(ptr_coef);
}

void jengine_dot_Et(double *vj, double *jvec,
                    int *jengine_loc, int *ao_loc, double *ctr_coef,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        int l2 = 2*LMAX;
        int Et_size = (l2+1)*(l2+2)*(l2+3)/6*NCART_MAX*NCART_MAX;
        int Ex_size = (2*LMAX+1)*(LMAX+1)*(LMAX+1);
        double *Et = malloc(sizeof(double) * (Et_size+3*Ex_size));
        double *buf = Et + Et_size;

        int *ptr_coef = malloc(sizeof(int) * nbas);
        _make_ctr_coeff_offsets(ptr_coef, bas, nbas);

        size_t nao = ao_loc[nbas];
        for (int ish = 0; ish < nbas; ish++) {
                int li = bas[ish*BAS_SLOTS+ANG_OF];
                int iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
                int ictr = bas[ish*BAS_SLOTS+NCTR_OF];
                int i0 = ao_loc[ish];
                double *ai = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *ci = ctr_coef + ptr_coef[ish];
                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                for (int jsh = 0; jsh <= ish; jsh++) {
                        int lj = bas[jsh*BAS_SLOTS+ANG_OF];
                        int jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
                        int jctr = bas[jsh*BAS_SLOTS+NCTR_OF];
                        int j0 = ao_loc[jsh];
                        double *aj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                        double *cj = ctr_coef + ptr_coef[jsh];
                        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
                        double *jvec_ij = jvec + jengine_loc[ish*nbas+jsh];
                        int lij = li + lj;
                        int nfi = (li + 1) * (li + 2) / 2;
                        int nfj = (lj + 1) * (lj + 2) / 2;
                        int Et_len = (lij + 1) * (lij + 2) * (lij + 3) / 6;
                        for (int ip = 0; ip < iprim; ip++) {
                        for (int jp = 0; jp < jprim; jp++) {
                                get_E_tensor(Et, li, lj, ai[ip], aj[jp], ri, rj, buf);
                                for (int ic = 0; ic < ictr; ic++) {
                                for (int jc = 0; jc < jctr; jc++) {
                                        double cc = ci[ip*ictr+ic] * cj[jp*jctr+jc];
                                        double *pj = vj + (i0+ic*nfi)*nao + j0+jc*nfj;
                                        for (int n = 0, t = 0; t < Et_len; t++) {
                                                double fac = cc * jvec_ij[t];
                                                for (int i = 0; i < nfi; i++) {
                                                for (int j = 0; j < nfj; j++, n++) {
                                                        pj[i*nao+j] += Et[n] * fac;
                                                } }
                                        }
                                } }
                                jvec_ij += Et_len;
                        } }
                }
        }
        free(Et);
        free(ptr_coef);
}
