#include <stdlib.h>
#include <stddef.h>

// slots of atm
#define CHARGE_OF       0
#define PTR_COORD       1
#define NUC_MOD_OF      2
#define PTR_ZETA        3
#define PTR_FRAC_CHARGE 4
#define RESERVE_ATMSLOT 5
#define ATM_SLOTS       6

// slots of bas
#define ATOM_OF         0
#define ANG_OF          1
#define NPRIM_OF        2
#define NCTR_OF         3
#define KAPPA_OF        4
#define PTR_EXP         5
#define PTR_COEFF       6
#define PTR_BAS_COORD   7
#define BAS_SLOTS       8

#define LMAX            5
#define NCART_MAX       ((LMAX+1)*(LMAX+2)/2)


#define MIN(x, y)       ((x) < (y) ? (x) : (y))
#define MAX(x, y)       ((x) > (y) ? (x) : (y))


#ifndef HAVE_DEFINED_MDINTENVVAS_H
#define HAVE_DEFINED_MDINTENVVAS_H
typedef struct {
        int natm;
        int nbas;
        int *atm;
        int *bas;
        double *env;
        double *Et_ij_cache;
        double *Et_kl_cache;
        int *Et_offsets;
        int *ao_loc;
        int *jengine_loc;
        double omega;
} MDIntEnvVars;

typedef struct {
        int nao;
        int n_dm;
        double *vj;
        double *vk;
        double *dm;
        double *Et_dm;
        double fac;
} JKMatrix;
#endif

void MD_jk_kernel(MDIntEnvVars *envs, JKMatrix *jk,
                  int ish, int jsh, int ksh, int lsh, double *buf);

int get_R_tensor(double *Rt, int l, double a, double fac, double *rpq, double *buf);
void get_Rt2(double *Rt2, int l1, int l2, double a, double fac, double *rpq, double *buf);
void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*,
            double*, double*, int*);
