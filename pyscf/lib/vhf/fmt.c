#include <float.h>
#include <math.h>

#define SML_FLOAT64   (DBL_EPSILON * .5)
#define SQRTPIE4      .8862269254527580136490837416705725913987747280611935641069038949264
#define ERFC_bound    200

// TODO: Chebyshev for fmt1 at m

/*
 * Relative errors of fmt1_erfc_like are of
 *      (2*t)**(m-1) / (2m-3)!! * machine_precision * fmt_val
 * Errors of the other choice are
 *      (2m-1)!! / (2*t)**(m-1) * machine_precision * fmt_val
 * Given m, the turn-over point for t should satisfy
 *      (2m-1)!! / (2*t)**(m-1) > (2m-1)**.5
 * t0 = .5 * ((2m-1)!!/(2m-1)**.5)**(1/(m-1))
 */
static double TURNOVER_POINT[] = {
        0.,
        0.,
        0.866025403784,
        1.295010032056,
        1.705493613097,
        2.106432965305,
        2.501471934009,
        2.892473348218,
        3.280525047072,
        3.666320693281,
        4.05033123037 ,
        4.432891808508,
        4.814249856864,
        5.194593501454,
        5.574069276051,
        5.952793645111,
        6.330860773135,
        6.708347923415,
        7.08531930745 ,
        7.461828891625,
        7.837922483937,
        8.213639312398,
        8.589013237349,
        8.964073695432,
        9.338846443746,
        9.713354153046,
        10.08761688545,
        10.46165248270,
        10.83547688448,
        11.20910439128,
        11.58254788331,
        11.95581900374,
        12.32892831326,
        12.70188542111,
        13.07469909673,
        13.44737736550,
        13.81992759110,
        14.19235654675,
        14.56467047710,
        14.93687515212
};

/*
 * Name
 *
 * fmtpse
 *
 * Synopsis
 *
 * double fmtpse(int m, double t)
 *
 * Description
 *
 * This function evaluates the auxiliary integral,
 *
 *             _ 1           2
 *            /     2 m  -t u
 * F (t)  =   |    u    e      du,
 *  m        _/  0
 *
 * by a power series expansion
 *
 *                    _                    2                     3                _
 *           exp(-t) |       t            t                     t                  |
 * F (t)  =  ------- | 1 + ----- + --------------- + ----------------------- + ... |
 *  m          2 b   |_    b + 1   (b + 1) (b + 2)   (b + 1) (b + 2) (b + 3)      _|,
 *
 * where b = m + 1 / 2. This power series expansion converges fast, when t is less than b + 1,
 * namely t < m + 3 / 2.
 *
 * Argument(s)
 *
 * int m:
 * F_m(t), see the Description section.
 *
 * double t:
 * F_m(t), see the Description section.
 *
 * Return Value
 * double:
 * F_m(t), see the Description section.
 *
 */

/*
 * Name
 *
 * fmt
 *
 * Synopsis
 *
 * double fmt(int m, double t)
 *
 * Description
 *
 * This function evaluates the auxiliary integral, see Eq. 2.11 in THO,
 *
 *             _ 1           2
 *            /     2 m  -t u
 * F (t)  =   |    u    e      du,
 *  m        _/  0
 *
 * where m replaces ν for more convenient typesetting.
 *
 * If t is less than SML16 or equals 0, then
 *
 *              1
 * F (t)  =  -------.
 *  m        2 m + 1
 *
 * If t is less than m + 3 / 2, the auxiliary integral is evaluated by
 * a power series expansion (see fmtpse.c for details).
 *
 * Otherwise F (t) is calculated first
 *            0
 *                    1
 *                    -
 *           1 /  π  \2       _
 * F (t)  =  - | --- |  erf( /t ).
 *  0        2 \  t  /
 *
 * Then the upward recurrence relation is used for F (t) of higher m
 *                                                  m
 *
 *            (2 m - 1) F     (t) - exp( -t )
 *                       m - 1
 *  F (t)  =  -------------------------------.
 *   m                      2 t
 *
 * Argument(s)
 *
 * int m:
 * F_m(t), see the Description section.
 *
 * double t:
 * F_m(t), see the Description section.
 *
 * Return Value
 *
 * double:
 * F_m(t), see the Description section.
 *
 */

static void fmt_downward(double *f, double t, int m)
{
        int i;
        double b = m + 0.5;
        double bi;
        double e = .5 * exp(-t);
        double x = e;
        double s = e;
        double tol = SML_FLOAT64 * e;
        for (bi = b + 1.; x > tol; bi += 1.) {
                x *= t / bi;
                s += x;
        }
        double fval = s / b;
        f[m] = fval;
        for (i = m-1; i >= 0; i--) {
                b -= 1.;
                fval = (e + t * fval) / b;
                f[i] = fval;
        }
}

static void gamma_inc_like(double *f, double t, int m)
{
        if (t == 0) {
                int i;
                f[0] = 1.;
                for (i = 1; i <= m; i++) {
                        f[i] = 1./(2*i+1);
                }
        } else if (t < TURNOVER_POINT[m]) {
                fmt_downward(f, t, m);
        } else {
                int i;
                double tt = sqrt(t);
                double fval = SQRTPIE4 / tt * erf(tt);
                f[0] = fval;
                if (m > 0) {
                        double e = .5 * exp(-t);
                        double b = 1. / t;
                        double b1 = .5;
                        for (i = 1; i <= m; i++) {
                                fval = b * (b1 * fval - e);
                                f[i] = fval;
                                b1 += 1.;
                        }
                }
        }
}

static inline double _pow(double base, int exponent)
{
        int i;
        double result = 1;
        for (i = 1; i <= exponent; i <<= 1) {
                if (i & exponent) {
                        result *= base;
                }
                base *= base;
        }
        return result;
}

void eval_boys(double *boys, int l, double a, double fac, double *rpq)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        double r2 = rx*rx + ry*ry + rz*rz;
        double a2 = -2. * a;
        gamma_inc_like(boys, a*r2, l);
        int i;
        for (int i = 0; i <= l; i++) {
                fac *= a2;
                boys[i] *= fac;
        }
}

// \int_0^b u^(2m) exp(-t u^2) du
// = b \int_0^1 (bv)^(2m) exp(-tb^2 v^2) dv
// = b^(2m+1) \int_0^1 v^(2m) exp(-(tb^2)v^2) dv
// = b^(2m+1) F(m, tb^2)
void eval_boys_erf(double *boys, int l, double a, double fac, double bound, double *rpq)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        double r2 = rx*rx + ry*ry + rz*rz;
        double a2 = -2. * a;
        gamma_inc_like(boys, a*r2*bound*bound, l);
        boys[0] *= bound;
        double a2b2 = a2 * bound*bound;
        for (int i = 0; i <= l; i++) {
                fac *= a2b2;
                boys[i] *= fac;
        }
}

void eval_boys_erfc(double *boys, int l, double a, double fac, double bound, double *rpq)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        double r2 = rx*rx + ry*ry + rz*rz;
        double a2 = -2. * a;
        gamma_inc_like(boys, a*r2*bound*bound, l);
        boys[0] *= (1. - bound);
        double a2b2 = a2 * bound*bound;
        double a2_pow = fac;
        double a2b2_pow = fac;
        for (int i = 0; i <= l; i++) {
                a2_pow *= a2;
                a2b2_pow *= a2b2;
                boys[i] *= a2_pow - a2b2_pow;
        }
}
