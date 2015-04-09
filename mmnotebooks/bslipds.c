#define ODE_DIM 6
#define ODE_AUX 0
#define ODE_NPAR 17
#define ODE_FUNC bslipds
#define ODE_DT 1e-3
#define ODE_MAX 1e5
#define ODE_MAX_EVENTS 1
#define ODE_SCAN_EVENTS bslipds_event_scan

#define ODE_RTOL 1e-11
#define ODE_ATOL 1e-11

#include <math.h>
#include <stdio.h>

/* define constants for indices of parameter vector */
#define P_k1 0
#define P_k2 1
#define P_a1 2
#define P_a2 3
#define P_l01 4
#define P_l02 5
#define P_b1 6
#define P_b2 7
#define P_m 8
#define P_g 9
#define P_f1x 10
#define P_f1y 11
#define P_f1z 12
#define P_f2x 13
#define P_f2y 14
#define P_f2z 15
#define P_lleg 16

void bslipds(unsigned n, double t, double *Y, double *p, double *f) {
  double l1, l2, F1, Fx1, Fy1, Fz1, Fx2, Fy2, Fz2, F2;
  double xr1, zr1, xr2, zr2;
  // force of leading leg
  xr1 = Y[0] - p[P_f1x];
  zr1 = Y[2] - p[P_f1z];
  l1 = sqrt(xr1*xr1 + Y[1] * Y[1] + zr1*zr1);
  if (p[P_lleg] == 1.) // leading leg is leg1?
  {F1 = - p[P_k1] * (l1 - p[P_l01]);}
  else {F1 = -p[P_k2] * (l1 - p[P_l02]); }
  if (F1 < 0.){ F1=0.;}

  // force of trailing leg
  xr2 = Y[0] - p[P_f2x];
  zr2 = Y[2] - p[P_f2z];
  l2 = sqrt(xr2*xr2 + Y[1] * Y[1] + zr2*zr2);
  if (p[P_lleg] == 1.) // leading leg is leg1 -> trailing leg is leg2?
  { F2 = - p[P_k2] * (l2 - p[P_l02]);}
  else
  { F2 = -p[P_k1] * (l2 - p[P_l01]);}
  if (F2 < 0.){ F2=0.;}

  Fx1 = xr1 / l1 * F1;
  Fy1 = Y[1] / l1 * F1;
  Fz1 =  zr1 / l1 * F1;

  Fx2 = xr2 / l2 * F2;
  Fy2 = Y[1] / l2 * F2;
  Fz2 =  zr2 / l2 * F2;

  f[0] = Y[3];
  f[1] = Y[4];
  f[2] = Y[5];
  f[3] = (Fx1 + Fx2) / p[P_m];
  f[4] = (Fy1 + Fy2) / p[P_m] + p[P_g];
  f[5] = (Fz1 + Fz2) / p[P_m];
};

double bslipds_event_scan( double t, double *Y, int n, double *pars, double *res) {
/* event is takeoff condition of trailing leg. Failure will be checked in python */
    /* output is actually ignored -> could be set to void (?) */
  double l; 
  double xr, zr;
  xr = Y[0] - pars[P_f2x];
  zr = Y[2] - pars[P_f2z];
  l = sqrt(xr*xr + Y[1] * Y[1] + zr*zr);
  if (pars[P_lleg] == 1.) // leading leg is leg1 -> scan for TO of leg 2
  {
    *res = l - pars[P_l02];
  }
  else
  {
    *res = l - pars[P_l01];
  }
};

