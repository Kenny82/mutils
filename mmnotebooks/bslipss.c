#define ODE_DIM 6
#define ODE_AUX 0
#define ODE_NPAR 17
#define ODE_FUNC bslipss
#define ODE_DT 1e-3
#define ODE_MAX 1e5
#define ODE_MAX_EVENTS 1
#define ODE_SCAN_EVENTS bslipss_event_scan

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

void bslipss(unsigned n, double t, double *Y, double *p, double *f) {
  double l, F, Fx, Fy, Fz;
  double xr, zr;
  xr = Y[0] - p[P_f1x]; // x relative to foot
  zr = Y[2] - p[P_f1z];

  l = sqrt(xr*xr + Y[1] * Y[1] + zr*zr);
  if (p[P_lleg] == 1.)
  {
      F = - p[P_k1] * (l - p[P_l01]);
      if (F < 0.){ F=0.;}
  }
  else
  {
      F = - p[P_k2] * (l - p[P_l02]);
      if (F < 0.){ F=0.;}
  }
  Fx = xr / l * F;
  Fy = Y[1] / l * F;
  Fz = zr / l * F;
  f[0] = Y[3];
  f[1] = Y[4];
  f[2] = Y[5];
  f[3] = Fx / p[P_m];
  f[4] = Fy / p[P_m] + p[P_g];
  f[5] = Fz / p[P_m]; 
};

double bslipss_event_scan( double t, double *Y, int n, double *pars, double *res) {
/* event is touchdown condition. Failure will be checked in python */
    /* output is actually ignored -> could be set to void (?) */
  if (pars[P_lleg] == 1.) // leading leg is leg1 -> scan for touchdown of leg2
  {
      *res = -(Y[1] - pars[P_l02] * sin(pars[P_a2])); 
  }
  else
  {
      *res = -(Y[1] - pars[P_l01] * sin(pars[P_a1])); 
  }

};

