#define ODE_DIM 4
#define ODE_AUX 0
#define ODE_NPAR 1
#define ODE_FUNC mmslipstance
#define ODE_DT 1e-3
#define ODE_MAX 1e5
#define ODE_MAX_EVENTS 1
#define ODE_SCAN_EVENTS mmslipstance_event_scan

#define ODE_RTOL 1e-10
#define ODE_ATOL 1e-10

#include <math.h>
#include <stdio.h>

#define alpha 68.0*3.141597/180.

static double sinalpha = sin(alpha);  /* sin 68deg -> stable for k=20k */
static double cosalpha = cos(alpha);


void mmslipstance(unsigned n, double t, double *Y, double *p, double *f) {
  double l, F, Fx, Fy;
  double xr = Y[0] -  p[0]; // x relative to foot
  l = sqrt(xr*xr + Y[1] * Y[1]);
  F = 20000. * (1. - l);
  Fx = xr / l * F;
  Fy = Y[1] / l * F;
  f[0] = Y[2];
  f[1] = Y[3];
  f[2] = Fx / 80.;
  f[3] = Fy / 80. - 9.81;
};

double mmslipstance_event_scan( double t, double *Y, int n, double *pars, double *res) {

    /* output is actually ignored -> could be set to void (?) */
  double l;
  double xr = Y[0] -  pars[0]; // x relative to foot
  l = sqrt(xr*xr + Y[1] * Y[1]);
  *res = l - 1.;
//printf("res: %2.5f\n", *res);

};

