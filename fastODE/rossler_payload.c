#define ODE_DIM 3
#define ODE_AUX 0
#define ODE_NPAR 3
#define ODE_FUNC rossler
#define ODE_DT 1e-3
#define ODE_MAX 1e5
#define ODE_EVENT rossler_event
#define ODE_MAX_EVENTS 1
#define ODE_SCAN_EVENTS rossler_event_scan

void rossler(unsigned n, double t, double *Y, double *p, double *f) {
  f[0] = -Y[1]-Y[2];
  f[1] = Y[0] + p[0]*Y[1];
  f[2] = p[1] + Y[2]*(Y[0] - p[2]);
};

double rossler_event( int evtId, double t, double *Y, int n, double *pars) {
  return t - pars[3];
};

double rossler_event_scan( double t, double *Y, int n, double *pars, double *res) {
  *res = (t - pars[3]);
};

#ifdef MAIN

#include <malloc.h>
#include <stdio.h>

#define NMAX 1000000

int main( int argc, char **argv ) {
  double par[] = { 0.2,  0.2,  5.7, 1000.0 };
  double Y0[] = {0, 5, -5, 2};
  double t0 = 0;
  double t1 = 1000;  
  double *res = (double*)malloc(NMAX*(ODE_DIM+1)*sizeof(double));
  long i,k,n;
  
  if (!res)
    return -fprintf(stderr,"Couldn't allocate memory for result\n");
  
  n = odeOnce( Y0, ODE_DIM+1, res, NMAX, ODE_DIM+1, 0,  t1, 1.0, par, 4 );
  for (i=0;i<n;i++) { 
    for (k=0;k<(ODE_DIM+1);k++)
      fprintf(stdout,"%10g ",res[(ODE_DIM+1)*i+k] );
    fprintf(stdout,"\n");
  };
  return 0;
};

#endif /* MAIN */

/* jEdit :mode=c:indentSize=2:tabSize=2:folding=indent:wrap=hard: */
