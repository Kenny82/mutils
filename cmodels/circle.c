#define ODE_DIM 2
#define ODE_AUX 0
#define ODE_NPAR 1
#define ODE_FUNC circle
#define ODE_DT 1e-3
#define ODE_MAX 1e5
#define ODE_MAX_EVENTS 1
#define ODE_SCAN_EVENTS circle_event_scan

void circle(unsigned n, double t, double *Y, double *p, double *f) {
  f[0] = -Y[1];
  f[1] = Y[0];
};

double circle_event_scan( double t, double *Y, int n, double *pars, double *res) {
  *res = Y[0]+ pars[0]; //0.1;
};

/* jEdit :mode=c:indentSize=2:tabSize=2:folding=indent:wrap=hard: */
