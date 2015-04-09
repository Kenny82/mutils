#define ODE_NO_EVENT ~0

int odeEventId(
  double *y0, 
  int y0_dim
  );

void solout(
  long nr, 
  double xold, 
  double x, 
  double* y, 
  unsigned n, 
  int* irtrn
  );

int odeComputeAux(
  double *y0,
  int y0_dim,
  double *pars,
  int pars_dim
  );

long odeOnce(
  double *yOut,
  int nmax,
  int yOut_dim,
  int startRow,
  double tEnd,
  double dt,
  double *pars,
  int pars_dim
  );

/* jEdit :mode=c:indentSize=2:tabSize=2:folding=indent:wrap=hard: */
