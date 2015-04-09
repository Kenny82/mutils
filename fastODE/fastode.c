#include <assert.h>
#include <math.h>
#include "fastode.h"

/* Forward declaration for ode function wrapper */
inline static void safeOdeFunc( 
  unsigned n, double t, double *state, double *par, double *deriv
);

inline static void alwaysCheckEvt( int evid );

#include "ode_payload.c" /* source include */
#include "integro.c"  /* source include */

static double *traj;
static long trajLen;
static long trajMaxLen;
static double *params;

#ifndef ODE_ATOL
#define ODE_ATOL 1e-9
#endif

#ifndef ODE_RTOL
#define ODE_RTOL 1e-6
#endif

#ifndef ODE_EVTTOL
#define ODE_EVTTOL 1e-12
#endif

#define ODE_WIDTH (ODE_DIM+ODE_AUX+1)

#ifdef ODE_ALWAYS_ABORT
#define OOPS abort()
#else 
#define OOPS
#endif

/* ODE_FUNC -- computes flow, i.e. the ODE function itself */
/* ODE_AUXFUNC -- computes extra values associated with a state (optional) */
/* ODE_SCAN_EVENTS -- compute all event functions with positive crossings for events */
/* ODE_EVENT -- compute sigle event function for precise detection */
/* ODE_MAX_EVENTS -- number of event types used */

const int ode_max_events = ODE_MAX_EVENTS;
const int ode_dim = ODE_DIM;
const int ode_width = ODE_WIDTH;
const int ode_aux = ODE_AUX;
const int ode_npar = ODE_NPAR;
double ode_atol = ODE_ATOL;
double ode_rtol = ODE_RTOL;
double ode_evttol = ODE_EVTTOL;
double ode_max = ODE_MAX;

double eventValue [ODE_MAX_EVENTS];
int eventActive [ODE_MAX_EVENTS];

inline static void alwaysCheckEvt( int evid ) {
  assert( evid>=0 && evid<ODE_MAX_EVENTS );
  eventActive[evid] = ~0;
}; 


static int currentEvent = ODE_NO_EVENT;

#ifndef isfinite
inline static int isfinite( double x ) { return (x == x) && (2*x != x); };
#else
#warning "builtin isfinite macro found"
#endif

inline static void safeOdeFunc( 
  unsigned n, double t, double *state, double *par, double *deriv
  ) 
{
  int k;
  const char *msg = NULL;
  do {
    if ( n != ODE_DIM ) { 
      msg = "Dimension %d does not match code"; 
      k = n; 
      goto fail;
    };
    if (0!=t && !isfinite(t)) {
      msg = "t is not finite or %d";
      k = 0;
      goto fail;
    };
    for (k=0;k<ODE_DIM;k++)
      if (0!=state[k] && !isfinite(state[k])) {
        msg = "Nan in state[%d]";
        goto fail;
      };
    ODE_FUNC( n, t, state, par, deriv );
    for (k=0;k<ODE_DIM;k++)
      if (0!=deriv[k] && !isfinite(deriv[k])) {
        msg = "Nan in deriv[%d]";
        goto fail;
      };
    return;
    /* never reach this point */
  } while(0);
fail:
  fprintf(stderr,msg,k);
  __asm__("int $3");
}; /* ENDS: safeOdeFunc */

/** Return the ID number of the event that terminated integration 
    and all event detector function results.
    
    NOTE: after the first positive event, no futher functions are called
      so their values will be out of date
*/
int odeEventId(double *y0, int y0_dim) {
  int i;
  if (y0 && y0_dim>0) {
    if (y0_dim>ODE_MAX_EVENTS)
      y0_dim=ODE_MAX_EVENTS;
    for (i=0; i<y0_dim; i++) {
      y0[i] = eventValue[i];
    };
  };
  return currentEvent;
};

#ifndef ODE_MAX_EVENTS
#error "Must define ODE_MAX_EVENTS to the number of events"
#endif

/* ODE_EVENT( int evtId, double x, double *y, int n, double *p )
   Should compute crossover event function for event ID evtId.
   If there's no optimized version --> we run a full event scane, and
   return only the relevant result. 
   Code uses scan function for binary search event detection */
#ifndef ODE_EVENT
#define ODE_EVENT ode_event_from_scan
static double ode_event_from_scan(
  int evtId, double x, double *y, int n, double *p
  ) 
{
  ODE_SCAN_EVENTS( x, y, n, p, eventValue );
  return eventValue[evtId];
};
#endif /* ODE_EVENT */

static double multiEventWrapper( double x, double *y, int n, double *p ) {
  int k;
  double evt = -ODE_EVTTOL;
  double cand;

  ODE_SCAN_EVENTS( x, y, n, p, eventValue );
  for (k=0; k<ODE_MAX_EVENTS; k++) {
    cand = eventValue[k];
    if (cand>ODE_EVTTOL) { /* possible event */ 
      if (eventActive[k]) { /* verify crossing direction */
        evt = cand; 
        currentEvent = k; 
        break;
      };
    } else if (cand<-ODE_EVTTOL) { /* on pre-event side of boundary --> activate it */
      eventActive[k] = ~0;
      /* If cand is closer to threshold --> use it */
      if (cand>evt) { 
        evt = cand; 
      };
    };
  }; /* ENDS: loop to find events */
  
  return evt;
}; /* ENDS: multiEventWrapper */

int findEvent( double *csr, double e1 ){
  double e0, *y0, *y1, x0, x1, ec, xc;
  int k;

  assert( currentEvent>=0 && currentEvent<ODE_MAX_EVENTS && "event was detected" );
  
  if (trajLen<2) {
    fprintf(stderr,"WARNING: event called on trajectory with less than two points\n");
    OOPS;
    return -1;
  };

  /* last two entries in output */
  y1 = csr;
  y0 = csr - ODE_WIDTH;
  
  /* split into x and y parts */
  x1 = *y1; 
  y1++;
  x0 = *y0; 
  y0++;

  /* The event function has value e0 at x0 and e1 at x1
     For an event to occur, e0 < 0 <= e1, and we search the interval [x0 x1]
     for the root of the event function */     
  if (e1<0) {
    fprintf(stderr,"WARNING: event hasn't occured yet\n");
    OOPS;
    return -1;
  };
  
  e0 = ODE_EVENT( currentEvent, x0, y0, ODE_DIM, params );
  /* Make sure that event didn't occur *before* x0 */
  if (e0>=0) {
    fprintf(stderr,"WARNING: event %d happened *before* previous time step\n"
      , currentEvent);
    OOPS;
    /* memcpy(y1,y0,n*sizeof(double)); */
    return -2;
  };
  
  //!!!printf("Refining %d...\n", currentEvent);//!!!
  
  /* Loop until we found a "point" */
  int iter = 0;
  while (fabs(x0-x1)>ode_evttol || fabs(ec)>ode_atol) {
    /* Sanity: break-out after 64 rounds*/
    if (iter++>64) 
      break;
    /* Go to midpoint */
    xc = (x0+x1)/2;
    
    /* Interpolate trajectory at midpoint */
    for (k=0; k<ODE_DIM; k++)
      y1[k] = contd5(k,xc);
    
    /* Evaluate event function */
    ec = ODE_EVENT( currentEvent, xc, y1, ODE_DIM, params );
    
    /* Do binary search */
    if (ec==0) 
      break;
    if (ec<0)
      x0 = xc;
    else
      x1 = xc;
  };/* end of binary search loop */
  
  /* store event value */
  eventValue[currentEvent] = ec;
    
  /* time is at *csr; y1 was incremented */
  *csr = xc;

  //!!!  printf("Current event: %d time %g value %g iter %d\n", currentEvent, xc, ec, iter ); 
  
  return 0;
}; /* ENDS: findEvent */ 

void solout(long nr, double xold, double x, double* y, unsigned n, int* irtrn) {
  double *csr;
  double evt;
  unsigned long mask = ~0;
  
  if (nr>=trajMaxLen) { /* If out of storage for output --> bail out */
    *irtrn = -1;
    return;
  };
  
  /* Store the new time and value */
  if (nr != trajLen+1 && nr != 1) OOPS;
  trajLen = nr;
  csr = &traj[ODE_WIDTH*(nr-1)];
  *csr = x;
  memcpy( csr+1, y, sizeof(double)*n );

#ifdef ODE_AUXFUNC
  ODE_AUXFUNC( (ODE_DIM+ODE_AUX), *csr, csr+1, params, csr+1+ODE_DIM );
#endif /* ENDS: ODE_AUXFUNC */

#ifdef ODE_EVENT    
  /* Compute value of event function */
  evt = multiEventWrapper( *csr, csr+1, n, params ); 
  
  /* If positive --> an event occurred; search for it */
  if (evt>0) {
    int evtId = findEvent( csr, evt );
    /* event was found --> abort integration */
    *irtrn = -1;
    if (evtId<0)
      currentEvent = evtId;
  };
#endif /* ENDS: ODE_EVENT */
}; /* ENDS: solout */ 

int odeComputeAux(
  double *y0,
  int y0_dim,
  double *pars,
  int pars_dim
  )
{
  if (y0_dim != ODE_WIDTH) {
    fprintf(stderr,"ERROR: Output storage for dimension %d instead of %d\n",
        y0_dim, (int)ODE_WIDTH );
    OOPS;
    return -2;
  };
  
#ifdef ODE_NPAR
  if (pars_dim != ODE_NPAR) {
    fprintf(stderr,"ERROR: Parameters dimension %d instead of %d\n",
        pars_dim, (int)ODE_NPAR );
    OOPS;
    return -3;
  };
#endif /* ENDS: ODE_NPAR */
#ifdef ODE_AUXFUNC
  ODE_AUXFUNC( (ODE_DIM+ODE_AUX), *y0, y0+1, pars, y0+1+ODE_DIM );
#endif /* ENDS: ODE_AUXFUNC */
  return 0;
}; /* ENDS: odeComputeAux */

long odeOnce(
  double *yOut,
  int nmax,
  int yOut_dim,
  int startRow,
  double tEnd,
  double dt,
  double *pars,
  int pars_dim
  )
{
  static double y0 [ ODE_WIDTH ];
  
  currentEvent = ODE_NO_EVENT;
  
  if (yOut_dim != ODE_WIDTH) {
    fprintf(stderr,"ERROR: Output storage for dimension %d instead of %d\n",
        yOut_dim, (int)ODE_WIDTH );
    OOPS;
    return -2;
  };
  
#ifdef ODE_NPAR
  if (pars_dim != ODE_NPAR) {
    fprintf(stderr,"ERROR: Parameters dimension %d instead of %d\n",
        pars_dim, (int)ODE_NPAR );
    OOPS;
    return -3;
  };
#endif /* ENDS: ODE_NPAR */

  if (nmax<=startRow+1) {
    fprintf(stderr,"ERROR: starting beyond end of the array\n");
    OOPS;
    return -4;
  };
  
  if (pars_dim==0)
    pars=NULL;

  /* Event detector should have access to params too */
  params = pars;

  /* Output location. Contains initial condition */
  traj = yOut + ODE_WIDTH*startRow;
  
  /* Disable all eventss until we see a point at the "right" side */
  int k;
  for (k=0; k<ODE_MAX_EVENTS; k++) {
    eventActive[k] = 0;
  };
  
  /* Compute value of event function 
  if (multiEventWrapper( *traj,traj+1, ODE_DIM, params )>=0) {
    fprintf(stderr,"WARNING: Event occured before initial point was computed.\n" );
    OOPS;
    return -5;
  }; */
  
#ifdef ODE_AUXFUNC
  /* Compute auxiliary values for initial condition */
  ODE_AUXFUNC( (ODE_DIM+ODE_AUX), *traj, traj+1, params, traj+1+ODE_DIM );
#endif /* ENDS: ODE_AUXFUNC */

 /* We already have one point (ICS); start output at pos 1 */
  trajLen = 1;
  trajMaxLen = nmax-startRow;
  
  double atol = ODE_ATOL;
  double rtol = ODE_RTOL; 
  int rc;
  
  /* Poison output array for downstream testing */
  //!!! memset( traj+ODE_WIDTH, 0xee, (trajMaxLen-1)*ODE_WIDTH );
  /* Copy initial condition to temp storage b/c dopri code abuses it */
  memcpy( y0, traj, (1+ODE_DIM) * sizeof(double));
  //!!! printf("Integration start\n");
  rc = dopri5(
    ODE_DIM,  /*  unsigned n,       dimension of the system <= UINT_MAX-1*/
    *y0,      /*  double x,         initial x-value */
    y0+1,     /*  double* y,        initial values for y */
    pars,     /*  double *pars,     parameters */
    tEnd,     /*  double xend,      final x-value (xend-x may be positive or negative) */
    &ode_rtol,/*  double* rtoler,   relative error tolerance */
    &ode_atol,/*  double* atoler,   absolute error tolerance */
    0,        /*  int itoler,       switch for rtoler and atoler: SCALARS */
    2,        /*  int iout,         switch for calling solout: DENSE */
    stdout,   /*  FILE* fileout,    messages stream */
    2.3E-16,  /*  double uround,    rounding unit */
    0.9,      /*  double safe,      safety factor */
    0.2,      /*  double fac1,      parameters for step size selection (lower ratio) */
    10.0,     /*  double fac2,      (upper ratio) */
    0.04,     /*  double beta,      for stabilized step size control */
    dt,       /*  double hmax,      maximal step size */
    dt/2,     /*  double h,         initial step size */
    nmax,     /*  long nmax,        maximal number of allowed steps */
    ode_max,  /*  double magboud,   bound on magnitude of solution components */
    1,        /*  int meth,         switch for the choice of the coefficients */
    -1,       /*  long nstiff,      test for stiffness: NEVER */
    ODE_DIM,  /*  unsigned nrdens,  number of components for which dense outpout is required */
    NULL,     /*  unsigned* icont,  indexes of components for which dense output is required, >= nrdens */
    0         /*  unsigned licont   declared length of icont */
   );
  //!!!printf("Integration ended with %d samples\n", trajLen);
  
  return trajLen-1;
}; /* ENDS: odeOnce */


/* jEdit :mode=c:indentSize=2:tabSize=2:folding=indent:wrap=hard: */
