#include <stdio.h>

/* A NOTE ON STYLE:

  This file is written in "Fortran Style" -- everything is done in global 
  variables under the assumption that the code runs as a singleton and we want
  the compiler's optimizer to have a go at aggresively inlining everything.
  
  You'd be amazed how much trouble you save when few parameters are passed
*/

#define ODE_DIM (sizeof(struct CTSLIP_state)/sizeof(double))
#define ODE_AUX (sizeof(struct CTSLIP_aux)/sizeof(double))
#define ODE_NPAR (sizeof(struct CTSLIP_param)/sizeof(double))
#define ODE_FUNC(n,t,s,p,d) ctslip_flow((n),(t),(s),(p),(d),NULL)  
#define ODE_AUXFUNC(n,t,s,p,a) ctslip_flow((n),(t),(s),(p),NULL,(a))
#define ODE_DT 1e-5
#define ODE_MAX 1e5
#define ODE_SCAN_EVENTS ctslip_scan_events
#define EPS 1e-20
//! #define ODE_ALWAYS_ABORT 1

struct CTSLIP_events {
  double crash;
  double plane;
  double apex;
  double land0;
  double land1;
  double lift0;
  double lift1;
};  /* ENDS: struct CTSLIP_events */

#define ODE_MAX_EVENTS (sizeof(struct CTSLIP_events)/sizeof(double))

#define NUM_LEGS 2
#define LEG_MASK ((1<<NUM_LEGS)-1)

struct CTSLIP_state {
  double com_x;
  double com_z;
  double com_vx;
  double com_vz;
  double leg_x [NUM_LEGS];
  double leg_z [NUM_LEGS];
  double clk;
}; /* ENDS: struct CTSLIP_state */

struct CTSLIP_aux {
  double ref_x [NUM_LEGS];
  double ref_dx [NUM_LEGS];
  double ref_z [NUM_LEGS];
  double ref_dz [NUM_LEGS];
  double ddt [NUM_LEGS];
  double ddl [NUM_LEGS];
  double fz [NUM_LEGS];
  double theta [NUM_LEGS];
  double dTheta [NUM_LEGS];
  double zOfs;
}; /* ENDS: struct CTSLIP_aux */

struct CTSLIP_param {
  double mass;
  double len0;
  double stK;
  double stMu;
  double stNu;
  double tqKth;
  double tqKxi;
  double tqOmRatio;
  double xi0;
  double tq0;
  double flKp;
  double flKd;
  double omega;
  double stHalfDuty;
  double stHalfSweep;
  double stOfs;
  double gravityX;
  double gravityZ;
  double xiUpdate;
  double xiLiftoff;
  double bumpW;
  double bumpZ0;
  double bumpZ1;
  double minZ;
  double maxZ;
  double co_com_x;
  double co_com_z;
  double co_com_vx;
  double co_com_vz;
  double co_clk;
  double co_value;
  double domain;
}; /* ENDS: struct CTSLIP_param */

static struct CTSLIP_state *s, *ds, dsTemp;
static struct CTSLIP_aux *x, auxTemp;
static struct CTSLIP_param *p;

#define PI 3.1415926
/* Phase offset as a function of leg number */
#define phaseOfs( k ) ((1&(int)k)? 0 : PI)
#define FORLEG( K ) for (K=0; K<NUM_LEGS; K++ )

/* Buehler clock -- leg angle theta as function of clock xi */
static void refFromPhase( double xi, double *theta, double *dTheta ) {
  xi -= floor((xi+PI)/(2*PI)) * (2*PI);
  
  double dc = p->stHalfDuty; /* Half Duty cycle (radians) */
  double sw = p->stHalfSweep; /* Half sweep angle (radians)*/
  /* If corrected clock is outside duty-cycle * PI --> swing */
  if (xi>dc || xi<-dc) {
    *dTheta = (PI-sw)/(PI-dc);
    if (xi>0) 
      *theta = sw + (*dTheta)*(xi-dc);
    else
      *theta = -sw + (*dTheta)*(xi+dc);
  } else { /* else --> stance */
    *dTheta = sw / dc;
    *theta = (*dTheta) * xi ;
  };
  *theta += p->stOfs;
  *theta -= floor((*theta+PI)/(2*PI)) * (2*PI);
}; /* ENDS: refFromPhase */

/* Reference trajectory for flight legs, relative to to COM frame
   This trajectory is *ONLY* a function of the leg angle and leg
   identity. The equations are derived with maxima by: 
 */
static void refLegFlight( int leg,  double xi ) 
{
  refFromPhase( xi, &x->theta[leg], &x->dTheta[leg] );
  double C = cos(x->theta[leg]);
  double S = sin(x->theta[leg]);
  
  x->ref_x[leg] = C * p->len0;
  x->ref_z[leg] = S * p->len0;
  x->ref_dx[leg] = -S * x->dTheta[leg] * p->len0;
  x->ref_dz[leg] = C * x->dTheta[leg] * p->len0;
}; /* ENDS: refLegFlight */

/** Compute the dynamics of a leg in flight */
static void flightLeg( int leg ) {
  /* Phase reference for leg */
  double phi = s->clk + phaseOfs(leg);
  
  /* Obtain reference trajectory */
  refLegFlight( leg, phi );
  
  /* x reference tracking - 1st order*/
  ds->leg_x[leg] = -(s->leg_x[leg] - x->ref_x[leg])*p->flKp+x->ref_dx[leg];
  
  /* z reference tracking - 1st order*/
  ds->leg_z[leg] = -(s->leg_z[leg] - x->ref_z[leg])*p->flKp+x->ref_dz[leg];

  /* zero out torque aux */
  x->ddt[leg] = 0;
  x->ddl[leg] = 0;
  x->fz[leg] = 0;
}; /* ENDS: flightLeg */

/** Model of axial force developed along a leg 
  \param l length of the leg
  \param dl rate of extension of the leg
  \return axial force */
static double legForceModel( double l, double dl, double l0 ) {
  /* Force along the leg -- bilinear spring damper model */
  return  - p->stK * ( l - l0 ) * (1 + p->stNu * dl ) - p->stMu * dl;
}; /* ENDS: legForceModel */

/** Model of ground shape 
  \param x x coordinate of query point
  \param z z coordinate of query point
  \return distance outside / inside ground surface */
static double groundLevel( double x, double z ) {
  return -z; 
}; /* ENDS: groundLevel */

/** Compute the forces on a stance leg */ 
static void stanceLeg( int leg ) {
  /* Phase reference for leg */
  double xi = s->clk + phaseOfs(leg);
  /* Torque */
  double th0;
  double dth0;
  /* Obtain reference trajectory */
  refLegFlight( leg, xi );
  th0 = x->theta[leg];
  dth0 = x->dTheta[leg];
  
  /* Reference leg length */
  double l0 = EPS+sqrt(x->ref_x[leg]*x->ref_x[leg]+x->ref_z[leg]*x->ref_z[leg]);

  /* Leg vector components */
  double lx = s->leg_x[leg];
  double lz = s->leg_z[leg];
  /* Leg length */
  double l = EPS+sqrt( lx*lx + lz*lz );
  /* Unit vector along the leg */
  double ul_x = lx / l;
  double ul_z = lz / l;
  /* Velocity along the leg */
  double dl = -ul_x * s->com_vx -ul_z * s->com_vz;
  /* Force along the leg -- spring damper model */
  double ddl = legForceModel( l, dl, l0 );
  /* Leg angle */
  double th = atan2( lz, lx );
  double ddt = p->tq0
             - p->tqKth * sin(th-th0);   /* Torsional spring around ref */

  /* If we are close enough to xi0 --> apply schedule */           
  if (cos(xi-p->xi0)>cos(PI/p->tqOmRatio))      
         ddt -= p->tqKxi * sin(p->tqOmRatio*(xi-p->xi0)); /* schedule */
             
  /* Emit aux data for ddl and ddt */
  x->ddl[leg] = ddl;
  x->ddt[leg] = ddt;
  
  /* Convert force and torque to accellerations */
  ddl /= p->mass;
  ddt /= (l * p->mass);

  /* compute force and "matrix multiply" by rotation */
  x->fz[leg]  = -ddl*ul_z -ddt*ul_x;
  ds->com_vx += -ddl*ul_x +ddt*ul_z;
  ds->com_vz += x->fz[leg]; 
  /* Leg length changes */
  ds->leg_x[leg] = -ds->com_x;
  ds->leg_z[leg] = -ds->com_z;  
}; /* ENDS: stanceLeg */

static void ctslip_flow( 
  unsigned n, double t, double *state, double *par, double *deriv, double *aux
  ) 
{ 
  s = (struct CTSLIP_state *)state;
  p = (struct CTSLIP_param *)par;
  int domain = (int)p->domain;
  int k;
  
  /* If we're in the integrator code --> dim should match state, aux is local */
  if (!deriv) {
    /* ds was null -- we use local temp storage to avoid conditionals
       in the critical sections  of the code */
    ds = &dsTemp;
  } else {
    ds = (struct CTSLIP_state *)deriv;
    if (n != sizeof(*s)/sizeof(double)) {
      fprintf(stderr,"ERROR: model dimension does not match CTSLIP\n" );
      assert(!"dimension");
      return;
    };
  };
  
  /* If we're collecting output --> dim should include aux, which is exported */
  if (!aux) {
    x = &auxTemp;
  } else {
    x = (struct CTSLIP_aux *) aux;
    if (n != ((sizeof(*s)+sizeof(*x))/sizeof(double))) {
      fprintf(stderr,"ERROR: model dimension does not match CTSLIP+AUX\n" );
      __asm__("int $3");
      return;
    };
  };
    
  for (k=0;k<ODE_DIM;k++)
    if (isnan(state[k])) {
      printf("Error --> Nan in state[%d]",k);
      __asm__("int $3"); 
    };    
  if (isnan(t)) { __asm__("int $3"); };
  
  /* Velocities propagate the positions */
  ds->com_x = s->com_vx; 
  ds->com_z = s->com_vz;
  /* Initial estimate of forces on COM is zero (we accumulate as we go) */
  ds->com_vx = p->gravityX;
  ds->com_vz = p->gravityZ;
  /* domain bits tell us which leg is in stance */
  FORLEG(k) {    
    if (domain & (1<<k)) {
      stanceLeg(k);
    } else {
      flightLeg(k);
    };
  };
  /* Clock progresses with constant phase rate */
  ds->clk = p->omega;
  
  if (deriv)
    for (k=0;k<ODE_DIM;k++)
      if (isnan(deriv[k])) {
        printf("Error --> Nan in state[%d]",k);
        __asm__("int $3"); 
      };    
}; /* ENDS: ctslip_flow */

double ctslip_scan_events( 
  double t, double *state, int n, double *pars, double *val
  )
{
  struct CTSLIP_state *s = (struct CTSLIP_state*)state;
  struct CTSLIP_param *p = (struct CTSLIP_param *)pars;
  struct CTSLIP_events *ev = (struct CTSLIP_events *)val;
  struct CTSLIP_state acc;
  int k;
  
  for (k=0; k<sizeof(*ev)/sizeof(double); k++) 
    val[k] = -1;
  
  /* Crash events are always active */
  alwaysCheckEvt(&ev->crash - (double*)ev);
  /* All states may have crashes */
  if (s->com_z>(p->maxZ+p->minZ)/2.0) {
    ev->crash = s->com_z-p->maxZ;
  } else {
    ev->crash = p->minZ - s->com_z;
  };
  
  /* general purpose hyperplane events */
  ev->plane = p->co_value + s->clk*p->co_clk      
    + s->com_z*p->co_com_z + s->com_vz*p->co_com_vz
    + s->com_x*p->co_com_x + s->com_vx*p->co_com_vx;
  
  /*
  static double evp = 1;
  if (evp<0 && ev->plane>0) {
    printf("<!!> Event: %g value %g\n",t,ev->plane);
  };
  evp = ev->plane;
  */
  
  switch ((int)p->domain) {
#warning " ignore liftoff events if reference would not lift off and truncate negative ddl-s"
  case 0: /* Fly-down state */
    ev->land0 = groundLevel( 
      s->com_x + s->leg_x[0], s->com_z + s->leg_z[0] );
    ev->land1 = groundLevel( 
      s->com_x + s->leg_x[1], s->com_z + s->leg_z[1] );
    break;
    
  case 1: /* leg0 stance */
    ev->land1 = groundLevel( 
      s->com_x + s->leg_x[1], s->com_z + s->leg_z[1] );
    /* If motion is up --> compute forces for this state to detect liftoff */
    ctslip_flow( n, t, state, pars, (double*)&acc, NULL );    
    ev->lift0 = -x->fz[0];
    ev->apex = -s->com_vz;
    break;
    
  case 2: /* leg1 stance */
    ev->land0 = groundLevel( 
      s->com_x + s->leg_x[0], s->com_z + s->leg_z[0] );
    /* If motion is up --> compute forces for this state to detect liftoff */
    ctslip_flow( n, t, state, pars, (double*)&acc, NULL );    
    ev->lift1 = -x->fz[1];
    ev->apex = -s->com_vz;
    break;
    
  case 3: /* Double stance */
    /* If motion is up --> compute forces for this state to detect liftoff */
    ctslip_flow( n, t, state, pars, (double*)&acc, NULL );    
    ev->lift0 = -x->fz[0];
    ev->lift1 = -x->fz[1];
    ev->apex = -s->com_vz;
    break;
    
  case 4: /* Fly up state */
    ev->apex = -s->com_vz;
    break;
    
  default:
    assert(!"Domain number is invalid");
  }; /* ENDS: switch on domain */
  //!!!printf("!!! %6e %6p Events %7e %7e %7e %7e %7e %7e\n", t,state,val[0],val[1],val[2],val[3],val[4],val[5] );
}; /* ENDS: ctslip_events */


#ifdef MAIN

#include <malloc.h>
#include <stdio.h>

#define NMAX 10000

#define LL 0.17

#define ODE_WIDTH (ODE_DIM+ODE_AUX+1)

int main( int argc, char **argv ) {
  double *res = (double*)calloc(NMAX,ODE_WIDTH*sizeof(double));
  struct CTSLIP_events *evp = (struct CTSLIP_events *)malloc(sizeof(*evp));
  struct CTSLIP_param *pp = (struct CTSLIP_param *)malloc(sizeof(*pp));
  struct CTSLIP_param p0 = {
    mass : 7.0, 
    len0 : LL,
    stK : 3.75*4500,
    stMu : 0,
    stNu : 0,
    tqKth : 0,     
    tqKxi : 0,
    xi0 : 0,
    flKp : 10000,
    flKd : 1,
    flX0 : 0,         
    flZ0 : 0,
    omega : -39.6170776, 
    stHalfDuty : PI*0.2,
    stHalfSweep : 0,   
    stOfs : -2*PI/5,
    gravityX : 0,
    gravityZ : -9.81,
    bumpW : 0,
    bumpZ0 : 0,
    bumpZ1 : 0,
    domain : 0
  };
  struct CTSLIP_state s0 = {
    com_x : 0,
    com_z : 0.2,
    com_vx : 2.4,
    com_vz : -1e-5,
    leg_x : { 0.6 * LL, -0.6 * LL},
    leg_z : { 0.8 * LL, -0.8 * LL},
    clk : 0
  };
  /* Initial state leaves space for timestamp inline first position */
  struct CTSLIP_state *sp = (struct CTSLIP_state *)(res+1);
  /* Initial aux is rigth after that */
  struct CTSLIP_aux *a0 = (struct CTSLIP_aux *)(sp+1);
  double t0 = 0;
  double t1 = 1000;  
  long i,k,n;
  
  memcpy( pp, &p0, sizeof(p0) );
  memcpy( sp, &s0, sizeof(s0) );
  n = odeOnce( res, NMAX, ODE_WIDTH, 
      0,  t1, 1.0, 
      (double*)pp, (sizeof(*pp)/sizeof(double)) );
  n = odeOnce( res, NMAX, ODE_WIDTH, 
      0,  t1, 1.0, 
      (double*)pp, (sizeof(*pp)/sizeof(double)) );
  for (i=0;i<n;i++) { 
    for (k=0;k<ODE_WIDTH;k++)
      fprintf(stdout,"%10g ",res[(ODE_DIM+1)*i+k] );
    fprintf(stdout,"\n");
  };
  return 0;
};

#endif /* MAIN */

/* jEdit :mode=c:indentSize=2:tabSize=2:folding=indent:wrap=none: */
