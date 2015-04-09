#define ODE_DIM 4
#define ODE_AUX 0
#define ODE_NPAR 1
#define ODE_FUNC mmslipflight
#define ODE_DT 1e-3
#define ODE_MAX 1e5
#define ODE_MAX_EVENTS 1
#define ODE_SCAN_EVENTS mmslipflight_event_scan

/* this will be done by EVENT_FROM_SCAN in the integrator ... #define ODE_EVENT mmslipflight_event */

#define ODE_RTOL 1e-10
#define ODE_ATOL 1e-10

#include <math.h>

#define alpha 68.0*3.141597/180.

//static double sinalpha = sin(alpha);  /* sin 68deg -> stable for k=20k */
//static double cosalpha = cos(alpha);



void mmslipflight(unsigned n, double t, double *Y, double *p, double *f) {
  f[0] = Y[2];
  f[1] = Y[3];
  f[2] = 0.;
  f[3] =  - 9.81;
};


double mmslipflight_event_scan( double t, double *Y, int n, double *pars, double *res) {
    /* touchdown event: foot := hip - l0 * sin(alpha) = 0 */
  // *res = - (Y[1] - sinalpha); /* trigger is POSITIVE zero crossing */
  *res = - (Y[1] - pars[0]); /* trigger is POSITIVE zero crossing */
  return 1;
  

};


//double mmslipflight_event ( int evtId, double t, double *Y, int n, double *pars) {
//    /* touchdown event: foot := hip - l0 * sin(alpha) = 0 */
//  // *res = - (Y[1] - sinalpha); /* trigger is POSITIVE zero crossing */
//  return  - (Y[1] - pars[0]); /* trigger is POSITIVE zero crossing */
//  
//
//};

