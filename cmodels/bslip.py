from libshai import integro
from pylab import (norm, pi, hstack, vstack, array, sign, sin, cos, arctan2,
        sqrt, zeros, 
        figure, subplot, plot, legend, xlabel, ylabel)
from numpy import float64
from copy import deepcopy
import mutils.io as mio

import fastode # local!

class SimulationError(Exception):
    pass

class BSLIP(mio.saveable):
    """ Class of the bipedal walking SLIP """
    
    def __init__(self, params=None, IC=None):
        """ 
        The BSLIP is a bipedal walking SLIP model.

        params (mutils.misc.Struct): parameter of the model
        IC (array): initial conditions. [x, y, z, vx, vy, vz] 
            *NOTE* the system starts in single stance and *must* have
            positive vertical velocity ("vy > 0")

        """
        super(BSLIP, self).__init__()
        self.params = deepcopy(params)
        self.state = deepcopy(IC)
        self.odd_step = True # leg 1 or leg two on ground?
        self.dt = .01
        

        self.odess = fastode.FastODE('bslipss')
        self.odeds = fastode.FastODE('bslipds')
        self.buf = zeros((2000, self.odess.WIDTH), dtype=float64)

        self.t = 0
        self.t_td = 0
        self.t_to = 0
        self.singleStance = True
        self.failed = False
        self.skip_forces = False
        self.ErrMsg = ""
        
        # storage for ode solutions
        self.feet1_seq = []
        self.feet2_seq = []
        self.t_ss_seq = []
        self.t_ds_seq = []
        self.y_ss_seq = []
        self.y_ds_seq = []
        self.forces_ss_seq = []
        self.forces_ds_seq = []
        self.DEBUG = False
        if self.params is not None:
            self.feet1_seq.append(self.params['foot1'])
            self.feet2_seq.append(self.params['foot2'])
        
    def init_ode(self):
        """ re-initialize the ODE solver """
        self.ode = integro.odeDP5(self.dy_Stance, pars=self.params)
        self.ode.ODE_RTOL = 1e-9
    
    def restore(self, filename):
        """
        update the restore procedure: re-initialize the ODE solver!
        
        :args:
            filename (str): the filename where the model information is stored
        """
        super(BSLIP, self).restore(filename)
        self.ode = integro.odeDP5(self.dy_Stance, pars=self.params)
        self.ode.ODE_RTOL = 1e-9
    
    def legfunc1(self, t, y, pars):
        """
        Force (scalar) function of leg 1: Here, spring function

        :args:
            t (float): time (ignored)
            y (6x float): CoM state [position, velocity]
            pars (dict): parameters of the model. Must include 
                'foot1' (3x float) foot1 position
                'lp1' (4x float) parameters of leg 1

        :returns:
            f (float): the axial leg force  ["f = k * (l - l0)"]

        NOTE: Overwrite this function to get different models.
              The signature must not change.
        """
        #DEBUG:
        #print 'pf1: ', pars['foot1']
        l1 = norm(array(y[:3]) - array(pars['foot1']))
        return -pars['lp1'][0] * (l1 - pars['lp1'][1])

    def legfunc2(self, t, y, pars):
        """
        leg function of leg 2: a spring function
        
        :args:
            t (float): time (ignored)
            y (6x float): CoM state [position, velocity]
            pars (dict): parameters of the model. Must include 
                'foot1' (3x float) foot1 position
                'lp1' (4x float) parameters of leg 1

        :returns:
            f (float): the axial leg force  ["f = k * (l - l0)"]

        NOTE: Overwrite this function to get different models.
              The signature must not change.
        """
        l2 = norm(array(y[:3]) - array(pars['foot2']))
        return -pars['lp2'][0] * (l2 - pars['lp2'][1])    
        
    def evt_vy0(self, t, states, traj, p):
        """
        triggers the  vy=0 event

        :args:
            t (2x float): list of time prior to and after event
            states (2x array): list of states prior to and after event
            traj (trajectory patch): a trajectory patch (ignored here)

        :returns:
            (bool) vy=0 detected? (both directions)
        """
        return sign(states[0][4]) * sign(states[1][4]) != 1
    
    def update_params_ss(self):
        """ 
        Updates the model parameters in the single stance vy=0 event.
        Here, this function does nothing.
        Overwrite it in derived models to enable e.g. control actions.
        """

        pass

    def update_params_ds(self):
        """ 
        Updates the model parameters in the double stance vy=0 event.
        Here, this function does nothing.
        Overwrite it in derived models to enable e.g. control actions.
        """
        
        pass
    
    def update_params_td(self):
        """
        Updates the model parameters at touchdown events.
        Here, this function does nothing.
        Overwrite it in derived models to enable e.g. control actions.
        """
        pass
    
    def update_params_to(self):
        """
        Updates the model parameters at takeoff events.
        Here, this function does nothing.
        Overwrite it in derived models to enable e.g. control actions.
        """
        pass
    
    def takeoff_event(self, t, states, traj, pars, legfun):
        """ 
        triggers the take off of a leg 
        Hint: use a lambda function to adapt the call signature

        This function is force-triggered. The parameter format (pars) must
        be the same as for legfun (which is called from here!)

        *NOTE* you can overwrite this method for derived models. However, 
        this is not required if the takeoff condition is "zero force".

        :args:
            t (2x float): list of time prior to and after event
            states (2x array): list of states prior to and after event
            traj (trajectory patch): a trajectory patch (ignored here)
            pars (<any>): the leg functions parameters
            legfun (function of (t, y, pars) ): the leg force function.
        
        :returns:
            (bool) takeoff detected? (force has falling zero crossing)
        """

        F0 = legfun(t[0], states[0], pars)
        F1 = legfun(t[1], states[1], pars)

        return F0 > 0 and F1 <= 0
        
    def touchdown_event(self, t, states, traj, pars):
        """
        triggers the touchdown of the leading leg.
        Hint: use a lambda function to adapt the call signature

        :args:
            t (2x float): list of time prior to and after event
            states (2x array): list of states prior to and after event
            traj (trajectory patch): a trajectory patch (ignored here)
            pars (4x float): the leg functions parameters. Format:
                [l0, alpha, beta, floorlevel]            

        pars format:
            [l0, alpha, beta, floorlevel]
        
        :returns:
            (bool) takeoff detected? (force has falling zero crossing)
        
        """
        def zfoot(state, pars):
            foot = state[1] - pars[0] * sin(pars[1])
            return foot - pars[3]
        return zfoot(states[0], pars) > 0 and zfoot(states[1], pars) <= 0
    
    def touchdown_event_refine(self, t, state, pars):
        """ 
        The touchdown event function for refinement of touchdown detection.
        The zero-crossing of the output is defined as instant of the event.
        Hint: use a lambda function to adapt the call signature
       
        :args:
            t (float): time (ignored)
            y (6x float): CoM state [position, velocity]
            pars (4x float): the leg functions parameters. Format:
                [l0, alpha, beta, floorlevel]    

        :returns:
            f (float): the axial leg force  ["f = k * (l - l0)"]


        """
        foot = state.squeeze()[1] - pars[0] * sin(pars[1])
        return foot - pars[3] # foot - ground level        
        
    def dy_Stance(self, t, y, pars, return_force = False):
        """
        This is the ode function that is passed to the solver. Internally, it calles:
            legfunc1 - force of leg 1 (overwrite for new models)
            legfunc2 - force of leg 2 (overwrite for new models)
        
        :args:
            t (float): simulation time
            y (6x float): CoM state
            pars (dict): parameters, will be passed to legfunc1 and legfunc2.
                must also include 'foot1' (3x float), 'foot2' (3x float), 'm' (float)
                and 'g' (3x float) indicating the feet positions, mass and direction of
                gravity, respectively.
            return_force (bool, default: False): return [F_leg1, F_leg2] (6x
                float) instead of dy/dt.
        """
        
        f1 = max(self.legfunc1(t, y, pars), 0) # only push
        l1 = norm(array(y[:3]) - array(pars['foot1']))
        f1_vec = (array(y[:3]) - array(pars['foot1'])) / l1 * f1
        f2 = max(self.legfunc2(t, y, pars), 0) # only push
        l2 = norm(array(y[:3]) - array(pars['foot2']))
        f2_vec = (array(y[:3]) - array(pars['foot2'])) / l2 * f2
        if return_force:
            return hstack([f1_vec, f2_vec])
        return hstack([y[3:], (f1_vec + f2_vec) / pars['m'] + pars['g']])

    
    def get_touchdown(self, t, y, params):
        """
        Compute the touchdown position of the leg. Overwrite this for different leg parameters!
        
        :args:
            t (float): time
            y (6x float): state of the CoM
            params (4x float): leg parameter: stiffness, l0, alpha, beta
        
        :returns:
            [xFoot, yFoot, zFoot] the position of the leg tip
        """
        k, l0, alpha, beta = params
        xf = y[0] + l0 * cos(alpha) * cos(beta)
        yf = y[1] - l0 * sin(alpha)
        zf = y[2] - l0 * cos(alpha) * sin(beta)
        
        return array([xf, yf, zf])

    def checkSim(self):
        """
        Raises an error if the model failed.
        Overwrite in derived classes to avoid raised errors.
        """

        if self.failed:
            raise SimulationError("simulation failed!")
    
    def do_step(self):
        """ 
        Performs a step from the current state, using the current parameters.
        The simulation results are also stored in self.[y|t]_[s|d]s_seq, 
        the states and times of single and double support phases.

        *requires*: 
            self.
                - params (dict): model and leg function parameters
                - odd_step (bool): whether or not to trigger contact of leg2 (leg1 otherwise)
                - state (6x float): the  initial state
            
            
        :args:
            (None)

        :returns:
            t_ss, y_ss, t_ds, y_ds: time and simulation results for single stance and double stance 
            phases

        :raises:
            TypeError - invalid IC or parameter
            SimulationError - if the simulation fails.
        """
        
        # test initial conditions.

        # test wether there is a current state and current parameters
        if self.params is None:
            raise TypeError("parameters not set")
        if self.state is None:
            raise TypeError("state (initial condition) not set")
        if self.failed:
            raise SimulationError("Simulation failed previously.")
#demo_p_reduced = [13100, 12900, 68.5 * pi / 180., -.05] # [k1, k2, alpha, beta]
#demo_p = { 'foot1' : [0, 0, 0],
#     'foot2' : [-1.5, 0, 0],
#     'm' : 80,
#     'g' : [0, -9.81, 0],
#     'lp1' : [13100, 1, 68.5 * pi / 180, -0.05],  # leg params: stiffness, l0, alpha, beta
#     'lp2' : [12900, 1, 68.5 * pi / 180, 0.1],
#     'delta_beta' : .05
#     }
        p = self.params # shortcut
        leadingleg = 1. if self.odd_step else 2.
        pars = [p['lp1'][0],
                p['lp2'][0],
                p['lp1'][2],
                p['lp2'][2],
                p['lp1'][1],
                p['lp2'][1],
                p['lp1'][3],
                p['lp2'][3],
                p['m'],
                p['g'][1],
                p['foot1'][0],
                p['foot1'][1],
                p['foot1'][2],
                p['foot2'][0],
                p['foot2'][1],
                p['foot2'][2],
                leadingleg]

        # maximal time for simulation of single stance or double stance (each)
        max_T = 1. 

        # run single stance
        self.buf[0, 1:] = array(self.state) #.copy()
        N = self.odess.odeOnce(self.buf, self.t + max_T, dt=1e-3, pars = pars)
        self.state = self.buf[N,1:].copy()

        self.y_ss_seq.append(self.buf[:N+1, 1:].copy())
        self.t_ss_seq.append(self.buf[:N+1,0].copy())
# quick sanity check: simulation time not exceeded?
        if self.buf[N,0] - self.t >= max_T - 1e-2:
            self.failed=True
            print "N=", N
            raise SimulationError("Maximal simulation time (single stance) reached!")
        self.t = self.buf[N,0]
        

# touchdown detected:
# update foot parameters
# (1) foot2 = foot1
# (2) foot1 = [NEW]
# (3) leading_leg = ~leading_leg
# update leg positions; change trailing leg

        y = self.state # shortcut
        vx, vz = y[3], y[5]
        a_v_com = -arctan2(vz, vx) # correct with our coordinate system

        pars[13] = pars[10]
        pars[15] = pars[12]
        if pars[16] == 1.:
# stance leg is leg 1 -> update leg 2 params
            pars[10] = y[0] + cos(pars[3]) * cos(pars[7] + a_v_com) * pars[5]
            pars[12] = y[2] - cos(pars[3]) * sin(pars[7] + a_v_com) * pars[5]

            #pars[13] = res[N, 1] + cos(pars[3])*cos(pars[7])*pars[5]
            #pars[15] = res[N, 3] + cos(pars[3])*sin(pars[7])*pars[5]
            pars[16] = 2.;
        else:
            pars[10] = y[0] + cos(pars[2]) * cos(pars[6] + a_v_com) * pars[4]
            pars[12] = y[2] - cos(pars[2]) * sin(pars[6] + a_v_com) * pars[4]

            #pars[10] = res[N, 1] + cos(pars[2])*cos(pars[6])*pars[4]
            #pars[12] = res[N, 3] + cos(pars[2])*sin(pars[6])*pars[4]
            pars[16] = 1.;

        self.params['foot1'] = pars[10:13][:]
        self.params['foot2'] = pars[13:16][:]

        # run double stance
        self.buf[0, 1:] = array(self.state) #.copy()
        N = self.odeds.odeOnce(self.buf, self.t + max_T, dt=1e-3, pars = pars)
        self.state = self.buf[N,1:].copy()
        self.feet1_seq.append(self.params['foot1'])
        self.feet2_seq.append(self.params['foot2'])
        self.y_ds_seq.append(self.buf[:N+1, 1:].copy())
        self.t_ds_seq.append(self.buf[:N+1,0].copy())
# quick sanity check: simulation time not exceeded?
        if self.buf[N,0] - self.t >= max_T - 1e-2:
            self.failed=True
            raise SimulationError("Maximal simulation time (double stance) reached!")
        self.t = self.buf[N,0]

        #self.y_ds_seq.append(y2)
        #self.t_ds_seq.append(t2)

        self.odd_step = not self.odd_step


        return self.t_ss_seq[-1], self.y_ss_seq[-1], self.t_ds_seq[-1], self.y_ds_seq[-1]

        if self.odd_step:            
            td_pars = self.params['lp2'][1:] + [ground, ] # set touchdown parameters
            td_pars_2 = self.params['lp2'] # another format of touchdown parameters (for get_touchdown)
            newfoot = 'foot2' # which foot position to update?
            to_evt_fun = self.legfunc1 # force generation for takeoff trigger in double support
            to_evt_ds_refine = self.legfunc1 # function for refinement of DS
            
            self.odd_step = False # next step is "even": leg "2" in single stance on ground
        else:
            td_pars = self.params['lp1'][1:] + [ground, ] # set touchdown parameters
            td_pars_2 = self.params['lp1'] # another format of touchdown parameters (for get_touchdown)
            newfoot = 'foot1' # which foot position to update?            
            to_evt_fun = self.legfunc2 # force generation for takeoff trigger in double support
            to_evt_ds_refine = self.legfunc2 # function for refinement of DS
            
            self.odd_step = True # next step is "odd": leg "1" in single stance on ground
            
        # stage 1a: simulate until vy=0
        
        self.singleStance = True
        self.ode.event = self.evt_vy0
        if self.state[4] <= 0:
            self.failed = True
            self.ErrMsg = ("initial vertical velocity < 0: single " +
            "stance apex cannot be reached!")
        t0 = self.t
        tE = t0 + max_T
        t_a, y_a = self.ode(self.state, t0, tE, dt=self.dt)
        #d_pars_l2 = self.params['lp2'][1:] + [ground, ]
        if self.DEBUG:
            print "finished stage 1 (raw)"
        if t_a[-1] >= tE:
            self.failed = True
            self.ErrMsg = ("max. simulation time exceeded - " + 
             "this often indicates simulation failure")
        else:
            tt1, yy1 = self.ode.refine(lambda tf, yf: yf[4])
            if self.DEBUG:
                print "finished stage 1 (fine)"
            self.state = yy1
            # compute forces
        if not self.skip_forces:
            forces_ss = [self.dy_Stance(xt, xy, self.params, return_force=True) for
                xt, xy in zip(t_a, y_a)]
        #self.forces_ss_seq.append()

        t = []  # dummy, if next step is not executed
        y = array([[]])
        if not self.failed:
            self.update_params_ss()
            
            # stage 1b: simulate until touchdown of leading leg
            # touchdown event of leading leg
            self.ode.event = lambda t,states,traj,p: self.touchdown_event(t, states, traj, td_pars)
            t0 = tt1
            tE = t0 + max_T
            t, y = self.ode(self.state, t0, tE, dt=self.dt)
            if self.DEBUG:
                print "finished stage 2 (raw)"
            if t[-1] >= tE:
                self.failed = True
                self.ErrMsg = ("max. sim time exceeded in single stance - no "
                        + "touchdown occurred")
            else:
                #d_pars_l2 = self.params['lp2'][1:] + [ground, ]
                tt, yy = self.ode.refine(lambda tf, yf: self.touchdown_event_refine(tf, yf, td_pars))
                if self.DEBUG:
                    print "finished stage 2 (fine)"
                self.state = yy
            forces_ss.extend([self.dy_Stance(xt, xy, self.params, return_force=True) for
                xt, xy in zip(t[1:], y[1:, :])])
            if not self.skip_forces:
                self.forces_ss_seq.append(vstack(forces_ss))
        if not self.failed:
            # allow application of control law        
            self.t_td = tt
            self.singleStance = False
            self.update_params_td()
        
        # accumulate results from stage 1a and stage 1b
        if not self.failed:
            t = hstack([t_a, t[1:]])
            y = vstack([y_a, y[1:, :]])
        
        # stage 2: double support
        # compute leg 2 touchdown position        
        t2_a = []
        y2_a = array([[]])
        if not self.failed:
            xf, yf, zf = self.get_touchdown(tt, yy, td_pars_2)
            self.params[newfoot] = [xf, yf, zf]
            
            # stage 2a: simulate until vy=0
            self.ode.event = self.evt_vy0
            t0 = tt
            tE = t0 + max_T
            t2_a, y2_a = self.ode(self.state, t0, tE, dt=self.dt)
            if t2_a[-1] >= tE:
                self.failed = True
                self.ErrMsg = ("max. sim time exceeded - no nadir event " +
                "detected in double stance")
                if self.DEBUG:
                    print "finished stage 3 (raw)"
            else:
                tt2, yy2 = self.ode.refine(lambda tf, yf: yf[4])
                if self.DEBUG:
                    print "finished stage 3 (fine)"
                self.state = yy2
            if not self.skip_forces:
                forces_ds = [self.dy_Stance(xt, xy, self.params, return_force=True) for
                    xt, xy in zip(t2_a, y2_a)]

        if not self.failed:
            # allow application of control law
            self.update_params_ds()
          
        
        # stage 2b: double stance - simulate until takeoff of trailing leg
       
        # define and solve double stance ode
        #ode = integro.odeDP5(self.dy_Stance, pars=self.params)
        # event is takeoff of leg 1
        t2_b = []
        y2_b = array([[]])
        if not self.failed:
            self.ode.event = lambda t,states,traj,p: self.takeoff_event(t,
                    states, traj, p, legfun=to_evt_fun)
            t0 = tt2
            tE = t0 + max_T
            t2_b, y2_b = self.ode(self.state, t0, tE, dt=self.dt)
            if t2_b[-1] >= tE:
                self.failed = True
                self.ErrMsg = ("sim. time exeeded - takeoff of trailing leg " + 
                        "not detected")
                if self.DEBUG:
                    print "finished stage 4 (raw)"
            else:
        # refinement: force reaches zero
                tt, yy = self.ode.refine(lambda tf, yf: to_evt_ds_refine(tf, yf, self.params))
                if self.DEBUG:
                    print "finished stage 4 (fine)"
                self.state = yy

            if not self.skip_forces:
                forces_ds.extend([self.dy_Stance(xt, xy, self.params, return_force=True) for
                    xt, xy in zip(t2_b[1:], y2_b[1:, :])])

                self.forces_ds_seq.append(vstack(forces_ds))
            # allow application of control law
            self.t_to = tt
            self.singleStance = True
            self.update_params_to()
        
        # accumulate results from stage 1a and stage 1b
        if not self.failed:
            t2 = hstack([t2_a, t2_b[1:]])
            y2 = vstack([y2_a, y2_b[1:, :]])
        
        #store simulation results
        if not self.failed:
            self.y_ss_seq.append(y)
            self.y_ds_seq.append(y2)
            self.t_ss_seq.append(t)
            self.t_ds_seq.append(t2)
            self.feet1_seq.append(self.params['foot1'])
            self.feet2_seq.append(self.params['foot2'])
        if not self.failed:
            if len(t2) > 0:
                self.t = t2[-1]

        if self.failed:
            raise SimulationError(self.ErrMsg)

        return t, y, t2, y2

class BSLIP_newTD(BSLIP):
    """ derived from BSLIP. The get_touchdown function is overwritten
    such that the leg placement is w.r.t. walking direction.
    
    *NOTE* This is also a show-case how to use inheritance for modelling here.
    """
    
    def get_touchdown(self, t, y, params):
        """
        Compute the touchdown position of the leg w.r.t. CoM velocity
        
        :args:
            t (float): time
            y (6x float): state of the CoM
            params (4x float): leg parameter: stiffness, l0, alpha, beta
        
        :returns:
            [xFoot, yFoot, zFoot] the position of the leg tip
        """
        k, l0, alpha, beta = params
        vx, vz = y[3], y[5]
        
        a_v_com = -arctan2(vz, vx) # correct with our coordinate system
        #for debugging
        #print "v_com_angle:", a_v_com * 180. / pi
                
        xf = y[0] + l0 * cos(alpha) * cos(beta + a_v_com)
        yf = y[1] - l0 * sin(alpha)
        zf = y[2] - l0 * cos(alpha) * sin(beta + a_v_com)

        #for debugging
        #print "foot: %2.3f,%2.3f,%2.3f," % ( xf,yf, zf)
        
        return array([xf, yf, zf])



def ICeuklid_to_ICcircle(IC):
    """
    converts from IC in euklidean space to IC in circle parameters (rotational invariant).
    The formats are:
    IC_euklid: [x, y, z, vx, vy, vz]
    IC_circle: [y, vy, |v|, |l|, phiv], where |v| is the magnitude of CoM velocity, |l| 
        is the distance from leg1 (assumed to be at [0,0,0]) to CoM, and phiv the angle
        of the velocity in horizontal plane wrt x-axis
    *NOTE* for re-conversion, the leg position is additionally required
    
    :args:
        IC (6x float): the initial conditions in euklidean space

    :returns:
        IC (5x float): the initial conditions in circular coordinates
    
    """
    x,y,z,vx,vy,vz = IC
    v = sqrt(vx**2 + vy**2 + vz**2)
    l = sqrt(x**2 + y**2 + z**2)
    #phiv = arctan2(vz, vx)
    #phiv = arctan2(-vz, vx)
    phiv = -arctan2(-vz, vx)
    #phix = arctan2(-z, -x)
    phix = arctan2(z, -x)
    # warnings.warn('TODO: fix phi_x (add)')
    # print "phix:", phix * 180 / pi
    return [y, vy, v, l, phiv + phix]
    
def ICcircle_to_ICeuklid(IC):
    """
    converts from IC in cirle parameters to IC in euklidean space (rotational invariant).
    The formats are:
    IC_euklid: [x, y, z, vx, vy, vz]
    IC_circle: [y, vy, |v|, |l|, phiv], where |v| is the magnitude of CoM velocity, |l| 
        is the distance from leg1 (assumed to be at [0,0,0]) to CoM, and phiv the angle
        of the velocity in horizontal plane wrt x-axis
    *NOTE* for re-conversion, the leg position is additionally required, assumed to be [0,0,0]
    Further, it is assumed that the axis foot-CoM points in x-axis
    
    :args:
        IC (5x float): the initial conditions in circular coordinates

    :returns:
        IC (6x float): the initial conditions in euklidean space
    
    """
    y, vy, v, l, phiv = IC
    z = 0
    x = -sqrt(l**2 - y**2)
    v_horiz = sqrt(v**2 - vy**2)
    vx = v_horiz * cos(phiv)
    #vz = v_horiz * sin(phiv)
    vz = v_horiz * sin(phiv)
    return [x, y, z, vx, vy, vz]


def circ2normal_param(fixParams, P):
    """
    converts the set (fixParams, P) to a set of initial conditions for
    a BSLIP model.

    :args:
        fixParams (dict): set of parameters for BSLIP, plus "delta_beta" key
        P [4x float]: step parameters k1, k2, alpha, beta (last two: for both legs)
    """
    k1, k2, alpha, beta = P
    par = deepcopy(fixParams)
    par['foot1'] = [0, 0, 0]
    par['foot2'] = [-2*par['lp2'][1], 0, 0] # set x to some very negative value
    par['lp1'][0] = k1
    par['lp2'][0] = k2
    par['lp1'][2] = par['lp2'][2] = alpha
    par['lp1'][3] = beta
    par['lp2'][3] = -beta + par['delta_beta']
    return par
    
def pred_to_p(baseParams, P):
    """
    converts the set (fixParams, P) to a set of initial conditions for
    a BSLIP model.

    :args:
        fixParams (dict): set of parameters for BSLIP
        P [8x float]: step parameters k1, k2, alpha1, alpha2, beta1, beta2,
            l01, l02 
    """
    k1, k2, a1, a2, b1, b2, l01, l02  = P
    par = deepcopy(baseParams)
    par['foot1'] = [0, 0, 0]
    par['foot2'] = [-2*par['lp2'][1], 0, 0] # set x to some very negative value
    par['lp1'][0] = k1
    par['lp2'][0] = k2
    par['lp1'][1] = l01
    par['lp2'][1] = l02
    par['lp1'][2] = a1
    par['lp2'][2] = a2
    par['lp1'][3] = b1
    par['lp2'][3] = b2
    return par

def new_stridefunction(fixParams):
    """ returns a function that maps [IC, P] -> [FS],
    in the BSLIP_newTD model
    where IC: (reduced) initial conditions
          P:  reduced parameter vector (4x float)
          FS: final state
    """
    model = BSLIP_newTD(fixParams,[0,0,0,0,0,0])
    model.skip_force = True #speed up simulation a little bit
    def stridefun(IC, P):
        """ performs a stride of the given model.
        
        :args:
            IC: (reduced) initial conditions: [y, vy, v, l, phiv]
            P: (reduced) parameter set: [k1, k2, alpha, beta]

        :returns:
            FS: final state, same format as initial conditions
        """

        full_IC = ICcircle_to_ICeuklid(IC)
        par = circ2normal_param(fixParams, P)

        model.state = full_IC
        model.params = par
        
        model.init_ode()

        model.do_step()
        model.do_step()
        fs = model.state.copy() # final state of simulation

        fs[:3] -= model.params['foot1'] # set origin to location of foot1 (which is on ground)

        return array(ICeuklid_to_ICcircle(fs))
    
    return stridefun

def stridefunction(fixParams):
    """ returns a function that maps [IC, P] -> [FS],
    in the BSLIP_newTD model
    where IC: (reduced) initial conditions
          P:  reduced parameter vector (8x float): k1, k2, a1, a2, b1, b2, l01,
              l02
          FS: final state
    """
    model = BSLIP_newTD(fixParams,[0,0,0,0,0,0])
    model.skip_force = True #speed up simulation a little bit
    
    def stridefun2(IC, P):
        """ performs a stride of the given model.
        
        :args:
            IC: (reduced) initial conditions: [y, vy, v, l, phiv]
            P: (reduced) parameter set: (k1, k2, a1, a2, b1, b2, l01, l02)

        :returns:
            FS: final state, same format as initial conditions
        """

        full_IC = ICcircle_to_ICeuklid(IC)
        par = pred_to_p(fixParams, P)

        model.state = full_IC
        model.params = par
        
        model.init_ode()

        model.do_step()
        model.do_step()
        fs = model.state.copy() # final state of simulation

        fs[:3] -= model.params['foot1'] # set origin to location of foot1 (which is on ground)

        return array(ICeuklid_to_ICcircle(fs))
    
    return stridefun2
    
def vis_sim(mdl):
    """
    quick hack that visualizes the simulation results from a model

    :args:
        mdl (BSLIP): model that has run some steps
    """
    # visualize
    fig = figure(figsize=(18,8))
    fig.clf()
    subplot(1,2,1)
    rep = 0
    for ys, yd, f1, f2 in zip(mdl.y_ss_seq, mdl.y_ds_seq, mdl.feet1_seq[1:], mdl.feet2_seq[1:]):
        label1 = label2 = label3 = label4 = None
        if rep == 0:
            label1 = 'single stance'
            label2 = 'double stance'
            label3 = 'foot leg#1'
            label4 = 'foot leg#2'
    
        plot(ys[:, 0], ys[:, 1], 'b-', linewidth=1, label=label1)
        plot(yd[:, 0], yd[: ,1], 'g-', linewidth=3, label=label2)    
        plot(f1[0], f1[1], 'kd', label=label3)
        plot(f2[0], f2[1], 'cd', label=label4)
        rep += 1
    
    legend(loc='best')
    
    xlabel('horizontal position [m]')
    ylabel('vertical position [m]')
    subplot(1,2,2)
    rep = 0
    for ys, yd, f1, f2 in zip(mdl.y_ss_seq, mdl.y_ds_seq, mdl.feet1_seq[1:], mdl.feet2_seq[1:]):
        label1 = label2 = label3 = label4 = None
        if rep == 0:
            label1 = 'single stance'
            label2 = 'double stance'
            label3 = 'foot leg#1'
            label4 = 'foot leg#2'        
        plot(ys[:, 0], ys[:, 2], 'r-', linewidth=1, label=label1)
        plot(yd[:, 0], yd[: ,2], 'm-', linewidth=3, label=label2)
        plot(f1[0], f1[2], 'kd', label=label3)
        plot(f2[0], f2[2], 'cd', label=label4)
        rep += 1
    
    legend(loc='best')
    #axis('equal')
    
    xlabel('horizontal position [m]')
    ylabel('lateral position [m]')    
    return fig

# define some example values

demo_p_reduced = [13100, 12900, 68.5 * pi / 180., -.05] # [k1, k2, alpha, beta]
demo_p = { 'foot1' : [0, 0, 0],
     'foot2' : [-1.5, 0, 0],
     'm' : 80,
     'g' : [0, -9.81, 0],
     'lp1' : [13100, 1, 68.5 * pi / 180, -0.05],  # leg params: stiffness, l0, alpha, beta
     'lp2' : [12900, 1, 68.5 * pi / 180, 0.1],
     'delta_beta' : .05
     }
demo_IC = array([-0.153942, 0.929608, 0, 1.16798, 0.593798, -0.045518])
