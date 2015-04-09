# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Bipedal SLIP
# 
# 
# ## LE:  
# May 5th, 2013 MM - forked from "Walking SLIP" notebook  
# May 16th, 2013 MM - bughunting: comparison between "wrapped" (circular-invariant) and "original" model gives slightly different results - this must not be the case. [edit: solved - that was a tricky one!]  
# May 31th, 2013 MM - moved to server, added to SVN ... continued bughunting  
# June 4th, 2013 MM - fixed bug (finally?), found quasiperiodic circular walking solution  
# June 14th, 2013 MM - searched and analyzed some fixed points  
# June 27th, 2013 MM - started re-organization (for fixpoint mappings)  
# July 1st, 2013 MM - cleaned up notebook  
# July 10th, 2013 MM - continued preparations for mapping; introduced fixed energy solutions  
# July 11th, 2013 MM - hangling on the edge of stable solutions introduced  
# July 12th, 2013 MM - continuation for changed alpha introduced  
# July 15th, 2013 MM - defined and found starting fixpoints. **Major edit**: removed "old" code not used for new approach
# July 22-25th, 2013 MM - forked "C" version -> BSLIP now runs
# 
# ## TODO:
# 
# 
# * think again about splitting of "fixed params" and "variable params" (for mapping)
# * map fixpoints as a function of parameters!
# 
# <a name="toc"></a>
# ## Table of content
# 
# <a href="#step1">Step 1: initialize notebook</a>  
# 
# <a href="#vis">visualize selected solution</a>  
# 
# <a href="#step5">Step 5: New approach: "augment" Harti's solutions</a>  
# <a href="#notes">General notes</a>

# <markdowncell>

# ## Goal
# This notebook implements the goals for analyzing the 3D walking gait.
# 
# ## Hypotheses are:
# 
# * Asymmetry leads almost-always to walking in circles, however there is a set of asymmetric "straight-walking" solutions.
# * This property persists under (random) perturbations (-> test with uniform (not gaussian) noise!)
# * Walking in circles can be achieved using symmetric configuration but asymmetric noise magnitude.
# * These properties are also valid in non-SLIP models (e.g. constant-force leg function)
# 
# ## requirements:
# 
#   - models.bslip
#   
# ### parameter layout
# Model parameters have the following structure:
# 
# `
# param   
#    .foot1  (1x3 array)   location of foot 1
#    .foot2  (1x3 array)   location of foot 2
#    .m      (float)       mass
#    .g      (1x3 array)   vector of gravity
#    .lp1    (4x float)    list of leg parameters for leg 1
#         for SLIP: [l0, k, alpha, beta] (may be overwritten in derived models)
#    .lp2    (4x float)    list of leg parameters for leg 2
# `

# <markdowncell>

# <a name="step1"></a>
# # Step 1: initialize notebook
# <a href="#toc">table of content</a>

# <codecell>

# Import libraries
#from models import bslip
import bslip
from bslip import ICeuklid_to_ICcircle, ICcircle_to_ICeuklid, circ2normal_param, new_stridefunction, vis_sim, stridefunction
from copy import deepcopy # e.g. for jacobian calculation
import mutils.misc as mi
import sys

#define functions
def new_stridefunction_E(pbase, E_des, p_red):
    f = new_stridefunction(pbase)
    
    def innerfunction(IC_red):
        IC = ICe_to_ICc(IC_red, E_des, p_red[0], pbase['m'], l0 = pbase['lp1'][1])
        return f(IC, p_red)[[0,1,3,4]]
    return innerfunction

def new_stridefunction_E2(pbase, E_des, p_red):
    f = stridefunction(pbase)
    
    def innerfunction(IC_red):
        IC = ICe_to_ICc(IC_red, E_des, p_red[0], pbase['m'], l0 = pbase['lp1'][1])
        return f(IC, p_red)[[0,1,3,4]]
    return innerfunction


# IC_circle: [y, vy, |v|, |l|, phiv]
def getEnergy(IC_circle, k1, m=80, l0=1):
    """ returns the energy of the given state (and with specified params). (the constant term mc**2  is neglected)

    :args:
        IC_circle: the initial conditions in circular form
        k1 : leg stiffness of contact leg
        m : mass
        l0 : leg rest length

    :returns:
        E: the total energy of the system

    """
    E_kin = m * .5 * IC_circle[2]**2
    E_pot = 9.81 * m * IC_circle[0]
    E_spring = .5 * k1 * (IC_circle[3] - l0)**2
    return E_kin + E_pot + E_spring

def ICe_to_ICc(ICr, E_des,  k1, m=80, l0=1):
    """
    returns circular ICs with a fixed energy.

    :args:
        ICr: the reduced circular ICs: [y, vy, |l|, vphi]
        E: the desired energy
        k1 : leg stiffness of contact leg
        m : mass
        l0 : leg rest length

    :returns:
        ICc: (circular) initial conditions
    """
    ICc = zeros(5)
    ICc[[0,1,3,4]] = ICr
    # compute velocity magnitude separately
    vmag2 =  2 * (E_des - getEnergy(ICc, k1, m) ) / m
    if vmag2 >= 0:
        ICc[2] = sqrt(vmag2)
    else:
        raise ValueError("Velocity magnitude must be imaginary!")
        
    return ICc

def deltafun_E_base(ICe, p_red, p_base):
    """ returns the difference of the IC minus the final state """
    f = new_stridefunction(p_base)
    ICc = ICe_to_ICc(ICe, E_des, p_red[0], p_base['m'], l0 = p_base['lp1'][1])        
    return array(f(ICc, p_red))[[0,1,3,4]] - array(ICe)

def deltafun_E_base2(ICe, p_red, p_base):
    """ returns the difference of the IC minus the final state """
    f = stridefunction(p_base)
    ICc = ICe_to_ICc(ICe, E_des, p_red[0], p_base['m'], l0 = p_base['lp1'][1])        
    return array(f(ICc, p_red))[[0,1,3,4]] - array(ICe)


def getPS(ICr, pred, pbase, E_des, maxStep=.1, debug=False, rcond=1e-7, maxnorm=5e-6, maxrep_inner=12,
    get_EV = False, h=1e-4):
    """
    searches a periodic solution

    :args:
        ICr [array (4)]: reduced initial conditions to start from: [y, vy, |l|, vphi]
        pred: reduced set of parameters - either length 4 or 8
            length 4: k1, k2, alpha, beta
            length 8: k1, k2, alpha1, alpha2, beta1, beta2, l01, l02
        pbase: base set of parameters

    :returns:
        (stable, ICp): stable is True if the solution is stable, and ICp give the periodic solution
        in circular coordinates (5x)

    :raises: 
        RuntimeError: if too many iterations were necessary

    """    
    # set algorithm parameter
    
    
    stab_thresh = 1.00 # maximum value for largest EV such that solution is considered stable.

    all_norms = []
    if len(pred) == 4:    
        deltafun_E = lambda x: deltafun_E_base(x, pred, pbase)
    elif len(pred) == 8:
        deltafun_E = lambda x: deltafun_E_base2(x, pred, pbase)
    else:
        raise ValueError("illegal format of pred: length must be 4 or 8")
            
            
    IC_next_E = ICr.copy()
    
    n_bisect_max = 4
    nrep = 0
    # This is the Newton-Raphson algorithm (except that the inverse is replaced by a pseudo-inverse)
    r_norm = norm(deltafun_E(IC_next_E)) #some high value
    while r_norm > maxnorm and nrep < maxrep_inner:
        
        J = mi.calcJacobian(deltafun_E, IC_next_E, h=h)
        # compute step (i.e. next IC). limit stepsize (only if start is too far away from convergence)
        delta0 =  - dot(pinv(J, rcond=rcond), deltafun_E(IC_next_E)).squeeze()
        if norm(delta0) > maxStep:
            delta0 = delta0 / norm(delta0) * maxStep
            sys.stdout.write('!')
        else:
            sys.stdout.write('.')
        # update position
        IC_next_E = IC_next_E + delta0
        nrep += 1
        
        r_norm_old = r_norm        
        r_norm = norm(deltafun_E(IC_next_E))
        
        all_norms.append(r_norm)
        
        # check if norm decreased - else, do a bisection back to the original point
        if r_norm > r_norm_old:
            # error: distance INcreased instead of decreased!
            new_dsts = []
            smallest_idx = 0
            maxnorm_bs = r_norm
            sys.stdout.write('x(%1.2e)' % r_norm)
            for niter_bs in range(5):
                IC_next_E = IC_next_E - (.5)**(niter_bs + 1) * delta0
                new_dsts.append([IC_next_E.copy(), norm(deltafun_E(IC_next_E))])
                if new_dsts[-1][1] < maxnorm_bs:
                    maxnorm_bs = new_dsts[-1][1]
                    smallest_idx = niter_bs
            IC_next_E = new_dsts[smallest_idx][0]
            
   

    if r_norm < maxnorm:
        print " success!",
        is_stable = True
        IC_circle = ICe_to_ICc(IC_next_E, E_des, pred[0], pbase['m'],l0 = pbase['lp1'][1])
        if len(pred) == 4:
            f = new_stridefunction_E(pbase, E_des, pred)
        else:
            f = new_stridefunction_E2(pbase, E_des, pred)
        J = mi.calcJacobian(f, IC_next_E)
        if max(abs(eig(J)[0])) > stab_thresh:
            is_stable = False
        if get_EV:
            return eig(J)[0], IC_circle
        else:
            return is_stable, IC_circle
    else:
        print "number of iterations exceeded - aborting"
        print "IC:", IC_next_E
        raise RuntimeError("Too many iterations!")

        
        
def getEig(sol):
    """ returns the eigenvalues of a pair of [icc, pr] """
    icc, pr = sol
    f = new_stridefunction_E(pbase, E_des, pr)
    J = mi.calcJacobian(f, icc[[0,1,3,4]])
    return eig(J)[0]



def getR(ICc, pr, pbase):
    ICe_v = ICcircle_to_ICeuklid(ICc)
    mdl_v = bslip.BSLIP_newTD(bslip.pred_to_p(pbase, pr), ICe_v)
    #mdl_v.ode.ODE_ATOL = 1e-11
    #mdl_v.ode.ODE_RTOL = 1e-12
    #mdl_v.ode.ODE_EVTTOL = 1e-12
    # make first two steps for calculating walking radius
    for rep in range(2):
        _ = mdl_v.do_step()
        l_v = norm(mdl_v.state[:3] - ICe_v[:3]) # this works *only* because the height is equal!
        phi0_v = arctan2(ICe_v[5], ICe_v[3])
        phiE_v = arctan2(mdl_v.state[5], mdl_v.state[3])
        deltaPhi_v = phiE_v - phi0_v
        
    if abs(deltaPhi_v) < 1e-5:
        r = 1e9
    else:
        r = l_v / (2. * sin(.5 * deltaPhi_v))
    return r

# <markdowncell>

# ### example usage



# <markdowncell>

# ### NOTE - check the "asymmetry" of the "symmetric parameter" configurations! 
# 
# - are "asymmetric" solutions with symmetric parameters stable?
# - do "asymmetric" solutions with symmetric parameters walk in circles?

# <markdowncell>

# <a name="step5"></a> 
# ## Step 5: New approach: "augment" Harti's solutions
# <a href="#toc">content</a>  
# <a href="#vis">visualize</a>  
# 
# There are three selected solutions from Harti:  
# k = 14 kN/m, alpha= 69 $\deg$ (symmetric)  
# k = 14 kN/m, alpha= 73 $\deg$ (asymmetric)  
# k = 20 kN/m, alpha= 76 $\deg$ (flat force pattern - small area of stable configurations)  

# <codecell>

p_red = array(bslip.demo_p_reduced)
E_des = 816.
p_base = bslip.demo_p
p_base['delta_beta'] = 0

selection = 1

# periodic solution 1:
if selection == 1:
    p_red[0] = p_red[1] = 14000
    p_red[2] = 69. * pi / 180.
    p_red[3] = .05
    
    ## ?? IC0 = array([ 0.93358044,  0.43799566,  1.25842366,  0.94657333, -0.10969046]) # already periodic (?)
    IC0 = array([ 0.93358034,  0.43799844,  1.25842356,  0.94657321,  0.10969058]) # beta=.05
    #IC0 = array([ 0.93358039,  0.45195548,  1.26003517,  0.94679076,  0.21853576]) # beta=.1

elif selection == 2:
    # periodic solution 2:  (asymmetric stance phase)
    p_red[0] = p_red[1] = 14000
    p_red[2] = 72.5 * pi / 180.
    p_red[3] = 0.05
    
    #IC0 = array([ 0.92172543,  0.40671347 , 1.1950172,   0.92609043 , 0.  ]) # periodic for beta = 0, alpha=72.5
    IC0 = array([ 0.9308592,   0.3793116,   1.19155584,  0.9360028,   0.1597469 ]) # for beta=.05, alpha=72.5
    #IC0 = array([ 0.93011364,  0.39346135,  1.19332777,  0.93554008,  0.31541887]) # for beta=.1, alpha=72.5
    
    # periodic, for beta=0, alpha=73. very very marginally stable: |ev| = .992|.984 (step|stride):
    #IC0 = array([ 0.9278273,   0.34175418,  1.17852052 , 0.93208755 , 0.]) 
    
elif selection == 3:
    # periodic solution 3:
    p_red[0] = p_red[1] = 20000
    p_red[2] = 76.5 * pi / 180.
    p_red[3] = 0.1
    #p_red[3] = 0.0
    #IC0 = array([ 0.96030477,  0.30256976,  1.15633538,  0.985058303, -0.11240564])
    #IC0 = array([ 0.97043906,  0.29739433,  1.0840199,   0.97280541,  0.        ]) # for beta=0; not really periodic (2e-4)
    IC0 = array([ 0.97236992, 0.18072418,  1.0718928,   0.97368293,  0.        ]) # for beta=0, alpha=76.5
    IC0 = array([ 0.97236992,  0.17616847,  1.07200705,  0.97370148,  0.22353624]) # for beta=.5, alpha=76.5
    IC0 = array([ 0.97237007,  0.16336241,  1.07238146,  0.97376284,  0.44756444]) # for beta=.5, alpha=76.5
    #IC0 = array([0.9709, 0.34167, 1.0855, 0.973732, 0.15846])
    #IC0 = array([ 0.97028136, 0.30045474,  1.08604313,  0.97290092, 0.16489379])
    #ICe = IC0[[0,1,3,4]]
    #[ 0.97029372  0.2972158   0.97289854  0.16536238]
    #ICe = array([ 0.97050506,  0.30422253,  0.97301987,  0.06965177])
    ##print ICe_to_ICc(ICe, E_des, p_red[0])
    ##stop

else:
    raise NotImplementedError("No valid selection - select solution 1,2, or 3")


ICe = IC0[[0,1,3,4]] # remove |v| -> this will be determined from the system energy
evs, IC = getPS(ICe, p_red, p_base, E_des, get_EV = True, maxnorm=1e-5, rcond=1e-7,  maxStep=.1, maxrep_inner=10, h=1e-4)
print IC
print "max. ev:", max(abs(evs))

# <markdowncell>

# #### goto vis
# <a href="#vis">visualization</a>

# <markdowncell>

# ### check periodicity


# ## Step 5.2: map the $\Delta L_0 - \Delta k$ or $\Delta L_0 - \Delta \alpha$ plane (again until instability is found)C
# 
# **Choice**: $\alpha - L_0$

# <codecell>

import mutils.io as mio
# "A"
#solutions_fname = "A_sols5_new_al0.list"
#IC0 = array([ 0.93358034,  0.43799844,  1.25842356,  0.94657321,  0.10969058])
#pr0 = [14000, 14000, 69.*pi / 180, 69 * pi / 180, .05, -.05, 1., 1.]
#solutions_fname = "A_sols10_new_al0.list"
#IC0 = array([ 0.93358039,  0.45195548,  1.26003517,  0.94679076,  0.21853576])
#pr0 = [14000, 14000, 69.*pi / 180, 69 * pi / 180, .1, -.1, 1., 1.]

# "B"
#solutions_fname = "B_sols5_new_al0.list"
#pr0 = [14000, 14000, 72.5 *pi / 180, 72.5 * pi / 180, .05, -.05, 1., 1.]
#IC0 = array([ 0.9308592,   0.3793116,   1.19155584,  0.9360028,   0.1597469 ]) # for beta=.05, alpha=72.5
solutions_fname = "B_sols10_new_al0.list"
pr0 = [14000, 14000, 72.5 *pi / 180, 72.5 * pi / 180, .1, -.1, 1., 1.]
IC0 = array([ 0.93011364,  0.39346135,  1.19332777,  0.93554008,  0.31541887]) # for beta=.1, alpha=72.5

# "C"
#solutions_fname = "C_sols5_new_al0.list"
#pr0 = [20000, 20000, 76.5 *pi / 180, 76.5 * pi / 180, .05, -.05, 1., 1.]
#IC0 = array([ 0.97236992,  0.17616847,  1.07200705,  0.97370148,  0.22353624]) # for beta=.5, alpha=76.5
#pr0 = [20000, 20000, 76.5 *pi / 180, 76.5 * pi / 180, .1, -.1, 1., 1.]
#solutions_fname = "C_sols10_new_al0.list"
#IC0 = array([ 0.97237007,  0.16336241,  1.07238146,  0.97376284,  0.44756444]) # for beta=.1, alpha=76.5

# <codecell>

# init the loop


ICa = IC0.copy()


all_solutions = [] # format: [IC, px, r, evs]

signs = [1, -1]

#signs = [-1, ]

# <codecell>

delta_l = 0
step_l = .0001 # m

delta_alpha = 0
step_alpha = .1 * pi / 180. # radiant!

# <codecell>

for sgn in signs:
    delta_alpha = 0
    ICa = IC0.copy()

    #if True:

    while True:
        

        pr0a = array(pr0)
        pr0a[2] += delta_alpha / 2.
        pr0a[3] -= delta_alpha / 2.

        print "new round:",
        try:
            evs, IC = getPS(ICa[[0,1,3,4]], pr0a, p_base, E_des, get_EV = True, maxnorm=2e-5, rcond=2e-7,
              maxStep=.025, maxrep_inner=10, h=5e-5)
        except Exception:
            print "fixpoint search failed",
            print "delta alpha", delta_alpha, 
            if sgn > 0:
                print "(+)",
            else:
                print "(-)", "-> done!"
            break;
            
        if max(abs(evs)) > 1:
            print "delta alpha", delta_alpha, 
            if sgn > 0:
                print "(+)",
            else:
                print "(-)",
            print "lead to instability!"
            break
            
        r = getR(IC, pr0a, p_base)
        print "dl=0, da=", delta_alpha, " -> r=", r
        
        all_solutions.append([array(IC), array(pr0a), r, array(evs)])
           
        delta_l = step_l # for delta_l = 0: already found!
        n_solutions_dk = 0
        
        ICx = IC.copy()
        ICa = IC.copy()
        while True:
            prx = array(pr0a)
            prx[6] += delta_l / 2.
            prx[7] -= delta_l / 2.
            try:
                evs, IC = getPS(ICx[[0,1,3,4]], prx, p_base, E_des, get_EV = True, maxnorm=2e-5, rcond=2e-7, 
                 maxStep=.1, maxrep_inner=10, h=1e-4)
            except Exception:
                break
                
            if max(abs(evs)) > 1:
                break
            
            r = getR(IC, prx, p_base)
            print "l:", prx[6:8], " -> r=", r
            n_solutions_dk += 1
            all_solutions.append([array(IC), array(prx), r, array(evs)])
            
            # prepare new round
            ICx = array(IC)   
            delta_l += step_l
        
        if n_solutions_dk == 0:
            break
            
        delta_alpha += sgn * step_alpha
        
mio.msave(solutions_fname, all_solutions)    
    
    

# <markdowncell>

# <a name="notes"></a>
# # Notes
# <a href="#toc">table of content</a>  
# 
# ### radius of a quasi-periodic solution
# the radius of a quasi-periodic solution can be computed as
# 
# $r = \frac{\large l}{\large 2 \sin{(\Delta \varphi / 2)}}$,
# 
# where $l$ is the absolute distance between CoM at start and end of a stride, and $\Delta \varphi$ is the angle between initial and final CoM velocity in horizontal plane.
# 
# ### Story (elements)
# 
# * Here, we are mostly interested in "functional" asymmetries of the leg - asymmetries that are not apparent like different leg lengths. These can be differences in the leg strength, here represented by different leg stiffnesses $k$, or differences in the leg swing policies, here represented by different angles of attack $\alpha$.
# 
# * For completion, we also demonstrate that the "apparent" asymmetries like in leg length or lateral leg placement also yield circular walking behavior.
# 
# * We investigated three different kind of periodic motions, as already found by Geyer: two different kind of symmetric motions, and an asymmetric motion. We selected three parameter sets similar to Geyer's A,B,C points, but changed them slightly towards configurations with increased stability.
# 
# 

