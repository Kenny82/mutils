# -*- coding : utf8 -*-
"""
.. module:: 3seg
    :synopsis: Equations and solutions for the three-segment model

.. moduleauthor:: Moritz Maus <mmaus@sport.tu-darmstadt.de>

"""

# format: l1, l2, l3, c1, c2]

from pylab import (array, arccos, linspace, vstack, figure, clf, plot, xlabel,
        ylabel, show, savefig, sqrt, xlim, ylim, axis, arange)


def cfunc(x, params):
    """
    this function returns the constraint violation of the 3seg leg.
    it must be zeroed!

    Parameters
    ----------

    x
        configuration of the leg: Fy, h1, h2, tau1, tau2

    params
        parameter of the system: l, l1, l2, l3, c1, c2


    Returns
    -------
    eq : *array* (1x5)
        the non-fulfilment of the constraints (subject to root finding)

    """
    
    l, l1, l2, l3, c1, c2 = params
   # print l, l1, l2, l3, c1, c2 
    Fy, h1, h2, tau1, tau2 = array(x).squeeze() 
   # print Fy, h1, h2, tau1, tau2 
    
    if h1 > l1:
        print "warning: invalid h1"
        h1 = l1
        return [5000, ]*5
    if h2 > l3:
        print "warning: invalid h2"
        h2 = l2
        return [5000, ]*5
    if h1 + h2 > l2:
        print "warning: invalid h1 + h2"
        return [5000, ]*5
        while h1 + h2 > l2:
            h1 = .8 * h1
            h2 = .8 * h2
            

    eq1 = Fy * h1 - tau1
    eq2 = tau1 - Fy * h1 - Fy * h2 + tau2
    eq3 = Fy * h2 - tau2
    eq4 = -1. * c1 * (arccos(h1 / l1) + arccos( (h1 + h2) / l2) - .9 * pi) - tau1
    eq5 = -1. * c2 * (arccos(h2 / l3) + arccos( (h1 + h2) / l2) - .9 * pi) - tau2
    eq6 = sqrt(l1**2 - h1**2) + sqrt(l2**2 - (h1 + h2)**2) + sqrt(l3**2 -
            h2**2) - l

    # note: eq2 is omitted because of equality to - (eq1 + eq3)!
    return array([eq1, eq3, eq4, eq5, eq6])


if __name__ == '__main__':
    import scipy.optimize as opt

    x0 = array([  2.64347199e+03,   7.04878037e-02,   1.67474976e-01,
             1.86332534e+02,   4.42715408e+02])

# first parameter is L0
    params = [.999, .36, .45, .2, 110., 65.]

#IC = array([0., .00001, .00001, .0002, .003])
    IC = array([1., .001, .005, 1., 2.])

    res0 = opt.fsolve(cfunc, IC, args=params, xtol=1e-10)

    def qfun(x, p):
        """ 
        did root finding succeed?
        """
        return sum(cfunc(x, p) **2)

    all_res = [res0, ]
    all_ll = [params[0], ]
    all_q = [qfun(res0, params),] 
    all_params = [params[:], ]

    for leglength in linspace(.999, .5, 100):
        params[0] = leglength
        IC = all_res[-1]
        all_res.append(opt.fsolve(cfunc, all_res[-1], args=params, xtol=1e-10))
        all_ll.append(leglength)
        all_params.append(params[:])
        all_q.append(qfun(all_res[-1], all_params[-1]))
        print 'll:', leglength


    all_res = vstack(all_res)
    all_params = vstack(all_params)

    figure('force of the leg')
    clf()
    plot(all_ll, all_res[:,0],'b.-')
    xlabel('leg length')
    ylabel('Force')
    show()

def visualize(config, param):
    """
    .. note::
        plots the leg on the current axes

    parameters
    ----------
    config : *array* (1x5)
        of cfunc's x parameter type, describing the configuration of the leg
    param : *list*
        the list of model parameters, according to cfunc's definition

    Returns
    -------
    *None*

    """

    figure('anim figure')
    clf()
    
# plot ground
    plot([-1,1],[0,0], color='#000044', linewidth=8)

    x = [0, -1 * config[1], config[2], 0]
    y1 = sqrt(param[1]**2 - config[1]**2)
    y2 = sqrt(param[2]**2 - (config[1] + config[2])**2)
    y3 = sqrt(param[3]**2 - config[2]**2)
    y = [0, y1, y1 + y2, y1 + y2 + y3]
    plot(x, y, color='#000000', linewidth=3)
    plot(x, y, color='#982112', linewidth=2, linestyle='--')
    plot(x[-1], y[-1], 'o', markersize=13, color='#ffea93')
    
    xlim(-1,1)
    ylim(-.2,2)
    axis('equal')


def viz(until):
    for k in arange(until):
        visualize(all_res[k,:], all_params[k,:])
        savefig('fig_%02i.png' % k)



