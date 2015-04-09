# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:00:05 2011

@author: Christian

simulation of spring mass model for running including energy management

input parameters:
    vx0,y0,L0,a0,k,m,ygrd,t,fs,steps
output parameters:
    x_sim,y_sim,vx_sim,vy_sim,t_sim,reached_steps,betaTO_sim,xF,k2,L02

"""

from __future__ import division
from pylab import arange,zeros,sqrt,cos,sin,pi,append,zeros_like,ones_like
from pylab import isnan,isinf,arctan,s_,delete, linspace, ceil
from scipy.integrate import odeint

from scipy.optimize import fmin


def quad(zahl):
    return zahl**2

def s_dot(s,t0,parameter):
    sd = zeros(4)
    sd[0] = s[2]
    sd[1] = s[3]
    
    xF = parameter['xF']
    yF = parameter['yF']
    
    L = sqrt((s[0]-xF)**2+((s[1]-yF))**2)
    P = (parameter['k']/parameter['m'])*((parameter['L0']/L)-1)
    
    if P > 0:
        ax = P * (s[0] -xF)
        ay = P * (s[1] -yF) + parameter['g']
    else:
        ax = 0
        ay = 0
    
    sd[2] = ax
    sd[3] = ay
    
    return sd

def sim(vx0,y0,L01,a0,k1,dE,m,ygrd,t,fs,steps):
    
    def optfunc_dE(par):
        return abs(-.5*k1*c**2+.5*(k1+par)*(c-(c*par/(k1+par)))**2-dE)
       
    g = -9.81 # gravitational force
    betaTO_sim = zeros(steps)
    xF=0
    
    x_sim = [0]
    y_sim = [0]
    vx_sim = [0]
    vy_sim = [0]
    t_sim = [0]    
    x_sim_apex = [0]
    y_sim_apex = [0]
    vx_sim_apex = [0]
    vy_sim_apex = [0]
    t_sim_apex = [0]
    
    k2 = 0
    L02 = 0
    
    skip_all = False
    st_lo_idx = -1
    # leg orientation
    xL = L01 * cos(a0 * pi/180.)
    yL = L01 * sin(a0 * pi/180.)
    
    # first ground contact
    t_gc = sqrt(2.*(L01*sin(a0*pi/180.)-y0)/g)
    
    if isnan(t_gc) == True:
        
        skip_all = True
    
    if skip_all == False:
        x_gc = vx0*t_gc
        y_gc = .5*g*t_gc**2 + y0
        
        # system parameter for first step
        s = zeros(4)
        s[0] = 0
        s[1] = y0
        s[2] = vx0
        s[3] = 0
        
        # definitions for calculation
        end_sim = False
        
        res = zeros([1,4])
        res_app = []
        t_v = [0]
        skip_stance = False
        
        # start loop
        while end_sim == False:
            st_lo_idx += 1 
            
            if st_lo_idx != 0: # if not first step
                t_gc = (-res[-1,3]/g) + \
                      sqrt( ((res[-1,3]**2)/(g**2)) - \
                        ((2*res[-1,1])/g-(2*L01*sin(a0*pi/180)/g)) )
                #print t_gc
                if isnan(t_gc) == False and isinf(t_gc) == False:
                    
                    y_gc = 0.5*g*t_gc**2 + res[-1,3]*t_gc + res[-1,1]
                    
                    t_vec = linspace(0,t_gc,ceil(t_gc*fs)+1)
                    #print len(t_vec)
                                    
                    s[0] = res[-1,2]*t_gc + res[-1,0]
                    s[1] = y_gc
                    s[2] = res[-1,2]
                    s[3] = g*t_gc +res[-1,3] 
                else:
                    t_gc = 1
                    t_vec = linspace(0,t_gc,ceil(t_gc*fs)+1)
                    end_sim = True
                    skip_stance = True
                    #break
                
                vx = zeros_like(t_vec)
                vy = zeros_like(t_vec)
                x =  zeros_like(t_vec)
                y =  zeros_like(t_vec)    
                
                vx[0] = res[-1,2]
                vy[0] = res[-1,3]
                x[0] = res[-1,0]
                y[0] = res[-1,1]
            else: # if first step
                t_vec = linspace(0,t_gc,ceil(t_gc*fs)+1)
                #print t_vec
                vx = zeros_like(t_vec)
                vy = zeros_like(t_vec)
                x =  zeros_like(t_vec)
                y =  zeros_like(t_vec) 
                #print vx
                vx[0] = s[2]
                vy[0] = s[3]
                x[0] = s[0]
                y[0] = s[1]
                
        ### flight phase 
            x[1:] = vx[0]*t_vec[1:] + x[0]
            y[1:] = .5*g*(t_vec[1:]**2)+vy[0]*t_vec[1:] + y[0]
            vy[1:] = g*t_vec[1:] + vy[0]
            vx[1:] = vx[0]*ones_like(t_vec[1:])
            
            # end at apex
            if st_lo_idx == steps or skip_stance == True:
                tapex = vy[0]/-g
                #print 'y0=',y0,'vy0=',vy[0],' tapex=',tapex
                tvec_apex = linspace(0,tapex,ceil(tapex*fs)+1)
                vyapex = g*tvec_apex+vy[0]
                #print 'vyapex=',vyapex
                yapex = .5*g*(tvec_apex**2) + vyapex[0]*tvec_apex + y[0]
                xapex = vx[0] * tvec_apex + x[0]
                vxapex = vx
                
                x_sim = append(x_sim,xapex[1:])
                y_sim = append(y_sim,yapex[1:])
                vx_sim = append(vx_sim,vxapex[1:])
                vy_sim = append(vy_sim,vyapex[1:])
                t_sim = append(t_sim,tvec_apex[1:]+t_sim[-1])
                skip_stance = True
                
                """
                for kk in arange(2,len(y)):
                    if y[kk-2]<y[kk-1] and y[kk-1]>y[kk]:
                        end_sim = True
                        
                        x_sim = append(x_sim,x[1:kk])
                        y_sim = append(y_sim,y[1:kk])
                        vx_sim = append(vx_sim,vx[1:kk])
                        vy_sim = append(vy_sim,vy[1:kk])
                        t_sim = append(t_sim,t_vec[1:kk]+t_sim[-1])
                """
                    
        ### stance phase
            
            if skip_stance == False:
                # system parameter
                if st_lo_idx == 0: # if first step
                    s[0] = x_gc         # x
                    s[1] = y_gc         # y
                    s[2] = vx0          # vx
                    s[3] = g*t_gc + 0   # vy
                    
                xF = s[0]+xL
                yF = ygrd
                
                
                parameter = {'k':k1,'L0':L01,'m':m,'g':g,'xF':xF,'yF':yF}
                calc_length = 0.1*fs
                
                find_pos = False
                round_prec = 11 # precision of rounding
                #print 'round_prec = ', round_prec
                
                step_size = 1./fs
        
                res = zeros([1,4]) 
                t_v = [0] 
                
                res_app = False
                skip = False  
                
                L0 = L01
                
                FinMin = False
                # start integration of stance phase
                while find_pos == False:
                    t_vint = (step_size) * arange(calc_length)
                    res_int = odeint(s_dot,s,t_vint, args=(parameter,), rtol=1e-11)
                    
                    # stop condition vx or y < 0
                    if res_int[-1,1] < 0 or res_int[-1,2] < 0: 
                        find_pos = True
                        end_sim = True
                        skip = True  
                    
                    # if ymin: calculation of new k and L for energy changes
                    for idx in arange(1,len(res_int[:,1])-1):
                        if res_int[idx,1]<res_int[idx-1,1] and  res_int[idx,1]<res_int[idx+1,1]:
                            res_int=delete(res_int,s_[idx+1::],0)
                            t_vint = delete(t_vint,s_[idx+1::],0)
                            c= L0-(sqrt(res_int[-1,1]**2 + (xF-res_int[-1,0])**2))
                            dk = fmin(optfunc_dE,0,full_output=1,disp=0,xtol=1e-12)
                            #print dk
                            # if not find a minimum stop loop --> no step
                            if dk[-1] <> 0:
                                find_pos = True
                                end_sim = True
                                skip = True  
                                print 'no energy management possible'
                                break
                            k2 = k1+dk[0]
                            L02 = L01 - c*dk[0]/(k1+dk[0])
                            #print 'delta L = ',L0 - L02, ' delta k = ', dk[0]
                            L0 = L02
                            parameter = {'k':k2,'L0':L02,'m':m,'g':g,'xF':xF,'yF':yF}
                            FinMin = True
                            break
                    
                    # if not ymin: start one before to avoid ignoring while change
                    if FinMin == False:
                        res_int = res_int[:-1]
                        t_vint = t_vint[:-1]
                    
                    # virtual leg after integration
                    L_virt = sqrt(res_int[-1,1]**2 + (xF-res_int[-1,0])**2)
                    
                    if L_virt < L0 and skip == False:
                        s[0] = res_int[-1,0]
                        s[1] = res_int[-1,1]
                        s[2] = res_int[-1,2]
                        s[3] = res_int[-1,3]
                        
                    elif L_virt >= L0 and skip == False:
                        for jj in arange(1,len(res_int)):
                            L_virt_1 = sqrt(res_int[jj,1]**2 + abs(xF-res_int[jj,0])**2)
                            L_virt_2 = sqrt(res_int[jj-1,1]**2 + abs(xF-res_int[jj-1,0])**2)
                            
                            if round(L_virt_2,round_prec) == round(L0,round_prec):
                                find_pos = True
                                res_int = res_int[0:jj,:]
                                res_app = True                 
                                t_vint = t_vint[0:jj]
                                if res_int[-1,3] <=0: end_sim = True
                                break
                            elif L_virt_2 < L0 and L_virt_1 > L0:
                                s[0] = res_int[jj-1,0]
                                s[1] = res_int[jj-1,1]
                                s[2] = res_int[jj-1,2]
                                s[3] = res_int[jj-1,3]
                                res_int = res_int[0:jj,:]
                                res_app = True
                                t_vint = t_vint[0:jj]
                                step_size = step_size / 10.
                                break
                    # append arrays
                    if res_app == False:
                        res = res_int
                        t_v = t_vint 
                        res_app = True
                    else:
                        res = append(res,res_int[1::],0)
                        t_v = append(t_v,t_vint[1::]+t_v[-1])
                
                # take of angle
                #print st_lo_idx
                betaTO_sim[st_lo_idx] = arctan(res[-1,1]/(res[-1,0]-xF))*180./pi 
                
                # append arrays 
                x_sim = append(x_sim,append(x,res[:,0]))
                y_sim = append(y_sim,append(y,res[:,1]))
                vx_sim = append(vx_sim,append(vx,res[:,2]))
                vy_sim = append(vy_sim,append(vy,res[:,3]))
                t_sim = append(t_sim,append(t_vec,t_v+t_gc)+t_sim[-1])
            
            else: #skip_stance == True because of ending at apex
                end_sim = True
        
        # cut array
        x_sim = x_sim[1::]
        y_sim = y_sim[1::]
        vx_sim = vx_sim[1::]
        vy_sim = vy_sim[1::]
        t_sim = t_sim[1::]
    
    
    #print 'vor return'
    return x_sim,y_sim,vx_sim,vy_sim,t_sim,st_lo_idx,betaTO_sim,xF,k2,L02


