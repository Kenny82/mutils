# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:42:46 2010

@author: moritz2
"""



import time
import os
import numpy as np
from subprocess import Popen, PIPE
import misc 
import cStringIO

                
def nrlazy(data, m=5, d=50, iterations=20,v=0.05):
    """
    This is a stub!
    applies the tisean local projective noise recduction filter to the data
    refer to the tisean documentation for the meaning of m,d,iterations
    """
    # TODO: implement this!
    pass
       

def project(data, m=5,  iterations=2, r=None, k=30, q=2,x=None):
    """
    applies the tisean local projective noise reduction filter 'project'
    for parameter meanings, refer to tisean documentation
    """        
    
    if r == None:
        r = np.std(data.flatten())/100.
        
    rstring = ' -r%s ' % str(r)
    xstring = ''
    if x is not None:
        xstring = (' -x=%i' % x)
    cmd_string = 'project -m%i %s -k%i -q%i -i%i %s' \
                %(m,rstring,k,q,iterations,xstring )   
    p = Popen(cmd_string, shell = True, stderr = PIPE, stdout = PIPE, stdin = PIPE)
    
    for a in data:
        p.stdin.write(str(a)+'\n')
    
    (output_s,outError) = p.communicate()
    try:
        output = np.loadtxt(cStringIO.StringIO(output_s))    
    except IOError:
        output = outError

    return output



def ghkss(data, m=(1,5), d=1, iterations=5, r=0.05, k=5, q=2):
    """
    applies the tisean local projective noise reduction filter ghkss
    for parameter meanings, refer to tisean documentation
    """        
    
    if r == None:
        r = np.std(data.flatten())/100.
        
    rstring = ' -r%s ' % str(r)
    
    cmd_string = 'ghkss -m%i,%i -d%i %s -k%i -q%i -i%i ' \
                %(m[0],m[1],d,rstring,k,q,iterations )   
    
    p = Popen(cmd_string, shell = True, stderr = PIPE, stdout = PIPE, stdin = PIPE)
    
    for a in data:
        p.stdin.write(str(a)+'\n')
    
    (output_s,outError) = p.communicate()
    output = np.loadtxt(cStringIO.StringIO(output_s))                 

    return output

        
def lzo_test(data,d=20, nd=3, s=1, C = None,k=30 ,r=None, f=1.2):
    """
    performa the zero-order prediction average error test.
    This is a front-end for tisean lzo-test    
    usage: lzo_test (data, d=20, )
        data: data to test, given in a Nxm - matrix N:number of measurements, m: number of variables
        d: time delay (default: 20)
        nd: embedding dimension (default: 3)
        C: causality window (default: d*nd*s)
        k: minimal number of neighbors (default: 30)
        r: starting neighborhood size (default: std(data)/1000)
        f: factor to increase the neighborhoodsize (default: 1.2)
    """
    
    if C == None:
        C = d*nd*s

    rstring = ''       
    if r is not None:
        rstring = ('r=%f'% float(r))
        
    m = (data.shape[-1],nd)
    
    cmd_string = 'lzo-test -m%i,%i -d%i -s%i -k%i -C%i -V2 -f%f %s' \
                %(m[0],m[1],d,s,k,C,f,rstring)   
    print (cmd_string)
    p = Popen(cmd_string, shell=True, stdin = PIPE, stderr = PIPE, stdout = PIPE)        
    for a in data:
        mystring = np.array2string(a, separator = '\t',max_line_width = np.inf)[1:-1]
        p.stdin.write(mystring+'\n')        

    (output_s,outError_s) = p.communicate() 
    #print('lzo: communicated!')
    output = np.loadtxt(cStringIO.StringIO(output_s))  
    #print('lzo: output performed')
    del output_s
    
    return output
    
def lfo_test(data,d=1, nd=3, s=1, C = None,k=30 ,r=None, f=1.2):
    """
    performa the local linear fit prediction average error test.
    This is a front-end for tisean lzo-test    
    usage: lzo_test (data, d=20, )
        data: data to test, given in a Nxm - matrix N:number of measurements, m: number of variables
        d: time delay (default: 20)
        nd: embedding dimension (default: 3)
        C: causality window (default: d*nd*s)
        k: minimal number of neighbors (default: 30)
        r: starting neighborhood size (default: std(data)/1000)
        f: factor to increase the neighborhoodsize (default: 1.2)
    """
    
    if C == None:
        C = d*nd*s

    rstring = ''       
    if r is not None:
        rstring = ('r=%f'% float(r))
        
    m = (data.shape[-1],nd)
    
    cmd_string = 'lfo-test -m%i,%i -d%i -s%i -k%i -C%i -V2 -f%f %s ' \
                %(m[0],m[1],d,s,k,C,f,rstring)   
    print (cmd_string)
    p = Popen(cmd_string, shell=True, stdin = PIPE, stderr = PIPE, stdout = PIPE)        
    for a in data:
        mystring = np.array2string(a, separator = '\t',max_line_width = np.inf)[1:-1]
        p.stdin.write(mystring+'\n')        

    (output_s,outError_s) = p.communicate() 
    #print('lzo: communicated!')
    output = np.loadtxt(cStringIO.StringIO(output_s))  
    #print('lzo: output performed')
    del output_s
    
    return output

       
def mutual(data, D=20, b=16):
    """
    mutual information from the tisean package
    """
    
    if len(data.shape) > 1:
        c = data.shape[-1]
    else:
        c = 1
        
    cmd_string = 'mutual -c%i -D%i' %(c,D)   
    print cmd_string        
    p = Popen(cmd_string, shell = True, stderr = PIPE, stdout = PIPE, stdin = PIPE)
    
    for a in data:
        mystring = np.array2string(a, separator = '\t',max_line_width = np.inf)[1:-1]
        p.stdin.write(mystring+'\n')
        #print(mystring)
        
    (output,outError) = p.communicate()
    
    res = []
    try:
        res = np.array([[float(entry) for entry in o_line.split(' ') if len(entry) > 1] for o_line in output.splitlines()[1:]])
    except ValueError:
        print('Error in execution of tisean mutual\n')
        res = (output,outError)
     
    
    return res
        
        
    
def recurr(data, m=(1,2), d=1, r = None, percent=None, maxsize = np.inf):
    """        
    creates a recurrence plot using the TISEAN package
    usage: parameters like tisean, m=(1,2), r= None (=std/100), d=1
    returns an Lx2 array, where similar time indices are listed
    to visualize: r = tis_recurr(...)  plot(r[:,0],r[:,1],'.')
    """
    
    if r == None:
        r = np.std(data.flatten())/100.
        
    rstring = ' -r%s ' % str(r)
    if percent != None:
        rstring = ' -%%%i' % int(percent)
    

    
    misc.a2dat(data,'totisean.dat')    
    
    cmd_string = 'recurr -m%i,%i -d%i %s totisean.dat' \
                %(m[0],m[1],d,rstring )                 
    p = Popen(cmd_string, shell=True, stderr = PIPE, stdout = PIPE, close_fds = True)        
    (output,outError) = p.communicate()
    
    refined = False
    while len(output) > maxsize:            
        if refined == False:
            refined = True
            print('refining: ')
        r /= 2.
        rstring = ' -r%s ' % str(r)
        print ('r= %s' % str(r))
        cmd_string = 'recurr -m%i,%i -d%i %s totisean.dat' \
                %(m[0],m[1],d,rstring ) 
        p = Popen(cmd_string, shell=True, stderr = PIPE, stdout = PIPE, close_fds = True)
        (output,outError) = p.communicate() 
    if refined == True:    
        print('\n')
        
    res = np.array([[float(entry) for entry in o_line.split(' ')] for o_line in output.splitlines()])
  
    return res
        
    
def recurr2(data, m=(1,2), d=1, c = None, r = None, percent=None, minsize = None, verbose = False):
    """        
    creates a recurrence plot using the TISEAN package
    usage: parameters like tisean, m=(1,2), r= None (=std/100), d=1
    c:None or String naming the columns to read (e.g.: c='1,2,3,4'
    returns an Lx2 array, where similar time indices are listed
    to visualize: r = tis_recurr(...)  plot(r[:,0],r[:,1],'.')
    
    
    in contrast to recurr, recurr2 increases r until the filelength is at minsize    
    """
    
    if data.ndim == 1:
        dat = data.copy()[:,np.newaxis]
    else:
        dat = data
        
    if minsize == None:
        minsize = dat.shape[0]/100
    
    if r == None:        
        r = np.std(dat.flatten())/1000.
        print ('no r given - setting it to %f' % r)

    if c == None:
       c = '1'
    else:
       c = c + ' '
        
    rstring = ' -r%s ' % str(r)
    if percent != None:
        rstring = ' -%%%i' % int(percent)
    
    cmd_string = 'recurr -m%i,%i -d%i %s -c%s ' % (m[0],m[1],d,rstring,c )     
    print cmd_string
    p = Popen(cmd_string, shell=True, stdin = PIPE, stderr = PIPE, stdout = PIPE)        
    for a in dat:
        mystring = np.array2string(a, separator = '\t',max_line_width = np.inf)[1:-1]
        p.stdin.write(mystring+'\n')     
    
    (output_s,outError) = p.communicate()
    try:
        output = np.loadtxt(cStringIO.StringIO(output_s))
    except IOError:
        output = np.array([])
                
        
    del output_s
    # is the recurrence data long enough?
    coarsed = False
    while output.shape[0] < minsize:            
        if coarsed == False:
            coarsed = True
            if verbose:
                print('coarsing: ')
        r *= 1.5
        rstring = ' -r%s ' % str(r)
        if verbose:
            print ('r= %s' % str(r))
        cmd_string = 'recurr -m%i,%i -d%i %s  -c%s ' % (m[0],m[1],d,rstring,c )             
        p = Popen(cmd_string, shell=True, stdin = PIPE, stderr = PIPE, stdout = PIPE)        
        
        cnt = 0
        for a in data:
            mystring = np.array2string(a, separator = '\t',max_line_width = np.inf)[1:-1]
            p.stdin.write(mystring+'\n')  
            cnt += 1
         
        (output_s,outError) = p.communicate()
        try:
            output = np.loadtxt(cStringIO.StringIO(output_s))
        except IOError:
            output = np.array([])        



    return output
    
        
def surrogate(data, seed = None, make_spectra_exact = False):
    """
    generates surrogate data (tisean method)
    it assumes that the tisean executables are accessible in the system path
    it can take an array, assuming that the time series goes along one column
    (i.e. that an Nxm array consists of N repeated measurements of m 
    variables)
    """
    
    if make_spectra_exact == True:
        spec_string = '-S'
    else:
        spec_string = ''
    
    try:
        t_seed = str(float(seed)).split('.')[0]
    except (TypeError, ValueError):
        seed = None
    
    if seed == None:
        t_seed = str(time.time()-int(time.time())).split('.')[1][0:8]
    
    # generate input datafile:
    misc.a2dat(data,'totisean.dat')
    
    # call tisean
    # generate command string
    if len(data.shape) == 1:
       str_n_cols = 1
    else:
       str_n_cols = data.shape[-1]            
    cmd_string = 'surrogates %s -I%s -m%s totisean.dat -o fromtisean.dat' \
                %(spec_string, t_seed,str_n_cols ) 
    os.system(cmd_string)
            
    
    # read output
    t_output = misc.dat2a('fromtisean.dat')     
    return t_output
        
 

   
def load_lzo(filename):
    """ 
    loads the result of an tisean-lzo prediction
    """
    myfile = open(filename,'r')
    c = []
    while True:
        mystring = myfile.readline()    
        if mystring == '':
            break
        c.append( [float(x) for x in mystring.split(' ')[1:-1] ] )
        
    
    myfile.close()
    c = np.array(c)
    return c
   
def applyLazyFilter(self, data):
    """
    NOT IMPLEMENTED!
    applies the lazy nonlinear tisean filter to the data
    version 1: for each col separately
    """
    pass
            

