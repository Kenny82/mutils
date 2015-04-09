# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:49:40 2011

@author: moritz2
Version 0.1 - defines an object that handles the subject data for an individual
subject.

Version 0.1.1 - small updates in performance. some optional arguments
introduced

LE: 30.06.2011, MM
    12.07.2011, MM: added null-models in fit function
    27.07.2011, MM: added within-stride prediction
    03.08.2011, MM: added symmetries to definition
    17.10.2011, MM: added postgres access (name/password) to init function
    19.10.2011, MM: connection to SQLserver is now optional
    10.09.2012, MM: introduced caching in get_kin_from_idict
"""

import scipy.signal as si
from pylab import randint, split
import helper_classes as hp    
#import util
#import gc
from numpy import vstack, hstack, gradient, ceil, pi, linspace, interp, \
    array, arange, dot, diag, cov, mean, ceil, var, zeros, cumsum
from numpy.linalg import lstsq, svd          
import gzip
import pickle
import os
import uuid


#TODO:
    # in setData: use Kalman-Filter to calculate velocity
    # find memory leak :)
#DONE (in fitUtil.py):
    # implement: prediction with other matrices
    # implement: basline subtraction (set of functions)
    # in setData: also load and interpolate the force!


""" Copy-And-Paste from mutils.io """
def msave(filename, dat):
    """
    shorthand for putting some data (array, list, ...) into a zipped file
    """
    mfile = gzip.open(filename,mode='wb')
    pickle.dump(dat,mfile)
    mfile.close()

def mload(filename):
    """
    loads whatever is in the given file
    """
    mfile = gzip.open(filename,mode='rb')
    res = pickle.load(mfile)
    mfile.close()
    return res

class dObj:
    """
    A simple data container class
    """
    pass

class sdata():
    """
    Container class that stores and collect subject data and calculation results.
    """
    
    def __init__(self, selection = None, pg_user = None, pg_password = None, pg_db = None, connectDB = True):
        """
        initializes the object
         optional arguments:
             selection: sets the selection of coordinates
        """
        self.subject_id = None
        self.data_fz = []
        self.all_A = []
        self.vred = []
        self.relVol = [] #relative volume after prediction
        self.rv = [] # relative overall variance after prediction
        self.data = None
        self.data1D = None
        self.trial_type = None
        self.dataGathered = False
        self.nps = None # frames per cycle
        if selection is None:
            self.selection = ('l_kne_y - r_kne_y',  # selection of coordinates
                      'l_anl_y - l_kne_y',
                      'r_anl_y - r_kne_y',                  
                      'r_mtv_z - r_hee_z',
                      'l_mtv_z - l_hee_z',
                      'r_trc_y - r_kne_y',
                      'l_trc_y - l_kne_y',
                      'l_kne_x - r_kne_x',
                      'l_anl_x - l_kne_x',
                      'r_anl_x - r_kne_x',
                      'r_mtv_x - r_hee_x',
                      'l_mtv_x - l_hee_x',
                      'r_trc_x - r_kne_x',
                      'l_trc_x - l_kne_x',
                      'com_z',
    # 'l_elb_x - r_elb_x',
    # 'l_elb_y - r_elb_y',
    # 'l_elb_y - l_acr_y',
    # 'r_elb_y - r_acr_y',
    # 'l_elb_x - l_acr_x',
    # 'r_elb_x - r_acr_x',
    # 'r_elb_x - r_wrl_x',
    # 'l_elb_x - l_wrl_x',
    # 'r_elb_x - r_wrl_y',
    # 'l_elb_y - l_wrl_y',                       
    #                  'r_elb_y - r_acr_y',
    #                  'l_elb_y - l_acr_y',
    #                  'r_acr_y - r_sia_y',
    #                  'l_acr_y - l_sia_y',                                    
                      )
            raw_sym = ([0, 0, -1],
                       [1,2,1],
                       [3,4,1],
                       [5,6,1],
                       [7,7,-1],
                       [8,9,-1],
                       [10,11,-1],
                       [12,13,-1],
                       [14,14,1],
                       ) # coordinate exchange for single step analysis
                      
        else:
            self.selection = selection
        self.sym = None
        self._computeSym(raw_sym)
        if connectDB:
            self.db = hp.DB(username = pg_user, password = pg_password, dbname = pg_db)
        else:
            self.db = None
        
    def _computeSym(self,rawSym):
        """
        Computes the symmetry matrix (coordinate exchange) for step-to-step analysis
        The parameter rawSym is a tuple of 3-tuples [x1,x2,sign], indicating to
        switch coordinate #x1 with coordinate #x2, and exchanging the sign 
        (sign = -1) or not (sign = 1).
        
        Note: only coordinate exchange has to be given, velocities will be handled
        accordingly.
        """
        ndim = len(self.selection)
        self.sym = zeros((ndim*2,ndim*2))
        for rs in rawSym:
            self.sym[rs[0],rs[1]] = rs[2]
            self.sym[rs[1],rs[0]] = rs[2]          
        #copy to velocities
        self.sym[ndim:,ndim:] = self.sym[:ndim,:ndim].copy()
        
    def setSelection(self,selection, rawSym = None):
        """
        sets the selection of coordinates
        """
        self.selection = selection
        if rawSym is not None:
            self._computeSym(rawSym)
    
    def setData(self,subject_id, trial_type_id, reps = None, selection = None,
                nps = None, with_force = True, do_interp1D = True):
        """
        loads the subject data from the database
            required parameteres:
                subject_id: subject_id from database
                trial_type_id: trial_type_id from database
            optional parameters:
                reps (tuple): which repetitions to use; if omitted: all
                selection: which selection to use; if omitted: default selection 
                           (selection of the object)
                nps: for resampling: frames per stride                
        """
        
        if self.db is None:
            raise NameError, 'No DB connection in this instance!'
        
        self.data = []
        self.data_fz = []
        self.data1D = None
        #gc.collect()
        
        if nps is None:
            self.nps = 10.
        else:
            self.nps = nps
    
        if selection is None:
            selection = self.selection

        fs_kin = 250.
        fqvel = .125 # normalized filter frequency for velocity
        bf,af = si.butter(1,fqvel)
        skip_b = 5. # omit first # cycles
        skip_e= 5. # omit last (-#) cycles
        
        as1D = [] #all strides as "1D"-concatenated array
        nInT = [] #number of strides in pt. #    
    
        
        dbQuery = 'select trial_id from trial where trial_type = %i and subject_id = %i' % (trial_type_id, subject_id)
        
        if reps is not None:
            dbQuery += ' and repetition in ('
            for rep in reps:
                dbQuery += str(rep) + ', '
            dbQuery = dbQuery[:-2] + ' )'
            
        dbQuery += ' order by repetition;' 
        trialIDs = self.db.cmd(dbQuery)
        
        FloqDat = []    
        for tid in trialIDs:
            #gc.collect()                
            strGetSelection = 'SELECT '
            for pt in selection:
                strGetSelection += pt + ', '    
            strGetSelection += ' phi1, phi2, time from kin where trial_id = %i order by frame' % tid
            res = array(self.db.cmd(strGetSelection),dtype=float)
            #print 'res:', res.shape
            if with_force:
                fz_data = array(self.db.cmd('select fz_c,time from force where trial_id = %i order by frame' % tid),
                                dtype=float)            
                fz_kin = interp(res[:,-1],fz_data[:,1],fz_data[:,0])
            
            allv = []
            
            for col in range(len(selection)):
                v = gradient(si.filtfilt(bf,af,res[:,col]))*fs_kin
                allv.append(v)
            
            resv = array(allv,dtype=float).T
            
            # now: equally-phased data
            phi = res[:,-2]

            if do_interp1D:
                nstrides_total = ceil( (phi[-1]-phi[0]) / (2.*pi))    
                nstrides = nstrides_total - skip_b - skip_e
                send = skip_b + nstrides
                phivec = 2.*pi*linspace(skip_b,send-1./self.nps,self.nps*nstrides)
                interp_pos = []
                for col in range(len(selection)):
                    pp = interp(phivec,phi,res[:,col])
                    interp_pos.append(pp)
                if with_force:
                    interp_fz = interp(phivec,phi,fz_kin)
                
                interp_vel = []
                for col in range(len(selection)):
                    vp = interp(phivec,phi,resv[:,col])
                    interp_vel.append(vp)
                
                alldat = vstack((interp_pos,interp_vel,phivec)).T
                self.data.append(alldat)
                nInT.append(nstrides)
                for a in range(int(round(nstrides))):
                    as1D.append(hstack(alldat[a*self.nps:(a+1)*self.nps,:-1].T))

            if with_force and do_interp1D:
                self.data_fz.append(interp_fz)
                        
            fdat = dObj()
            fdat.data = vstack((res[:,:-3].T,allv))
            fdat.phi_kin =  phi
            fdat.kinTime = res[:,-1]
            if with_force:
                fdat.frcTime = fz_data[:,-1]
                fdat.fz = fz_data[:,0]
            FloqDat.append(fdat)
            if do_interp1D:
                del interp_pos, interp_vel
            
            del allv
            
            #gc.collect()
            
        self.rawData = FloqDat
        del FloqDat
        self.dataGathered  = True
        if do_interp1D:
            self.data1D = array(as1D,dtype=float)
        if with_force and do_interp1D:
            self.Fz1D = hstack(self.data_fz)
        #gc.collect()

    def get_kin_at_phase(self, sid, tid, lop, reps=None, reloadData=True):
        """
        returns the kinematic data, defined by self.selection, at phases given
        in a list of phases lop.

        '''''''''''
        Parameters:
        '''''''''''
        sid : *integer
            the subject id for which to set the data
        tid : *integer*
            trial id for which to set the data
        lop : *list*
            a list of an array phases at which the data should be computed
        reloadData : *boolean*
            **for internal use** if set to False, data will not be reloaded.
            This can cause inconsistent data, so be careful!

        """
        # set data quickly
#        print 'reload data?', reloadData
        if reloadData:
#            print 'setting data... sid = ', sid, ' tid = ', tid, 'reps =', reps
            self.setData(sid, tid, reps=reps, do_interp1D=False, 
                    with_force=False)
#            print 'done...'

        # walk through each trial, interpolate data at given phases
        all_dat = []
        for rep, phases in enumerate(lop):
            dat = self.rawData[rep].data
            phi = self.rawData[rep].phi_kin
            all_dat.append(vstack([interp(phases, phi, x) for x in dat]))

        return all_dat

    def get_kin_from_idict(self, sid, tid, idict, cachedir='./cache'):
        """
        This function is a wrapper for get_kin_at_phase, to compute the
        kinematic state at the apices which are mentioned in "idict" (idict
        stems from a stored SLIP data file).

        e.g.::

            import mutils.io
            import subjData 
            sd = subjData.sdata()
            _, _, _, _, idict = mutils.io.get_data(sid, tid, ... )
            kin = sd.get_kin_from_phase(sid, tid, idict)

        ''''''''''
        Parameter:
        ''''''''''
        sid : *integer* 
            subject id
        tid : *integer*
            trial-type id
        idict : *dictionary* (special format required, see description above)
            the 'information dictionary' that contains information about at
            which phases to get the data

        ''''''''
        Returns:
        ''''''''        
        dat_l : *list*
            a list of arrays that contain the states at a left apex, defined 
            by '*self*.selection', at the phases defined in idict.
        dat_r : *list*
            same as dat_l, only for right apices

        """

        # look for cached files
        cindexname = cachedir + os.sep + 'cachefiles.dict'
        try:
            mfile = gzip.open(cindexname, mode='rb')
            cfiles = pickle.load(mfile)
            mfile.close()
        except (IOError, EOFError, KeyError):
            print 'no cache index found'
            cfiles = {}

        # check marker list and subject Id and trial ID
        cfilefound = False
        for fname, fcontent in cfiles.iteritems():
            if (fcontent['selection'] == self.selection and
                    fcontent['sid'] == sid and
                    fcontent['tid'] == tid and
                    fcontent['reps'] == idict['reps']):
                print 'cached file (', fname, ') found in index'
                dat_l, dat_r = mload(cachedir + os.sep + fname)
                cfilefound = True
                break

        if not cfilefound:
            print 'no cache file found - accessing database'
            # create list of phases
            splitpoints = cumsum(idict['n_in_ws'])[:-1]
            lop_l = split(idict['phases_l'], splitpoints)
            lop_r = split(idict['phases_r'], splitpoints)
            
            dat_l = self.get_kin_at_phase(sid, tid, lop_l, reps=idict['reps'], 
                    reloadData=True)
            dat_r = self.get_kin_at_phase(sid, tid, lop_r, reps=idict['reps'], 
                    reloadData=False)
            nfnames = os.listdir(cachedir)
            fid = uuid.uuid4().hex
            while fid in nfnames:
                fid = uuid.uuid4().hex
            print 'storing data in cache'
            msave(cachedir + os.sep + fid, [dat_l, dat_r])
            cfiles.update({ fid : {'sid' : sid, 'tid' : tid, 'reps' :
                idict['reps'], 'selection' : self.selection} })
            msave(cindexname, cfiles)

        return dat_l, dat_r
        



    def fitmdl(self,nidx = 500,nrep = 500,psec = 0, rcond=0.03, dS = 1, dimVol = None, nullMdl = None):
        """
        performs a bootstrapped prediction (nrep times), based in nidx frames.
        also computes the out-of-sample variance reduction
        optionally, the number of the poincare-section can be given (psec). it
        must be a non-negative integer smaller than nps
        
        dS: number of strides to predict, typically: 1
        dimVol: how many singular values should be taken into account when comparing
            the relative volume? Default: all
        nullMdl: None -> normal fitting
                'inv' -> iDat <-> oDat switched (should have no effect in AR?)
                'rand' -> iDat, oDat have no relation (random selection)
        """
        if psec >= self.nps:
            raise ValueError, 'invalid poincare-section selected!'
            
        data = self.data1D[:,psec::self.nps]
        
        self.all_A = []
        #gc.collect()
        
        #section = psec
        #u0,s0,v0 = svd(aa1D[:,section::10].T,full_matrices = False)
        #vi = util.VisEig(256)
        self.vred = []
        self.rv = []
        self.relVol = []
        
        #vredpc = []
        #vredb = []
        in_idx = arange(self.data1D.shape[0]-dS)
        
        for rep in range(nrep):
            # create input and output data for regression, dependend on
            # null model
            if nullMdl is None:
                pred_idx = in_idx[randint(0,len(in_idx),nidx)]                
                idat = data[array(list(pred_idx),dtype=int),:]
                odat = data[array(list(pred_idx),dtype=int)+dS,:]
            elif nullMdl == 'inv':
                pred_idx = in_idx[randint(dS,len(in_idx),nidx)]
                idat = data[array(list(pred_idx),dtype=int),:]
                odat = data[array(list(pred_idx),dtype=int)-dS,:]
            elif nullMdl == 'rand':
                pred_idx = in_idx[randint(0,len(in_idx),nidx)]
                pred_idx_out = in_idx[randint(0,len(in_idx),nidx)]
                idat = data[array(list(pred_idx),dtype=int),:]
                odat = data[array(list(pred_idx_out),dtype=int),:]            
            else:
                raise ValueError, 'Error: Null model type not understood'
                
            test_idx = set(in_idx).difference(pred_idx)                        
            A = lstsq(idat,odat,rcond = rcond)[0]                
            test_idat = data[array(list(test_idx),dtype=int),:]
            test_odat = data[array(list(test_idx),dtype=int)+dS,:]
            pred = dot(test_idat,A)            
            res = test_odat - pred                                
            self.vred.append(diag(cov(res.T))/diag(cov(test_odat.T)))
            self.all_A.append(A)
            self.rv.append(var(res)/var(test_odat))
            
            # compute relative volume:
            # actually, another comparison might be usefull: compute along the
            # SAME projection, e.g. the principal component linear hull of odat!
            s_odat = svd(test_odat,full_matrices = False, compute_uv = False)
            s_pred = svd(res,full_matrices = False, compute_uv = False)
            volRatio =  reduce(lambda x,y: x*y, s_pred[:dimVol]) / \
                        reduce(lambda x,y: x*y, s_odat[:dimVol]) 
                      # /sqrt(N) cancels out
            self.relVol.append(volRatio)            
            
            #vredb.append(diag(cov(resb.T))/diag(cov(test_odat2.T)))
            #vredpc.append(diag(cov(dot(u0.T,res.T) ))/diag(cov(dot(u0.T,test_odat.T) )))        
            
        
    def fitmdl_step(self,nidx = 500,nrep = 500, psec_in = 0, psec_out = None,
                    dimVol = None, rcond=0.03, nullMdl = None,useSym = False):
        """
        similar to fitmdl, except that the Poincare-section for the input data
        (psec_in) and the output-data (psec_out) are within a single stride.
        
        if psec_out is not given, it is set to psec_in + nps/2
        
        if psec_out is < psec_in (mod nps), then the subsequent step is taken
        
        fits a single stride; usually from phase x to phase x+pi
        dimVol: how many singular values should be taken into account when comparing
            the relative volume? Default: all
            
        useSym: Apply a symmetrization of the data using the self.sym - matrix            
        """
        
        # first: compute in_dat and out_dat
        assert type(psec_in) is int, 'psec_in must be of type int (from 0 to nps)'
        if psec_out is None:
            psec_out = psec_in + int(self.nps/2)
        
        psec_out = psec_out % self.nps        
        
        # if psec_out would be in the next step -> shift out, cut in.
        # actually, for an ar-system this should not be relevant due to
        # time symmetrie of autocorrelation function, however, here we might
        # face a substantially different system.
        
        oneStrideAhead = False
        if psec_out < psec_in:  
            oneStrideAhead = True

        data_in = self.data1D[:,psec_in::self.nps] if not oneStrideAhead else \
                  self.data1D[:-1,psec_in::self.nps]
        
        data_out_pre = self.data1D[:,psec_out::self.nps] if not oneStrideAhead else \
                       self.data1D[1:,psec_out::self.nps]                  
                  
        if useSym:
            data_out = dot(self.sym,data_out_pre.T).T
        else:
            data_out = data_out_pre
            
                  
        assert data_in.shape[0] == data_out.shape[0], 'in- and out-data shape mismatch'
        
        self.all_A = []
        self.relVol = [] #relative volume after prediction
        self.vred = []        
        self.rv = [] # relative variance after prediction
        
        in_idx = arange(data_in.shape[0])
        #gc.collect()
        
        for rep in range(nrep):
            # create input and output data for regression, dependend on
            # null model
            if nullMdl is None:
                pred_idx = in_idx[randint(0,len(in_idx),nidx)]                
                idat = data_in[array(list(pred_idx),dtype=int),:]
                odat = data_out[array(list(pred_idx),dtype=int),:]
            elif nullMdl == 'inv': # acutally, this could be deleted
                pred_idx = in_idx[randint(len(in_idx),nidx)]
                idat = data_in[array(list(pred_idx),dtype=int),:]
                odat = data_out[array(list(pred_idx),dtype=int),:]
            elif nullMdl == 'rand':
                pred_idx = in_idx[randint(0,len(in_idx),nidx)]
                pred_idx_out = in_idx[randint(0,len(in_idx),nidx)]
                idat = data_in[array(list(pred_idx),dtype=int),:]
                odat = data_out[array(list(pred_idx_out),dtype=int),:]            
            else:
                raise ValueError, 'Error: Null model type not understood'
            
            test_idx = set(in_idx).difference(pred_idx)                        
            A = lstsq(idat,odat,rcond = rcond)[0]
            test_idat = data_in[array(list(test_idx),dtype=int),:]
            test_odat = data_out[array(list(test_idx),dtype=int),:]
            pred = dot(test_idat,A)            
            res = test_odat - pred                                
            self.vred.append(diag(cov(res.T))/diag(cov(test_odat.T)))
            self.all_A.append(A)
            self.rv.append(var(res)/var(test_odat))
            
            # compute relative volume:
            # actually, another comparison might be usefull: compute along the
            # SAME projection, e.g. the principal component linear hull of odat!
            s_odat = svd(test_odat,full_matrices = False, compute_uv = False)
            s_pred = svd(res,full_matrices = False, compute_uv = False)
            volRatio =  reduce(lambda x,y: x*y, s_pred[:dimVol]) / \
                        reduce(lambda x,y: x*y, s_odat[:dimVol])
                      # /sqrt(N) cancels out
            self.relVol.append(volRatio)         

    
    
