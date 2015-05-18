# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:27:05 2011

@author: moritz
"""


# import pickle
import cPickle as pickle # for speedup. if this leads to problems -> use pickle

import gzip
from pylab import (vstack, hstack, mean, mod, pi, array, std, sort, arange, 
        find, zeros, argmin, gradient, linspace, interp, polyfit, polyval)
# local modules
import misc as mi
#import FDatAn as fda
# other modules
import re
import os
import scipy.io as sio
import hashlib
import warnings
import numpy as np # required for datatype np.ndarray
from copy import deepcopy

# hint: use bz2 instead of gzip
import cPickle
import marshal
import inspect
import shelve
import tempfile

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

def get_data(sid, ttype, datdir = './SLIP_params', file_name_0 = 'params3D_', 
            detrend_hwlen = 15, detrend = True, normalize = False,
            exclude_ws = ['params3D_s8t1r2.dict',]):
    """
    returns a tuple (stateL, stateR, paramsL, paramsR, addInfo),
    such that the SLIP model that would start at stateL[0] with parameters
    paramsL[0] would result at stateR[0], and coninuing this would result in 
    the states stateL[1], stateR[1], stateL[2], stateL[3], ...
    A break might occur at these steps where different trials are concatenated.
    
    addInfo is a dict containing additional information such as the mass,
    minimal vertical excursion, step duration (required for SLIP parameter
    calculation).
    
    state is [k, alpha, L0, beta, dE]
    
    input: sid: subject-id [currently: 1,2,3,4,6(?),7,8]
           ttype: trial-type [currently: 1-free running, 2-metronome running]
           datdir: directory where to look for stored data files
           fileName0: common beginning of filenames
           detrend_hwlen: (half) window length of centered moving average
                          detrending
           detrend: whether or not to detrend
           normalize: whether or not to normalize parameters
           exclude_ws: list of filenames that must not be loaded
    """
    
    param_type = 'ESLIP_params' # alternatively: 'SLIP_params'

    # import other, locally defined library (FDatAn -> misc)
    #import misc as fda

    
    pattern = file_name_0 + ('s%it%ir' % (sid, ttype))
    files1 = [x for x in os.listdir(datdir) if x[:len(pattern)] == pattern and 
                x not in exclude_ws]
    files1.sort()
    reps = [x[-6] for x in files1]
    
    states_r = []
    states_l = []
    params_r = []
    params_l = []
    
    masses1 = []
    phases1 = []
    phases_r = []
    phases_l = []
    delta_e = []
    time_1 = []
    ymin = []
    ymin_r = []
    ymin_l = []
    delta_er = []
    delta_el = []
    time_r1 = []
    time_l1 = []
    
    n_in_ws = []
    #allStates = []
    avg_params_l = []
    avg_params_r = []
    for fname in files1:
        slip_params = mload(datdir + os.sep + fname)
        # normalization: k' = k*l0/(m*g)
        # process workspaces: 
        #  - make non-dimensional (following Blickhan+Full 1993):
        #    k' = k*l0/(m*g)
        #  - detrend [::2,:] !! detrend left and right steps separately, 
        #    to account for gait asymmetry
        #  - sort left / right: 0 < phi < pi: subsequent touchdown is left,
        #                     pi < phi < 2pi: subsequent touchdown is right.
        phases1.append(slip_params['phases'])
    
        all_params = slip_params[param_type]
        masses1.append(slip_params['mass'])
        phases1.append(slip_params['phases'])
        all_states = vstack( [slip_params['y0'], 
                                  slip_params['vx'],
                                  slip_params['vz'], ] ).T[:all_params.shape[0]]
        delta_e.append(slip_params['dE'])
        time_1.append(slip_params['T_exp'])
        ymin.append(slip_params['ymin'])
        offset_r = 0
        offset_l = 0
        # enforce that every dataset starts with a left step
        # -> important for regression L -> R, R -> L, because only then it is
        # guaranteed that stateR[i] follows stateL[i], and stateL[i+1] follows
        # stateR[i]    
        if mean(mod(phases1[-1][::2], 2.*pi)) < pi:
            # starts with left step
            offset_r = 1
        else:
            # starts with right step        
            offset_l = 1
            offset_r = 2
        nStrides1 = all_params[offset_r::2, :].shape[0]
        states_r.append(all_states[offset_r::2, :])
        states_l.append(all_states[offset_l::2, :][:nStrides1, :])
        params_r.append(all_params[offset_r::2, :])
        params_l.append(all_params[offset_l::2, :][:nStrides1, :])
        delta_er.append(slip_params['dE'][offset_r::2])
        delta_el.append(slip_params['dE'][offset_l::2][:nStrides1])
        time_r1.append(slip_params['T_exp'][offset_r::2])
        time_l1.append(slip_params['T_exp'][offset_l::2][:nStrides1])    
        ymin_r.append(slip_params['ymin'][offset_r::2][:nStrides1])    
        ymin_l.append(slip_params['ymin'][offset_l::2][:nStrides1])
        phases_r.append(phases1[-1][offset_r::2][:nStrides1])
        phases_l.append(phases1[-1][offset_l::2][:nStrides1])
        # normalize parameters
        l0r = mean(params_r[-1][:, 2])    
        l0l = mean(params_l[-1][:, 2])
        if normalize:
            params_r[-1][:, 0] = params_r[-1][:, 0] * l0r / (masses1[-1] * 9.81)
            params_l[-1][:, 0] = params_l[-1][:, 0] * l0l / (masses1[-1] * 9.81)
            params_r[-1][:, 4] = params_r[-1][:, 4] * l0r / (masses1[-1] * 9.81)
            params_l[-1][:, 4] = params_l[-1][:, 4] * l0l / (masses1[-1] * 9.81)
            params_r[-1][:, 2] = params_r[-1][:, 2] / l0r
            params_l[-1][:, 2] = params_l[-1][:, 2] / l0l        
            params_r[-1][:, 5] = params_r[-1][:, 5] / l0r
            params_l[-1][:, 5] = params_l[-1][:, 5] / l0l        
            delta_e[-1] = delta_e[-1]/( masses1[-1] * 9.81 * ( l0r + l0l) / 2.)
            delta_er[-1] = ( delta_er[-1]/( masses1[-1] * 9.81 
                            * ( l0r + l0l) / 2.))
            delta_el[-1] = ( delta_el[-1]/( masses1[-1] * 9.81 
                            * ( l0r + l0l) / 2.))
        
        params_r[-1][:, 4] = delta_er[-1]
        params_l[-1][:, 4] = delta_el[-1]
        avg_params_l.append( mean( params_r[-1], axis=0))
        avg_params_l.append( mean( params_l[-1], axis=0))       
        if detrend:
            params_r[-1] = mi.dt_movingavg( params_r[-1], detrend_hwlen)
            params_l[-1] = mi.dt_movingavg( params_l[-1], detrend_hwlen)

    
    n_in_ws = [len(sl) for sl in states_l]
    state_r = vstack(states_r)
    params_r = vstack(params_r)[:, :5]
    state_l = vstack(states_l)
    params_l = vstack(params_l)[:, :5]
    
    add_info = {'time_left' : hstack(time_l1),
               'time_right' : hstack(time_r1),
               'ymin_left' : hstack(ymin_l),
               'ymin_right' : hstack(ymin_r),
               'all_masses' : masses1,
               'mass' : mean(masses1),
               'avg_params_l' : avg_params_l,
               'avg_params_r' : avg_params_r,
               'n_in_ws' : n_in_ws,
               'phases_r' : hstack(phases_r),
               'phases_l' : hstack(phases_l),
               'reps': reps,
                }
    
    return state_l, state_r, params_l, params_r, add_info
    
class adat(object):
    """
    defines a data load and store object that accesses the database or uses a
    cache if present. 
    Here, only data at the apices are given.

    """

    def __init__(self, sid, tid=1, markers=['anl', ], loadData=True):
        """
        Initializes the object and sets the corresponding data.
        
        -----------
        Parameters:
        -----------
        sid : *integer*
            The subject's id to set the data for. Currently supported:
            [1,2,3,4,7,8].
        tid : *integer*
            The trial-type id (1: normal running, 2: metronome running, 3-6:
            some disturbance experiments)
        markers: *list*
            a list of markers to be read. For all markers, x-, y- and z-
            directions will be read (relative to CoM), for both left and right
            side. The CoM vertical position and speed will read additionally.

        """
        import misc
        self.LOG = misc.logger()
        self.sid = sid
        self.tid = tid
        self.markers = markers
        self.sthresh = 6. # deviation from limit cycle (in std-dev's) when a
            # stride is discarded
        self.sym = None
        if loadData:
            self._loadData()
            self.sym = self._sym()



    def _loadData(self, ):
        """
        **internally used function**

        Here, the data is loaded when __init__ is called. 

        """
        self.LOG('setting data')
        cfilename = ''.join([ ('s%it1_' % self.sid), ''.join(self.markers),
            '_symd.dict'])
        cfilefound = False

        if cfilename in os.listdir('cache'):
            res = mload(os.sep.join(['cache', cfilename]))
            if not res['markers'] == self.markers:
                self.LOG('Cache file is inconsistent - accessing database')
            else:
                self.dl = res['dl']
                self.dr = res['dr']
                self.labels = res['labels']
                self.ndim = self.dl.shape[1]
                cfilefound = True

        if not cfilefound:
            import subjData as sd
            a = sd.sdata()

            cfilefound = False
            self.LOG('No valid cache file - accessing database')

            selection = []
            for elem in self.markers:
                selection.append('l_' + elem + '_x - com_x')
                selection.append('r_' + elem + '_x - com_x')
                selection.append('l_' + elem + '_y - com_y')
                selection.append('r_' + elem + '_y - com_y')
                selection.append('l_' + elem + '_z - com_z')
                selection.append('r_' + elem + '_z - com_z')
            selection.extend( ['CoM_x', 'CoM_y', 'CoM_z', ])

            a.selection = selection
            ndim0 = len(selection)
            sl, sr, pl, pr, idict = get_data(self.sid, self.tid, 
                    datdir='./SLIP_params2/', detrend=False)
            dat_l, dat_r = a.get_kin_from_idict(self.sid, 1, idict)
            rmslice = array(list(set(arange(2 * ndim0)) - set([ndim0 - 2,
                ndim0 - 3])))
            rmslice.sort()

            labels = [x[:-8] for x in selection[:-3]]
            labels.append('com_z')
            labels.extend(['v_' + x[:-8] for x in selection[:-3]])
            labels.extend(['v_' + x for x in selection[-3:]])

            ndim = len(rmslice)
            dat_l = hstack(dat_l).T[:, rmslice]
            dat_r = hstack(dat_r).T[:, rmslice]
            #import misc as fda
            dl = mi.dt_movingavg(dat_l, 30)
            dr = mi.dt_movingavg(dat_r, 30)

# find 'good' indices - remove strides with very high data amplitude
            dln = dl / std(dl, axis=0)
            drn = dr / std(dr, axis=0)

            mal = array([max(abs(x)) for x in dln])
            mar = array([max(abs(x)) for x in drn])

            badidx = set(find(mal > self.sthresh)) | set(
                    find(mar > self.sthresh))
            goodidx = sort(list(set(arange(dln.shape[0])) - badidx))

# attention: strictly speaking, these data will now be inconsistent. There are
# some data points whose successors have been removed - consequently, their
# (new) successors will not be their true successors, and a regression on these
# data points will produce inconsistent results. However, when only a small
# fraction of points is removed, this effect should be rather small.
            self.dl = dl[goodidx, :]
            self.dr = dr[goodidx, :]
            self.ndim = ndim
            self.labels = labels

# optional step - rescale all velocities. A factor of ~11. leads to roughly
# similar variance in positions and velocities across all subjects.
# This does not alter results at all.
            #vscale = 1. / 11. 
            #dl[:, ndim0 - 2:] /= 11.
            #dr[:, ndim0 - 2:] /= 11.

# store results if they were not cached
            self.LOG('storing data in cache file')
            res = {'dl' : self.dl, 'dr' : self.dr, 'markers' : self.markers,
                    'labels' : self.labels}
            msave(os.sep.join(['cache', cfilename]), res)

    def _sym(self,):
        """
        Returns the 'symmetry' of the data. More precisely, it returns a matrix
        *S* with the properties that R ~ *S* L *S*, where R and L denote a map
        that maps the data from a left apex to a right apex.

        """

# The assumption of the symmetry is as follows: "A right step is a mirrored
# version of a left step. It is reflected at the sagittal plane. Left and right
# extremities are exchanged."
# This leads to the definition of the symmetry matrix:
# It exchanges the coordinates of left and right limb. Further, lateral
# positions and velocities (coordinate 'x') switch sign. (I am not sure that
# the latter part, changing sign, has any effect in a linear system.)

        sym = zeros((self.ndim, self.ndim), dtype=float)
# first: account for all 'markers'
        for nm, mname in enumerate(self.markers):
            bp = nm * 6 # base position of coordinate
            # x-direction (lateral):
            sym[bp, bp + 1] = -1
            sym[bp + 1, bp] = -1
            # y-direction (running direction):
            sym[bp + 2, bp + 2 + 1] = 1
            sym[bp + 2 + 1, bp + 2] = 1
            # z-direction (running direction):
            sym[bp + 4, bp + 4 + 1] = 1
            sym[bp + 4 + 1, bp + 4] = 1

# second: account for CoM height: leave it as is
        sym[len(self.markers) * 6, len(self.markers) * 6] = 1

# third: account for all velocities corresponding to the 'markers' (same order)
        for nm, mname in enumerate(self.markers):
            bp = nm * 6 + 6 * len(self.markers) + 1 # base position for velocities
            # x-direction (lateral):
            sym[bp, bp + 1] = -1
            sym[bp + 1, bp] = -1
            # y-direction (running direction):
            sym[bp + 2, bp + 2 + 1] = 1
            sym[bp + 2 + 1, bp + 2] = 1
            # z-direction (running direction):
            sym[bp + 4, bp + 4 + 1] = 1
            sym[bp + 4 + 1, bp + 4] = 1

# fourth: the CoM velocities
        sym[-3, -3] = -1
        sym[-2, -2] = 1
        sym[-1, -1] = 1
        return sym


def getcache(def_dict, cachedir='./cache', checkonly=False):
    """
    tries to read data defined in def_dict from cache.

    Parameters
    ----------
    def_dict : *dict*
        the definition of what to look for.
        must contain the key "type". All other keys are optional and are used
        to define the file, e.g. 'subject_id', 'trial_id' , 'repetition' ...

    cachedir : *str*
        (optional) directory to look for the cache file

    checkonly : *bool*
        (optional) if True, check only if cache file exists, do not load
        

    Returns
    -------
    the stored content
        - *OR* - 
    *True* if checkonly=True and the cached file is found
        - *OR* -
    *None* if no stored content could be found
    """
    hobj = hashlib.sha1()
    hobj.update(pickle.dumps(def_dict))
    fname = def_dict['type'] + '_' + hobj.hexdigest()
    if fname in os.listdir(cachedir):
        if checkonly:
            return True
        else:
            return mload(cachedir + os.sep + fname)
    else:
        return None


def setcache(def_dict, content, cachedir='./cache'):
    """
    sets the content, definded by def_dict, in the cache

    ==========
    Parameter:
    ==========
    def_dict : *dict*
        the definition of what to look for.
        must contain the key "type". All other keys are optional and are used
        to define the file, e.g. 'subject_id', 'trial_id' , 'repetition' ...

    content : *any*
        any content that should be stored. must be pickle-able

    cachedir : *str*
        (optional) directory to store the cache file
        
    ========
    Returns:
    ========
    (nothing)
    """
    hobj = hashlib.sha1()
    hobj.update(pickle.dumps(def_dict))
    fname = def_dict['type'] + '_' + hobj.hexdigest()
    msave(cachedir + os.sep + fname, content)


class KinData(object):
    """ Object to handle the MMCL kinematic data """
    
    def __init__(self, data_dir=None):
        """ Returns an object that contains kinematic data 

        :args:
            data_dir (str): directory relative to current path to look 
            for data. Default: ../data/2011-mmcl_mat

        """
        self.subject = None
        self.ttype = None
        self.reps = None
        self.raw_dat = []
        self.selection = []
        self._loaded_selection = None
        if data_dir is None:
            self.data_dir = '../data/2011-mmcl_mat'
        else:
            self.data_dir = data_dir
        
    def load(self, subject, ttype, reps=None, quickreload=True):
        """
        load the kinematic data for a specific subject and trial type
        
        :args:
            subject (int): the subject id
            ttype (int): the trial type (1: free running, others: 2-6)
            reps (tuple): the repetitions to load. 'None' indicates all 
                available repetitions.
            quickreload (bool): If true, do not reload data if request
                corresponds to already loaded data.
        :returns: 
            None
        :raises:
            OSError if an invalid subject id is given

        """
        
        # is reloading necessary?
        if (quickreload and subject == self.subject and ttype == self.ttype and reps == self.reps 
            and self.selection == self._loaded_selection):
            return
        
        path = os.sep.join([self.data_dir, 'subj' + str(subject), 'kin'])
        all_files = os.listdir(path)
        files = [fn for fn in all_files if fn.startswith('kin_t%i' % ttype) and
                                fn.endswith('.mat')]
        if reps is not None:
            files_sel = []
            for fn in files:
                file_rep = re.search('r([0-9]).mat', fn)
                if file_rep:
                    rep = int(file_rep.groups()[0])
                    if rep in reps:
                        files_sel.append(fn)
                        
        else:
            files_sel = files
        files_sel.sort()
        found_reps = [int(re.search('r([0-9]+)[.]mat', fn).groups()[0]) for fn in files_sel]
        
        #load data
        self.raw_dat = []
        for fn in files_sel:
            dat = sio.loadmat(os.sep.join([path, fn]))
            self.raw_dat.append(dat)

        # load forces
        path_force = os.sep.join([self.data_dir, 'subj' + str(subject), 'frc'])
        files_frc = ['frc' + fn[3:] for fn in files_sel]
        self.raw_frc = []
        for fn in files_frc:
            self.raw_frc.append(sio.loadmat(os.sep.join([path_force, fn])))
        
        self.subject = subject
        self.ttype = ttype
        if reps == None:
            self.reps = found_reps
        else:
            self.reps = reps
        self._loaded_selection = self.selection
        
    def make_1D(self, nps, phase='phi2', fps=250, phases_list=None):
        """
        interpolate the data with <nps> frames per stride

        *Note* The <object>.selection attribute must contain a valid list of selection
            items. A selection item can either be a single coordinate (e.g. 'com_z', 
            'r_anl_y') or the difference between two coordinates (e.g. 'l_anl_y - com_y')

        :args:
            nps (int): how many frames per stride should be sampled
            phase (str): 'phi1' or 'phi2' : which phase to use. Note: previous data mostly
                used 'phi2'.
            fps (int): how many frames per second do the original data have? This is only required
                for correct scaling of the velocities.
            phases_list (list of list of int): If present, use given phases
                instead of predefined sections. Data will *not* be sampled
                exactly at the given phases; instead there will be _nps_
                sections per stride, and strides start (on average only) at the
                given phases.
            
        """
        
        if len(self.raw_dat) == 0:
            print "No data loaded."
            return
        if len(self.selection) == 0:
            print "No data selected. Set <object>.selection to something!"
            
        
        # gather phases for each trial:
        i_phases =[]
        if not phases_list: # typical call: omit first and last strides;
            # strides go from phase 0 to phase 2pi (exluding 2pi)
            for raw in self.raw_dat:
                phi = raw[phase].squeeze()
                # cut lower phase and upper phase by ~4pi each
                first_step = (phi[0] + 6. * pi) // (2. * pi)
                last_step = (phi[-1] - 4. * pi) // (2. * pi)
                i_phases.append(linspace(first_step, last_step + 1, (last_step - first_step + 1) * nps,
                     endpoint=False) * 2. *pi)
        else:
            # phases are explicitely given
            avg_phasemod = mean([mean(mod(elem, 2.*pi)) for elem in
                phases_list])
            for elem in phases_list:
                firstphase = (elem[0] // (2. * pi)) * 2.*pi + avg_phasemod
                lastphase = (elem[-1] // (2. * pi)) * 2.*pi + avg_phasemod
                i_phases.append(linspace(firstphase, lastphase + 2*pi, len(elem)
                    * nps, endpoint=False))

        self.i_phases = i_phases
        
        # walk through each element of "selection"
        all_pos = []
        all_vel = []
        for elem in self.selection:
            items = [x.strip() for x in elem.split('-')] # 1 item if no "-" present
            dims = []
            markers = []
            for item in items:                
                if item.endswith('_x'):
                    dims.append(0)
                elif item.endswith('_y'):
                    dims.append(1)
                elif item.endswith('_z'):
                    dims.append(2)
                else:
                    print "invalid marker suffix: ", item
                    continue
                markers.append(item[:-2])
            
            all_elem_pos = []
            all_elem_vel = []
            for raw, phi in zip(self.raw_dat, self.i_phases):
                if len(items) == 1:
                    # only a single marker
                    dat = raw[markers[0]][:, dims[0]]

                else:
                    # differences between two markers
                    dat = raw[markers[0]][:, dims[0]] - raw[markers[1]][:, dims[1]]
                
                all_elem_pos.append(interp(phi, raw[phase].squeeze(), dat))
                all_elem_vel.append(interp(phi, raw[phase].squeeze(), gradient(dat) * fps))
            all_pos.append(hstack(all_elem_pos))
            all_vel.append(hstack(all_elem_vel))
            
        dat_2D = vstack([all_pos, all_vel])
        return mi.twoD_oneD(dat_2D, nps)
        
    def get_kin_apex(self, phases, return_times = False):
        """
        returns the kinematic state at the apices which are close to the given phases. Apex is re-calculated.
        
        :args:
            self: kin object (-> later: "self")
            phases (list): list of lists of apex phases. must match with length of "kin.raw_data". 
               The n'th list of apex phases will be assigned to the nth "<object>.raw_data" element.
            return_times (bool): if true, return only the times at which apex occurred.
    
        :returns:
           if lr_split is True:
              [[r_apices], [l_apices]]
           else:
              [[apices], ]
              where apices is the kinematic (from <object>.selection at the apices *around* the given phases.
              *NOTE* The apices themselves are re-computed for higher accuracy.
    
        """
        
        all_kin = []
        all_kin_orig = self.get_kin()
        all_apex_times = []
        if len(self.raw_dat) != len(phases):
            raise ValueError("length of phases list does not match number of datasets")
        for raw, phase, kin_orig in zip(self.raw_dat, phases, all_kin_orig):
            kin_apex = []
            kin_time = arange(len(raw['phi2'].squeeze()), dtype=float) / 250.
            # 1st: compute apex *times*
            apex_times = []
            for phi_apex in phase:
                # 1a: get "rough" estimate
                idx_apex = argmin(abs(raw['phi2'].squeeze() - phi_apex))
                # 1b: fit quadratic function to com_y
                idx_low = max(0, idx_apex - 4)
                idx_high = min(raw['com'].shape[0] - 1, idx_apex + 4)
                com_y_pt = raw['com'][idx_low:idx_high + 1, 2]            
                tp = arange(idx_high - idx_low + 1) # improve numerical accuracy: do not take absolute time
                p = polyfit(tp, com_y_pt, 2) # p: polynomial, highest power coeff first
                t0 = -p[1] / (2.*p[0]) # "real" index of apex (offset is 2: a value of 2
                           # indicates that apex is exactly at the measured frame
                t_apex = kin_time[idx_apex] + (t0 - 4.) / 250.
                apex_times.append(t_apex)
            
            if return_times:
                all_apex_times.append(array(apex_times))		    
            else:
                # 2nd: interpolate data
                dat = vstack([interp(apex_times, kin_time, kin_orig[row, :]) for row in arange(kin_orig.shape[0])])
                all_kin.append(dat)

        if return_times:
	    return all_apex_times

        return all_kin

    def get_kin(self, fps=250.):
        """
        returns a list of the selected kinematics (one list item for each repetition)
        
        :args:
            self: kin object
            fps (float, default 250): sampling frequency. Required to correctly compute the velocities.
    
        :returns:
            a list. Each element contains the selected (-> self.selection) data with corresponding 
               velocities (i.e. 2d x n elements per item)
        """
        # walk through each element of "selection"
        all_pos = []
        all_vel = []
        for raw in self.raw_dat:
            curr_pos = []
            curr_vel = []
            for elem in self.selection:
                items = [x.strip() for x in elem.split('-')] # 1 item if no "-" present
                dims = []
                markers = []
                for item in items:                
                    if item.endswith('_x'):
                        dims.append(0)
                    elif item.endswith('_y'):
                        dims.append(1)
                    elif item.endswith('_z'):
                        dims.append(2)
                    else:
                        print "invalid marker suffix: ", item
                        continue
                    markers.append(item[:-2])
                            
                if len(items) == 1: # single marker
                    curr_pos.append(raw[markers[0]][:, dims[0]])
                    curr_vel.append(gradient(raw[markers[0]][:, dims[0]]) * fps)
                else: # difference between two markers
                    curr_pos.append(raw[markers[0]][:, dims[0]] - raw[markers[1]][:, dims[1]])
                    curr_vel.append(gradient(raw[markers[0]][:, dims[0]] - raw[markers[1]][:, dims[1]]) * fps)
    
            all_pos.append(vstack(curr_pos + curr_vel))
            all_vel.append(vstack(curr_vel))  
            
        return all_pos        


class workspace(object):
    """ This class provides a basic workspace, which in future versions
    should also provide save / reload options """
    def __init__(self, src = None):
        """
        Initialize the class.
        Defines which datatypes of members should be stored. Overwrite attribute
        _saveables to change this.

        :args: 
            src (dict or filename): initialize from given dictionary or
            filename
        """
        if type(src) == dict:
            self.__dict__.update(src)
        elif type(src) == str:
            self.restore(src)
        elif hasattr(src, 'read'):
            if callable(src.read):
                self.restore2(src)
        self._saveables = [list, float, tuple, int, bool, complex, dict, str,
                np.ndarray, saveable]
        self._saveables_doc = 'list of types that can be stored'
        
    def __repr__(self):
        return self.display(returnString=True)

    def __getitem__(self, arg):
        
        if arg.endswith('*'): # special case: return an iterator
            refkey = arg[:-1]
            res = []
            for key, val in self.__dict__.iteritems():
                if key.startswith(refkey):
                    res.append((key, val))
            return res
        else:
            try:
                return getattr(self, arg)
            except AttributeError:
                setattr(self, arg, saveable())
                return self[arg]

    def __setitem__(self, arg, value):
        setattr(self, arg, value)

    def __iter__(self):
        for key, val in self.__dict__.iteritems():
            if not(str(key).startswith('_')):
                yield key, val

    def update(self, other):
        """
        updates the saveable object from another saveable object
        """
        for key, val in other:
            self[key] = val


    def dumps(self):
        """
        returns a pickle dump string for the object
        """
        return pickle.dumps(self)

    def dump(self, fdesc):
        """
        pickle-dumps the object to the file <fdesc>.
        *NOTE* if you want to have a compressed file, just pass a compressed
        file descriptor (a gzip file)
        
        :args:
            fdesc (file descriptor): the file descriptor to put data in.

        """
        pickle.dump(self, fdesc)
        #fdesc.write(self.dumps())

    def store(self, filename, excludes=[]):
        """
        Stores all relevant information of the object into the file 
        specified by filename. Can be loaded using the "restore" method.
        
        :args:
            filename (string): the filename of the file where to store
                the object
            excludes (list of strings): exclude items with names listed in here. 
        :returns:
            (None)
        """
        
        if not hasattr(self, '_saveables'):
            warnings.warn("_saveables attribute not found - using default")
            saveables = [list, float, tuple, int, bool, complex, dict, str,
                    np.ndarray, saveable]
        else:
            saveables = self._saveables

        ddict = {}
        ddict.update([(key, val) for key, val in self.__dict__.iteritems() 
                if type(val) in saveables and key not in excludes])
        msave(filename, ddict)
        

    def restore(self, filename):
        """
        Restores all relevant information of the object from the file 
        specified by filename (which is usually created using the
        "store" method.)
        
        :args:
            filename (string): the filename of the file from which the
                object should be restored
        :returns:
            (None)
        """
        
        ddict = mload(filename)
        self.__dict__.update(ddict)

    def restore2(self, fdesc):
        """
        like restore but reads a pickled file

        :args:
            fdesc (file descriptor): the file where the pickled data is in
        """
        tmp = pickle.load(fdesc)
        for key, val in tmp:
            self[key] = val


    def display(self, returnString=False, extra_sep=""):
        """
        displays (prints) the data stored in the saveable object. If the
        datatype is known (list, tuple, array), additional information is
        given.

        :args:
            returnString (bool): wether to print or to return the string
            extra_sep (string): extra separator between name, type and doc
        :returns:
            (None) or String
        """
        def pprint(string, length):
            l = len(string)
            return ''.join([string, " " * (length - l)])

        olines = []
        keys = [elem for elem in self.__dict__.keys()]
        keys.sort()
        maxlen = max([len(x) for x in keys])
        for k in keys:
            if str(k).startswith('_'):
                continue
            pt = []
            if isinstance(k, basestring):
                if k.endswith('_doc') and k[:-4] in keys:
                    continue
            pt.append(''.join( [k, " " * (maxlen + 2 - len(k)), extra_sep])) 
            if type(getattr(self, k)) == np.ndarray:
                pt.append(
                    pprint("array " + str(getattr(self, k).shape), 16))
            elif isinstance(getattr(self, k), saveable):
                pt.append(pprint("<workspace object>", 16))
            elif type(getattr(self, k)) == str:
                pt.append(
                    pprint("string  " + getattr(self,k)[:9], 16))
            elif type(getattr(self, k)) == list:
                pt.append(
                    pprint("list (" + str(len(getattr(self, k))) + ")", 16))
            elif type(getattr(self, k)) == tuple:
                pt.append(
                    pprint("tuple (" + str(len(getattr(self, k))) + ")", 16))
            elif type(getattr(self, k)) == bool:
                pt.append(
                    pprint("bool  " + str(getattr(self, k)) , 16))
            elif type(getattr(self, k)) == int:
                pt.append(
                    pprint("int  " + str(getattr(self, k)) , 16))
            elif issubclass(type(getattr(self, k)), float):
                pt.append(
                    pprint(("float  %1.5e" % getattr(self, k)) , 16))
            else:
                pt.append(
                    pprint(str(type(getattr(self, k))), 16))
            olines.append(''.join(pt))
            olines[-1] = olines[-1] + " "

            if ''.join([k, '_doc']) in keys:
                olines[-1] = olines[-1] + extra_sep + getattr(self, ''.join([k, '_doc']))

        if returnString:
            return '\n'.join(olines)
        else:
            print '\n'.join(olines)


# define an alias for "saveable"
#workspace = saveable
saveable = workspace # alias for old name (backward compatibility)

def read_kistler(fname):
    """
    reads the output text files from Kistler force plate data.

    :args:
        fname (str): file name of the data file (typically .txt export from
            Kistler bioware)

    :returns:
        dat (dict): dictionary containing the data ("Matlab(r)"-workspace)
    """
    data = []
    fieldnames = []

    desc_found = False
    n_past_desc = 0
    with open(fname) as f:
        for line in f:
            if desc_found:
                n_past_desc += 1
            elif 'description' in line.lower():
                desc_found = True

            if n_past_desc == 1:
                fieldnames = [fn.strip() for fn in line.split('\t')]
                if 'time' in fieldnames[0].lower(): # remove invalide name
                    fieldnames[0] = 'time'

            elif n_past_desc == 2:
                units = line.split('\t') # actually - this is ignored
            elif n_past_desc > 2:
                try:
                    numbers =[float(elem.replace(',','.')) for elem in
                        line.split('\t')]
                    data.append(numbers)
                except ValueError:
                    pass
    
    if not desc_found:
        raise ValueError('Line with "Description" not found - does not appear'
        ' to be a valid file!')

    data = vstack(data)
    d = {}
    for nr, fn in enumerate(fieldnames):
        d[fn] = data[:, nr]

    return d


# build common dataset

def normalize_m(SlipData):
    """
    normalizes the SlipData structure to the mass, that is, spring stiffness
    and energy is divided by mass, and mass is set to 1.

    :args:
        SlipData (io.Struct or saveable): data structure (e.g. from file)

    :returns:
        SlipData (same structure)
    """
    res = deepcopy(SlipData)
    res.ESLIP_params[:,0] /= res.mass
    res.ESLIP_params[:,4] /= res.mass

    # modify P only if it is a copy (not a view) of ESLIP_params!
    if id(res.P) != id(res.ESLIP_params):
        res.P[:,0] /= res.mass
        res.P[:,4] /= res.mass

    res.mass = 1.
    return res

def build_dataset(k, SlipData, allow_nonCoM=False, dt_window=30,
        dt_median=False):
    """
    This function builds a common dataset that combines SlipData 
    and corresponding kinematic data (matches the corresponding steps).

    :args:
        k (KinData)
        SlipData (list)
        allow_nonCoM (bool) : whether the first three values in k.selection may be other 
            than ['com_x', 'com_y', 'com_z']
        dt_window (int): (half) detrending window length
        dt_median (bool): wether to use median detrending (or mean)

    :returns:
        kdata (saveable object): a saveable object that contains the kinematic data with the 
            following elements:
            all_IC_r = vstack(all_IC_r) # SLIP apex data (interpolated)
            all_IC_l = vstack(all_IC_l) # SLIP apex data (interpolated)
            all_IC_rc     # kinematic apex data (close to SLIP apex data)
            all_IC_lc     # kinematic apex data (close to SLIP apex data)
            all_param_r = vstack(all_param_r)
            all_param_l = vstack(all_param_l)
            all_kin_r = vstack(all_kin_r)
            all_kin_l = vstack(all_kin_l)
            all_phases_r
            all_phases_l
            param_right
            param_left
            IC_right
            IC_left
            masses
            
            TR = d3orig.TR
            TL = d3orig.TL
            yminR = d3orig.yminR
            yminL = d3orig.yminL
            kin_labels
        
    """
    
    if k.selection[0].lower() != 'com_x' and k.selection[1].lower() != 'com_y' and k.selection[2] != 'com_z':
        raise ValueError("ERROR: the first elements of k.selection have to be: 'com_x', 'com_y', 'com_z'")
        
    res = saveable()
    
    indices_right = [mi.upper_phases(d.phases[:-1], sep=0, return_indices=True) for d in SlipData]
    indices_left = [mi.upper_phases(d.phases[:-1], sep=pi, return_indices=True) for d in SlipData]
    
    param_right = [ vstack(d.P)[idxr, :] for d, idxr in zip(SlipData, indices_right)]
    param_left = [ vstack(d.P)[idxl, :] for d, idxl in zip(SlipData, indices_left)]
    IC_right = [ vstack(d.IC)[idxr, :] for d, idxr in zip(SlipData, indices_right)]
    IC_left = [ vstack(d.IC)[idxl, :] for d, idxl in zip(SlipData, indices_left)]
    
    starts_right = [idxr[0] < idxl[0] for idxr, idxl in zip(indices_right, indices_left)]
    
    kin_right = k.get_kin_apex( [mi.upper_phases(d.phases[:-1], sep=0) for d in SlipData],)
    kin_left  = k.get_kin_apex( [mi.upper_phases(d.phases[:-1], sep=pi) for d in SlipData],)
    
    all_kin_r = []
    all_kin_l = []
    
    all_param_r = []
    all_param_l = []
    
    all_IC_r = []
    all_IC_l = []
    
    all_IC_rc = []
    all_IC_lc = []
    
    res.masses = []
    
    all_phases_r = []
    all_phases_l = []
    
    all_phases_r_uncut = [mi.upper_phases(d.phases[:-1], sep=0) for d in SlipData]
    all_phases_l_uncut = [mi.upper_phases(d.phases[:-1], sep=pi) for d in SlipData]

    all_minlen = []
    
    
    for rep in arange(len(starts_right)): #kr, kl, sr in zip(kin_right, kin_left, starts_right):
        res.masses.append(SlipData[rep].mass)
        # when repetition starts with right step: select         
        kin_r = vstack(kin_right[rep]).T
        kin_l = vstack(kin_left[rep]).T
        par_r = param_right[rep]
        par_l = param_left[rep]
        IC_r = IC_right[rep]
        IC_l = IC_left[rep]
        omit_first_phase_left = 0
        if not starts_right[rep]:
            # omit first value in kin_l!
            kin_l = kin_l[1:, :]
            par_l = par_l[1:, :]
            IC_l = IC_l[1:, :]
            omit_first_phase_left = 1
        
        minlen = min(kin_r.shape[0], kin_l.shape[0]) - 1
        all_minlen.append(minlen)
        kin_r = hstack([kin_r[:minlen, 2 : len(k.selection) + 2] ,
                        kin_r[:minlen, len(k.selection) + 3 :]])# remove absolute position + vertical velocity
        kin_l = hstack([kin_l[:minlen, 2 : len(k.selection) + 2] ,
                        kin_l[:minlen, len(k.selection) + 3 :]])# remove absolute position + vertical velocity
        par_r = par_r[:minlen, :]
        par_l = par_l[:minlen, :]
        IC_r = IC_r[:minlen, :]
        IC_l = IC_l[:minlen, :]
        IC_rc =  vstack([kin_r[:minlen, 0] ,                        # vertical position
                         kin_r[:minlen, len(k.selection) - 1],      # horizontal speed
                         kin_r[:minlen, len(k.selection) - 2]]).T   # lateral speed
        IC_lc =  vstack([kin_l[:minlen, 0] ,                        # vertical position
                         kin_l[:minlen, len(k.selection) - 1],      # horizontal speed
                         kin_l[:minlen, len(k.selection) - 2]]).T   # lateral speed
        if hasattr(SlipData[rep], 'vBelt'):
            IC_rc[:, 1] += SlipData[rep].vBelt
            IC_lc[:, 1] += SlipData[rep].vBelt
        elif hasattr(SlipData[rep], 'vb'):
            IC_rc[:, 1] += SlipData[rep].vb
            IC_lc[:, 1] += SlipData[rep].vb
        else:
            raise KeyError(
                "Belt speed must be present in SlipData ('vB' or 'vBelt' key)")
            

        
        all_IC_rc.append(IC_rc)
        all_IC_lc.append(IC_lc)
        all_IC_r.append(IC_r)
        all_IC_l.append(IC_l)
        all_param_r.append(par_r)
        all_param_l.append(par_l)
        all_kin_r.append(kin_r)
        all_kin_l.append(kin_l)
        all_phases_r.append(array(all_phases_r_uncut[rep][:minlen]))
        all_phases_l.append(array(all_phases_l_uncut[rep][omit_first_phase_left:minlen+omit_first_phase_left]))
    
    res.all_IC_rc = vstack(all_IC_rc)
    res.all_IC_lc = vstack(all_IC_lc)    
    res.all_IC_r = vstack(all_IC_r)
    res.all_IC_l = vstack(all_IC_l)    
    res.all_param_r = vstack(all_param_r)
    res.all_param_l = vstack(all_param_l)
    res.all_kin_r = vstack(all_kin_r)
    res.all_kin_l = vstack(all_kin_l)
    res.all_phases_r = all_phases_r
    res.all_phases_l = all_phases_l
    


    res.param_right = param_right
    res.param_left = param_left
    res.IC_right = IC_right
    res.IC_left = IC_left
    
    if hasattr(SlipData[0], 'T'):
        res.TR = [vstack(d.T)[idxr, :][:minlen, :] for d, idxr, minlen in
                zip(SlipData, indices_right, all_minlen)]
        res.TL = [vstack(d.T)[idxl, :][:minlen, :] for d, idxl, minlen in zip(SlipData,
            indices_left, all_minlen)]
    elif hasattr(SlipData[0], 'T_exp'):
        res.TR = [vstack(d.T_exp)[idxr, :][:minlen, :] for d, idxr, minlen in
                zip(SlipData, indices_right, all_minlen)]
        res.TL = [vstack(d.T_exp)[idxl, :][:minlen, :] for d, idxl, minlen in
                zip(SlipData, indices_left, all_minlen)]
    else:
        raise KeyError("SlipData must feature 'T' or 'T_exp' key!")
    res.yminR = [vstack(d.ymin)[idxr, :][:minlen, :] for d, idxr, minlen in zip(SlipData,
        indices_right, all_minlen)]
    res.yminL = [vstack(d.ymin)[idxl, :][:minlen, :] for d, idxl, minlen in zip(SlipData,
        indices_left, all_minlen)]    

    #print "dt_window:", dt_window, "dt_median:", dt_median
    res.s_param_r = mi.dt_movingavg(res.all_param_r, dt_window, dt_median)
    res.s_param_r /= std(res.s_param_r, axis=0)
    res.s_param_l = mi.dt_movingavg(res.all_param_l, dt_window, dt_median)
    res.s_param_l /= std(res.s_param_l, axis=0)    
    res.s_kin_r = mi.dt_movingavg(res.all_kin_r, dt_window, dt_median)
    res.s_kin_l = mi.dt_movingavg(res.all_kin_l, dt_window, dt_median)
    res.s_kin_r /= std(res.s_kin_r, axis=0)
    res.s_kin_l /= std(res.s_kin_l, axis=0)
    
    res.kin_labels = k.selection[2 : ] + ['v_' + elem for elem in (k.selection[:2] + k.selection[3:])]
    
    return res




def cacherun_marshal(fun, *args, **kwargs):
    """ 
    This function works essentially just as cacherun, but it uses marshal for cache file I/O.
    """
    # compute cache
    # for the function: inspect.findsource(fun)
    # for non-arrays: cPickle.dumps
    # for arrays: hash array.data
    # look up cache
    # run or load
    # return
    
    try:
        os.mkdir('.cache')
    except OSError:
        pass
    # open persistent dict
    index = shelve.open(os.sep.join(['.cache','shelf_marshal']))
 
    hf = hashlib.sha256()
    
    # update hashfunction with function
    hf.update(cPickle.dumps(inspect.findsource(fun)))
    for arg in args:
        hf.update(cPickle.dumps(arg))
    for key in sorted(kwargs.keys()):
        hf.update(cPickle.dumps(key))
        hf.update(cPickle.dumps(kwargs[key]))
    
    hash_ = hf.hexdigest()
    if hash_ in index.keys():
        ffname = os.sep.join(['.cache', index[hash_]])
        with open( ffname , 'rb') as f:
            res = marshal.load(f)
    else:
        # compute
        res = fun(*args, **kwargs)
        # store
        # get new file name
        fh, fn = tempfile.mkstemp(suffix='.marshal', prefix='cache', dir='.cache', text=False)
        os.close(fh)
        fname = fn.split(os.sep)[-1]
        ffname = os.sep.join(['.cache', fname])
        with open( ffname, 'wb') as f:
            marshal.dump(res, f)
        index[hash_] = fname
        index.sync()
    
    index.close()
    return res
    
    
def cacherun(fun, *args, **kwargs):
    """ 
    This function runs the given function with the passed arguments.
    However, when the same function (function source is checked) has been run with the same arguments
    before, its result is loaded from a cache file.
    If this function was run before with the same arguments, return cached results instead.
    
    :args:
        fun (function): the function to be run. Must be a python function (no build-in function). 
            Its source code is hashed.
        *args : sequence of positional arguments
        **kwargs : sequence of keyword arguments
    
    :returns:
        result of fun(*args, **kwargs)
    
    """
    
    
    try:
        os.mkdir('.cache')
    except OSError:
        pass
    # open persistent dict
    index = shelve.open(os.sep.join(['.cache','shelf_pickle']))
 
    hf = hashlib.sha256()
    
    # update hashfunction with function
    hf.update(cPickle.dumps(inspect.findsource(fun)))
    for arg in args:
        hf.update(cPickle.dumps(arg))
    for key in sorted(kwargs.keys()):
        hf.update(cPickle.dumps(key))
        hf.update(cPickle.dumps(kwargs[key]))
    
    hash_ = hf.hexdigest()
    if hash_ in index.keys():
        ffname = os.sep.join(['.cache', index[hash_]])
        with open( ffname , 'rb') as f:
            res = cPickle.load(f)
    else:
        # compute
        res = fun(*args, **kwargs)
        # store
        # get new file name
        fh, fn = tempfile.mkstemp(suffix='.pickle', prefix='cache', dir='.cache', text=False)
        os.close(fh)
        fname = fn.split(os.sep)[-1]
        ffname = os.sep.join(['.cache', fname])
        with open( ffname, 'wb') as f:
            cPickle.dump(res, f)
        index[hash_] = fname
        index.sync()
    
    index.close()
    return res

