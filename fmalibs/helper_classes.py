# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:32:14 2011

@author: Moritz Maus

defines some nice helper classes
LE: May 10, 2011  Moritz Maus
    May 11, 2011   -"-
    May 12, 2011   -"- extended, changed to use the _data-tables
    May 13, 2011   -"- adapted to use the _nr - data-tables, improved phaser functionality    
    May 14, 2011   -"- added "update" functionality
    
V.: 0.11 - some minor bugfixes   
    0.12 - changed to new table format
    0.13 - 
    0.14 - added "update" functionality
    

"""

import numpy as np
import scipy.integrate
import scipy.fftpack as fp
import pylab as pl
import cStringIO as sio
import psycopg2 as pg
import scipy.signal as si
import fmalibs.phaser2 as phaser2
import copy
import csv

class DB:
    """
    handy shortcut for data retrieval (and storage? will see ...)
    connections parameters dbname, username, host and/or password can be given
    or be omitted - then, some values for my local system are assumed ^^ (FIXME)
    
    --- W A R N I N G ---
    This class is handy and therefore _NOT_ secure!
    
    """
    
    def __init__(self,dbname = None, username = None, host = None, password = None):
        """
        if dbname, username, host and/or password are omitted, some values for my
        local system are assumed.
        TODO: fix this if this class is published ^^ 
        """
        self.config_pguser = 'mm_pg' if username is None else username
        self.config_pghost = 'localhost' if host is None else host
        self.config_pgpass = 'mm_pg_pw' if password is None else password
        self.config_dbname = 'll_data3' if dbname is None else dbname
        self.conn = None
        self.datastring = sio.StringIO()
        self.connect()
        
    def __del__(self):
        """
        tidy up: close connection
        """        
        self.conn.close()
        
    def connect(self):
        """
        establish the connection
        """
        self.conn = pg.connect("dbname=%s user=%s password=%s host=%s" % 
                    (self.config_dbname, self.config_pguser,
                     self.config_pgpass, self.config_pghost))
        self.create_tmpViews()
        
    def cmd(self, command):
        """
        executes the given SQL-command
        """
        cur = self.conn.cursor()
        res = None
        try:
            cur.execute(command)
            if cur.statusmessage[:6].upper() == 'SELECT' and cur.description is not None:
                res = cur.fetchall()    
            self.conn.commit()
        except pg.ProgrammingError as err:
            # something has failed
            self.conn.rollback()
            cur.close()        
            print 'Error: ' + err.pgerror + '\nrolling back.'
            
        return res
        
    def get_trial(self,subject, trial_type, repetition):
        """
        returns the trial_id for a given subject_short (or -id),  
        trial_type (or -id), and repetition
        """
        try:
            if not isinstance(subject,int): # get from data
                subject = self.cmd('select subject_id from subject where \
                       shortname=\'%s\' limit 1' % subject)
                if len(subject) == 0:
                    print 'subject not found\n'
                    return None
                subject = subject[0][0]
            if not isinstance(trial_type,int): # get from data
                trial_type = self.cmd('select trial_type_id from trial_type where \
                          shortname=\'%s\' limit 1' % trial_type)
                if len(trial_type) == 0:
                    print 'trial type not found\n'
                    return None
                trial_type = trial_type[0][0]
    
            res = self.cmd('select trial_id from trial where subject_id = %i and \
                            trial_type = %i and repetition = %i ' % (subject,trial_type, repetition) )
            if len(res) == 0:
                print 'trial not found\n'
                return None
        except pg.ProgrammingError as err:
            print 'error in query: ' + err.pgerror + '\nrolling back.'
            self.conn.rollback()
            return None
            
        return res[0]
                     
    def get_kin(self,fields, trial_id, condition = 'True', asDict = True):
        """
        get_kin(fields,trial_id, [condition], [asDict=True]):
            retrieves the selected fields in "fields" (list) from the database, where
            the expression "condition" is matched.
            The return value is a dictionary (asDict = True, default) or an array.     
            
            ATTENTION: This version is suited for the MMCL_LT data, because it relies
            on a fixed list of markers. 
            EDIT THIS FUNCTION FOR YOU NEEDS, IT'S EASY AND STRAIGHT-FORWARD!
        """
        if condition is None:
            condition = 'True'
        # initially: create database-specific data
        markerlist = ['R_MtV', 'R_MtI', 'R_Hee', 'R_Kne', 'R_WrL', 'R_WrM',
                               'R_Elb', 'R_Acr', 'R_Trc', 'R_Sia', 'R_Sip', 'R_AnL',
                               'R_Hea', 'R_AnM', 'L_Hea', 'L_Acr', 'L_Sia', 'L_Sip',
                               'L_Trc', 'L_Elb', 'L_WrL', 'L_WrM', 'L_Kne', 'L_Hee',
                               'L_MtI', 'L_MtV', 'L_AnL', 'L_AnM', 'CVII', 'Front',
                               'Sacr','CoM']    
        mlow = [x.lower() for x in markerlist]
        # now: create SQL-query
        cur = self.conn.cursor()
        lengths = []
        Query_string = 'SELECT '
        for field0 in fields:
            field = field0.strip()
            if len(lengths) == 0:
                prefix = ''
            else:
                prefix = ','
            if field.lower() in mlow:
                Query_string += prefix + field + '_x, ' + field + '_y, ' + field + '_z, ' + field + '_r ' 
                lengths.append(4)
            else:
                Query_string += prefix + field
                lengths.append(1)            
        Query_string += ' FROM kin_' + str(trial_id) + ' WHERE ' + condition + ' ORDER BY frame;'
        # now: execute SQL-Query and retrieve results        
        try:
            cur.execute(Query_string)
            if cur.statusmessage[:6].upper() == 'SELECT':
                res_list = cur.fetchall()    
            self.conn.commit()
        except pg.ProgrammingError as err:
            # something has failed
            self.conn.rollback()            
            print 'Error: ' + err.pgerror + '\nrolling back.'
            cur.close()
            return None
        cur.close()
        # now: format output
        res_array = np.array(res_list)
        if asDict:
            cols_passed = 0
            res_dict = {}
            for nfield,field in enumerate(fields):
                res_dict[field] = res_array[:,cols_passed:cols_passed+lengths[nfield]]
                cols_passed += lengths[nfield]
            return res_dict
        else:
            return res_array


    def get_frc(self,fields,trial_id, condition = 'True', asDict = True):
        """
        get_frc(fields,[condition], [asDict=True]):
            retrieves the selected fields in "fields" (list) from the database, where
            the expression "condition" is matched.
            The return value is a dictionary (asDict = True, default) or an array.             
        """       
        
        # now: create SQL-query
        cur = self.conn.cursor()
        lengths = []
        Query_string = 'SELECT '
        for field0 in fields:
            field = field0.strip()
            if len(lengths) == 0:
                prefix = ''
            else:
                prefix = ','
            if 'cop' in field.lower():
                Query_string += prefix + field + '[1], ' + field + '[2], ' + field + '[3] '
                lengths.append(3)
            else:
                Query_string += prefix + field
                lengths.append(1)            
        
        
        # yes, I know that this % - magic- style is very very very bad in psycopg2-programming...
        cur.execute('create temporary view Fz_sums_%i (Fz, FzR, FzL, Fx, Fy,frame,t_id)\
           AS SELECT FzR1 + FzR2 + FzR3 + FzR4 + FzL1 + FzL2 + FzL3 + FzL4,\
           FzR1 + FzR2 + FzR3 + FzR4, FzL1 + FzL2 + FzL3 + FzL4, \
           Fx1 + Fx2, Fy1 + Fy2, frame,trial_id from force_%i;' % (trial_id,trial_id)) 
        Query_string += (' FROM force_' + str(trial_id) + ' as fd join Fz_sums_' + str(trial_id) + ' as fsum on ' + 
                        '(fd.frame = fsum.frame) ' +                  
                        'WHERE ' + condition + ' ORDER BY fd.frame;')
        # now: execute SQL-Query and retrieve results
        try:
            cur.execute(Query_string)
            if cur.statusmessage[:6].upper() == 'SELECT':
                res_list = cur.fetchall()    
            self.conn.commit()
        except pg.ProgrammingError as err:
            # something has failed
            self.conn.rollback()            
            print 'Error: ' + err.pgerror + '\nrolling back.'
            cur.close()
            return None
        cur.close()
        # now: format output
        res_array = np.array(res_list)
        if asDict:
            cols_passed = 0
            res_dict = {}
            for nfield,field in enumerate(fields):
                res_dict[field] = res_array[:,cols_passed:cols_passed+lengths[nfield]]
                cols_passed += lengths[nfield]
            return res_dict
        else:
            return res_array

    def update(self,data,table,trial_id):
        """
        updates a given table with a given trial_id from data
        data must be a dictionary, including a "frame"-field.
        """
        # for this, a cursor is required. cmd() won't work with temporary tables
        #
        # first: delete old table if exists        
        fields = data.keys()
        if 'frame' not in fields:
            print('WARNING: frame not specified in dict - aborting!')
            return
        cur = self.conn.cursor()
        
        cur.execute('DROP TABLE IF EXISTS tmp_upload')
        
        self.datastring.seek(0,0)
        self.datastring.truncate(0)
        
        # 1st: order array items
        fields.remove('frame')
        # 2nd: build array from dictionary
        #         
        if data['frame'].ndim == 1:
            all_dat = data['frame'][:,np.newaxis].copy()
        elif data['frame'].shape[0] > 1:
            all_dat = data['frame'].copy()
        else:
            all_dat = data['frame'].T.copy()    

        formatstring = 'frame integer, '
        updatestring = ''
        for field in fields:
            formatstring += field + ' double precision, '
            updatestring += field + ' = t.' + field + ', '
            pt = data[field].copy()
            if pt.ndim == 1:
                pt = pt[:,np.newaxis]
            elif pt.shape[1] > 1:
                pt = pt.T
            if pt.shape[1] > 1:
                cur.close()
                print 'Error: only 1D-data must be present in the dictionary. aborting'
                return
            all_dat = np.hstack((all_dat,pt))
        formatstring = formatstring[:-2]
        updatestring = updatestring[:-2]
        
        # print 'query:\n' + querystring + '\n\n'
        #print 'format:\n' + formatstring + '\n\n'
        
        # create temporary table: everything is double precision...
        # conversion can be done in "update"-db-commsnd
        createTblQuery = 'CREATE TABLE tmp_upload (' + formatstring + ');'
        cur.execute(createTblQuery)
        
        #ofile = open('tmp_data.dat','w')
        writer = csv.writer(self.datastring)
        #writer = csv.writer(ofile)
        for x in range(all_dat.shape[0]):
            writer.writerow(all_dat[x,:])
        
        #ofile.close()
        #ofile = open('tmp_data.dat','r')
        self.datastring.seek(0,0)       
        
        cur.copy_from(self.datastring,'tmp_upload',sep=',',columns=['frame'] + fields)
        #cur.copy_from(ofile,'tmp_upload',sep=',',columns=['frame'] + fields)
        #ofile.close()        
        
        self.datastring.seek(0,0)       
        self.datastring.truncate(0)
        
        updateTableQuery = 'update ' + table + ' set ' + updatestring
        updateTableQuery += ' from tmp_upload t where ' + table + '.trial_id = '
        updateTableQuery += ('%i and ' % (trial_id,)) + table + '.frame = t.frame::integer'
        
        cur.execute(updateTableQuery)
        n_update =  int(cur.statusmessage[7:])
        if n_update != all_dat.shape[0]:
            self.conn.rollback()
            print 'Error: number of updated rows does not match number of uploaded rows. rolling back\n'
        else:
            self.conn.commit()
        
        cur.close()
        del all_dat

    def create_tmpViews(self):
        """
        creates some usefull temporary views of the data.
        These views will be deleted when the connection is closed.
        """
        # create "Fz_sums"
        cur = self.conn.cursor()
        cur.execute('create temporary view Fz_sums (Fz, FzR, FzL, Fx, Fy,frame,t_id)\
           AS SELECT FzR1 + FzR2 + FzR3 + FzR4 + FzL1 + FzL2 + FzL3 + FzL4,\
           FzR1 + FzR2 + FzR3 + FzR4, FzL1 + FzL2 + FzL3 + FzL4, \
           Fx1 + Fx2, Fy1 + Fy2, frame,trial_id from force;')    
        cur.close()



class PhaseMaker:
    """
    creates an object that facilitates the phase computation
    """

    def __init__(self, phaseCoords=None, dbUsername=None, dbPassword=None):
        """
        initializes the PhaseMaker object
        """
        self.phaseCoords = None
        if phaseCoords is None:
            # phase coordinate style:
            # (marker1 - marker2, dimensions (pos), dimensions (vel), scaling (pos), scaling(vel)
            
            self.phaseCoords =[ ('R_Kne','L_Kne',[1],[],[1.],[]),
                    ('R_Trc','R_AnL',[1],[],[2.],[]),
                    ('L_Trc','L_AnL',[1],[],[2.],[]),                    
                    ('L_Elb','L_MtV',[],[1],[],[10.]),
                    ('R_Elb','R_MtV',[],[1],[],[10.]),
                    ('R_MtV','L_MtV',[1,2],[1],[1.5,1.],[6.])
                    ]
            self.useButter = True
            self.f_butter = 0.2 # average over 5 frames ~50 Hz
        else:
            self.phaseCoords = phaseCoords
        self.all_data = []
        self.all_time = []
        self.all_psecdata_TDR = []        
        self.all_psecdata_kin = []        
        self.hDB = DB(username=dbUsername, password=dbPassword)
        self.fs_kin = 250.
        self.phaserTDR = None
        self.phaserkin = None
        self.phases = []
        self.trial_ids = []
            
    def get_data(self,subject,trial_type,repetitions):
        """
        retrieves the data from the database
        usage: get_data (subject, trial_type, repetitions as list)
        """
        del self.trial_ids[0:]
        trials = []
        for rep in repetitions:
            res = self.hDB.get_trial(subject,trial_type,rep)            
            if res is not None:
                trials.append(res)
                self.trial_ids.append(res)
                 
        #empty the list if data would be present
        del self.all_data[0:]
        del self.all_time[0:]
        del self.all_psecdata_TDR[0:], self.all_psecdata_kin[0:]
        
        #now: create sql_query_string
        sql_pt = ''
        n_posdim = 0
        n_veldim = 0
        scalings_pos = []
        scalings_vel = []
        coord_replace = ['_x','_y','_z']
        #first run: only positions
        for coord in self.phaseCoords:
            for n,pdim in enumerate(coord[2]):
                sql_pt += ' %s%s - %s%s,' % (coord[0],coord_replace[pdim],coord[1],coord_replace[pdim])
                scalings_pos.append(coord[4][n])
                n_posdim += 1                
        #second run: only velocities
        for coord in self.phaseCoords:
            for n,vdim in enumerate(coord[3]):
                sql_pt += ' %s%s - %s%s,' % (coord[0],coord_replace[vdim],coord[1],coord_replace[vdim])
                scalings_vel.append(coord[5][n])
                n_veldim += 1         
        
        sql_pt = sql_pt[:-1] + ' '
        for trial_id in trials:
            SQLquery = 'SELECT ' + sql_pt;
            SQLquery += (' , time from kin WHERE trial_id = %i order by frame' % trial_id)
            #print SQLquery
            res = np.vstack(self.hDB.cmd(SQLquery))
            if self.useButter:
                bf, af = si.butter(1,self.f_butter)
                for dim in range(res.shape[1]):
                    res[:,dim] = si.filtfilt(bf,af,res[:,dim])
            for dim in range(n_posdim):                
                res[:,dim] = res[:,dim]*scalings_pos[dim]                
            for dim in range(n_posdim,n_veldim+n_posdim):                
                res[1:,dim] = (np.diff(res[:,dim]))*scalings_vel[dim-n_posdim]
                res[:1,dim]  = 0
            
            self.all_data.append(res[:,:-1])
            self.all_time.append(res[:,-1])
            # interpolate vertical force to "kinematic time", use as psec-data
            SQLquery = ('SELECT Fz_c, time from force WHERE trial_id = %i order by frame' % trial_id)
            res_f = np.vstack(self.hDB.cmd(SQLquery))            
            Fz = np.interp(res[:,-1],res_f[:,-1],res_f[:,0])
            
            # split forces to Fzl and Fzr
            drl = res[:,4]
            Fzl = np.zeros_like(Fz)
            Fzr = np.zeros_like(Fz)
            vz = np.diff(np.hstack(self.hDB.cmd('select com_z from kin where trial_id = %i order by frame;' % trial_id)))
            apices = pl.find((vz[:-1] >= 0) * (vz[1:] < 0))

            for apx in range(apices.size - 1):
                pt = (apices[apx], apices[apx+1])  
                # right is higher (vertically) -> "left" force
                if np.mean(drl[pt[0]:pt[1]]) > 0: 
                    Fzl[pt[0]:pt[1]] = Fz[pt[0]:pt[1]].copy()
                else:
                    Fzr[pt[0]:pt[1]] = Fz[pt[0]:pt[1]].copy()            
            
            
            # here: SHIFT these data - then, the time-lag to touchdown/takeoff
            # is approximately zero with -3 resp. +8 frames
            # print 'workaroud?'
            # Fzr = Fz - Fzl
            FzrTD = np.hstack((Fzr[3:], Fzr[-1]*np.ones(3))  ) - 200.
                       
            self.all_psecdata_TDR.append(FzrTD)
                        
            res0 = np.vstack(self.hDB.cmd('select r_mtv_y - l_mtv_y from kin where trial_id = %i order by frame' % trial_id))
            self.all_psecdata_kin.append(res0[:,0])
            
            
            del FzrTD, Fzr, Fzl, Fz            
            del res, res_f, 

    def build_phasers(self):
        """
        builds the phaser objects
        """
        try:
            del self.phaserTDR, self.phaserkin        
        except NameError:
            pass
    
        allPhrIn_TDR = [(x - np.mean(x,axis=0)).T for x in self.all_data]
        allPhrIn_kin = copy.deepcopy(allPhrIn_TDR)
        
        psec_TDR = [x for x in self.all_psecdata_TDR]
        psec_kin = [x for x in self.all_psecdata_kin]

        #allPhrIn: list of dxn-array!
        print('creating TDR phaser...\n')
        self.phaserTDR = phaser2.Phaser2(y = allPhrIn_TDR, psecData = psec_TDR)
        print('creating kin phaser...\n')
        self.phaserkin = phaser2.Phaser2(y = allPhrIn_kin, psecData = psec_kin)

        del allPhrIn_TDR, allPhrIn_kin,
        del psec_TDR, psec_kin
        

    def computePhases(self,verbose=False):
        """
        applies the phaser to the data
        """
        del self.phases[0:]        
        if verbose:
            print 'applying phaser to data:\n'
        for k in range(len(self.all_data)):
            if verbose:
                print '.'
            phase0 = np.zeros((self.all_data[0].shape[0],2))
            phrIn = copy.deepcopy(self.all_data[k]).T
            psecIn = copy.deepcopy(self.all_psecdata_TDR[k])
            phase0[:,0] = self.phaserTDR.phaserEval(phrIn,psecIn).squeeze()
            phrIn = copy.deepcopy(self.all_data[k]).T
            psecIn = copy.deepcopy(self.all_psecdata_kin[k])            
            phase0[:,1] = self.phaserkin.phaserEval(phrIn,psecIn).squeeze()
            self.phases.append(phase0)
            del phase0
    
        if verbose:
            print ' done\n'
        
    def upload_phases(self):
        """
        inserts
        """
        pass

    
    
    