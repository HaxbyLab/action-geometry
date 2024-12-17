# drive_hyperalignment.py
# edit Jane Han, Jan 2022
#
# ref: drive_hyperalignment_cross_validation.py erica busch, 2020
# this script takes no command line argument
#
# use condor_submit ./drive_hyperalignment.submit            
# python ./drive_hyperalignment.
#
# ---------------------------------------------------
# NO cross validation on the hyperalignment training.
# 1) type of hyperalignment [either RHA, H2A, or CHA]
# 2) dataset to use [either raiders, whiplash, or budapest]
# 3) the run to test on (which means you're training the HA model on the other runs)

import os, sys
import numpy as np
from scipy.sparse import save_npz
from mvpa2.base.hdf5 import h5save
import HA_prep_functions as prep
import hybrid_hyperalignment as h2a
from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine 


N_LH_NODES_MASKED = 37476 # previous fsaverage5 = 9372
N_JOBS = 32 
N_BLOCKS = 256
TOTAL_NODES = 40962 #fsaverage6
SPARSE_NODES = 642  
HYPERALIGNMENT_RADIUS = 20


def save_transformations(transformations, outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    h5save(outdir+'/all_subjects_mappers_N.hdf5', transformations)
    for T, s in zip(transformations, utils.subjects):
        save_npz(outdir+"/subj{}_mapper_all.npz".format(s), T.proj)


if __name__ == '__main__':
    ha_type = 'h2a'
    dataset = 'raiders'
    import raiders_utils_noruns as utils
    
    print('running {a} on {b}'.format(a=ha_type,b=dataset))    
    all_runs = np.arange(1, 4+1) #no validate
    #all_runs = np.arange(1, utils.TOT_RUNS+1) #no validate
    data_t = np.setdiff1d(all_runs,1)
    
    # get_data
    dss = utils.get_data('b', num_subjects=23)
    print('dss_node_indices_shape:', dss[0].fa.node_indices.shape) ##debug
    print('dss_node_indices:', dss[0].fa.node_indices) ##debug   
    
    # get the node indices to run SL HA, both hemis
    node_indices = np.concatenate(prep.get_node_indices('b', surface_res=TOTAL_NODES)) #both hemi
    print('node_indices:',node_indices) ##debug
    
    ## 
    target_indices = prep.get_node_indices('b', surface_res=SPARSE_NODES) ##debug
        
    # get the surfaces for both hemis
    surface = prep.get_freesurfer_surfaces('b') #both hemi

    # make the surface QE 
    qe = SurfaceQueryEngine(surface, HYPERALIGNMENT_RADIUS)
    
    # run hybrid hyperalignment
    if ha_type == 'h2a':
        
        outdir = os.path.join(utils.h2a_dir)
        
        ha = h2a.HybridHyperalignment(ref_ds=dss[0],
                                      nblocks=N_BLOCKS,
                                      nproc=N_JOBS,
                                      mask_node_indices=node_indices,
                                      seed_indices=node_indices,
                                      target_indices=target_indices,
                                      target_radius=utils.HYPERALIGNMENT_RADIUS,
                                      surface=surface)
        Ts = ha(dss)

    else:
        print('ha_type must be h2a')
        sys.exit()
    
    save_transformations(Ts, os.path.join(outdir, 'transformations')) 
