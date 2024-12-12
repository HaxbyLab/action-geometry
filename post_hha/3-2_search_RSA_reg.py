#!/usr/bin/env python
# 3-2_search_RSA_reg.py
# November 2024, Jane Han
#
# Purpose
# [Figure S2] to calculate combined regression 
#
# How to run this code: 
# conda activate action-python2
# condor_submit ./3-2_search_RSA_reg.submit
#

## Environment
from os.path import join
import numpy as np
import mvpa2.suite as mv
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, zscore
from itertools import combinations
import pandas as pd

## Participant
participants = {'1': 'sid000021', '2': 'sid000120',
                '3': 'sid000005', '4': 'sid000029',
                '5': 'sid000010', '6': 'sid000013',
                '7': 'sid000020', '8': 'sid000024',
                '9': 'sid000009', '10': 'sid000012',
                '11': 'sid000034', '12': 'sid000007',
                '13': 'sid000416', '14': 'sid000433',
                '15': 'sid000134', '16': 'sid000522',
                '17': 'sid000114', '18': 'sid000102',
                '19': 'sid000499', '20': 'sid000142',
                '21': 'sid000535', '22': 'sid000278',
                '23': 'sid000052'}

n_conditions = 90
n_vertices = 40962

## Directories
base_dir = '/backup/data/social_actions'
scripts_dir = join(base_dir, 'scripts')
data_dir = join(base_dir, 'fmri')
suma_dir = join(data_dir, 'suma-fsaverage6')
mvpa_dir = join(data_dir, 'pymvpa')

## Load in searchlight neural RDMs
# this takes time
condition_order = mv.h5load(join(scripts_dir, 'condition_order.hdf5'))

sl_rdms = {}
for hemi in ['lh', 'rh']:
    sl_rdms[hemi] = {}
    for participant in participants.keys():
        # load post hha glm coef
        sl_result = mv.h5load(join(mvpa_dir, 'post_hha_no_roi_ids', 'search_RDMs_sq_zscore_p{0}_{1}.hdf5'.format(participant, hemi)))
        # reshape to squareform 
        sl_sq = sl_result.samples.reshape(n_conditions, n_conditions,  n_vertices)
        
        sl_tri = []
        for sl in sl_sq.T:
            sl_tri.append(squareform(sl, checks=False))
        sl_tri = np.array(sl_tri).T
        assert sl_tri.shape == (n_conditions * (n_conditions - 1) / 2, n_vertices)
        sl_tri = mv.Dataset(sl_tri,
                            sa={'conditions': list(combinations(condition_order['original_condition_order'], 2))},
                            fa=sl_result.fa, a=sl_result.a)
        sl_tri.sa['participants'] = [int(participant)] * sl_tri.shape[0]
        sl_rdms[hemi][participant] = sl_tri
        print("Loaded searchlight RDMs for participant {0} "
              "hemisphere {1}".format(participant, hemi))

## Load in target RDMs 
motion_rdm = np.load(join(scripts_dir, 'RDMs', 'pymoten_motion_energy_rdm.npy'))
gaze_rdm = np.load(join(scripts_dir, 'RDMs', 'gaze_rdm.npy'))
verb_rdm = np.load(join(scripts_dir, 'RDMs', 'verb_rdm.npy'))
nonverb_rdm = np.load(join(scripts_dir, 'RDMs', 'nonverb_rdm.npy'))
object_rdm = np.load(join(scripts_dir, 'RDMs', 'object_rdm.npy'))
social_rdm = np.load(join(scripts_dir, 'RDMs', 'social_rdm.npy'))
static_object_rdm = np.load(join(scripts_dir, 'RDMs', 'frame_arrangement_object_RDM.npy'))
static_scene_rdm = np.load(join(scripts_dir, 'RDMs', 'frame_arrangement_scene_RDM.npy'))
static_person_rdm = np.load(join(scripts_dir, 'RDMs', 'frame_arrangement_person_RDM.npy'))

# combined models
# Dictionary of interest: verbs + nonverbs / object + person + scene/ transitivity + sociality / motion + gaze
model_groups = {#'ts': {'transitivity': object_rdm, 'sociality': social_rdm},
                #'ops': {'object': static_object_rdm,  'person': static_person_rdm, 'scene': static_scene_rdm},
                #'vn': {'verbs': verb_rdm, 'nonverbs': nonverb_rdm},
                #'all': {'transitivity': object_rdm, 'sociality': social_rdm, 
                #        'object': static_object_rdm,'person':static_person_rdm, 'scene': static_scene_rdm, 
                #        'verbs': verb_rdm, 'nonverbs': nonverb_rdm, 
                #        'gaze':gaze_rdm, 'motion': motion_rdm}}
                'mg': {'motion': motion_rdm, 'gaze':gaze_rdm}}
###########################################################
model_group = 'mg' ### CHANGE THIS TO RUN DIFFERENT MODELS'
###########################################################
model_names = model_groups[model_group]

## Compute Regression 
# regression code from 5_ROI_RSA_python3.py

# Standardized rank regression using OLS
# assuming neural_square = True, standardize=True, rank=False
import statsmodels.formula.api as smf

for hemi in ['lh', 'rh']:
    for participant in sorted(participants.keys()):

        # standardize and rank both True
        variables = {name : zscore(rankdata(rdm))
                     for name, rdm in model_names.items()}
        
        neural_rdms = sl_rdms[hemi][participant].samples
        
        sl_params, sl_r2s = [], []
        for v in np.arange(neural_rdms.shape[1]):
            variables['neural'] = zscore(rankdata(neural_rdms[:, v]))
            df = pd.DataFrame(variables)
            formula = 'neural ~ ' + ' + '.join(model_names.keys())
            
            # compute multiple regression
            sl_regression = smf.ols(formula, data=df).fit()
            
            # store results
            sl_params.append(sl_regression.params.get_values()[1:])
            sl_r2s.append(sl_regression.rsquared)
            
        sl_params = np.column_stack(sl_params)
        sl_r2s = np.array(sl_r2s)[np.newaxis, :]
        
        ## save
        mv.niml.write(join(mvpa_dir, 'sl_participants',
                           'sl_reg-{0}_{1}_p{2}_params.niml.dset'.format(
                               model_group, hemi, participant)),
                      np.nan_to_num(sl_params))
        mv.niml.write(join(mvpa_dir, 'sl_participants',
                           'sl_reg-{0}_{1}_p{2}_r2.niml.dset'.format(
                               model_group, hemi, participant)),
                      np.nan_to_num(sl_r2s))
            
        print("saved regression calculation for model group {0} participant {1} hemisphere {2}".format(model_group, participant, hemi))
        
        # calculate across participant average glm coef
        #sl_mean = np.nan_to_num(np.nanmean(np.vstack(sl_spearmans[hemi]), axis=0))[None, :]
        #assert sl_mean.shape == (1, n_vertices)
        #mv.niml.write(join(mvpa_dir, 'sl_{0}_mean_{1}_regression.niml.dset'.format(model, hemi)), sl_mean)