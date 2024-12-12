#!/usr/bin/env python
# 3-1a_search_RSA_spearman.py
# 2024 August, Jane Han
# 
# Purpose
#
# conda environment necessary:
# conda activate action-python2

## Import environments
from os.path import join
import numpy as np
import mvpa2.suite as mv
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from itertools import combinations

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

condition_order = mv.h5load(join(scripts_dir, 'condition_order.hdf5'))
reorder = condition_order['reorder']
sparse_ordered_labels = condition_order['sparse_ordered_labels']

## Load in searchlight RDMs
sl_rdms = {}
for hemi in ['lh', 'rh']:
    sl_rdms[hemi] = {}
    for participant in participants.keys():
        # post hha
        sl_result = mv.h5load(join(mvpa_dir, 'post_hha_no_roi_ids', 'search_RDMs_sq_zscore_p{0}_{1}.hdf5'.format(participant, hemi)))
        # glm coef
        #sl_result = mv.h5load(join(mvpa_dir, 'no_roi_ids', 'search_RDMs_sq_zscore_p{0}_{1}.hdf5'.format(participant, hemi)))
        
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
static_object_rdm = mv.h5load(join(scripts_dir, 'RDMs', 'frame_arrangement_object_RDM.hdf5'))
static_person_rdm = mv.h5load(join(scripts_dir, 'RDMs', 'frame_arrangement_person_RDM.hdf5'))
static_scene_rdm = mv.h5load(join(scripts_dir, 'RDMs', 'frame_arrangement_scene_RDM.hdf5'))
motion_rdm = np.load(join(scripts_dir, 'RDMs', 'pymoten_motion_energy_rdm.npy'))
gaze_rdm = mv.h5load(join(scripts_dir, 'RDMs', 'gaze_target_RDM.hdf5'))
verb_rdm = mv.h5load(join(scripts_dir, 'RDMs', 'word2vec_verbs_RDM.hdf5'))
nonverb_rdm = mv.h5load(join(scripts_dir, 'RDMs','word2vec_nonverbs_RDM.hdf5'))
object_rdm = mv.h5load(join(scripts_dir, 'RDMs', 'arrangement_object_RDM.hdf5'))
social_rdm = mv.h5load(join(scripts_dir, 'RDMs', 'arrangement_social_RDM.hdf5'))

# combined models
models = {'object': static_object_rdm,  'person': static_person_rdm, 'scene': static_scene_rdm, 'motion': motion_rdm,  'gaze': gaze_rdm, 'verbs': verb_rdm, 'nonverbs': nonverb_rdm, 'transitivity': object_rdm, 'sociality': social_rdm}

## Compute Spearman correlations per subject
for model in ['object', 'person', 'scene', 'motion', 'gaze', 'verbs', 'nonverbs', 'transitivity', 'sociality']:
    sl_spearmans = {}
    for hemi in ['lh', 'rh']:
        sl_spearmans[hemi] = []
        for participant in sorted(participants.keys()):
            # spearman correlation calculation
            sl_spearman = np.array([spearmanr(np.nan_to_num(sl), models[model])[0]
                                    for sl in sl_rdms[hemi][participant].samples.T])[None, :]
            sl_spearmans[hemi].append(sl_spearman)
            print("Finished searchlight Spearman correlations for "
                  "participant {0}, hemisphere {1}, model {2}".format(
                    participant, hemi, model))
            
            ## save
            # post hha
            mv.niml.write(join(mvpa_dir, 'sl_participants', 
                          'sl_{0}_{1}_p{2}.niml.dset'.format(model, hemi, participant)), sl_spearman)
            # glm coef
            #mv.niml.write(join(mvpa_dir, 'sl_participants_glm', 
            #              'sl_{0}_{1}_p{2}_glm_coef.niml.dset'.format(model, hemi, participant)), sl_spearman)
        
        sl_mean = np.nan_to_num(np.nanmean(np.vstack(sl_spearmans[hemi]), axis=0))[None, :]
        assert sl_mean.shape == (1, n_vertices)
        
        ## save
        # post hha
        mv.niml.write(join(mvpa_dir, 'sl_{0}_mean_{1}.niml.dset'.format(model, hemi)), sl_mean)

        # glm coef
        #mv.niml.write(join(mvpa_dir, 'sl_{0}_mean_{1}_glm_coef.niml.dset'.format(model, hemi)), sl_mean)

        print("Finished searchlight Spearman correlations for model {0}".format(model))