# 4-1a_search_group.py
# 2024 August, Jane Han
#
# Purpose
# [Figure 3] permutation & FDR 
#
# How to use
# conda activate action-python2
# condor_submit 4-1a_search_group.submit

## Import environments
import numpy as np
from os.path import join
from itertools import product
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
import mvpa2.suite as mv

# Function for computing Fisher Z-transformed means
def fisher_mean(correlations, axis=None):
    return np.tanh(np.nanmean(np.arctanh(correlations), axis=axis))

# LOADING
base_dir = '/backup/data/social_actions'
mvpa_dir = join(base_dir, 'fmri', 'pymvpa')

n_participants = 23
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

assert len(participants.keys()) == n_participants
hemis = ['lh', 'rh']
n_vertices = 40962

# Load in masks excluding medial wall
lh_mask = np.load(join(mvpa_dir, 'cortical_mask_lh.npy'))
rh_mask = np.load(join(mvpa_dir, 'cortical_mask_rh.npy'))
    
models = ['gaze', 'motion', 'nonverbs', 'verbs', 'object', 'person', 'scene', 'sociality','transitivity']

# Load searchlight correlation data
for model in models:
    print("Started computing {0}".format(model))
    
    dss = {}
    for participant in sorted(participants.keys()):
        dss[participant] = {}
        for hemi in hemis:
            dss[participant][hemi] = mv.niml.read(join(mvpa_dir, 'sl_participants','sl_{0}_{1}_p{2}.niml.dset'.format(model, hemi, participant))).samples


    # Fisher mean without permutation
    ds_test = {'lh': fisher_mean([np.nan_to_num(dss[p]['lh']) for p in sorted(participants.keys())], axis=0),
               'rh': fisher_mean([np.nan_to_num(dss[p]['rh']) for p in sorted(participants.keys())], axis=0)}
    
    # Get subject-level permutation
    n_permutations = 10000
    permutations = [np.random.choice([-1, 1], n_participants) for i in np.arange(n_permutations)]

    null_distribution = {'lh': [], 'rh': []}
    
    for n, permutation in enumerate(permutations):
        for hemi in hemis:
            null_distribution[hemi].append(fisher_mean([sign * dss[p][hemi] for sign, p in zip(permutation, sorted(participants.keys()))], axis=0))
        if n % 100 == 0:
            print("Finished permutation {0}".format(n))
    
    null_distribution['lh'] = np.vstack(null_distribution['lh'])
    null_distribution['rh'] = np.vstack(null_distribution['rh'])
    # -----------------------
    # P-values for one-sided test (for searchlight RDM correlations)
    #one_sided = False # not expecting positive correlation between beh and neuro
    one_sided = True # expecting positive correlation between beh and neuro
    # -----------------------
    if one_sided:
        p_values = {}
        for hemi in hemis:
            p_values[hemi] = ((np.sum(null_distribution[hemi] >= np.maximum(ds_test[hemi], -ds_test[hemi]), 
                                  axis=0) + 1) / (float(n_permutations) + 1))[None, :]
    else:
        p_values = {}
        for hemi in hemis:
            left_tail = ((np.sum(null_distribution[hemi] <= np.minimum(ds_test[hemi], -ds_test[hemi]),axis=0) + 1) / (float(n_permutations) + 1))[None, :]
            right_tail = ((np.sum(null_distribution[hemi] >= np.maximum(ds_test[hemi], -ds_test[hemi]),axis=0) + 1) / (float(n_permutations) + 1))[None, :]
        
            p_values[hemi] = left_tail + right_tail


    # Apply masks and compute FDR
    lh_ids = np.where(lh_mask > 0)[0].tolist()
    rh_ids = np.where(rh_mask > 0)[0].tolist()
    
    n_lh_ids = len(lh_ids)
    n_rh_ids = len(rh_ids)

    combined_p = np.hstack((p_values['lh'][0, lh_ids], p_values['rh'][0, rh_ids]))[None, :]
    assert combined_p.shape[1] == n_lh_ids + n_rh_ids

    # -----------------------
    # fdr correction: q_values and z_values  
    fdr = multipletests(combined_p[0, :], method='fdr_bh')[1]
    # 'by' more conservative, using more standard 'bh' here
    # -----------------------
    
    zval = np.abs(norm.ppf(fdr))

    q_values = {'lh': np.zeros((1, n_vertices)), 'rh': np.zeros((1, n_vertices))}
    z_values = {'lh': np.zeros((1, n_vertices)), 'rh': np.zeros((1, n_vertices))}

    np.put(q_values['lh'][0, :], lh_ids, fdr[:n_lh_ids])
    np.put(q_values['rh'][0, :], rh_ids, fdr[n_lh_ids:])
    np.put(z_values['lh'][0, :], lh_ids, zval[:n_lh_ids])
    np.put(z_values['rh'][0, :], rh_ids, zval[n_lh_ids:])
    
    for hemi in hemis:
        z_values[hemi][z_values[hemi] == np.inf] = 0
    
        # four maps per one brain
        results = np.vstack((ds_test[hemi], p_values[hemi],q_values[hemi], z_values[hemi]))
        assert results.shape == (4, n_vertices)
        
        # save
        mv.niml.write(join(mvpa_dir, 'sl_post_hha_results_1sided_bh_{0}_{1}.niml.dset'.format(model, hemi)), results) 