# 4-2_search_group_reg.py
# ran with all models including pymoten motion Jane Han 2024.08.29
# ran with R2 bootstrap Jane Han 2024 Oct
#
# Purpose
# permutation & FDR for model groups multiple regression
#
# How to use
# ssh head8 
# han@head8: conda activate action-python2
# python ./4-2_search_group_reg.py
# or
# han@head1: conda activate action-python2
# condor_submit ./4-2_search_group_reg.submit
#
# reference code
# /backup/data/social_actions/scripts/post_hha/4-1a_search_group.py
# /backup/data/social_actions/scripts/post_hha/4-1b_search_group_zdist.py


## import environment
import numpy as np
from os.path import join
from itertools import product
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
import mvpa2.suite as mv

## LOAD IN DATA
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

### change this part -----------------------------
sqr_r2 = False # flag for calculating R2 or sqr(R2) 
 
#model_groups = {'ts': ['transitivity', 'sociality']}
#model_groups = {'ops': ['object', 'person', 'scene']}
#model_groups = {'vn': ['verbs', 'nonverbs']}
model_groups = {'all': [
    'transitivity', 'sociality', 'object', 'person', 'scene', 'verbs', 'nonverbs', 'gaze', 'motion']}
#model_groups = {'control': [
#    'object', 'person', 'scene', 'verbs', 'nonverbs', 'gaze', 'motion']}


model_group = next(iter(model_groups))
#model_group = 'ops'

# ------------------------------------------------

## calculate the mean R^2 (or square root of R^2) across subjects 
# only for simple visualization
        
# load in r2
for hemi in hemis:
    sl_r2s = []
    for participant in sorted(participants.keys()):
        sl_r2s.append(mv.niml.read(join(mvpa_dir, 'sl_participants', 
                                        'sl_reg-{0}_{1}_p{2}_r2.niml.dset'.format(
                                            model_group, hemi, participant))).samples)
        
    if sqr_r2 == True:
        sl_mean = np.nan_to_num(np.nanmean(np.vstack(np.sqrt(sl_r2s)), axis=0))[None, :]
    else:
        sl_mean = np.nan_to_num(np.nanmean(np.vstack(sl_r2s), axis=0))[None, :]
    
    assert sl_mean.shape == (1, n_vertices)
    
    ## save
    if sqr_r2 == True:
        mv.niml.write(join(mvpa_dir, 
                           'sl_reg-{0}_{1}_mean_sqrR2.niml.dset'.format(model_group, hemi)), sl_mean)
    else:
        mv.niml.write(join(mvpa_dir, 
                           'sl_reg-{0}_{1}_mean_r2.niml.dset'.format(model_group, hemi)), sl_mean)

## calculate for params
# load in r2 in dss
dss = {}
for participant in sorted(participants.keys()):
    dss[participant] = {}
    for hemi in hemis:
        dss[participant][hemi] = mv.niml.read(join(
            mvpa_dir, 'sl_participants','sl_reg-{0}_{1}_p{2}_r2.niml.dset'.format(
                model_group, hemi, participant))).samples

# calculate average across subjects to ds_test by hemisphere (after nan_to_num)
if sqr_r2 == True:
    for p in sorted(participants.keys()):
        for hemi in hemis:
            dss[p][hemi] = np.sqrt(np.abs(dss[p][hemi]))


ds_test = {'lh': np.nanmean([dss[p]['lh'] 
                          for p in sorted(participants.keys())], axis=0),
           'rh': np.nanmean([dss[p]['rh'] 
                          for p in sorted(participants.keys())], axis=0)}

## non-parametric test
n_np = 10000
null_distribution = {'lh': [], 'rh': []}

if sqr_r2 == True:
    # bootstrap 
    # sqr(R2) is above zero so permutation didn't work
    # reference: 4-4-3_search_diff_reg_unique.py
    boot_distribution = {'lh': [], 'rh': []}
    participant_keys = sorted(participants.keys())
    for n, b in enumerate(np.arange(n_np)):
        # random sample participants with replacement
        boot_ids = np.random.choice(
            np.arange(n_participants), size=n_participants, replace=True)
        
        for hemi in hemis:
            boot_sample = [dss[participant_keys[i]][hemi] for i in boot_ids]
            boot_distribution[hemi].append(np.nanmean(boot_sample, axis=0))         
        if n % 100 == 0:
            print("Computed {0} sqr(R2) bootstrap {1}".format(model_group, n))
            
    null_distribution['lh'] = np.vstack(boot_distribution['lh']) - ds_test['lh']
    null_distribution['rh'] = np.vstack(boot_distribution['rh']) - ds_test['rh']
    
else: 
    # bootstrap for R2
    # reference: 4-4-3_search_diff_reg_unique.py
    boot_distribution = {'lh': [], 'rh': []}
    participant_keys = sorted(participants.keys())
    for n, b in enumerate(np.arange(n_np)):
        # random sample participants with replacement
        boot_ids = np.random.choice(
            np.arange(n_participants), size=n_participants, replace=True)
        
        for hemi in hemis:
            boot_sample = [dss[participant_keys[i]][hemi] for i in boot_ids]
            boot_distribution[hemi].append(np.nanmean(boot_sample, axis=0))         
        if n % 100 == 0:
            print("Computed {0} R2 bootstrap {1}".format(model_group, n))
            
    null_distribution['lh'] = np.vstack(boot_distribution['lh']) - ds_test['lh']
    null_distribution['rh'] = np.vstack(boot_distribution['rh']) - ds_test['rh']
    
    '''
    # Original option: Get subject-level permutation
    permutations = [np.random.choice([-1, 1], n_participants) for i in np.arange(n_np)]
    for n, permutation in enumerate(permutations):
        for hemi in hemis:
            null_distribution[hemi].append(np.mean(
                [sign * dss[p][hemi] 
                 for sign, p in zip(permutation, sorted(participants.keys()))], axis=0))
        if n % 100 == 0:
            print("Computed {0} R2 permutation {1}".format(model_group, n))
    
    null_distribution['lh'] = np.vstack(null_distribution['lh'])
    null_distribution['rh'] = np.vstack(null_distribution['rh'])
    '''
print("Computed non-parametric test")


# change this part if necessary -----------------------
## calculate p-values for one-sided test (for searchlight RDM correlations)
one_sided = True # we ARE expecting positive correlation between beh and neuro
#one_sided = False # not expecting positive correlation between beh and neuro
# -----------------------------------------------------
if one_sided:
    p_values = {}
    for hemi in hemis:
        p_values[hemi] = ((np.sum(null_distribution[hemi] >= np.maximum(ds_test[hemi], -ds_test[hemi]), 
                              axis=0) + 1) / (float(n_np) + 1))[None, :]
else:
    p_values = {}
    for hemi in hemis:
        left_tail = ((np.sum(
            null_distribution[hemi] <= np.minimum(
                ds_test[hemi], -ds_test[hemi]),axis=0) + 1) / (float(n_np) + 1))[None, :]
        right_tail = ((np.sum(
            null_distribution[hemi] >= np.maximum(
                ds_test[hemi], -ds_test[hemi]),axis=0) + 1) / (float(n_np) + 1))[None, :]

        p_values[hemi] = left_tail + right_tail


## Apply masks and compute FDR
# apply masks
lh_ids = np.where(lh_mask > 0)[0].tolist()
rh_ids = np.where(rh_mask > 0)[0].tolist()

n_lh_ids = len(lh_ids)
n_rh_ids = len(rh_ids)

combined_p = np.hstack((p_values['lh'][0, lh_ids], p_values['rh'][0, rh_ids]))[None, :]
assert combined_p.shape[1] == n_lh_ids + n_rh_ids

# caculate for fdr correction -----------------------------
# for q_values and z_values
fdr = multipletests(combined_p[0, :], method='fdr_bh')[1]
# 'by' more conservative, using 'bh' here
# fdr = multipletests(combined_p[0, :], method='fdr_by')[1]
# ----------------------------------------------------------

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
    results = np.vstack((ds_test[hemi], p_values[hemi], q_values[hemi], z_values[hemi]))
    assert results.shape == (4, n_vertices)

    ## save output
    if sqr_r2 == True:
        mv.niml.write(
            join(mvpa_dir, 
                 'sl_post_hha_boot_1sided_bh_reg-{0}_{1}_sqrR2.niml.dset'.format(model_group, hemi)), results)
    else:
        mv.niml.write(
            join(mvpa_dir,
                 'sl_post_hha_boot_1sided_bh_reg-{0}_{1}_r2.niml.dset'.format(model_group, hemi)), results)
                 # permutation file 
                 #'sl_post_hha_results_1sided_bh_reg-{0}_{1}_r2.niml.dset'.format(model_group, hemi)), results)
    print("Saved model {0} for hemi {1}".format(model_group, hemi))