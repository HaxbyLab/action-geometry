# 4-4-3_search_diff_reg_unique.py
# Jane 2024 Sep 
#
# for non-parametric test for unique variance difference between multiple regression groups
#
# How to run this code: 
# ssh head8
# conda activate action-python2
# ./python 4-4-3_search_diff_reg_between.py
# Note: run 4-4-2_search_diff_reg_between.py for t-test to see the difference in R^2
# reference t-test_v2.ipynb in ./trash


## Environment
from gifti_io import read_gifti, write_gifti
import numpy as np
from os.path import join
from itertools import product, combinations
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
from scipy.stats import norm, ttest_rel, ttest_1samp
import mvpa2.suite as mv
from nipype.interfaces import afni

## directory
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

## Load in masks excluding medial wall
cortical_masks = {'lh': np.load(join(mvpa_dir, 'cortical_mask_lh.npy')),
                  'rh': np.load(join(mvpa_dir, 'cortical_mask_rh.npy'))}
cortical_coords = {'lh': np.where(cortical_masks['lh']), 
                   'rh': np.where(cortical_masks['rh'])}


'''
PART 1
caculate the plain-old mean differences in correlation
- subtract within the subject to get the difference values
- average the values across the subject with the value above
'''
## Load each participants' searchlight correlation data
# change this part############
model_groups = {'all': ['transitivity', 'sociality', 'person', 'object', 'scene','verb', 'nonverb', 'gaze','motion'],
                #'all-ts': ['person', 'object', 'scene', 'verb', 'nonverb', 'gaze','motion']}
                #'all-sociality': ['transitivity', 'object', 'person', 'scene', 'verb', 'nonverb', 'gaze','motion']}
                'all-transitivity': ['sociality', 'object', 'person', 'scene', 'verb', 'nonverb', 'gaze','motion']}
    
model_full = 'all'
model_nested = 'all-transitivity'

sqr_r2 = False ### change this flag for calculating R2 or sqr(R2)
#########################################

dss = {}
for model_group in model_groups.keys():
    print("Started computing model_group: {0}".format(model_group))
    dss[model_group] = {}
    for hemi in hemis:
        dss[model_group][hemi] = []
        for participant in sorted(participants.keys()):
            ds = mv.niml.read(join(
                mvpa_dir, 'sl_participants', 'sl_reg-{0}_{1}_p{2}_r2.niml.dset'.format(
                    model_group, hemi, participant))).samples
            if sqr_r2 == True:
                ds = np.sqrt(ds)
            #mask
            dss[model_group][hemi].append(np.nan_to_num(ds[0][cortical_masks[hemi]]))
print("Finished loading all model groups r2")

## non-parametric test
# mean difference without permutation        
ds_test = {'lh': np.mean([np.nan_to_num(dss[model_full]['lh'][p] - dss[model_nested]['lh'][p]) for p in np.arange(n_participants)], axis=0),
           'rh': np.mean([np.nan_to_num(dss[model_full]['rh'][p] - dss[model_nested]['rh'][p]) for p in np.arange(n_participants)], axis=0)}

'''
# Get subject-level permutation
# reference: 4-1a_search_group.py permutation
n_permutations = 10000
permutations = [np.random.choice([-1, 1], n_participants) for i in np.arange(n_permutations)]

null_distribution = {'lh': [], 'rh': []}
for n, permutation in enumerate(permutations):
    for hemi in hemis:
        null_distribution[hemi].append(
            np.mean([sign * (dss[model_full][hemi][p] - dss[model_nested][hemi][p]) 
                     for sign, p in zip(permutation, np.arange(n_participants))], axis=0))
    if n % 100 == 0:
        print("Finished permutation {0}".format(n))

null_distribution['lh'] = np.vstack(null_distribution['lh'])
null_distribution['rh'] = np.vstack(null_distribution['rh'])
# muted because somehow all p-values were 0
'''

## bootstrap - sam 
diff_subj = {'lh': np.array(dss[model_full]['lh']) -  np.array(dss[model_nested]['lh']),
             'rh': np.array(dss[model_full]['rh']) -  np.array(dss[model_nested]['rh'])}

n_bootstraps = 10000
null_distribution = {'lh': [], 'rh': []}
for n, b in enumerate(np.arange(n_bootstraps)):
    for hemi in hemis:
        boot_ids = np.random.choice(np.arange(n_participants), size=n_participants, replace=True)
        diff_boot = diff_subj[hemi][boot_ids]
        null_distribution[hemi].append(np.mean(diff_boot, axis=0))
    if n % 100 == 0:
        print("Finished bootstrap {0}".format(n))
        
null_distribution['lh'] = np.vstack(null_distribution['lh']) - np.mean(null_distribution['lh'], axis=0)
null_distribution['rh'] = np.vstack(null_distribution['rh']) - np.mean(null_distribution['rh'], axis=0)


## P-values for one-sided test (for searchlight RDM correlations)
# -----------------------
one_sided = True # expecting positive correlation between beh and neuro
#one_sided = False # not expecting positive correlation between beh and neuro
# -----------------------
if one_sided:
    p_values = {}
    for hemi in hemis:
        p_values[hemi] = ((np.sum(null_distribution[hemi] 
                                  >= np.maximum(ds_test[hemi], -ds_test[hemi]), axis=0) + 1) / (float(n_bootstraps) + 1))[None, :]
else:
    p_values = {}
    for hemi in hemis:
        left_tail = ((np.sum(null_distribution[hemi] <= np.minimum(ds_test[hemi], -ds_test[hemi]),axis=0) + 1) / (float(n_bootstraps) + 1))[None, :]
        right_tail = ((np.sum(null_distribution[hemi] >= np.maximum(ds_test[hemi], -ds_test[hemi]),axis=0) + 1) / (float(n_bootstraps) + 1))[None, :]

        p_values[hemi] = left_tail + right_tail


## Apply masks and compute FDR
#lh_ids = np.where(cortical_masks['lh'] > 0)[0].tolist()
#rh_ids = np.where(cortical_masks['rh'] > 0)[0].tolist()

n_lh_ids = cortical_coords['lh'][0].shape[0]
n_rh_ids = cortical_coords['rh'][0].shape[0]

combined_p = np.hstack((p_values['lh'][0], 
                        p_values['rh'][0]))[None, :]
assert combined_p.shape[1] == n_lh_ids + n_rh_ids

## fdr correction
# q_values and z_values

# fdr = multipletests(combined_p[0, :], method='fdr_by')[1]
fdr = multipletests(combined_p[0, :], method='fdr_bh')[1]
# 'by' more conservative, using more standard 'bh' here

zval = np.abs(norm.ppf(fdr))

q_values = {'lh': np.zeros((1, n_vertices)), 'rh': np.zeros((1, n_vertices))}
z_values = {'lh': np.zeros((1, n_vertices)), 'rh': np.zeros((1, n_vertices))}
p_values = {'lh': np.zeros((1, n_vertices)), 'rh': np.zeros((1, n_vertices))}
mean_values = {'lh': np.zeros((1, n_vertices)), 'rh': np.zeros((1, n_vertices))}

np.put(q_values['lh'][0, :], cortical_coords['lh'][0], fdr[:n_lh_ids])
np.put(q_values['rh'][0, :], cortical_coords['rh'][0], fdr[n_lh_ids:])
np.put(z_values['lh'][0, :], cortical_coords['lh'][0], zval[:n_lh_ids])
np.put(z_values['rh'][0, :], cortical_coords['rh'][0], zval[n_lh_ids:])
np.put(p_values['lh'][0, :], cortical_coords['lh'][0], combined_p[:n_lh_ids])
np.put(p_values['rh'][0, :], cortical_coords['rh'][0], combined_p[n_lh_ids:])
np.put(mean_values['lh'][0, :], cortical_coords['lh'][0], ds_test['lh'])
np.put(mean_values['rh'][0, :], cortical_coords['rh'][0], ds_test['rh'])

for hemi in hemis:
    z_values[hemi][z_values[hemi] == np.inf] = 0

    # four maps per one brain
    results = np.vstack((mean_values[hemi], p_values[hemi], q_values[hemi], z_values[hemi]))
    assert results.shape == (4, n_vertices)

    ## save
    mv.niml.write(join(mvpa_dir, 'sl_post_hha_boot{0}_results_1sided_bh_{1}-{2}_{3}.niml.dset'.format(n_bootstraps, model_full, model_nested, hemi)), results)
    
    ## save gifti
    out_f = join(mvpa_dir, 'sl_post_hha_boot{0}_results_1sided_bh_{1}-{2}_{3}.gii'.format(n_bootstraps, model_full, model_nested, hemi))
    #
    gifti_template = ('/backup/data/social_actions/fmri/pymvpa/'
                  'sub-sid000535_ses-actions2_task-actions_'
                  'hemi-rh_desc-hha_coef.gii') #arbitrary selection
    write_gifti(results, out_f, gifti_template)
    print('saved')
    print(out_f)


    
    
    
    
    
'''
# calculate t-test for model_group pairs
model_group_pairs = list(combinations(model_groups.keys(), 2))

t_results = {}
for model_pair in model_group_pairs:
    pair_label = "{0} - {1}".format(model_pair[0], model_pair[1]) #check correct direction
    t_results[pair_label] = {'lh': {}, 'rh': {}}
    for hemi in hemis:
        # mean difference calculation
        diff = np.vstack(dss[model_pair[0]][hemi]) - np.vstack(dss[model_pair[1]][hemi])
        mean = np.mean(diff, axis=0)
        
        # t-test calculation
        t_value, p_value = ttest_1samp(diff, 0, axis=0)
        # [] need update: save it as gifti -> python3:ttest_1samp with 'alternative=greater' 
        
        # store results
        t_results[pair_label][hemi]['mean'] = mean
        t_results[pair_label][hemi]['t-value'] = t_value
        t_results[pair_label][hemi]['p-value'] = p_value
        
    print("Finished running t-test for pair: {0}".format(pair_label))

print("Finished caculating t-test for all group pairs")

#FDR correction
for model_pair in model_group_pairs:
    pair_label = "{0} - {1}".format(model_pair[0], model_pair[1])
    lh_width = t_results[pair_label]['lh']['p-value'].shape[0]
    p_stack = np.hstack((t_results[pair_label]['lh']['p-value'],
                         t_results[pair_label]['rh']['p-value']))
    
    # fdr caculation
    threshold, q_value, _, _ = multipletests(p_stack, method='fdr_bh')
    
    # store results
    t_results[pair_label]['lh']['threshold'] = threshold[:lh_width]
    t_results[pair_label]['rh']['threshold'] = threshold[lh_width:]
    t_results[pair_label]['lh']['q-value'] = q_value[:lh_width]
    t_results[pair_label]['rh']['q-value'] = q_value[lh_width:]
    
    print("Finished FDR corrections for pair: {0}".format(pair_label))

print("Finished FDR correction for all group pairs")

#thresholded mean map
#save it also in niml.dset

convertdset = afni.ConvertDset()

for model_pair in model_group_pairs: 
    pair_label = "{0} - {1}".format(model_pair[0], model_pair[1])
    for hemi in hemis:
        mean_thresh = np.zeros(t_results[pair_label][hemi]['mean'].shape)
        mean_thresh[t_results[pair_label][hemi]['threshold']] = \
            t_results[pair_label][hemi]['mean'][t_results[pair_label][hemi]['threshold']]
        #
        mean_map = np.zeros(40962)
        mean_map[cortical_coords[hemi]] = mean_thresh
        ## save gifti
        out_f = join(mvpa_dir, 't_test_reg_' + "{0}-{1}_r2".format(model_pair[0], model_pair[1]) + "_{0}.gii".format(hemi))
        if sqr_r2 == True:
            out_f = join(mvpa_dir, 't_test_reg_' + "{0}-{1}_sqrR2".format(model_pair[0], model_pair[1]) + "_{0}.gii".format(hemi))
        print(out_f)
        #
        gifti_template = ('/backup/data/social_actions/fmri/pymvpa/'
                      'sub-sid000535_ses-actions2_task-actions_'
                      'hemi-rh_desc-hha_coef.gii') #arbitrary selection
        write_gifti(mean_map, out_f, gifti_template)
        ## save niml
        convertdset.inputs.in_file = out_f
        convertdset.inputs.out_type = 'niml_asc'
        if sqr_r2 == True:
            convertdset.inputs.out_file  = join(mvpa_dir, 't_test_reg_' + "{0}-{1}_sqrR2".format(model_pair[0], model_pair[1]) + "_{0}.niml.dset".format(hemi))
        else:
            convertdset.inputs.out_file  = join(mvpa_dir, 't_test_reg_' + "{0}-{1}_r2".format(model_pair[0], model_pair[1]) + "_{0}.niml.dset".format(hemi))
        convertdset.cmdline
        res = convertdset.run()
'''