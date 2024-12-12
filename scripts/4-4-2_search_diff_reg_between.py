# 4-4-2_search_diff_reg_between.py
# 2024 May, Jane Han 
#
# Purpose 
# [Figure S2] for t-test for difference between multiple regression groups
# 
# Note 
# for all&nested use the permutation 4-4-3_search_diff_reg_unique.py
#
# How to run this code: 
# conda activate action-python2


## Import environments
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

model_groups = {'ts': ['transitivity', 'sociality'],
                'mg': ['motion','gaze']}
                #'ops': ['object', 'person', 'scene'], 
                #'vn': ['verb', 'nonverb']}

## Load each participants' searchlight correlation data
# loading input coming from 3-2_search_RSA_reg.py
#########################################
sqr_r2 = True ### change this flag for calculating R2 or sqr(R2)
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
            dss[model_group][hemi].append(np.nan_to_num(ds[0][cortical_masks[hemi]]))
print("Finished loading all model groups r2")
    
## calculate t-test for model_group pairs
model_group_pairs = list(combinations(model_groups.keys(), 2)) 

t_results = {}
for model_pair in model_group_pairs:
    pair_label = "{0} - {1}".format(model_pair[0], model_pair[1])
    t_results[pair_label] = {'lh': {}, 'rh': {}}
    for hemi in hemis:
        # mean difference calculation
        mean = np.mean(np.vstack(dss[model_pair[0]][hemi]) - np.vstack(dss[model_pair[1]][hemi]), axis=0)
        
        # t-test calculation
        t_value, p_value = ttest_rel(np.vstack(dss[model_pair[0]][hemi]), np.vstack(dss[model_pair[1]][hemi]), axis=0)
        
        # store results
        t_results[pair_label][hemi]['mean'] = mean
        t_results[pair_label][hemi]['t-value'] = t_value
        t_results[pair_label][hemi]['p-value'] = p_value
        
    print("Finished running t-test for pair: {0}".format(pair_label))

print("Finished caculating t-test for all group pairs")


'''
FDR correction
'''
for model_pair in model_group_pairs:
    pair_label = "{0} - {1}".format(model_pair[0], model_pair[1])
    lh_width = t_results[pair_label]['lh']['p-value'].shape[0]
    p_stack = np.hstack((t_results[pair_label]['lh']['p-value'],
                         t_results[pair_label]['rh']['p-value']))
    # fdr caculation
    threshold, q_value, _, _ = multipletests(p_stack, method='fdr_bh')
    # note: multipletests has the default alpha=0.05
    
    # store results
    t_results[pair_label]['lh']['threshold'] = threshold[:lh_width]
    t_results[pair_label]['rh']['threshold'] = threshold[lh_width:]
    t_results[pair_label]['lh']['q-value'] = q_value[:lh_width]
    t_results[pair_label]['rh']['q-value'] = q_value[lh_width:]
    
    print("Finished FDR corrections for pair: {0}".format(pair_label))

print("Finished FDR correction for all group pairs")


'''
thresholded mean map
'''
convertdset = afni.ConvertDset()

for model_pair in model_group_pairs: 
    pair_label = "{0} - {1}".format(model_pair[0], model_pair[1])
    for hemi in hemis:
        mean_thresh = np.zeros(t_results[pair_label][hemi]['mean'].shape)
        mean_thresh[t_results[pair_label][hemi]['threshold']] = \
            t_results[pair_label][hemi]['mean'][t_results[pair_label][hemi]['threshold']]
        
        mean_map = np.zeros(40962)
        mean_map[cortical_coords[hemi]] = mean_thresh
        
        ## save gii
        out_f = join(mvpa_dir, 't_test_reg_' + "{0}-{1}_r2".format(model_pair[0], model_pair[1]) 
                    + "_{0}.gii".format(hemi))
        if sqr_r2 == True:
            out_f = join(mvpa_dir, 't_test_reg_' + "{0}-{1}_sqrR2".format(model_pair[0], model_pair[1]) 
                        + "_{0}.gii".format(hemi))
        print(out_f)
        
        gifti_template = ('/backup/data/social_actions/fmri/pymvpa/'
                      'sub-sid000535_ses-actions2_task-actions_'
                      'hemi-rh_desc-hha_coef.gii') #arbitrary selection
        write_gifti(mean_map, out_f, gifti_template)
        
        ## save niml.dset
        convertdset.inputs.in_file = out_f
        convertdset.inputs.out_type = 'niml_asc'
        if sqr_r2 == True:
            convertdset.inputs.out_file  = join(mvpa_dir, 't_test_reg_' + "{0}-{1}_sqrR2".format(model_pair[0], model_pair[1]) 
                                                + "_{0}.niml.dset".format(hemi))
        else:
            convertdset.inputs.out_file  = join(mvpa_dir, 't_test_reg_' + "{0}-{1}_r2".format(model_pair[0], model_pair[1]) 
                                                + "_{0}.niml.dset".format(hemi))
        convertdset.cmdline
        res = convertdset.run()
