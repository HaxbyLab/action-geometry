'''
Jane Feb 13, 2024
calculate the motion energy RDM from pymoten npy

how to run this code
(action-python3) han@head1:/backup/data/social_actions/scripts$ python ./pymoten_rdm_debug.py

reference code: motion_energy_sam.py
'''
import os
import numpy as np
from glob import glob
import re
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from scipy.stats import zscore

rdm_dir = '/backup/data/social_actions/scripts/RDMs'
npy_dir = '/backup/data/social_actions/pymoten_data'
ha_dir = '/backup/data/social_actions/scripts/post_hha'

# read video list in from the set list 
# same as fMRI read: conditions = mv.h5load('condition_order.hdf5')['original_condition_order']
condition_order = ['arts_crafts_1', 'arts_crafts_3', 'arts_crafts_4', 'arts_crafts_6',
       'arts_crafts_7', 'assembly_line_10', 'assembly_line_12',
       'assembly_line_13', 'assembly_line_16', 'assembly_line_9',
       'conversation_26', 'conversation_29', 'conversation_30',
       'conversation_31', 'conversation_33', 'cooking_35', 'cooking_37',
       'cooking_38', 'cooking_39', 'cooking_40', 'cosmetics_grooming_41',
       'cosmetics_grooming_43', 'cosmetics_grooming_44',
       'cosmetics_grooming_46', 'cosmetics_grooming_48',
       'cosmetics_grooming_49', 'cosmetics_grooming_50',
       'cosmetics_grooming_51', 'cosmetics_grooming_55',
       'cosmetics_grooming_56', 'dancing_183', 'dancing_59', 'dancing_60',
       'dancing_61', 'dancing_62', 'dancing_63', 'dancing_64',
       'dancing_65', 'dancing_68', 'dancing_69', 'eating_180',
       'eating_181', 'eating_71', 'eating_72', 'eating_73', 'eating_76',
       'eating_81', 'eating_85', 'eating_86', 'eating_90',
       'exercising_102', 'exercising_104', 'exercising_106',
       'exercising_108', 'exercising_111', 'exercising_92',
       'exercising_93', 'exercising_95', 'exercising_98', 'exercising_99',
       'gardening_115', 'gardening_117', 'gardening_118', 'gardening_120',
       'gardening_123', 'hugging_kissing_124', 'hugging_kissing_125',
       'hugging_kissing_126', 'hugging_kissing_127',
       'hugging_kissing_130', 'playing_instrument_141',
       'playing_instrument_143', 'playing_instrument_146',
       'playing_instrument_149', 'playing_instrument_151', 'teaching_152',
       'teaching_154', 'teaching_155', 'teaching_158', 'teaching_159',
       'using_tools_162', 'using_tools_164', 'using_tools_166',
       'using_tools_168', 'using_tools_169', 'using_tools_170',
       'using_tools_172', 'using_tools_173', 'using_tools_180',
       'using_tools_181']

# load in pymoten npy
motion_vecs = [] 
for video_name in condition_order:
    motion_vec = np.load(os.path.join(npy_dir, f'moten_{video_name}.npy')) #(75, 2530)
    #raise #debug
    
# =================================
    # 8 different combinations
    # [log&zscore] x [method 1/method 2] x [zscore]

    ## log per video clip + zscore across video clip frames
    # solution0 no log
    # solution1 could add a constant (np.amin(motion_vec)+0.001) instead of doing nan_to_num
    #motion_vec = motion_vec + np.abs(np.amin(motion_vec)) + 0.001 #amin
    #motion_vec = np.log(motion_vec) 
    # solution 2 nan_to_num: could distort data 
    #motion_vec = np.nan_to_num(np.log(motion_vec)) #shape (75, 2530) #commented out for debug
    
    #motion_vec = zscore(motion_vec, axis=0) #shape (75, 2530)
    
    ## ------
    # method 1 (m1): averaging motion energy within one frame
    #motion_vec = np.mean(motion_vec, axis=0) #shape (2530,)
    #motion_vecs.append(motion_vec) #shape (90, 2530)
    
    # method 2 (m2): concatenate across frames rather than average                    
    #motion_vecs.append(motion_vec) #shape (90, 75x2530=189750)
    motion_vecs.append(motion_vec.reshape(-1))
    #print(np.shape(motion_vecs))
    # -------
    
motion_vecs = np.vstack(motion_vecs) 
#print(f'motion_vecs: {np.shape(motion_vecs)}') #(90, 2530) or (90, 75, 2530)

## z-score again, across the 90 video clips before computing rdm
motion_vecs = zscore(motion_vecs, axis=0) #shape (90, 2530) or (90, 75, 2530)
# ==========================================================

## rdm magic
motion_rdm = pdist(motion_vecs, 'correlation') #shape (4005,)
#print(f'motion_rdm: {motion_rdm.shape}') 

plt.matshow(squareform(1 - motion_rdm), vmin=-1, vmax=1, cmap='RdYlBu_r')
plt.colorbar()

## save <-  final! all debugged.
test_name = 'motion_energy' #correct video order
name_npy = f'pymoten_{test_name}_rdm.npy'
np.save(os.path.join(rdm_dir,name_npy), motion_rdm)
print(f'saved {os.path.join(rdm_dir,name_npy)}')

## plot
from scipy.stats import spearmanr
from os.path import join

sparse_ordered_labels = np.load(os.path.join(ha_dir,'sparse_ordered_labels.npy')).astype(str)

plt.figure(figsize=(8, 6))
ax = sns.heatmap(squareform(rankdata(motion_rdm) / len(motion_rdm) * 100)[reorder][:, reorder], vmin=0, vmax=100,
            square=True, cmap='RdYlBu_r', xticklabels=sparse_ordered_labels, yticklabels=sparse_ordered_labels)
ax.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.yticks(va='top')
plt.tight_layout()
plt.show()


## spearman check
cv_rdm_dir = '/backup/data/social_actions/fmri/cv_rdm'

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


hemi = 'lh'

roi = 'EV'
p_stack = []
for participant in participants.values():
    cv_rdm_fn = f'cv_rdm_sub-{participant}_{hemi}_{roi}.npy'
    p_stack.append(np.load(join(cv_rdm_dir, cv_rdm_fn)))
ev_rdm = np.mean(p_stack, axis=0)

roi = 'LO'
p_stack = []
for participant in participants.values():
    cv_rdm_fn = f'cv_rdm_sub-{participant}_{hemi}_{roi}.npy'
    p_stack.append(np.load(join(cv_rdm_dir, cv_rdm_fn)))
lo_rdm = np.mean(p_stack, axis=0)

print(f'EV: {spearmanr(squareform(ev_rdm, checks=False), motion_rdm)}')
print(f'LO: {spearmanr(squareform(lo_rdm, checks=False), motion_rdm)}')

# previous note
## save
#test_name tries
#simple_mean : no log no zscore whatsoever. just mean axis=0 and append
#nolog_zscore_mean : log doesn't seem necessary when amax is 14 and amin is -3?
#aminlog_zscore_mean_zscore: amin+log, zscore + mean, zscore across video clips
#aminlog_zscore_mean: amin+log, zscore + mean
#---
#np.save(os.path.join(rdm_dir,'pymoten_mean_target_RDM.npy'), motion_rdm)

#np.save(os.path.join(rdm_dir,'pymoten_logzscore_m1_zscore_target_RDM.npy'), motion_rdm)
#np.save(os.path.join(rdm_dir,'pymoten_logzscore_m1_nozscore_target_RDM.npy'), motion_rdm) #used this!
#np.save(os.path.join(rdm_dir,'pymoten_nologzscore_m1_zscore_target_RDM.npy'), motion_rdm)
#np.save(os.path.join(rdm_dir,'pymoten_nologzscore_m1_nozscore_target_RDM.npy'), motion_rdm)

#np.save(os.path.join(rdm_dir,'pymoten_logzscore_m2_zscore_target_RDM.npy'), motion_rdm)
#np.save(os.path.join(rdm_dir,'pymoten_logzscore_m2_nozscore_target_RDM.npy'), motion_rdm)
#np.save(os.path.join(rdm_dir,'pymoten_nologzscore_m2_zscore_target_RDM.npy'), motion_rdm)
#np.save(os.path.join(rdm_dir,'pymoten_nologzscore_m2_nozscore_target_RDM.npy'), motion_rdm)

#hdf5 save
#h5save(os.path.join(rdm_dir,'motionpymoten_target_RDM.npy'), motion_rdm)
