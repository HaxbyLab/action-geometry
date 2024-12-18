#!/usr/bin/env python

# Run from ~/social_actions/fmri with something like
# ./actions_glm_clean.py sid000021 1 1 |& tee afni/logs/glm_p1_s1_log.txt &

import sys
import json
from os import chdir, makedirs, system
from os.path import exists, join
from subprocess import call
from mvpa2.base.hdf5 import h5load
from natsort import natsorted
import numpy as np

participant = sys.argv[1]
participant_id = sys.argv[2]
session = sys.argv[3]

## WARNING
# Note that we need to use the onsets labeled p9 for session 1
# for subject 8 and the onsets labeled p8 s1 for subject 9

base_dir = '/home/han/social_actions'
scripts_dir = join(base_dir, 'scripts')

##### fMRI data directory
fmri_dir = join(base_dir, 'fmri')
prep_dir = join('/backup/home/nastase/fmriprep', 'fmriprep', 'sub-'+participant, 'ses-actions'+session, 'func') #2021-06-01
#fMRIprep directory:/backup/home/nastase/fmriprep/fmriprep 
#prep_dir = join(fmri_dir, 'fmriprep', 'sub-'+participant, 'ses-actions'+session, 'func')

##### GLM directory
glm_dir = join(fmri_dir, 'afni', 'sub-'+participant, 'ses-actions'+session, 'func')
if not exists(glm_dir):
    makedirs(glm_dir)
    copy_dir = join(prep_dir,'*')
    system("cp " + copy_dir + " " + glm_dir)
    #2021-06-01
reg_dir = join(glm_dir, 'regressors')
if not exists(reg_dir):
    makedirs(reg_dir)

##############################################
## Convert fmriprep's confounds.tsv for 3dDeconvolve -ortvec
ortvecs = []
keep_a = []
for run in [1, 2, 3, 4]:
    with open(join(prep_dir, f'sub-{participant}_ses-actions{session}_task-actions_run-{run}_desc-confounds_regressors.tsv')) as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        
    with open(join(prep_dir, f'sub-{participant}_ses-actions{session}_task-actions_run-{run}_desc-confounds_regressors.json')) as f:
        conf_json = json.load(f)
        
    csf_comps, wm_comps = [], []

    a_comps = natsorted([a for a in conf_json.keys()
                         if 'a_comp_cor' in a])
    for a_comp in a_comps:
        if conf_json[a_comp]['Mask'] == 'CSF' and len(csf_comps) < 5:
            csf_comps.append(a_comp)
        if conf_json[a_comp]['Mask'] == 'WM' and len(wm_comps) < 5:
            wm_comps.append(a_comp)

    confounds = {}
    for confound_i, confound in enumerate(lines[0]):
        confound_ts = []
        for tr in lines[1:]:
            confound_ts.append(tr[confound_i])
        confounds[confound] = confound_ts

    keep = ['trans_x', 'trans_x_derivative1','trans_y','trans_y_derivative1', 'trans_z', 'trans_z_derivative1',
            'rot_x','rot_x_derivative1', 'rot_y','rot_y_derivative1', 'rot_z','rot_z_derivative1',
            'cosine01', 'cosine02', 'cosine03', 'cosine04', 'cosine05', 'cosine06']
    keep = keep + csf_comps + wm_comps
    keep_a.append(keep)
        
    ortvec = {c: confounds[c] for c in keep}

    assert len(ortvec) == 28 ###05-20-2021
    ortvecs.append(ortvec)

    with open(join(reg_dir, 'sub-{0}_ses-actions{1}_task-actions_run-0{2}_bold_ortvec.1D'.format(participant, session, run)), 'w') as f:
        rows = []
        for tr in range(len(ortvec[keep[0]])):
            row = []
            for confound in keep:
                if ortvec[confound][tr] == 'n/a':
                    row.append('0')
                else:
                    row.append(ortvec[confound][tr])
            row = '\t'.join(row)
            rows.append(row)
        f.write('\n'.join(rows))
###################END RUN LOOP###########################
# total 535 TRs per Run = len(ortvec[keep[0]])
# 4 runs
# total TR = 2140 TRs
keep_a  = np.array(keep_a).T.tolist()

## Concatenate ortvec time series across runs
comp_id = 1
rall_ortvec = {}
for confound in keep_a:
    rall_ts = []
    for r, run_ortvec in enumerate(ortvecs):
        rall_ts.extend(run_ortvec[confound[r]])
        if 'a_comp_cor' in confound[r]:
            confound[r] = f'a_comp_cor_{comp_id}'
            #print(f'a_comp_cor_{comp_id}')#check
    assert len(rall_ts) == 2140
    if 'a_comp_cor' in confound[r]:
        comp_id += 1

    rall_ortvec[confound[0]] = rall_ts
    
assert len(rall_ortvec.keys()) == len(keep)
keep = [k[0] for k in keep_a]

with open(join(reg_dir, 'sub-{0}_ses-actions{1}_task-actions_rall_bold_ortvec.1D'.format(participant, session)), 'w') as f:
    rows = []
    for tr in range(len(rall_ortvec[keep[0]])):
        row = []
        for confound in keep:
            if rall_ortvec[confound][tr] == 'n/a':
                row.append('0')
            else:
                row.append(rall_ortvec[confound][tr])
        row = '\t'.join(row)
        rows.append(row)
    f.write('\n'.join(rows))
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#
## Convert stimulus onsets to 3dDeconvolve stim_times format
stim_times = {}
prep_times = []
question_times = []
for run in [1, 2, 3, 4]:
    # Switch session 1 onsets for participants 8 and 9 (ugh...)
    if participant_id == '8' and session == '1':
        trial_order = h5load(join(scripts_dir, 'trial_orders', 'trial_order_p{0}_s{1}_r{2}.hdf5'.format(9, 1, run)))
    elif participant_id == '9' and session == '1':
        trial_order = h5load(join(scripts_dir, 'trial_orders', 'trial_order_p{0}_s{1}_r{2}.hdf5'.format(8, 1, run)))
    else:
        trial_order = h5load(join(scripts_dir, 'trial_orders', 'trial_order_p{0}_s{1}_r{2}.hdf5'.format(participant_id, session, run)))
    run_prep = []
    run_question = []
    for trial_i, trial in enumerate(trial_order):
        if trial[2] == b'fixation':
            continue
        elif trial[2] == b'question':
            run_question.append(trial[0])
            continue
        elif trial_i < 3:
            run_prep.append(trial[0])
            continue
        elif trial[2] not in stim_times.keys():
            stim_times[trial[2]] = [trial[0]]
        elif trial[2] in stim_times.keys():
            stim_times[trial[2]].append(trial[0])
    prep_times.append(run_prep)
    question_times.append(run_question)
assert len(stim_times.keys()) == 90
for times in stim_times.values():
    assert len(times) == 4

with open(join(reg_dir, 'sub-{0}_ses-actions{1}_prep_reg.txt'.format(participant, session)), 'w') as f:
    prep_times = [' '.join([str(onset) for onset in prep]) for prep in prep_times]
    f.write('\n'.join(prep_times))
prep_reg = ("-stim_times 91 {0} 'BLOCK(2.5,1)' -stim_label 91 prep".format(
                join(reg_dir, 'sub-{0}_ses-actions{1}_prep_reg.txt'.format(participant, session))))

with open(join(reg_dir, 'sub-{0}_ses-actions{1}_question_reg.txt'.format(participant, session)), 'w') as f:
    question_times = [' '.join([str(onset) for onset in question]) for question in question_times]
    f.write('\n'.join(question_times))
question_reg = ("-stim_times 92 {0} 'BLOCK(2,1)' -stim_label 92 question".format(
                join(reg_dir, 'sub-{0}_ses-actions{1}_question_reg.txt'.format(participant, session))))

for stimulus, onsets in stim_times.items():
    stim_label = stimulus[:-10].decode('utf-8')
    with open(join(reg_dir, 'sub-{0}_ses-actions{1}_{2}_reg.txt'.format(participant, session, stim_label)), 'w') as f:
    #with open(join(reg_dir, 'sub-{0}_ses-actions{1}_{2}_reg.txt'.format(participant, session, stimulus[:-10])), 'w') as f: #2021-06-02
        f.write('\n'.join([str(onset) for onset in onsets]))

stimulus_labels = [stimulus[:-10] for stimulus in sorted(stim_times.keys())]
stimulus_regs = []
for stim_i, stim_label in enumerate(stimulus_labels):
    stim_label = stim_label.decode('utf-8')
    stimulus_regs.append("-stim_times {0} {1} 'BLOCK(2.5,1)' -stim_label {0} {2}".format(
            stim_i + 1, join(reg_dir, 'sub-{0}_ses-actions{1}_'.format(participant, session) + stim_label) + '_reg.txt', stim_label))
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#
## Change directory for AFNI
chdir(glm_dir)
### make sure this directory makes sense for the new directories and new fmriprep files

## Run AFNI's 3dDeconvolve
for side, hemi in [('L', 'lh'), ('R', 'rh')]: 
    cmd = ("3dDeconvolve -polort A -jobs 8 "
                "-input "
                "{0}/sub-{1}_ses-actions{2}_task-actions_run-1_space-fsaverage6_hemi-{3}_bold.func.gii "
                "{0}/sub-{1}_ses-actions{2}_task-actions_run-2_space-fsaverage6_hemi-{3}_bold.func.gii "
                "{0}/sub-{1}_ses-actions{2}_task-actions_run-3_space-fsaverage6_hemi-{3}_bold.func.gii "
                "{0}/sub-{1}_ses-actions{2}_task-actions_run-4_space-fsaverage6_hemi-{3}_bold.func.gii "
                "-local_times -num_stimts 92 ".format(prep_dir, participant, session, side) +
                ' '.join(stimulus_regs) + ' ' + prep_reg + ' ' + question_reg + ' ' +
                "-ortvec {4} "
                "-fout -tout -x1D {3}/sub-{1}_ses-actions{2}_glm.{0}.X.xmat.1D "
                "-xjpeg {3}/sub-{1}_ses-actions{2}_glm.{0}.X.jpg "
                "-fitts {3}/sub-{1}_ses-actions{2}_glm.{0}.fitts "
                "-errts {3}/sub-{1}_ses-actions{2}_glm.{0}.errts "
                "-bucket {3}/sub-{1}_ses-actions{2}_glm.{0}.stats".format(hemi,
                    participant, session, glm_dir, join(reg_dir, 
            'sub-{0}_ses-actions{1}_task-actions_rall_bold_ortvec.1D'.format(participant, session, run))))
    call(cmd, shell=True)
    #run(cmd, shell=True)

    ## Run AFNI's 3dREMLfit
    ### redoing fancier GLM temporal autocorrelation signal 
    ### [_]compare it and understand the diff with 3dDeconvolve
    
    chdir(glm_dir)
    with open(join(glm_dir, 'sub-{0}_ses-actions{1}_glm.REML_cmd'.format(participant, session))) as f:
        reml_cmd = '3dREMLfit' + f.read().split('3dREMLfit')[-1]
    call(reml_cmd, shell=True)

    ## Grab coefficients
    chdir(glm_dir)
    call("3dbucket -prefix sub-{0}_ses-actions{1}_glm.{2}.coefs_REML.gii "
         "'sub-{0}_ses-actions{1}_glm.{2}.stats_REML.gii[1..179(2)]'".format(participant, session, hemi), shell=True)

    call("ConvertDset -o_niml_asc -input sub-{0}_ses-actions{1}_glm.{2}.coefs_REML.gii "
         "-prefix sub-{0}_ses-actions{1}_glm.{2}.coefs_REML".format(participant, session, hemi), shell=True)
