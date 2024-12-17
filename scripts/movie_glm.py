#!/usr/bin/env python

# Run from ~/social_actions/scripts with something like
# ./movie_glm.py sid000021 |& tee ../logs/movie_glm_p1_s1_log.txt &

import sys
import json
from os import chdir, makedirs
from os.path import exists, join
from subprocess import call
from natsort import natsorted

participant = sys.argv[1]

base_dir = '/home/nastase/social_actions'
scripts_dir = join(base_dir, 'scripts')
prep_dir = join(base_dir, 'fmri', 'afni', 'sourcedata', 'fmriprep',
                f'sub-{participant}', 'ses-raiders', 'func')
glm_dir = join(base_dir, 'fmri', 'afni', 'afni',
               f'sub-{participant}', 'ses-raiders', 'func')
if not exists(glm_dir):
    makedirs(glm_dir)

# Convert fmriprep's confounds.tsv for 3dDeconvolve -ortvec
ortvecs = []
for run in [1, 2, 3, 4]:
    with open(join(prep_dir,
                   (f'sub-{participant}_ses-raiders_task-movie_'
                    f'run-{run}_desc-confounds_regressors.tsv'))) as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        
    with open(join(prep_dir,
                   (f'sub-{participant}_ses-raiders_task-movie_'
                    f'run-{run}_desc-confounds_regressors.json'))) as f:
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

    keep = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
            'cosine01', 'cosine02', 'cosine03', 'cosine04', 'cosine05',
            'cosine06', 'cosine07', 'cosine08', 'cosine09', 'cosine10',
            'cosine11']
    keep = keep + csf_comps + wm_comps
    
    ortvec = {c: confounds[c] for c in keep}

    # Save confounds in 1D file for AFNI
    with open(join(glm_dir, (f'sub-{participant}_ses-raiders_task-movie_'
                   f'run-{run}_desc-model_regressors.1D')), 'w') as f:
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

# Concatenate ortvec time series across runs
rall_ortvec = {}
for confound in keep:
    rall_ts = []
    for run_ortvec in ortvecs:
        rall_ts.extend(run_ortvec[confound])
    #assert len(rall_ts) == 2140
    rall_ortvec[confound] = rall_ts
assert len(rall_ortvec.keys()) == len(keep)

ort_fn = join(glm_dir, (f'sub-{participant}_ses-raiders_task-movie_'
                        'run-all_desc-model_regressors.1D'))

with open(ort_fn, 'w') as f:
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

# Change directory for AFNI's sanity
chdir(glm_dir)

# Run AFNI's 3dTproject
for hemi in ['L', 'R']:
    
    input_fns = ' '.join([join(prep_dir, (f'sub-{participant}_ses-raiders_'
                                          f'task-movie_run-{run}_'
                                          'space-fsaverage6_'
                                          f'hemi-{hemi}_bold.func.gii'))
                          for run in [1, 2, 3, 4]])

    output_fn = join(glm_dir, (f'sub-{participant}_ses-raiders_task-movie_'
                               f'run-{run}_space-fsaverage6_hemi-{hemi}_'
                               'desc-clean_bold.func.gii'))
    
    run(f'3dTproject -polort 2 -TR 1.0 -input {input_fns} '
          f'-ort {ort_fn} -prefix {output_fn}', shell=True)
    print("Finished out confounds using AFNI's 3dTproject for"
          f'{participant} (hemi {hemi})')
