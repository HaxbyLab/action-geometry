#!/usr/bin/env python3
# conda activate action-python3

import os
import numpy as np
import nibabel as nib
from scipy.stats import zscore
from scipy.sparse import load_npz
from nibabel.gifti import GiftiDataArray, GiftiImage 

## subject
sub_nums = ['5', '7', '9', '10', '12', 
            '13', '20', '21', '24', '29',
            '34', '52', '102', '114', '120',
            '134', '142','278', '416', '433', 
            '499', '522','535']
subjects = ['{:0>6}'.format(subid) for subid in sub_nums]

## run separately for session 1 and session 2 ####################
ses = 2
##################################################################

## data shape
n_sample = 90
n_vertices = 40962

## directory
basedir = '/backup/data/social_actions'
coef_dir = os.path.join(basedir, 'fmri', 'afni')
mvpa_dir = os.path.join(basedir, 'fmri', 'pymvpa')
trans_dir = os.path.join(basedir, 'fmri', 'hha', 'transformations') 

## we deal with actions1 and actions2 separately not concat
cortical_lh = np.load(f'{mvpa_dir}/cortical_mask_lh.npy')
cortical_rh = np.load(f'{mvpa_dir}/cortical_mask_rh.npy')

# Helper functions to read/write in GIfTI files - Sam Nastase
def read_gifti(gifti_fn):
    gii = nib.load(gifti_fn)
    data = np.vstack([da.data[np.newaxis, :]
                      for da in gii.darrays])
    return data

def write_gifti(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    gda = nib.gifti.GiftiDataArray(data)
    gii.add_gifti_data_array(gda)
    #nib.gifti.giftiio.write(gii, output_fn) #deprecated error
    nib.save(gii, output_fn)

## loop through subjects and apply mappers to coefficients
for s in subjects:
    ## GLM coefficients
    # load coef
    coef_lh_gii = nib.load(f'{coef_dir}/sub-sid{s}/ses-actions{ses}/func/'
                           f'sub-sid{s}_ses-actions{ses}_glm.lh.coefs_REML.gii')
    coef_rh_gii = nib.load(f'{coef_dir}/sub-sid{s}/ses-actions{ses}/func/'
                           f'sub-sid{s}_ses-actions{ses}_glm.rh.coefs_REML.gii')
    print(f"Finished loading coefficients for subject {s}")
    
    coef_lh = np.vstack(coef_lh_gii.agg_data())
    coef_rh = np.vstack(coef_rh_gii.agg_data())
    
    # remove medial wall vertices 
    coef_lh = coef_lh[:, cortical_lh]
    coef_rh = coef_rh[:, cortical_rh]
    
    # concat both hemisphere
    coef_bh = np.hstack([coef_lh, coef_rh])
    print(f'input data shape: {coef_bh.shape}')
    
    # z-score coefficients each vertex across stimuli
    coef_bh = zscore(coef_bh, axis=0) #shape (90, 74947)

    ## apply transformation matrix to GLM coefficient
    mapper = load_npz(f'{trans_dir}/subj{s}_mapper_all.npz')
    print(f'loaded mapper for subj{s}')
    
    # z-score coefficients after applying the transformation 
    print(f"Applying HHA mapper for subject {s}")
    aligned = zscore(np.nan_to_num(coef_bh @ mapper), axis=0) #(90, 74947)
    
    # re-split the hemisphere
    aligned_lh = aligned[:, :coef_lh.shape[1]]
    aligned_rh = aligned[:, coef_lh.shape[1]:]
    
    # put zeros back into medial wall
    template_lh = np.zeros((n_sample, n_vertices))
    template_rh = np.zeros((n_sample, n_vertices))

    template_lh[:, np.where(cortical_lh)[0]] = aligned_lh
    template_rh[:, np.where(cortical_rh)[0]] = aligned_rh
    
    ## save hyperaligned coefficients
    write_gifti(template_lh, (f'{mvpa_dir}/sub-sid{s}_ses-actions{ses}_'
                              f'task-actions_hemi-lh_desc-hha_coef.gii'), 
                f'{coef_dir}/sub-sid{s}/ses-actions{ses}/func/'
                f'sub-sid{s}_ses-actions{ses}_glm.lh.coefs_REML.gii')
    write_gifti(template_rh, (f'{mvpa_dir}/sub-sid{s}_ses-actions{ses}_'
                              f'task-actions_hemi-rh_desc-hha_coef.gii'), 
                f'{coef_dir}/sub-sid{s}/ses-actions{ses}/func/'
                f'sub-sid{s}_ses-actions{ses}_glm.rh.coefs_REML.gii')
    print(f"Finished applying HHA to subject {s}")
    