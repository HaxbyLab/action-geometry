#!/usr/bin/env python
'''
# 5-2_ROI_RSA_python3-unique.py
last edit May 31, 2026
save pickle for source data for NatComm Fig 4, using 9 models (excluding people, category, body), not 12 models

last edit Feb Jan 2026
added in ppl_rdm, actioncat_rdm, sns_rdm
exclusing sns_rdm, total of 12 models

* how to run this code: 
ssh head8, 
open a new tmux 
activate action-python3, 
then 
(action-python3) han@head8:/backup/data/social_actions/scripts/post_hha$ 
python ./5_ROI_RSA_python3-unique.py

# note
for unique variance calculation
this code is based on following codes:
3-3_search_RSA_reg-nested.py
4-4-3_search_diff_reg_unique.py
5_ROI_RSA_python3-regression.py

'''
from os import chdir, makedirs
from os.path import exists, join
from copy import deepcopy
from subprocess import call
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, rankdata, spearmanr, wilcoxon, zscore
from scipy.sparse import load_npz
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

import nibabel as nib
import h5py
from gifti_io import read_gifti
import pickle

## all subjects 
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

rois = ['EV', 'LO', 'VT', 'pSTS', 'TPJ', 'PPC', 'AIP', 'VPM', 'PMC'] #glasser #reordered
hemis = ['lh', 'rh']
model_names = ['motion', 'gaze', 'nonverbs', 'verbs', 'transitivity', 
               'sociality', 'person', 'scene', 'object']#, total 9 models for Fig4
              # 'people', 'category', 'body'] #total 12 models
              # 'number of people', 'action category', 'social nonsocial'] 
              # space doesn't work for formula 

center_features = False
zscore_features = True
hyperalign = True

n_conditions = 90
n_vertices = 40962
n_pairs = n_conditions * (n_conditions - 1) // 2

base_dir = '/backup/data/social_actions'
scripts_dir = join(base_dir, 'scripts')
data_dir = join(base_dir, 'fmri')
mvpa_dir = join(data_dir, 'pymvpa')
trans_dir = join(base_dir, 'fmri', 'hha', 'transformations')
figures_dir = join(base_dir, 'figures')
# we deal with actions1 and actions2 separately not concat #todo!
cortical_lh = np.load(f'{mvpa_dir}/cortical_mask_lh.npy')
cortical_rh = np.load(f'{mvpa_dir}/cortical_mask_rh.npy')

reorder = [10, 11, 12, 13, 14, 65, 66, 67, 68, 69, 75, 76, 77, 78, 79,
           5, 6, 7, 8, 9, 40, 41, 42, 44, 45, 31, 32, 33, 34, 35,
           55, 56, 57, 58, 59, 25, 26, 27, 28, 29, 80, 81, 82, 83, 84,
           43, 46, 47, 48, 49, 30, 36, 37, 38, 39, 50, 51, 52, 53, 54,
           20, 21, 22, 23, 24, 85, 86, 87, 88, 89, 15, 16, 17, 18, 19,
           60, 61, 62, 63, 64, 0, 1, 2, 3, 4, 70, 71, 72, 73, 74]

sparse_ordered_labels = np.load('sparse_ordered_labels.npy') 

'''
## SAVED###############################################################
# Load in neural data and compute RDMs
dss = {}
for participant_id, participant in participants.items():
    glm_dir = join(data_dir, 'afni', 'sub-'+participant)
    dss[participant] = {}
    dss[participant]['lh'] = {}
    dss[participant]['rh'] = {}
    
    s = participant[3:]
    for ses in [1, 2]:
    ## GLM coefficients
        coef_lh_gii = nib.load(f'{data_dir}/afni/sub-sid{s}/ses-actions{ses}/func/'
                               f'sub-sid{s}_ses-actions{ses}_glm.lh.coefs_REML.gii')
        coef_rh_gii = nib.load(f'{data_dir}/afni/sub-sid{s}/ses-actions{ses}/func/'
                               f'sub-sid{s}_ses-actions{ses}_glm.rh.coefs_REML.gii')
        print(f"Finished loading coefficients for sid{s}")
    
        coef_lh = np.vstack(coef_lh_gii.agg_data())
        coef_rh = np.vstack(coef_rh_gii.agg_data())

        # remove medial wall vertices 
        coef_lh = coef_lh[:, cortical_lh]
        coef_rh = coef_rh[:, cortical_rh]

        # concat both hemisphere
        coef_bh = np.hstack([coef_lh, coef_rh])
        print(f'input data shape: {coef_bh.shape}') #(90, 74947)

        # z-score coefficients each vertex across stimuli
        coef_bh = zscore(coef_bh, axis=0) 

        # apply transformation matrix to GLM coefficient
        mapper = load_npz(f'{trans_dir}/subj{s}_mapper_all.npz')
        print(f'loaded mapper for sid{s}')

        # z-score coefficients after applying the transformation 
        print(f"Applying HHA mapper for sid{s}")
        aligned = zscore(np.nan_to_num(coef_bh @ mapper), axis=0) 

        # re-split the hemisphere
        aligned_lh = aligned[:, :coef_lh.shape[1]]
        aligned_rh = aligned[:, coef_lh.shape[1]:]

        dss[participant]['lh'][ses] = aligned_lh
        dss[participant]['rh'][ses] = aligned_rh

neural_rdm_dir = '/backup/data/social_actions/scripts/RDMs/neural_rdm.pkl'

with open(neural_rdm_dir, "wb") as f:
    pickle.dump(dss, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved:", neural_rdm_dir)

# saved data above  ###############################################

'''

### load in neural rdm
neural_rdm_dir = '/backup/data/social_actions/scripts/RDMs/neural_rdm.pkl'
with open(neural_rdm_dir, "rb") as f:
     dss = pickle.load(f)

# Compute neural RDMs per ROI
cortical_masks = {'lh': cortical_lh, 'rh': cortical_rh}

neural_rdms = {}
for roi in rois:
    neural_rdms[roi] = {}
    for hemi in hemis:
        neural_rdms[roi][hemi] = {}
        for participant_id, participant in participants.items():

            # Compute cross-validated RDMs for each ROI
            mask = read_gifti(join(mvpa_dir, '{0}_mask_{1}.gii'.format(hemi, roi))) #glasser
            mask = mask[cortical_masks[hemi]]
            
            ds_roi_ses1 = dss[participant][hemi][1][:, (mask > 0)] #
            ds_roi_ses2 = dss[participant][hemi][2][:, (mask > 0)] #

            # Non-symmetric (and non-zero diagonal) matrix
            cv_nonsym = cdist(ds_roi_ses1, ds_roi_ses2, 'correlation')
            assert not np.array_equal(cv_nonsym.T, cv_nonsym)
            
            # Transpose and average, gets mean of both cross-validation directions
            cv_rdm = np.mean(np.dstack((cv_nonsym, cv_nonsym.T)), axis=2)
            assert np.array_equal(cv_rdm, cv_rdm.T)
            neural_rdms[roi][hemi][participant_id] = cv_rdm

    print("ROI {0}: Computed cross-validated RDMs".format(roi))

print("[done] finished computing neural RDMs per ROI")
    
### Load in target/model RDMs
motion_rdm = np.load(join(scripts_dir, 'RDMs', 'pymoten_motion_energy_rdm.npy'))
gaze_rdm = np.load(join(scripts_dir, 'RDMs', 'gaze_rdm.npy'))
verb_rdm = np.load(join(scripts_dir, 'RDMs', 'verb_rdm.npy'))
nonverb_rdm = np.load(join(scripts_dir, 'RDMs', 'nonverb_rdm.npy'))
object_rdm = np.load(join(scripts_dir, 'RDMs', 'object_rdm.npy'))
social_rdm = np.load(join(scripts_dir, 'RDMs', 'social_rdm.npy'))

static_object_rdm = np.load(join(scripts_dir, 'RDMs', 'frame_arrangement_object_RDM.npy'))
static_scene_rdm = np.load(join(scripts_dir, 'RDMs', 'frame_arrangement_scene_RDM.npy'))
static_person_rdm = np.load(join(scripts_dir, 'RDMs', 'frame_arrangement_person_RDM.npy'))

# ppl_rdm
# ppl_rdm = np.load(join(scripts_dir, "RDMs", "ppl_rdm.npy"))
# actioncat_rdm
# actioncat_rdm = np.load(join(scripts_dir, "RDMs", "action_category_rdm.npy"))
# body_rdm
# body_rdm = np.load(join(scripts_dir, "RDMs", "bodypart_rdm_JasonDazea.npy"))
# sns_rdm
# sns_rdm = np.load(join(scripts_dir, "RDMs", "social_nonsocial_rdm.npy"))
# object_direction_rdm (optional, not asked from R1)

model_rdms = {'motion': motion_rdm, 'gaze': gaze_rdm,
              'verbs': verb_rdm, 'nonverbs': nonverb_rdm,
              'transitivity': object_rdm, 'sociality': social_rdm,
              'object': static_object_rdm, 'scene': static_scene_rdm,
              'person': static_person_rdm} #,
              #'people': ppl_rdm, 
              #'category': actioncat_rdm, 
              #'body': body_rdm} 
              #'number of people': ppl_rdm, 
              #'action category': actioncat_rdm, 
              #'body parts': body_rdm} 
              #'social nonsocial': sns_rdm #optional

assert set(model_names) == set(model_rdms.keys()) 

#####
# Standardized rank regression using OLS
def unique_var(neural_rdms, model_rdms, neural_square=True, standardize=True, rank=True):
    """
    Returns
    -------
    results : dict
        results[roi][hemi]['full_r2'][pid] = R²(full)
        results[roi][hemi]['unique'][model_name][pid] = ΔR² for that model
        plus ['mean'] entries and bootstrap CI fields (added later).
    df_long : pd.DataFrame
        Long-form table with one row per (roi, hemi, pid, model_name).
    """
    
    import statsmodels.formula.api as smf
    
    predictors = list(model_names)
    
    def _as_condensed(rdm):
        rdm = np.asarray(rdm)
        if rdm.ndim == 2:
            return squareform(rdm, checks=False)
        return rdm
    
    rows = []
    results = {}
    for roi in rois:
        results[roi] = {}
        for hemi in hemis:
            results[roi][hemi] = {
                'full_r2': {},
                'unique': {m: {} for m in predictors},
            }

            
            for pid in participants.keys(): 
                variables = {m: _as_condensed(model_rdms[m]) for m in predictors}
                
                if neural_square:
                    variables['neural'] = squareform(neural_rdms[roi][hemi][pid], checks=False)
                else:
                    variables['neural'] = neural_rdms[roi][hemi][pid]
                
                # regression code
                if standardize and rank:
                    variables = {name: zscore(rankdata(v)) for name, v in variables.items()}
                elif standardize and not rank:
                    variables = {name: zscore(v) for name, v in variables.items()}
                elif rank and not standardize:
                    variables = {name: rankdata(v) for name, v in variables.items()}
                    
                df = pd.DataFrame(variables)
                
                # todo 
                # for a given model_name 
                # in the given roi & in the given hemi 
                # 1. run the full model -> get R2 of the full model 
                formula_full = "neural ~ " + " + ".join(predictors)
                res_full = smf.ols(formula_full, data=df).fit()
                r2_full = float(res_full.rsquared)
                results[roi][hemi]['full_r2'][pid] = r2_full

                
                # 2. run the nested model -> excluding model_name -> get R2 of this nested model
                # Nested + unique for each target
                for m in predictors:
                    pred_nested = [p for p in predictors if p != m]
                    formula_nested = "neural ~ " + " + ".join(pred_nested)
                    res_nested = smf.ols(formula_nested, data=df).fit()
                    r2_nested = float(res_nested.rsquared)

                    delta = r2_full - r2_nested
                    results[roi][hemi]['unique'][m][pid] = delta

                    rows.append({
                        "roi": roi,
                        "hemi": hemi,
                        "participant_id": pid,
                        "model": m,
                        "full_r2": r2_full,
                        "nested_r2": r2_nested,
                        "unique_delta_r2": delta,
                    })
                            
            # means across parcitipants 
            pids = list(participants.keys())
            results[roi][hemi]['full_r2']['mean'] = float(np.mean([results[roi][hemi]['full_r2'][p] for p in pids]))
            for m in predictors:
                results[roi][hemi]['unique'][m]['mean'] = float(np.mean([results[roi][hemi]['unique'][m][p] for p in pids]))

            print(f"[done] ROI={roi} hemi={hemi}: computed full and drop-one unique ΔR² for {len(pids)} participants")

    df_long = pd.DataFrame(rows)
    return results, df_long

            
### main ################################################ 
# 1) compute per-participant unique ΔR²
results, df_long = unique_var(neural_rdms, model_rdms, 
                              neural_square=True, standardize=True, rank=True)

# Save to p
# Fig4
out_pickle = join(mvpa_dir, 'ROI_unique_9models.p')
out_csv    = join(mvpa_dir, 'ROI_unique_9models.csv')

# supplementary 
#out_pickle = join(mvpa_dir, 'ROI_unique_12models.p')
#out_csv    = join(mvpa_dir, 'ROI_unique_12models.csv')


# 2) bootstrap across participants ONLY ################## 
# 1) compute per-participant unique ΔR²
#    (no permutations, no condition bootstrap)
def bootstrap_unique_participants(results, n_bootstraps=1000, seed=0):
    rng = np.random.default_rng(seed)
    pids = list(participants.keys())
    
    for _ in range(n_bootstraps):
        sample_pids = rng.choice(pids, size=len(pids), replace=True)
        
        for roi in rois:
            for hemi in hemis:
                for m in model_names:
                    if 'bootstrap_participants' not in results[roi][hemi]['unique'][m].keys():
                        results[roi][hemi]['unique'][m]['bootstrap_participants'] = []
                        
                    vals = np.array([results[roi][hemi]['unique'][m][pid] for pid in sample_pids], dtype=float)
                    boot_mean = np.mean(vals)
                    results[roi][hemi]['unique'][m]['bootstrap_participants'].append(boot_mean)
    
    for roi in rois:
        for hemi in hemis:
            for m in model_names:
                boots = results[roi][hemi]['unique'][m]['bootstrap_participants']
                results[roi][hemi]['unique'][m]['bootstrap_participants_ci'] = (
                np.percentile(boots, 2.5),
                np.percentile(boots, 97.5)
                )
    
    return results

results = bootstrap_unique_participants(results, n_bootstraps=1000, seed=0)

# Save outputs
with open(out_pickle, "wb") as f:
    pickle.dump(results, f)

df_long.to_csv(out_csv, index=False)
print(f"""Saved: {out_pickle} {out_csv} """)

### for plotting ################################################ 
# Load pickle results

# supplementary 
# results = pickle.load(open(join(mvpa_dir, 'ROI_unique_12models.p'), 'rb')) 


# Plot model test results 
def plot_unique_tests(results, roi, save_fn=None, ax=None):
    plot_order = [
        'motion', 'gaze', 'nonverbs', 'verbs',
        'scene', 'person', 'object',
        #'body', 'people', 'category', 
        'sociality', 'transitivity'
    ]
    
    # Build a tidy dataframe of means (one point per model × hemi)
    true_df = {'Unique ΔR²': [], 'hemisphere': [], 'model': []}
    for model_name in plot_order:
        for hemi in hemis:
            true_df['Unique ΔR²'].append(results[roi][hemi]['unique'][model_name]['mean'])
            true_df['model'].append(model_name)
            true_df['hemisphere'].append(hemi)
    true_df = pd.DataFrame(true_df)

    created_fig = False
    if ax is None:
        created_fig = True
        f, ax = plt.subplots(figsize=(4.5, 8))

    sns.pointplot(
        x='model', y='Unique ΔR²', data=true_df,
        hue='hemisphere', dodge=.37, join=False,
        order=plot_order,
        palette=['darkblue', 'darkred'],
        ax=ax
    )


    plt.setp(ax.lines, zorder=1000000)
    plt.setp(ax.collections, zorder=1000000, label="", sizes=[30])
    ax.axhline(y=0, linewidth=4, color='.75', linestyle='-', zorder=1)

    # Add participant-bootstrap CIs (same idea as your original plotting code)
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)

    participants_errorbars = []
    y_i = 0
    for hemi in hemis:
        for model_name in plot_order:
            ci = results[roi][hemi]['unique'][model_name]['bootstrap_participants_ci']
            participants_errorbars.append([y_coords[y_i] - ci[0], ci[1] - y_coords[y_i]])
            y_i += 1
    participants_errorbars = np.array(participants_errorbars).T

    # Left half of points are first hemi, right half are second hemi (as produced by seaborn)
    ax.errorbar(
        x_coords[:len(plot_order)], y_coords[:len(plot_order)],
        yerr=participants_errorbars[:, :len(plot_order)], fmt=' ',
        elinewidth=4, ecolor='darkblue'
    )
    ax.errorbar(
        x_coords[len(plot_order):], y_coords[len(plot_order):],
        yerr=participants_errorbars[:, len(plot_order):], fmt=' ',
        elinewidth=4, ecolor='darkred'
    )

    ax.set(ylabel='Unique variance (ΔR²)', title=roi)
    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.tick_params(axis='x', rotation=90)
    
    if created_fig:
        plt.tight_layout()
        if save_fn:
            plt.savefig(save_fn + '.svg', format='svg', transparent=True)
            plt.savefig(save_fn + '.png', format='png', dpi=300, transparent=True)
        else:
            plt.show()
        plt.close()

for roi in rois:
    #plot_unique_tests(results, roi)
    plot_unique_tests(results, roi, save_fn=f'/backup/data/social_actions/figures/unique_roi-{roi}_9models')
    #plot_unique_tests(results, roi, save_fn=f'/backup/data/social_actions/figures/unique_roi-{roi}_12models')

### one-shot view plot ################################################ 
def plot_all_rois_grid(results, rois, save_fn=None):
    fig, axes = plt.subplots(2, 5, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    n_panels = 10
    n_rois = len(rois)  # should be 9

    for i in range(n_panels):
        if i < n_rois:
            roi = rois[i]
            plot_unique_tests(results, roi, ax=axes[i])
            '''
            axes[i].set_xlabel("")  # no per-panel x label

            # Hide x tick labels except bottom row (row indices 4 => panels 8 and 9)
            if i < 8:
                axes[i].set_xticklabels([])
                axes[i].tick_params(axis='x', length=0)  # optional: also hide tick marks
            '''
        else:
            axes[i].axis('off')  # last empty panel

    plt.tight_layout()
    if save_fn:
        plt.savefig(save_fn + '.svg', format='svg', transparent=True)
        plt.savefig(save_fn + '.png', format='png', dpi=300, transparent=True)
    else:
        plt.show()
    plt.close()

plot_all_rois_grid(
    results, rois,
    save_fn='/backup/data/social_actions/figures/unique_roiGrid_9models'
    #save_fn='/backup/data/social_actions/figures/unique_roiGrid_12models'
)
#plot_all_rois_grid(results, rois)
