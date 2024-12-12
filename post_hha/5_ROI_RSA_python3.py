#!/usr/bin/env python
'''
5_ROI_RSA_python3.py
2024 February, Jane Han

Purpose
[Figure 4] ROI analysis

* how to run this code: 
ssh head8
activate action-python3
python ./5_ROI_RSA_python3.py
'''

## Import environments
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


rois = ['EV', 'LO', 'VT', 'AIP', 'VPM', 'pSTS', 'TPJ', 'PPC', 'PMC']# glasser 
hemis = ['lh', 'rh']
model_names = ['motion', 'gaze', 'nonverbs', 'verbs', 'transitivity', 
               'sociality', 'person', 'scene', 'object']

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

## Load in neural data and compute RDMs
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

## Compute neural RDMs per ROI
cortical_masks = {'lh': cortical_lh, 'rh': cortical_rh}

neural_rdms = {}
for roi in rois:
    neural_rdms[roi] = {}
    for hemi in hemis:
        neural_rdms[roi][hemi] = {}
        for participant_id, participant in participants.items():

            # Compute cross-validated RDMs for each ROI
            #mask = read_gifti(join(mvpa_dir, '{0}.mask_{1}.gii'.format(hemi, roi))) # hand-drawn rois 
            mask = read_gifti(join(mvpa_dir, '{0}_mask_{1}.gii'.format(hemi, roi))) #glasser rois
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

    print("Computed cross-validated RDMs for ROI {0}".format(roi))

## Load in target/model RDMs
motion_rdm = np.load(join(scripts_dir, 'RDMs', 'pymoten_motion_energy_rdm.npy'))
gaze_rdm = np.load(join(scripts_dir, 'RDMs', 'gaze_rdm.npy'))
verb_rdm = np.load(join(scripts_dir, 'RDMs', 'verb_rdm.npy'))
nonverb_rdm = np.load(join(scripts_dir, 'RDMs', 'nonverb_rdm.npy'))
object_rdm = np.load(join(scripts_dir, 'RDMs', 'object_rdm.npy'))
social_rdm = np.load(join(scripts_dir, 'RDMs', 'social_rdm.npy'))

static_object_rdm = np.load(join(scripts_dir, 'RDMs', 'frame_arrangement_object_RDM.npy'))
static_scene_rdm = np.load(join(scripts_dir, 'RDMs', 'frame_arrangement_scene_RDM.npy'))
static_person_rdm = np.load(join(scripts_dir, 'RDMs', 'frame_arrangement_person_RDM.npy'))

model_rdms = {'motion': motion_rdm, 'gaze': gaze_rdm,
              'verbs': verb_rdm, 'nonverbs': nonverb_rdm,
              'transitivity': object_rdm, 'sociality': social_rdm,
              'object': static_object_rdm, 'scene': static_scene_rdm,
              'person': static_person_rdm}
assert set(model_names) == set(model_rdms.keys()) 

# Compute Spearman correlations between ROI RDMs
def inter_roi_correlations(neural_rdms, average_late=True):
    if average_late:
        rois_rdm = np.mean(np.vstack([pdist(
                           [rankdata(squareform(neural_rdms[roi][h][p],
                                                 checks=False))
                             for roi in rois for h in hemis], 'correlation')
                           for p in participants.keys()]), axis=0)
    elif not average_late:
        rois_rdm = pdist([rankdata(np.mean(np.vstack(
                            [squareform(neural_rdms[roi][h][p], checks=False)
                             for p in participants.keys()]),
                                           axis=0))
                          for roi in rois for h in hemis], 'correlation')
    return rois_rdm

# Compute Spearman correlations between subject RDMs (ISCs)
def inter_subject_correlations(neural_rdms, roi, hemi, pairwise=True):
    if pairwise:
        roi_iscs = pdist([rankdata(squareform(neural_rdms[roi][hemi][p],
                                              checks=False))
                          for p in sorted(participants.keys())],
                         'correlation')
        assert len(roi_iscs) == (len(participants) *
                                 (len(participants) - 1) / 2)
        print("Mean Spearman correlation = {0}, SD = {1}".format(
                np.mean(1 - roi_iscs), np.std(1 - roi_iscs)))
        return roi_iscs

    elif not pairwise:
        roi_iscs = []
        for participant in sorted(participants.keys()):
            lo_rdm = squareform(neural_rdms[roi][hemi][participant], checks=False)
            mean_rdm = np.mean(np.vstack(
                        [squareform(neural_rdms[roi][hemi][p], checks=False)
                         for p in sorted(participants.keys())
                         if p is not participant]), axis=0)
            assert len(mean_rdm) == n_pairs
            assert lo_rdm.shape == mean_rdm.shape
            roi_iscs.append(spearmanr(lo_rdm, mean_rdm)[0])
        print("Mean spearman correlation = {0}, SD = {1}".format(
                np.mean(roi_iscs), np.std(roi_iscs)))
        return roi_iscs

# Compute Spearman correlations between target/model RDMs
def inter_model_correlations(model_rdms, model_names=None):
    if model_names:
        models_rdm = pdist([rankdata(model_rdms[name]) for name in model_names], 'correlation')
    elif not model_names:
        model_names = sorted(model_rdms.keys())
        models_rdm = pdist([rankdata(model_rdms[name]) for name in model_names], 'correlation')
    return models_rdm

# Compute mean (across participants) neural RDM for ROI
def mean_neural_rdm(neural_rdms, roi, hemi, save_fn=None):
    assert roi in neural_rdms.keys()
    mean_rdm = np.mean(np.dstack([neural_rdms[roi][hemi][p]
                                  for p in participants.keys()]), axis=2)
    if save_fn:
        pickle.dump(results, open(save_fn, 'wb'))
    return mean_rdm

# Plot correlation matrix
def plot_correlation_matrix(rdm, save_fn=None, **kwargs):
    ax = sns.heatmap(1 - squareform(rdm), square=True, annot=True,
                     cmap='RdYlBu_r', vmin=-1., vmax=1., fmt='.2f',
                     cbar_kws={'label': 'Spearman correlation'},
                     **kwargs)
    ax.xaxis.tick_top()
    #plt.xticks(rotation=45, ha='left')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Spearman correlation',
                   rotation=270)
    cbar.set_ticks([-1, -.5, 0, .5, 1.])
    #plt.xticks(rotation=45, ha='left')
    plt.tight_layout()
    if save_fn:
        plt.savefig(save_fn + '.svg', format='svg', transparent=True)
        plt.savefig(save_fn + '.png', format='png', dpi=300, transparent=True)
    else:
        plt.show()
    plt.close('all')

# Plot average (across participants) cross-validated neural RDM
def plot_percentile_RDM(rdm, square=True, reorder=False, save_fn=None, **kwargs):
    if square:
        assert rdm.shape[0] == rdm.shape[1]
        n = rdm.shape[0]
        n_pair = n * (n - 1) // 2
        percents = (rankdata(np.concatenate((squareform(rdm, checks=False), 
                                             np.diag(rdm))))
                            / (n_pair + n) * 100)
        rdm = squareform(percents[:n_pair])
        np.fill_diagonal(rdm, percents[n_pair:])
    elif not square:
        rdm = rankdata(rdm) / len(rdm) * 100
    plt.figure(figsize=(8, 6))
    if reorder:
        rdm = rdm[reorder][:, reorder]
    ax = sns.heatmap(rdm, vmin=0, vmax=100, square=True, linewidth=.5, cmap='RdYlBu_r',
                     **kwargs)
    cbar = ax.collections[0].colorbar
    cbar.set_label('dissimilarity (percentile correlation distance)',
                   rotation=270)
    cbar.set_ticks([0, 100])
    cbar.set_ticklabels(['0%', '100%'])
    ax.xaxis.tick_top()
    plt.xticks(rotation=45, ha='left')
    plt.yticks(va='top')
    plt.tight_layout()
    if save_fn:
        plt.savefig(save_fn + '.svg', format='svg', transparent=True)
        plt.savefig(save_fn + '.png', format='png', dpi=300, transparent=True)
    else:
        plt.show()
    plt.close('all')

### Inter-subject Spearman correlations for an ROI
for roi in rois:
    for hemi in hemis:
        roi_iscs = inter_subject_correlations(neural_rdms, roi, hemi, pairwise=True)
        plot_correlation_matrix(roi_iscs, save_fn=join(figures_dir, 'ISC_hyper_{0}_{1}'.format(roi, hemi)),
                                xticklabels=sorted(participants.keys()),
                            yticklabels=sorted(participants.keys()))

### Inter-ROI Spearman correlations
rois_rdm = inter_roi_correlations(neural_rdms, average_late=True)
labels = ['{0} {1}'.format(hemi, roi) for roi in rois for hemi in ['left', 'right']]
plot_correlation_matrix(rois_rdm, save_fn=join(figures_dir, 'inter_ROI_correlations_hyper'), 
                        xticklabels=labels,
                        yticklabels=labels)

# Inter-model Spearman correlations
model_names = ['motion', 'gaze', 'nonverbs', 'verbs', 'transitivity', 'sociality','object','scene','person']
models_rdm = inter_model_correlations(model_rdms, model_names)
plot_correlation_matrix(models_rdm, save_fn=join(figures_dir, 'inter_model_correlations'),
                        xticklabels=model_names,
                        yticklabels=model_names)
#####    
### Mean neural RDM across subjects    
for roi in rois:
    for hemi in hemis:
        mean_rdm = mean_neural_rdm(neural_rdms, roi, hemi)
        plot_percentile_RDM(mean_rdm, reorder=reorder,
                            xticklabels=sparse_ordered_labels,
                            yticklabels=sparse_ordered_labels)
###
# Compute group Spearman correlations
def spearman(neural_rdms, model_rdms, neural_square=True):
    results = {}
    for roi in rois:
        results[roi] = {}
        for hemi in hemis:
            results[roi][hemi] = {}
            for model_name in model_names:
                results[roi][hemi][model_name] = {}
                for participant_id in participants.keys():
                    if neural_square:
                        neural_rdm = squareform(neural_rdms[roi][hemi][participant_id],
                                                checks=False)
                    elif not neural_square:
                        neural_rdm = neural_rdms[roi][hemi][participant_id]
                    results[roi][hemi][model_name][participant_id] = \
                            spearmanr(neural_rdm, model_rdms[model_name])[0]
                results[roi][hemi][model_name]['mean'] = np.mean(
                                    [results[roi][hemi][model_name][p]
                                     for p in participants.keys()])
    return results

# Partial Spearman correlations
def partial_spearman(neural_rdms, model_rdms, neural_square=True):
    from partial_corr import partial_corr
    results = {}
    for roi in rois:
        results[roi] = {}
        for hemi in hemis:
            results[roi][hemi] = {}
            for model_name in model_names:
                results[roi][hemi][model_name] = {}
            for participant_id in participants.keys():
                if neural_square:
                    neural_rdm = squareform(neural_rdms[roi][hemi][participant_id], checks=False)
                elif not neural_square:
                    neural_rdm = neural_rdms[roi][hemi][participant_id]
                constant = np.ones(neural_rdm.shape)
                variables = [neural_rdm, constant]
                for model_name in model_names:
                    variables.append(model_rdms[model_name])
                variables = np.column_stack(variables)
                partials = partial_corr(variables)
                assert len(partials[2:]) == len(model_rdms) == 9
                for model_i, model_name in enumerate(sorted(model_names)):
                    results[roi][hemi][model_name][participant_id] = partials[0, 2:][model_i]
            for model_name in model_names:
                results[roi][hemi][model_name]['mean'] = np.mean([results[roi][hemi][model_name][p]
                                                             for p in participants.keys()])
    return results

# Standardized rank regression using OLS
def regression(neural_rdms, model_rdms, neural_square=True,
               standardize=True, rank=True):
    import pandas as pd
    import statsmodels.formula.api as smf
    
    results = {}
    for roi in rois:
        results[roi] = {}
        for hemi in hemis:
            results[roi][hemi] = {}
            results[roi][hemi]['R-squared'] = {}
            for model_name in model_names:
                results[roi][hemi][model_name] = {}
            for participant_id in participants.keys():
                variables = {model_name: model_rdm for model_name, model_rdm
                             in model_rdms.items()}
                if neural_square:
                    variables['neural'] = squareform(
                            neural_rdms[roi][hemi][participant_id], checks=False)
                elif not neural_square:
                    variables['neural'] = neural_rdms[roi][hemi][participant_id]
                    
                if standardize and rank:
                    variables = {name: zscore(rankdata(rdm))
                                 for name, rdm in variables.items()} 
                elif standardize and not rank:            
                    variables = {name: zscore(rdm)
                                 for name, rdm in variables.items()}
                elif rank and not standardize:
                    variables = {name: rankdata(rdm)
                                 for name, rdm in variables.items()}
                    
                df = pd.DataFrame(variables)
                formula = 'neural ~ ' + ' + '.join(model_rdms.keys())
                result = smf.ols(formula, data=df).fit()
                for model_name in model_names:
                    results[roi][hemi][model_name][participant_id] = result.params[model_name]
                results[roi][hemi]['R-squared'][participant_id] = result.rsquared
            for model_name in model_names:
                results[roi][hemi][model_name]['mean'] = np.mean([results[roi][hemi][model_name][p]
                                                        for p in participants.keys()])
            results[roi][hemi]['R-squared']['mean'] = np.mean([results[roi][hemi]['R-squared'][p]
                                                        for p in participants.keys()])
    return results

# Compute permutation distribution randomizing condition labels
def permutation_distribution(neural_rdms, model_rdms, test,
                             results=None, n_permutations=1000, **kwargs):
    if not results:
        results = test(neural_rdms, model_rdms, neural_square=True, **kwargs)
    
    for roi in rois:
        for hemi in hemis:
            for model_name in model_names:
                results[roi][hemi][model_name]['permutation_distribution'] = []
    
    for iteration in np.arange(n_permutations):
        np.random.seed(iteration)
        shuffler = np.random.permutation(np.arange(n_pairs)).tolist()
        permuted_neural_rdms = {}
        for roi in rois:
            permuted_neural_rdms[roi] = {}
            for hemi in hemis:
                permuted_neural_rdms[roi][hemi] = {}
                for participant_id in participants.keys():
                    permuted_neural_rdms[roi][hemi][participant_id] = \
                        squareform(neural_rdms[roi][hemi][participant_id], checks=False)[shuffler]
        permuted_results = test(permuted_neural_rdms, model_rdms,
                                neural_square=False, **kwargs)
        
        for roi in rois:
            for hemi in hemis:
                for model_name in model_names:
                    results[roi][hemi][model_name]['permutation_distribution'].append(
                        permuted_results[roi][hemi][model_name]['mean'])
        
        if iteration % 100 == 0:
            print("Permutation Iteration {0} out of {1}".format(iteration, n_permutations))
    return results

# Compute p-values based on permutation distribution
def permutation_test(results, save_fn=None):
    for roi in rois:
        for hemi in hemis:
            for model_name in model_names:
                result = results[roi][hemi][model_name]
                p_value = ((np.sum(np.abs(result['permutation_distribution']) >=
                            np.abs(result['mean'])) + 1) /
                            float((len(result['permutation_distribution']) + 1)))
                results[roi][hemi][model_name]['p-value'] = p_value
    if save_fn:
        pickle.dump(results, open(save_fn, 'wb'))
    return results

# Compute bootstrap distributions for model tests
def bootstrap_distribution(neural_rdms, model_rdms, test, results=None,
                           bootstrap_participants=True,
                           bootstrap_conditions=True,
                           n_bootstraps=1000, **kwargs):
    assert bootstrap_participants or bootstrap_conditions
    
    if bootstrap_participants and not bootstrap_conditions:
        bootstrap_type = 'bootstrap_participants'
    elif bootstrap_conditions and not bootstrap_participants:
        bootstrap_type = 'bootstrap_conditions'
    elif bootstrap_participants and bootstrap_conditions:
        bootstrap_type = 'bootstrap_both'
    
    if not results:
        results = test(neural_rdms, model_rdms, neural_square=True, **kwargs)
    
    for roi in rois:
        for hemi in hemis:
            for model_name in model_names:
                results[roi][hemi][model_name][bootstrap_type] = {}
                for p in list(participants.keys()) + ['mean']:
                    results[roi][hemi][model_name][bootstrap_type][p] = []
    
    if bootstrap_conditions:
        conditions = [str(c) for c in np.arange(squareform(
                                    model_rdms['motion']).shape[0])]
        pairs = ['-'.join(p) for p in combinations(conditions, 2)]
        diagonals = ['-'.join((c, c)) for c in conditions]
    
    for iteration in np.arange(n_bootstraps):
        bootstrap_neural_rdms = {}
        for roi in rois:
            bootstrap_neural_rdms[roi] = {}
            for hemi in hemis:
                bootstrap_neural_rdms[roi][hemi] = {}
        if bootstrap_participants:
            np.random.seed(iteration)
            bootstrap_participant_ids = np.random.choice(list(participants.keys()),
                                            len(participants.keys()), replace=True)
            for roi in rois:
                for hemi in hemis:
                    for p, boot_p in zip(participants.keys(), bootstrap_participant_ids):
                        bootstrap_neural_rdms[roi][hemi][p] = \
                            squareform(neural_rdms[roi][hemi][boot_p], checks=False)
                    assert len(bootstrap_neural_rdms[roi][hemi].keys()) == len(participants.keys())
        if bootstrap_conditions:
            np.random.seed(iteration)
            boot_conditions = [str(c) for c in sorted(np.random.choice(conditions,
                                        len(conditions), replace=True).astype(int))]
            boot_pairs = ['-'.join(p) for p in
                                combinations(boot_conditions, 2)]
            boot_indices = [pairs.index(boot_pair)
                                 for boot_pair in boot_pairs
                                 if boot_pair not in diagonals]
            for roi in rois:
                for hemi in hemis:
                    for participant_id in participants.keys():
                        if not bootstrap_participants:
                            bootstrap_neural_rdms[roi][hemi][participant_id] = \
                                squareform(bootstrap_neural_rdms[roi][hemi][participant_id],
                                           checks=False)[boot_indices]
                        elif bootstrap_participants:
                            bootstrap_neural_rdms[roi][hemi][participant_id] = \
                                bootstrap_neural_rdms[roi][hemi][participant_id][boot_indices]
            
            bootstrap_model_rdms = {}
            for model_name in model_names:
                bootstrap_model_rdms[model_name] = model_rdms[model_name][boot_indices]
        elif not bootstrap_conditions:
            bootstrap_model_rdms = model_rdms

        bootstrap_results = test(bootstrap_neural_rdms, bootstrap_model_rdms,
                                 neural_square=False, **kwargs)
        for roi in rois:
            for hemi in hemis:
                for model_name in model_names:
                    for p in list(participants.keys()) + ['mean']:
                        results[roi][hemi][model_name][bootstrap_type][p].append(
                                bootstrap_results[roi][hemi][model_name][p])
        
        if iteration % 100 == 0:
            print("Bootstrap Iteration {0} out of {1}".format(iteration, n_bootstraps))
    
    return results

# Compute 95% confidence interval from bootstrap distribution
def bootstrap_ci(results, bootstrap_type='bootstrap_participants', save_fn=None):
    assert bootstrap_type == 'bootstrap_participants' or \
        bootstrap_type == 'bootstrap_conditions' or \
        bootstrap_type == 'bootstrap_both'
    for roi in rois:
        for hemi in hemis:
            for model_name in model_names:
                means = results[roi][hemi][model_name][bootstrap_type]['mean']
                ci = (np.percentile(means, 2.5), np.percentile(means, 97.5))
                results[roi][hemi][model_name][bootstrap_type + '_ci'] = ci
    if save_fn:
        pickle.dump(results, open(save_fn, 'wb'))
    return results

# Compute Wilcoxon signed-rank test against zero
def wilcoxon_test(results, save_fn=None):
    for roi in rois:
        for hemi in hemis:
            for model_name in model_names:
                p_value = wilcoxon([results[roi][hemi][model_name][p]
                                    for p in participants.keys()], correction=True)[1]
                results[roi][hemi][model_name]['wilcoxon_p-value'] = p_value
    if save_fn:
        pickle.dump(results, open(save_fn, 'wb'))
    return results

# Compute pairwise Wilcoxon signed-rank tests between models
def wilcoxon_model_comparison(results, save_fn=None):
    model_pairs = list(combinations(model_names, 2))
    for roi in rois:
        for hemi in hemis:
            results[roi][hemi]['wilcoxon_model_comparisons'] = {}
            for model_one, model_two in model_pairs:
                p_value = wilcoxon([results[roi][hemi][model_one][p]
                                    for p in sorted(participants.keys())],
                                   [results[roi][hemi][model_two][p]
                                    for p in sorted(participants.keys())],
                                   correction=True)[1]
                results[roi][hemi]['wilcoxon_model_comparisons'][' vs. '.join((
                                                            model_one, model_two))] = p_value
    if save_fn:
        pickle.dump(results, open(save_fn, 'wb'))
    return results

# Compute noise ceiling
def noise_ceiling(neural_rdms, results):
    for roi in rois:
        for hemi in hemis:
            biased_mean_rdm = np.mean(np.vstack([squareform(neural_rdms[roi][hemi][p], checks=False)
                                                 for p in participants.keys()]), axis=0)
            upper_bound = np.mean([spearmanr(biased_mean_rdm, squareform(neural_rdms[roi][hemi][p],
                                             checks=False))[0] for p in participants.keys()])
            unbiased_spearmans = []
            for participant in participants.keys():
                unbiased_mean_rdm = np.mean(np.vstack([squareform(neural_rdms[roi][hemi][p], checks=False)
                                                    for p in participants.keys() if p != participant]), axis=0)
                unbiased_spearmans.append(spearmanr(unbiased_mean_rdm,
                                        squareform(neural_rdms[roi][hemi][participant], checks=False))[0])
            lower_bound = np.mean(unbiased_spearmans)
            assert lower_bound < upper_bound
            results[roi][hemi]['noise_ceiling'] = {'upper': upper_bound,
                                                   'lower': lower_bound}
    return results

# Correct for multiple tests by controlling FDR at .05
def fdr_correction(results):
    for roi in rois:
        for hemi in hemis:
            results[roi][hemi]['wilcoxon_model_comparisons_fdr'] = {}
            comparisons, p_values = [], []
            for comparison, p_value in results[roi][hemi]['wilcoxon_model_comparisons'].items():
                comparisons.append(comparison)
                p_values.append(p_value)
            p_values_fdr = multipletests(p_values, alpha=.05, method='fdr_bh')[1].tolist()
            results[roi][hemi]['wilcoxon_model_comparisons_fdr'] = {
                comparison: p_fdr for comparison, p_fdr
                in zip(comparisons, p_values_fdr)}
            model_p_values = []
            for model_name in model_names:
                model_p_values.append(results[roi][hemi][model_name]['wilcoxon_p-value'])
            model_p_values_fdr = multipletests(model_p_values, alpha=.05, method='fdr_bh')[1].tolist()
            for model_name, p_fdr in zip(model_names, model_p_values_fdr):
                results[roi][hemi][model_name]['wilcoxon_p-value_fdr'] = p_fdr
    return results
            
### run line by line ################################################ 
# spearman, partial_spearman, regression
results = permutation_distribution(neural_rdms, model_rdms, spearman)
#results = permutation_distribution(neural_rdms, model_rdms, partial_spearman)

results = permutation_test(results)
results = noise_ceiling(neural_rdms, results)

results = bootstrap_distribution(neural_rdms, model_rdms, spearman,
                                 results=results, n_bootstraps=1000,
                                 bootstrap_conditions=False)
results = bootstrap_distribution(neural_rdms, model_rdms, spearman,
                                 results=results, n_bootstraps=1000,
                                 bootstrap_participants=True,
                                 bootstrap_conditions=True)

results = bootstrap_ci(results, bootstrap_type='bootstrap_participants')
results = bootstrap_ci(results, bootstrap_type='bootstrap_both') 
results = wilcoxon_test(results)

results = wilcoxon_model_comparison(results, save_fn=join(mvpa_dir, 'ROI_spearman_zscore_hyper_glasser_all9rois.p'))
#results = wilcoxon_model_comparison(results, save_fn=join(mvpa_dir, 'ROI_spearman_zscore_hyper_glasser.p'))
######################################################################
# Load pickle results
# results = pickle.load(open(join(mvpa_dir, 'ROI_spearman_zscore_hyper_glasser_all9rois.p'), 'rb')) #2024
# results = pickle.load(open(join(mvpa_dir, 'ROI_spearman_zscore_hyper_glasser.p'), 'rb'))
# results = pickle.load(open(join(mvpa_dir, 'ROI_partial_spearman_zscore_hyper_glasser.p'), 'rb'))

# Plot model test results 
def plot_model_tests(results, roi, save_fn=None):
    true_df = {'Spearman correlation': [], 'hemisphere': [], 'model': []}
    for model_name in model_names:
        for hemi in hemis:
            true_df['Spearman correlation'].append(
                results[roi][hemi][model_name]['mean'])
            true_df['model'].append(model_name)
            true_df['hemisphere'].append(hemi) 
    true_df = pd.DataFrame(true_df)

    n_permutations = 1000
    n_bootstraps = 1000

    null_df = {'Spearman correlation': [], 'hemisphere': [], 'model': []}
    for model_name in model_names:
        for hemi in hemis:
            null_df['Spearman correlation'].extend(
                results[roi][hemi][model_name]['permutation_distribution'])
            null_df['model'].extend([model_name] * n_permutations) 
            null_df['hemisphere'].extend([hemi] * n_permutations) 
    null_df = pd.DataFrame(null_df)

    group_df = {'Spearman correlation': [], 'participant': [],
                'hemisphere': [], 'model': []}
    for model_name in model_names:
        for hemi in hemis:
            for participant_id in results[roi][hemi][model_name].keys():
                if participant_id not in ('mean',
                                          'permutation_distribution',
                                          'p-value'):
                    group_df['Spearman correlation'].append(
                        results[roi][hemi][model_name][participant_id])
                    group_df['participant'].append(participant_id)
                    group_df['model'].append(model_name)
                    group_df['hemisphere'].append(hemi)
    group_df = pd.DataFrame(group_df)

    model_order = ['motion', 'gaze', 'nonverbs', 'verbs', 'scene', 'person', 'object', 'sociality', 'transitivity'] #2024
    
    sns.set_style('whitegrid')
    sns.set_context('talk', font_scale=1)
    f, ax = plt.subplots(figsize=(4.5, 8))
    ax = sns.pointplot(x='model', y='Spearman correlation', data=true_df,
                       hue='hemisphere', dodge=.37, join=False,
                       order=model_order,
                       palette=['darkblue', 'darkred'])
    
    plt.setp(ax.lines, zorder=1000000)    
    plt.setp(ax.collections, zorder=1000000, label="", sizes=[30])
    plt.axhline(y=0, linewidth=4, color='.75', linestyle='-', zorder=1)
    #plt.axhline(y=results[roi][hemi]['noise_ceiling']['lower'], linewidth=3,
    #            color='.75', linestyle='--', zorder=1) commented out for no ceiling

    x_coords = []
    y_coords = []
    errorbars = []
    
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)
    # '''
    both_errorbars = []
    y_i = 0
    for hemi in hemis:
        for model_name in model_order:
            ci = results[roi][hemi][model_name]['bootstrap_both_ci']
            
            assert ci[0] < y_coords[y_i] and ci[1] > y_coords[y_i]
            both_errorbars.append([y_coords[y_i] - ci[0],
                                   ci[1] - y_coords[y_i]])
            y_i += 1
    both_errorbars = np.array(both_errorbars).T
    ax.errorbar(x_coords, y_coords, yerr=both_errorbars,
                fmt=' ', elinewidth=2, ecolor='.5')
    
    participants_errorbars = []
    y_i = 0
    for hemi in hemis:
        for model_name in model_order:
            ci = results[roi][hemi][model_name]['bootstrap_participants_ci']
            assert ci[0] < y_coords[y_i] and ci[1] > y_coords[y_i]
            participants_errorbars.append([y_coords[y_i] - ci[0],
                                           ci[1] - y_coords[y_i]])
            y_i += 1
    participants_errorbars = np.array(participants_errorbars).T
    ax.errorbar(x_coords[:len(model_names)], y_coords[:len(model_names)],
                yerr=participants_errorbars[:, :len(model_names)], fmt=' ',
                elinewidth=4, ecolor='darkblue')
    ax.errorbar(x_coords[len(model_names):], y_coords[len(model_names):],
                yerr=participants_errorbars[:, len(model_names):], fmt=' ',
                elinewidth=4, ecolor='darkred')
    # '''
    sns.stripplot(x='model', y='Spearman correlation', data=null_df,
                  hue='hemisphere', jitter=0.12, size=5, alpha=.5,
                  linewidth=0, palette=['.75', '.75'], dodge=0.37,
                  order=model_order, zorder=1, ax=ax)
    
    ax.set(ylabel='Spearman correlation', title=roi)
    #ax.set(ylabel='Partial Spearman correlation', title = roi)
    #ax.set(ylabel='Standardized rank regression coefficient')
    ax.legend_.remove()
    
    plt.xticks(rotation=90, ha='center')
    plt.ylim(-.055, .5)
    plt.tight_layout()
    if save_fn:
        plt.savefig(save_fn + '.svg', format='svg', transparent=True)
        plt.savefig(save_fn + '.png', format='png', dpi=300, transparent=True)
    else:
        plt.show()
    plt.close()

for roi in rois:
    plot_model_tests(results, roi)
    plot_model_tests(results, roi, save_fn=f'/backup/data/social_actions/figures/glasser_spearman_roi-{roi}_no_ceiling_9_errorbar')
    
#plot_model_tests(results, roi, save_fn=f'/backup/data/social_actions/figures/glasser_spearman_roi-{roi}')
    #plot_model_tests(results, roi, save_fn=f'/backup/data/social_actions/figures/spearman_roi-{roi}')
    #plot_model_tests(results, roi)
