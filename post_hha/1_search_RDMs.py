#!/usr/bin/env python
# 1_search_RDMs.py
# 2022 August, Jane
# conda activate action-python2

## Import environments
from sys import argv
from os.path import join
from scipy.spatial.distance import cdist, squareform
import numpy as np
import mvpa2.suite as mv 
import mvpa2.support as sup
import mvpa2.datasets as datasets
import nibabel as nib 

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

participant_id = argv[1]
participant = participants[participant_id]
hemi = argv[2]

n_conditions = 90
n_vertices = 40962

n_medial = {'lh': 3486, 'rh': 3491}

## Directories
base_dir = '/backup/data/social_actions'
scripts_dir = join(base_dir, 'scripts')
data_dir = join(base_dir, 'fmri')
suma_dir = join(base_dir, 'suma-fsaverage6')
mvpa_dir = join(base_dir, 'fmri', 'pymvpa')

condition_order = mv.h5load(join(scripts_dir, 'condition_order.hdf5'))
reorder = condition_order['reorder']
sparse_ordered_labels = condition_order['sparse_ordered_labels']

## Load surface and create searchlight query engine
surf = mv.surf.read(join(suma_dir, '{0}.pial.gii'.format(hemi)))
qe = mv.SurfaceQueryEngine(surf, 10.0, distance_metric='dijkstra')

## Load in hyperaligned data #######
gifti_fn1 = join(mvpa_dir, 'sub-'+participant+'_ses-actions1_task-actions_hemi-'+hemi+'_desc-hha_coef.gii')
gii1 = nib.load(gifti_fn1)
data1 = np.vstack([da.data
                  for da in gii1.darrays])
ds_ses1 = mv.Dataset(data1)

gifti_fn2 = join(mvpa_dir, 'sub-'+participant+'_ses-actions2_task-actions_hemi-'+hemi+'_desc-hha_coef.gii')
gii2 = nib.load(gifti_fn2)
data2 = np.vstack([da.data
                  for da in gii2.darrays])
ds_ses2 = mv.Dataset(data2)
###################################

## Exclude medial wall
medial_wall = np.where(np.sum(ds_ses1.samples == 0, axis=0) == n_conditions)[0].tolist()
cortical_vertices = np.where(np.sum(ds_ses1.samples == 0, axis=0) < n_conditions)[0].tolist()
assert len(medial_wall) == n_medial[hemi]
assert len(medial_wall) + len(cortical_vertices) == n_vertices

##
for ses, ds in enumerate([ds_ses1, ds_ses2]):
    mv.zscore(ds, chunks_attr=None)

    if 'stats' in ds.sa:
        ds.sa.pop('stats')
    if 'history' in ds.a:
        ds.a.pop('history')
    
    if 'conditions' in ds.sa:
        ds.sa['conditions'] = [c[:-7] for c in ds.sa.labels]
    else:
        conditions = mv.h5load('condition_order.hdf5')['original_condition_order']
        ds.sa['conditions'] = conditions
    
    ds.sa['targets'] = ds.sa.conditions
    ds.sa['sessions'] = [ses] * ds.shape[0]
    ds.fa['node_indices'] = range(ds.shape[1])
    ds.fa['center_ids'] = range(ds.shape[1])
    if 'labels' in ds.sa:
        ds.sa.pop('labels')
    assert ds.shape == (n_conditions, n_vertices)


ds_both = mv.vstack((ds_ses1, ds_ses2), fa='update')

# Set up cross-validated RSA
cv_rsa_ = mv.CrossValidation(mv.CDist(pairwise_metric='correlation'),
                             mv.HalfPartitioner(attr='sessions'),
                             errorfx=None)

# cv_rsa above would return all kinds of .sa which are important
# but must be the same across searchlights. so we first apply it
# to the entire ds to capture them
cv_rsa_out = cv_rsa_(ds_both)
target_sa = cv_rsa_out.sa.copy(deep=True)

## And now create a postproc which would verify and strip them off
# to just return samples
from mvpa2.testing.tools import assert_collections_equal
from mvpa2.base.collections import SampleAttributesCollection
from mvpa2.base.node import Node

def lean_errorfx(ds):#Node):
    #def __call__(self, ds):
        assert_collections_equal(ds.sa, target_sa)
        # since equal, we could just replace with a blank one
        ds.sa = SampleAttributesCollection()
        return ds

# the one with the lean one
cv_rsa = mv.CrossValidation(mv.CDist(pairwise_metric='correlation'),
                             mv.HalfPartitioner(attr='sessions'),
                             errorfx=None, postproc=lean_errorfx)

sl = mv.Searchlight(cv_rsa, queryengine=qe, enable_ca=['roi_sizes'],
                    nproc=1, results_backend='native')

mv.debug.active += ['SLC']
sl_result = sl(ds_both)
assert len(sl_result.sa) == 0  # we didn't pass any
sl_result.sa = target_sa

print '>>>', np.mean(sl.ca.roi_sizes), np.std(sl.ca.roi_sizes)
                            
sl_means = np.mean(np.dstack((sl_result.samples[:n_conditions**2, :],
                              sl_result.samples[n_conditions**2:, :])),
                   axis=2)

sl_final = mv.Dataset(
     sl_means,
     sa={'conditions': sl_result.sa.conditions[:sl_means.shape[0], :].tolist(),
         'participants': [int(participant_id)] * sl_means.shape[0]},
     fa=sl_result.fa, a=sl_result.a)
#assert sl_result.shape[0] == n_conditions**2

## save output
mv.h5save(join(mvpa_dir, 'post_hha_no_roi_ids', 'search_RDMs_sq_zscore_p{0}_{1}.hdf5'.format(participant_id, hemi)), sl_final)