#!/usr/bin/env python

from os.path import join
import numpy as np
from mvpa2.base.hdf5 import h5save, h5load
from itertools import combinations
from scipy.spatial.distance import euclidean, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

base_dir = '/home/nastase/social_actions'
arrng_dir = join(base_dir,  'frame_arrangement', 'final_arrangements')
scripts_dir = join(base_dir, 'scripts')

subset_size = 30
n_subsets = 12

# Note that participants 1 and 3 are switched here (sid000021=3, sid000005=1)
participants = {1: {'person': 3, 'object': 4, 'scene': 5},
                2: {'person': 3, 'object': 4, 'scene': 5},
                4: {'person': 3, 'object': 4, 'scene': 5},
                6: {'person': 3, 'object': 5, 'scene': 4},
                7: {'person': 3, 'object': 5, 'scene': 4},
                8: {'person': 3, 'object': 5, 'scene': 4},
                9: {'person': 4, 'object': 3, 'scene': 5},
                10: {'object': 3},
                11: {'person': 4, 'object': 3, 'scene': 5},
                12: {'person': 4, 'object': 3, 'scene': 5},
                14: {'person': 5, 'object': 3, 'scene': 4},
                17: {'scene': 3},
                23: {'person': 5, 'object': 4, 'scene': 3}}

arrangements = {}
for task in ['person', 'object', 'scene']:
    arrangements[task] = {}
    for participant in participants.keys():
        if task in participants[participant]:
            arrangement = h5load(join(arrng_dir, 'final_arrangements_n{0}_r{1}_p{2}_s{3}.hdf5'.format(
                                subset_size, n_subsets, participant, participants[participant][task])))
            arrangements[task][participant] = arrangement
            print("Loaded arrangement for participant {0}, task {1}".format(participant, task))

condition_order = h5load(join(scripts_dir, 'condition_order.hdf5'))
stimuli = condition_order['original_condition_order']
reorder, sparse_ordered_labels = condition_order['reorder'], condition_order['sparse_ordered_labels']

distances = {}
mean_distances = {}
for task in ['person', 'object', 'scene']:
    distances[task] = {}
    for participant in participants.keys():
        if task in participants[participant]:
            distances[task][participant] = {}
            for arrangement_i, arrangement in arrangements[task][participant].items():
                arrangement_pairs = list(combinations(arrangement, 2))
                for pair in combinations(stimuli, 2):
                    if arrangement_i == 0:
                        assert len(arrangement) == 90
                        assert len(arrangement_pairs) == 90*89/2
                        distances[task][participant][pair] = []
                        distances[task][participant][pair].append(euclidean(arrangement[pair[0]]['finish'],
                                                                        arrangement[pair[1]]['finish']))
                    elif arrangement_i > 0 and pair in arrangement_pairs:
                        distances[task][participant][pair].append(euclidean(arrangement[pair[0]]['finish'],
                                                                        arrangement[pair[1]]['finish']))
                    elif arrangement_i > 0 and pair not in arrangement_pairs:
                        pass
            print("Computed distances for participant {0}, task {1}".format(participant, task))

RDMs = {}
for task in ['person', 'object', 'scene']:
    RDMs[task] = {}
    for participant in distances[task].keys():
        if task in participants[participant]:
            RDM = [np.mean(distances[task][participant][pair]) for pair in sorted(distances[task][participant].keys())]
            assert len(RDM) == 4005
            RDMs[task][participant] = RDM
            #        distances
            #                 = [np.mean(distances[pair]) for pair in sorted(distances.keys())] 
            #distances[task][participant]['mean'] = [np.mean(distances[pair]) for pair in sorted(distances.keys())] 
            #distances[task][participant] = mean_distances
            print("Computed RDM for participant {0}, task {1}".format(participant, task))

person_RDM = np.mean(np.vstack([RDMs['person'][p] for p in participants.keys() if 'person' in participants[p]]), axis=0)
object_RDM = np.mean(np.vstack([RDMs['object'][p] for p in participants.keys() if 'object' in participants[p]]), axis=0)
scene_RDM = np.mean(np.vstack([RDMs['scene'][p] for p in participants.keys() if 'scene' in participants[p]]), axis=0)

h5save(join(scripts_dir, 'RDMs', 'frame_arrangement_person_RDM.hdf5'), person_RDM)
h5save(join(scripts_dir, 'RDMs', 'frame_arrangement_object_RDM.hdf5'), object_RDM)
h5save(join(scripts_dir, 'RDMs', 'frame_arrangement_scene_RDM.hdf5'), object_RDM)

target_RDM = person_RDM
plt.figure(figsize=(8, 6))
ax = sns.heatmap(squareform(rankdata(target_RDM) / len(target_RDM) * 100)[reorder][:, reorder], vmin=0, vmax=100,
            square=True, cmap='RdYlBu_r', xticklabels=sparse_ordered_labels, yticklabels=sparse_ordered_labels)
ax.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.yticks(va='top')
plt.tight_layout()
plt.show()

# Compute MDS solution on average arrangement RDM
from sklearn.manifold import MDS

mds = {}
for n in range(1,11):
    mds[n] = {}
    solution = MDS(n_components=n, metric=True,
                   dissimilarity='precomputed', eps=1e-9, n_init=10,
                   max_iter=10000, verbose=1).fit(
                        squareform(target_RDM))
    mds[n]['stress'] = solution.stress_
    mds[n]['positions'] = solution.embedding_

#h5save(join(scripts_dir, 'frame_arrangment_person_MDS.hdf5'), mds)
mds = h5load(join(scripts_dir, 'frame_arrangement_person_MDS.hdf5'))

condition_order = h5load(join(scripts_dir, 'condition_order.hdf5'))
stimuli = condition_order['original_condition_order']
reorder, sparse_ordered_labels = condition_order['reorder'], condition_order['sparse_ordered_labels']

# Scree plot for stress by dimensionality
def plot_mds_scree(mds, max_dim=8):
    sns.set_style('white')
    stresses = np.array([mds[n]['stress'] for n in sorted(mds.keys())])
    plt.plot(np.arange(1, len(stresses + 1))[:max_dim],
             stresses[:max_dim]/1000000, '-o', lw=4, ms=10, clip_on=False)
    sns.despine(offset=10)
    plt.xlabel('Number of dimensions')
    plt.ylabel('Metric stress (1e6)')
    plt.tight_layout()
    plt.show()

plot_mds_scree(mds, max_dim=8)

# Plot MDS positions
def plot_mds_embedding(mds, n_dim=2, x_dim=0, y_dim=1, standardize=True):
    positions = mds[n_dim]['positions'][reorder, :]
    if standardize:
        from scipy.stats import zscore
        positions = zscore(positions, axis=0)

    # Create custom palette for 18 categories
    custom_palette = np.vstack((sns.color_palette('cubehelix', 8)[:4],
                                sns.color_palette("Set1", 5),
                                sns.color_palette('Pastel1', 5),
                                sns.color_palette('cubehelix', 8)[4:]))

    # Plot positions with scaled axes, no labels
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.scatter(positions[:, x_dim], positions[:, y_dim],
               c=np.repeat(custom_palette, 5, axis=0), s=150,
               marker='o', edgecolors='face')
    plt.axis('scaled')
    sns.despine()
    plt.xlabel('MDS dimension {0}'.format(x_dim + 1))
    plt.ylabel('MDS dimension {0}'.format(y_dim + 1))
    plt.xticks([])
    plt.yticks([])

    # Plot fancy legend, adjust to plot size
    from matplotlib import patches
    from matplotlib.transforms import Bbox
    patch_labels = ['conversation', 'intimacy', 'teaching', 'manufacturing',
                    '', '', '', '', '',
                    'eating', 'dancing', 'exercise', 'grooming', 'tool use',
                    'cooking', 'gardening', 'arts and crafts', 'musical performance']
    social_patches = [patches.Patch(color=c, label=l)
                      for c, l in zip(custom_palette, patch_labels)[:9]]
    nonsocial_patches = [patches.Patch(color=c, label=l)
                      for c, l in zip(custom_palette, patch_labels)[9:]]
    social_legend = ax.legend(bbox_to_anchor=(.9, 1), loc=2, handles=social_patches)
    social_legend._legend_box.align = "left"
    social_legend.set_title('social', prop={'weight': 'semibold'})
    ax.add_artist(social_legend)
    plt.draw()
    extent = social_legend.get_window_extent().get_points()
    x0 = Bbox(extent + 16).inverse_transformed(ax.transAxes).x0
    nonsocial_legend = ax.legend(bbox_to_anchor=(x0, .859), loc=2, handles=nonsocial_patches)
    nonsocial_legend._legend_box.align = "left"
    nonsocial_legend.get_title().set_position((0, -159))
    nonsocial_legend.set_title('nonsocial', prop={'weight': 'semibold'})

    plt.tight_layout(rect=(0, 0, .85, 1))
    plt.show()

plot_mds_embedding(mds, n_dim=2, x_dim=0, y_dim=1)


# Plot MDS positions
def plot_mds_embedding(mds, n_dim=2, x_dim=0, y_dim=1, standardize=True):
    positions = mds[n_dim]['positions'][reorder, :]
    if standardize:
        from scipy.stats import zscore
        positions = zscore(positions, axis=0)

    ordered_labels = ['conversation', 'intimacy', 'teaching', 'manufacturing',
                      'eating ', 'dancing ', 'exercise ', 'grooming ', 'tool use ',
                      'eating', 'dancing', 'exercise', 'grooming', 'tool use',
                      'cooking', 'gardening', 'arts and crafts', 'musical performance']
    import pandas as pd
    df = pd.DataFrame({'category': np.repeat(ordered_labels, 5),
                       'x': positions[:, x_dim], 'y': positions[:, y_dim]})

    custom_palette = np.vstack((sns.color_palette('cubehelix', 8)[:4],
                                sns.color_palette("Set1", 5),
                                sns.color_palette('Pastel1', 5),
                                sns.color_palette('cubehelix', 8)[4:]))

    g = sns.FacetGrid(df, hue="category", size=7, aspect=1.3,
                      palette=custom_palette)
    g.map(plt.scatter, "x", "y", s=150, edgecolor="face")
    ax = g.facet_axis(0, 0)
    from matplotlib import patches
    patch_labels = ['conversation', 'intimacy', 'teaching', 'manufacturing',
                    '', '', '', '', '',
                    'eating', 'dancing', 'exercise', 'grooming', 'tool use',
                    'cooking', 'gardening', 'arts and crafts', 'musical performance']
    social_patches = [patches.Patch(color=c, label=l)
                      for c, l in zip(custom_palette, patch_labels)[:9]]
    nonsocial_patches = [patches.Patch(color=c, label=l)
                      for c, l in zip(custom_palette, patch_labels)[9:]]
    social_legend = plt.legend(handles=social_patches, loc=(.8, .715), title='social')
    social_legend._legend_box.align = "left"
    ax.add_artist(social_legend)
    nonsocial_legend = plt.legend(handles=nonsocial_patches, loc=(.85, .59), title='nonsocial')
    #nonsocial_legend._legend_box.align = "left"
    #nonsocial_legend.get_title().set_va('bottom')
    #nonsocial_legend.set_title('nonsocial')
    plt.axis('scaled')
    sns.despine()
    plt.xlabel('MDS dimension {0}'.format(x_dim + 1))
    plt.ylabel('MDS dimension {0}'.format(y_dim + 1))
    plt.xticks([])
    plt.yticks([])
    plt.show()

plot_mds_embedding(mds, n_dim=2, x_dim=0, y_dim=1)

colors = np.repeat(range(18), 5)
from matplotlib.colors import ListedColormap
from matplotlib import patches
cmap = ListedColormap(custom_palette)
fig, ax = plt.subplots()
plt.scatter(positions[:, x_dim], positions[:, y_dim], c=colors, s=200,
                 marker='o', edgecolors='face', cmap=cmap, label='sc')
sns.despine()
plt.xlabel('MDS dimension {0}'.format(x_dim + 1))
plt.ylabel('MDS dimension {0}'.format(y_dim + 1))
plt.xticks([])
plt.yticks([])
social_patches = [patches.Patch(color=c, label=l)
                  for c, l in zip(custom_palette, ordered_labels)[:9]]
nonsocial_patches = [patches.Patch(color=c, label=l)
                  for c, l in zip(custom_palette, ordered_labels)[9:]]
social_legend = plt.legend(handles=social_patches, loc=(0, 0))
ax.add_artist(social_legend)
plt.legend(handles=nonsocial_patches, loc=(0, 0))
plt.tight_layout()
plt.show()

ordered_labels = ['conversation', 'intimacy', 'teaching', 'manufacturing',
                  'eating ', 'dancing ', 'exercise ', 'grooming ', 'tool use ',
                  'eating', 'dancing', 'exercise', 'grooming', 'tool use',
                  'cooking', 'gardening', 'arts and crafts', 'musical performance']

sociality = ['social', 'nonsocial']

df = pd.DataFrame({'category': np.repeat(ordered_labels, 5), 'sociality': np.repeat(sociality, 45), 'x': positions[:, 0], 'y': positions[:, 1]})

custom_palette = np.vstack((np.array(sns.color_palette('Dark2', 8))[[0, 5, 6, 7], :],
                            sns.color_palette("Set1", 5),
                            sns.color_palette('Pastel1', 5),
                            np.array(sns.color_palette('Set3', 8))[[4, 5, 6, 7], :]))

custom_palette = np.vstack((sns.color_palette('cubehelix', 8)[:4],
                            sns.color_palette("Set1", 5),
                            sns.color_palette('Pastel1', 5),
                            sns.color_palette('cubehelix', 8)[4:]))

g = sns.FacetGrid(df, hue="category", size=8, aspect=1.3,
                  palette=custom_palette)
g.map(plt.scatter, "x", "y", s=150, edgecolor="face")
g.add_legend(title='')
plt.axis('scaled')
sns.despine()
plt.xlabel('MDS dimension {0}'.format(x_dim + 1))
plt.ylabel('MDS dimension {0}'.format(y_dim + 1))
plt.xticks([])
plt.yticks([])
plt.show()

sns.palplot(np.vstack((np.array(sns.color_palette('Dark2', 8))[[0, 5, 6, 7], :], sns.color_palette("Set1", 5), sns.color_palette('Pastel1', 5), np.array(sns.color_palette('Set3', 8))[[4, 5, 6, 7], :]))); plt.show()
