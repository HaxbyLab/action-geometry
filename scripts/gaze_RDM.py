#!/usr/bin/env python

# Run with singularity container with new scipy
# singularity exec /backup/singularity/neurodebian-v2.0.img /bin/bash

from os.path import join
import numpy as np
from mvpa2.base.hdf5 import h5save, h5load
import pandas as pd

base_dir = '/home/nastase/social_actions'
gaze_dir = join(base_dir, 'gaze')

participants = {2: [1, 3, 4], # half of block 2 is gone
                3: [1, 2, 3, 4],
                4: [1, 2, 4], # missing 2, 8, 14, 2
                5: [1, 2, 3, 4], # lost a couple in 2
                6: [1, 2, 3, 4],
                7: [1, 2, 3, 4],
                8: [1, 2, 3, 4], # good! 
                9: [1, 2, 3, 4],
                10: [3, 4], # okay
                11: [2, 3, 4], # okay
                12: [1, 2, 3, 4],
                15: [1, 3], # missing 2, 16, 7, 19
                16: [1, 2, 3, 4], # good!
                17: [1, 2, 3, 4], # good!
                18: [1, 2, 3, 4], # okay
                19: [1, 2, 3, 4], # good!
                20: [1, 2, 3, 4]} # good!
example_participant, example_block = 8, 0

all_trajectories = {}
all_missed = {}
for participant in sorted(participants.keys()):
    data_dir = join(gaze_dir, 'data_eyetracker', 'Sub{0}'.format(participant), 'Output')

    participant_trajectories = []
    participant_missed = []
    for block in participants[participant]:
        with open(join(data_dir, 'S{0}B{1}.asc'.format(participant, block))) as f:
            samples = [line.strip().split() for line in f.readlines()]

        with open(join(data_dir, 'Out_S{0}B{1}.csv'.format(participant, block))) as f:
            trials = [line.strip().split(',') for line in f.readlines()][1:]

        annotations = ['MSG', 'SFIX', 'EFIX',
                       'SSACC', 'ESACC', 'SBLINK', 'EBLINK']

        # Collect all samples, excluding blinks, repetitions
        trial_start, clip_start, blink = False, False, False
        block_samples = {}
        for sample_i, sample in enumerate(samples):
            if not trial_start:
                try:
                    if sample[2] == 'TRIALID:':
                        trial_i = int(sample[3]) - 1
                        clip = trials[trial_i][0]
                        if trials[trial_i][2] == '1' and not repetition:
                            repetition = True
                        elif trials[trial_i][2] == '1' and repetition:
                            print("Caught a repetition for trial {0}, starting sample {1}".format(
                                    trial_i + 1, sample_i))
                            repetition = False
                            trial_start, clip_start = False, False
                            continue
                        elif trials[trial_i][2] == '0':
                            repetition = False
                        trial_start = True
                        clip_start = False
                        trial_samples = []
                        start_blink, end_blink = [], []
                        print "Started trial {0} at sample {1}".format(sample[3], sample_i)
                    else:
                        trial_start = False
                        continue
                except IndexError:
                    continue
            elif trial_start and not clip_start:
                try:
                    if sample[2] == 'FRAME' and sample[3] == 'ON':
                        clip_start = True
                        print "Started clip {0}, {1}, at sample {2}".format(trial_i + 1, clip, sample_i)
                    else:
                        clip_start = False
                        continue
                except IndexError:
                    continue
            else:
                if sample[0] == 'MSG':
                    if sample[2] == 'FRAME' and sample[3] == 'OFF':
                        frame_off = int(sample[1])
                    if sample[2] == 'FIX2' and sample[3] == 'ON':
                        trial_start, clip_start, blink = False, False, False
                        for sblink in start_blink:
                            for s in trial_samples:
                                if int(s[0]) in range(sblink - 30, sblink + 1):
                                    s[1:3] = None, None
                        for eblink in end_blink:
                            for s in trial_samples:
                                if int(s[0]) in range(eblink, eblink + 100 + 1):
                                    s[1:3] = None, None
                        trial_samples_trim = [ts for ts in trial_samples if ts[0] < frame_off]
                        assert np.isclose(len(trial_samples_trim), 2500, rtol=1)
                        block_samples[clip] = trial_samples_trim 
                        print("Finished trial {0}".format(trial_i + 1))
                        continue
                elif sample[0] == 'SBLINK':
                    blink = True
                    start_blink.append(int(sample[2]))
                    print "Beginning of a blink!"
                    continue
                elif sample[0] == 'EBLINK':
                    blink = False
                    end_blink.append(int(sample[2]))
                    print "End of a blink!"
                    continue
                elif sample[0] not in annotations:
                    if sample_i % 500 == 0:
                        print "Collecting trials! {0}".format(sample[:3]) 
                    time_point = int(sample[0])
                    if blink:
                        trial_samples.append([time_point, None, None])
                        if sample_i % 500 == 0:
                            print "Appending blink samples {0}".format([time_point, None, None])
                    elif not blink:
                        if sample[1:3] == ['.', '.']:
                            trial_samples.append([time_point, None, None])
                            blink = True
                            print "Caught an unexpected blink!!! {0}".format([time_point, None, None])
                        else:
                            trial_samples.append([time_point] + [float(s) for s in sample[1:3]])
                            if sample_i % 500 == 0: 
                                print "Appending samples {0}".format([time_point] + [float(s) for s in sample[1:3]])
        assert len(block_samples) == 100

        # Trim clip names, number of samples, and remove unused categories
        for clip, trial_samples in block_samples.items():
            if 'board_game' in clip:
                del block_samples[clip]
            if 'phone_laptop' in clip:
                del block_samples[clip]

        for clip, trial_samples in block_samples.items():
            if len(trial_samples) > 2500:
                block_samples[clip] = trial_samples[:2500]
                assert len(block_samples[clip]) == 2500
            elif len(trial_samples) < 2500:
                while len(trial_samples) < 2500:
                    trial_samples.append(trial_samples[-1])
                block_samples[clip] = trial_samples
                assert len(block_samples[clip]) == 2500
            if clip[-4:] == '_NON':
                clip_trim = clip[:-4]
                block_samples[clip_trim] = block_samples.pop(clip)
        assert len(block_samples) == 90

        # Preprocess gaze trajectories (i.e., smooth and interpolate)
        import pandas as pd
        trajectories = {}
        n_missed = 0
        for clip, trial_samples in block_samples.items():
            arr = np.array(trial_samples, dtype=np.float)
            assert arr.shape == (2500, 3)
            df = pd.DataFrame({'x': arr[:, 1],
                              'y': arr[:, 2]})

            # Window width of approximately 2 frames (84 ms)
            trajectory = df.rolling(window=84, center=False).median().interpolate(method='linear')
            ### TODO WE NEED TO CHECK FOR ALL NANS HERE!!!
            trajectories[clip] = trajectory.as_matrix()
            if np.all(np.isnan(trajectory.as_matrix())):
                print("Lost an entire trial!!!\n"
                      "\tParticipant {0}, block {1}, clip {2}".format(participant, block, clip))
                import time
                time.sleep(1)
                n_missed += 1
        participant_trajectories.append(trajectories)
        participant_missed.append(n_missed)
    all_missed[participant] = participant_missed
    all_trajectories[participant] = participant_trajectories
h5save('/home/nastase/social_actions/gaze/all_gaze_trajectories.hdf5', all_trajectories)

# Downsample (decimating) trajectories
from scipy.signal import decimate
all_downsampled = {}
for participant in sorted(all_trajectories.keys()):
    participant_trajectories = []
    for trajectories in all_trajectories[participant]:
        for clip, trajectory in trajectories.items():
            downsampled = decimate(trajectory[83:, :], 7, n=0, ftype='iir', axis=0, zero_phase=True)
            downsampled = decimate(downsampled, 6, n=0, ftype='iir', axis=0, zero_phase=True)
            downsampled = np.vstack((np.array([[np.nan, np.nan], [np.nan, np.nan]]), downsampled))
            assert len(downsampled) == 60
            trajectories[clip] = downsampled
        participant_trajectories.append(trajectories)
    all_downsampled[participant] = participant_trajectories
h5save('/home/nastase/social_actions/gaze/all_gaze_trajectories_downsampled.hdf5', all_downsampled)
all_trajectories = all_downsampled

# Re-load trajectories
all_trajectories = h5load('/home/nastase/social_actions/gaze/all_gaze_trajectories_downsampled.hdf5')

# Plot some trajectories
import matplotlib.pyplot as plt
import seaborn as sns

trajectories = all_trajectories[example_participant][example_block]

clips = ['conversation_26_final', 'hugging_kissing_125_final', 'teaching_155_final', 'assembly_line_10_final',
         'cooking_39_final', 'gardening_115_final', 'arts_crafts_7_final', 'playing_instrument_151_final',
         'eating_76_final', 'eating_72_final', 'dancing_63_final', 'dancing_64_final']
clip_aliases = ['conversation 1', 'intimacy 2', 'teaching 3', 'manufacturing 1',
                'cooking 4', 'gardening 1', 'arts and crafts 5', 'musical performance 5',
                'social eating 3', 'nonsocial eating 1', 'social dancing 5', 'nonsocial dancing 1'] 
trajectories_df = {'position': [], 'dimension': [],
                 'clip': []}
for clip, alias in zip(clips, clip_aliases):
    trajectory = trajectories[clip]
    trajectories_df['position'].extend(np.hstack((trajectory[:, 0], trajectory[:, 1])))
    trajectories_df['dimension'].extend(['x'] * len(trajectory) + ['y'] * len(trajectory))
    trajectories_df['clip'].extend([alias] * (len(trajectory)*2))
trajectories_df = pd.DataFrame(trajectories_df)

sns.set_style('white')
sns.set_context('notebook', font_scale=1.15)
g = sns.FacetGrid(trajectories_df, col='clip', hue='dimension', col_wrap=4, size=3)
g.map(plt.plot,  'position', lw=3).set_axis_labels('time', 'position')
g.fig.subplots_adjust(left=.07)
g.axes[0].legend(loc='upper right')
plt.savefig('gaze_trajectories_8stim.png', dpi=300)
plt.show()

# Plot same trajectory across participants
example_trajectories = {p: v[0] for p, v in all_trajectories.items() if p in [8, 16, 5, 12]}
clip = 'dancing_183_final'
trajectories_df = {'position': [], 'dimension': [], 'participant': []}
for participant in sorted(example_trajectories.keys()):
    trajectory = example_trajectories[participant][clip]
    trajectories_df['position'].extend(np.hstack((trajectory[:, 0], trajectory[:, 1])))
    trajectories_df['dimension'].extend(['x'] * len(trajectory) + ['y'] * len(trajectory))
    trajectories_df['participant'].extend([participant] * (len(trajectory)*2))
trajectories_df = pd.DataFrame(trajectories_df)

sns.set_style('white')
sns.set_context('notebook', font_scale=1.15)
g = sns.FacetGrid(trajectories_df, col='participant', hue='dimension', col_wrap=2, size=3)
g.map(plt.plot,  'position', lw=3).set_axis_labels('time', 'position')
#g.fig.subplots_adjust(left=.06)
g.axes[0].legend(loc='lower right')
#g.fig.subplots_adjust(left=.03)
plt.tight_layout()
plt.savefig('gaze_trajectories_4subj.png', dpi=300)
plt.show()

# Compute pairwise Euclidean distances
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

# Combinations should match order of squareform
all_RDMs = {}
for participant in sorted(participants.keys()):
    participant_RDMs = []
    for trajectories in all_trajectories[participant]:
        distances = []
        for pair in combinations(sorted(trajectories.keys()), 2):
            t1, t2 = trajectories[pair[0]], trajectories[pair[1]]
            if np.all(np.isnan(t1)) or np.all(np.isnan(t2)):
                distance = np.nan
            else:
                distance = np.nansum(np.sqrt(((trajectories[pair[0]] - trajectories[pair[1]]) ** 2).sum(-1)))
            distances.append(distance)
        for distance in distances:
            if distance == 0.:
                distance = np.nan
        gaze_RDM = np.array(distances)
        participant_RDMs.append(gaze_RDM)
    all_RDMs[participant] = participant_RDMs
h5save('/home/nastase/social_actions/gaze/all_gaze_RDMs.hdf5', all_RDMs)

# Check intra-participant RDM correlation across blocks
mean_RDMs = {}
mean_correlations = {}
#for participant in sorted(participants.keys()):
for participant in [2, 5, 7, 8, 12, 16, 17, 19, 20]:
    intra_distances = np.vstack(all_RDMs[participant])
    intra_mean = np.nanmean(1 - pdist(intra_distances, 'correlation'))
    print("Mean intra-participant Pearson correlation\n"
          "\tof gaze RDMs = {0:.3f} for participant {1}".format(intra_mean, participant))
    #print("Missed trials for participant {0}: {1}".format(participant, all_missed[participant]))
    mean_RDMs[participant] = np.nanmean(np.vstack(all_RDMs[participant]), axis=0)
    mean_correlations[participant] = intra_mean
print "Mean intra:", np.nanmean(mean_correlations.values())
print "SD intra:", np.nanstd(mean_correlations.values())
overall_mean = np.nanmean(1 - pdist(np.vstack((all_RDMs.values())), 'correlation'))

# Plot RDM
sns.heatmap(squareform((all_RDMs[example_participant][example_block] / 
                        np.nanmax(all_RDMs[example_participant][example_block]))),
            square=True, cmap='YlGnBu', xticklabels=False, yticklabels=False)
plt.show()

# Create final target RDM
target_RDM = np.nanmean(np.vstack(mean_RDMs.values()), axis=0)

# Cleaner version (participants with inter-block correlation > .1)
clean_RDMs = {p: v for p, v in all_RDMs.items() if p in [2, 5, 7, 8, 12, 16, 17, 19, 20]}
target_RDM = np.nanmean(np.vstack(clean_RDMs.values()), axis=0)
h5save('/home/nastase/social_actions/scripts/gaze_target_RDM.hdf5', target_RDM)

condition_order = h5load('/home/nastase/social_actions/scripts/condition_order.hdf5')
reorder, sparse_ordered_labels = condition_order['reorder'], condition_order['sparse_ordered_labels']

plt.figure(figsize=(8, 6))
ax = sns.heatmap(squareform(rankdata(target_RDM) / len(target_RDM) * 100)[reorder][:, reorder], vmin=0, vmax=100,
            square=True, cmap='RdYlBu_r', xticklabels=sparse_ordered_labels, yticklabels=sparse_ordered_labels)
ax.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.yticks(va='top')
plt.tight_layout()
plt.show()
