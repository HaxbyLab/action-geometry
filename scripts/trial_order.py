#!/usr/bin/env python

# Run with: trial_order.py 1 |& tee trial_orders/trial_order_log_1.txt
# or: for i in {1..30}; do trial_order.py $i |& tee trial_orders/trial_order_log_$i.txt; done
# If verb selection hangs, try changing seed_increment

import sys
from time import time
from os.path import join
from copy import deepcopy
import numpy as np
from mvpa2.base.hdf5 import h5load, h5save

base_dir = '/home/nastase/social_actions'
scripts_dir = join(base_dir, 'scripts')
exp_dir = join(base_dir, 'experiment')
stim_dir = join(scripts_dir, 'stimuli')
seq_dir = join(scripts_dir, 'sequences')
timing_dir = join(scripts_dir, 'timing')

if len(sys.argv) == 1:
    participant = 99
    print "WARNING: Test run with participant set to {0}!".format(participant)
else:
    participant = int(sys.argv[1]) 

# Load jittered timing and T1I1 sequences
timing = h5load(join(timing_dir, 'timing_final_{0}.hdf5'.format(participant)))
sequence = h5load(join(seq_dir, 'sequence_final_{0}.hdf5'.format(participant)))

assert timing.shape == sequence.shape

categories = {i: [] for i in range(1, 19)}
with open(join(scripts_dir, 'stimuli.csv')) as f:
    for line in f.readlines():
        categories[int(line.split(',')[1])].append(line.strip().split(',')[2])

assert len(categories) == 18
for stimulus_fns in categories.values():
    assert len(stimulus_fns) == 5

prep_categories = {i: [] for i in range(1, 19)}
with open(join(scripts_dir, 'prep_stimuli.csv')) as f:
    for line in f.readlines():
        prep_categories[int(line.split(',')[1])].append(line.strip().split(',')[2])

assert len(prep_categories) == 18
for stimulus_fns in prep_categories.values():
    assert len(stimulus_fns) == 1

timing_sequence = []
for run_timing, run_sequence in zip(timing, sequence):
    timing_sequence.append(np.column_stack((run_timing, run_sequence)).tolist())
for run_i in range(len(timing_sequence)):
    for trial_i in range(len(timing_sequence[run_i])):
        timing_sequence[run_i][trial_i][1] = int(timing_sequence[run_i][trial_i][1]) + 1 

timing_sequence_1 = timing_sequence[:4]
timing_sequence_2 = timing_sequence[4:]
assert len(timing_sequence_1) == len(timing_sequence_2) == 4

# Randomly assign stimuli (without replacement) based on conditions
seed_increment = 0
for timing_sequence in [timing_sequence_1, timing_sequence_2]:
    initial_three_first_run = {}
    final_three = {}
    for run_i, run in enumerate(timing_sequence):
        categories_shuffle = deepcopy(categories)
        
        for category_id in categories_shuffle.keys():
            np.random.seed(participant * 1000 +  seed_increment)
            np.random.shuffle(categories_shuffle[category_id])
        seed_increment += 1

        for trial_i, trial in enumerate(run):
            trial_category = trial[1]

            if trial_category == 19:
                trial.append('fixation')
                continue
            elif trial_category == 20:
                if trial_i == 0:
                    trial.append('fixation')
                else:
                    trial.append('question')
                continue
            
            if trial_i < 3:
                trial.append(prep_categories[trial_category][0]) 
            else:
                trial.append(categories_shuffle[trial_category].pop(0))

        #assert len(initial_three_first_run) == 3
        for category_id in categories_shuffle.keys():
            assert len(categories_shuffle[category_id]) == 0

# Now we insert question probe verbs
with open(join(scripts_dir, 'verbs.csv')) as f:
        verbs = {int(line.split(',')[0]):
                    {'category': line.split(',')[1],
                     'sociality': line.split(',')[2],
                     'verbs': line.strip('\n').split(',')[3:],
                     'foils_used': 0}
                 for line in f.readlines()}

for verb_category in verbs.values():
    assert len(verb_category['verbs']) == 4

from copy import deepcopy
verbs_reference = deepcopy(verbs)

# Change this if verb selection hangs, e.g., to 50
seed_increment = 50
start_time = time()
for sequence_i, timing_sequence in enumerate([timing_sequence_1, timing_sequence_2]):
    initial_three_first_run = {}
    final_three = {}
    for run_i, run in enumerate(timing_sequence):
        for trial_i, trial in enumerate(run):
            if trial[2] == 'question':
                if previous_trial[2] == 'question' or previous_trial[2] == 'fixation':
                    trial[2] = 'fixation'
                    print run_i, trial_i, previous_trial, "Switching question to fixation!!!"
                else:
                    if run_i > 0 and trial_i <= 2:
                        correct, foil = final_three[run_i - 1][category_id]
                        print run_i, trial_i, previous_trial, correct, foil, "Question in first three trials!!!"
                    elif run_i == 3 and trial_i in range(len(timing_sequence[0]))[-3:]:
                        assert set(initial_three_first_run[category_id]) <= set(sum([v['verbs'] for v in verbs.values()], []))
                        correct, foil = initial_three_first_run[category_id]
                        print run_i, trial_i, previous_trial, correct, foil, "Question in last three trials of last run!!!"
                    else:
                        category_id = previous_trial[1]
                        correct_verbs = verbs[category_id]
                        np.random.seed(participant * 1000 +  seed_increment)
                        np.random.shuffle(correct_verbs['verbs'])
                        if category_id in initial_three_first_run:
                            if correct_verbs['verbs'][0] in initial_three_first_run[category_id]:
                                correct = correct_verbs['verbs'].pop(1)
                        else:
                            correct = correct_verbs['verbs'].pop(0)
                        foil_verbs = verbs[np.random.choice(verbs.keys())]
                        while foil_verbs['category'] == correct_verbs['category'] or foil_verbs['foils_used'] > sequence_i:
                            foil_verbs = verbs[np.random.choice(verbs.keys())]
                            if time() - start_time > 5:
                                raise Exception, ("Got stuck trying to assign verbs "
                                    " on participant {0}, try a different random seed!!!".format(participant))
                        if correct_verbs['category'] == 'assembly' or correct_verbs['category'] == 'using':
                            while foil_verbs['category'] == 'assembly' or foil_verbs['category'] == 'using' or foil_verbs['foils_used'] > sequence_i:
                                foil_verbs = verbs[np.random.choice(verbs.keys())]
                                if time() - start_time > 5:
                                    raise Exception, ("Got stuck trying to assign verbs "
                                        " on participant {0}, try a different random seed!!!".format(participant))
                        foil_verbs['foils_used'] += 1
                        np.random.seed(participant * 10000 +  seed_increment)
                        np.random.shuffle(foil_verbs['verbs'])
                        if category_id in initial_three_first_run:
                            if foil_verbs['verbs'][0] in initial_three_first_run[category_id]:
                                foil = foil_verbs['verbs'].pop(1)
                        else:
                            foil = foil_verbs['verbs'].pop(0)

                        if run_i == 0 and trial_i < 3:
                            initial_three_first_run[category_id] = [correct, foil]
                            print run_i, trial_i, previous_trial, "Grabbing question in first three trials of first run!!!"
                        elif trial_i in range(len(timing_sequence[0]))[-3:]:
                            if run_i not in final_three.keys():
                                final_three.update({run_i: {}})
                            final_three[run_i].update({category_id: [correct, foil]})
                            print run_i, trial_i, previous_trial, "Grabbing question in final three trials!!!"
                        
                        print run_i, trial_i, previous_trial, correct, correct_verbs['category'], foil, foil_verbs['category']

                    assert trial[2] == 'question'
                    probe = [correct, foil]
                    np.random.shuffle(probe)
                    trial.append(probe)
    
            previous_trial = trial      
            seed_increment += 1

for i, j in zip(verbs.values(), verbs_reference.values()):
    print i['foils_used'], len(i['verbs']), i['verbs'], j['verbs']

# Run some tests
verbs_used = []
for s in [timing_sequence_1, timing_sequence_2]:
    question_count = 0
    for r in s:
        for t in r[3:]:
            if t[2] == 'question':
                question_count += 1
                verbs_used.extend(t[3])
    assert question_count == 18
assert len(verbs_used) == len(np.unique(verbs_used)) == 18*4

for verb_category in verbs.values():
    assert verb_category['foils_used'] == 2
    assert len(verb_category['verbs']) == 0

for sequence_i, timing_sequence in enumerate([timing_sequence_1, timing_sequence_2]):
    for run_i, run in enumerate(timing_sequence):
        h5save(join(scripts_dir, 'trial_orders',
                    'trial_order_p{0}_s{1}_r{2}.hdf5'.format(
                         participant, sequence_i + 1, run_i + 1)), run)

print "Successfully finished creating trial order"
