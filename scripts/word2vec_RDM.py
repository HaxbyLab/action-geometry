#!/usr/bin/env python

from os.path import join
import numpy as np
from mvpa2.base.hdf5 import h5load, h5save
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns
import gensim

base_dir = '/home/nastase/social_actions'
scripts_dir = join(base_dir, 'scripts')
w2v_dir = join(base_dir, 'word2vec')

# replace /home/rebecca/Downloads with location of download
# gensim.models.KeyedVectors.load_word2vec_format('/home/rebecca/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format('/home/nastase/social_actions/word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)

word_type = 'verbs'

with open(join(w2v_dir, 'annotation_{0}.txt'.format(word_type))) as f:
    lines = [line.strip().replace(' ', '').split(',') for line in f.readlines()]

weighted_average = True
if weighted_average:
    from collections import Counter
    lines = [list(set(line)) for line in lines]
    vocabulary = [word for line in lines for word in line]
    a = .001
    frequencies = Counter(vocabulary)
    probabilities = {word: frequencies[word] / float(len(vocabulary)) for 
                     word in frequencies.keys()}
    weights = {word: (a / (probabilities[word] + a)) for word in probabilities.keys()}


condition_order = h5load('/home/nastase/social_actions/scripts/condition_order.hdf5')

annotation = {c: w for c, w in zip(condition_order['original_condition_order'], lines)}

# create an empty list
vectors = {}

# get an an array for a list of verbs
for stimulus, words in annotation.items():
    stim_vectors = []
    for word in words: 
        if weighted_average:
            stim_vectors.append(model.word_vec(word) * weights[word])
        elif not weighted_average:
            stim_vectors.append(model.word_vec(word))
    if weighted_average:
        avg_vec = np.sum(np.vstack(stim_vectors), axis=0)
    elif not weighted_average:
        avg_vec = np.mean(np.vstack(stim_vectors), axis=0)
    assert len(avg_vec) == 300
    vectors[stimulus] = avg_vec

vectors_arr = np.vstack([vectors[stimulus] for stimulus
                         in condition_order['original_condition_order']])
# check shape
print vectors_arr.shape


# distances will give (n*(n-1))/2 values
target_RDM = pdist(vectors_arr, 'cosine')
if weighted_average:
    fn = 'word2vec_weighted_{0}_RDM.hdf5'.format(word_type)
elif not weighted_average:
    fn = 'word2vec_{0}_RDM.hdf5'.format(word_type)
h5save(join(scripts_dir, fn), target_RDM)

reorder, sparse_ordered_labels = condition_order['reorder'], condition_order['sparse_ordered_labels']

plt.figure(figsize=(8, 6))
ax = sns.heatmap(squareform(rankdata(target_RDM) / len(target_RDM) * 100)[reorder][:, reorder], vmin=0, vmax=100,
            square=True, cmap='RdYlBu_r', xticklabels=sparse_ordered_labels, yticklabels=sparse_ordered_labels)
ax.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.yticks(va='top')
plt.tight_layout()
plt.show()
