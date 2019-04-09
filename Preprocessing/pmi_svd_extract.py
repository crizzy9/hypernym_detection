import glob
import nltk
import string
from collections import Counter
import itertools
import nltk
import pickle
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


punct = set(string.punctuation)
stopwords_set = set(stopwords.words('english'))


'''
corpus = glob.glob("corpus" + "/*.xml")

data = []
for text_file in corpus:
    with open(text_file, "r", errors="ignore", encoding="utf-8") as fp:

        fp = fp.readlines()
        line = []
        for i in fp:
            token = i.strip().split("\t")[0]
            if token == "<s>":
                line = []
            elif token == "</s>":
                if len(line) > 5:
                    data.append(line)
                else:
                    pass
            elif token in punct or token in stopwords_set:
                pass
            else:
                line.append(i.strip().split("\t")[0])
print(data)
print(len(data))


## SKip grams
tok2indx = dict()
unigram_counts = Counter()
for ii, line in enumerate(data):
    for token in line:
        unigram_counts[token] += 1
        if token not in tok2indx:
            tok2indx[token] = len(tok2indx)
indx2tok = {indx:tok for tok,indx in tok2indx.items()}
print('done')
print('vocabulary size: {}'.format(len(unigram_counts)))
print('most common: {}'.format(unigram_counts.most_common(10)))

back_window = 2
front_window = 2
skipgram_counts = Counter()
for iline, line in enumerate(data):
    for ifw, fw in enumerate(line):
        icw_min = max(0, ifw - back_window)
        icw_max = min(len(line) - 1, ifw + front_window)
        icws = [ii for ii in range(icw_min, icw_max + 1) if ii != ifw]
        for icw in icws:
            skipgram = (line[ifw], line[icw])
            skipgram_counts[skipgram] += 1

print('done skipgram')
print('number of skipgrams: {}'.format(len(skipgram_counts)))
print('most common: {}'.format(skipgram_counts.most_common(10)))
'''

tok2indx = pickle.load( open("tok2indx.pkl", "rb" ) )
indx2tok = pickle.load( open("indx2tok.pkl", "rb" ) )


skipgram_counts = pickle.load( open("skipgram_counts.pkl", "rb" ))






row_indxs = []
col_indxs = []
dat_values = []
ii = 0
for (tok1, tok2), sg_count in skipgram_counts.items():
    ii += 1
    tok1_indx = tok2indx[tok1]
    tok2_indx = tok2indx[tok2]
    row_indxs.append(tok1_indx)
    col_indxs.append(tok2_indx)
    dat_values.append(sg_count)
wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
print('done')

wwcnt_norm_mat = normalize(wwcnt_mat, norm='l2', axis=1)

num_skipgrams = wwcnt_mat.sum()
assert(sum(skipgram_counts.values())==num_skipgrams)

# for creating sparce matrices
row_indxs = []
col_indxs = []

pmi_dat_values = []
ppmi_dat_values = []
spmi_dat_values = []
sppmi_dat_values = []

# smoothing
alpha = 0.75
nca_denom = np.sum(np.array(wwcnt_mat.sum(axis=0)).flatten()**alpha)
sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()
sum_over_words_alpha = sum_over_words**alpha
sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()

ii = 0
for (tok1, tok2), sg_count in skipgram_counts.items():
    ii += 1
    tok1_indx = tok2indx[tok1]
    tok2_indx = tok2indx[tok2]
    
    nwc = sg_count
    Pwc = nwc / num_skipgrams
    nw = sum_over_contexts[tok1_indx]
    Pw = nw / num_skipgrams
    nc = sum_over_words[tok2_indx]
    Pc = nc / num_skipgrams
    
    nca = sum_over_words_alpha[tok2_indx]
    Pca = nca / nca_denom
    
    pmi = np.log2(Pwc/(Pw*Pc))
    ppmi = max(pmi, 0)
    
    #spmi = np.log2(Pwc/(Pw*Pca))
    #sppmi = max(spmi, 0)
    
    row_indxs.append(tok1_indx)
    col_indxs.append(tok2_indx)
    pmi_dat_values.append(pmi)
    ppmi_dat_values.append(ppmi)
    #spmi_dat_values.append(spmi)
    #sppmi_dat_values.append(sppmi)
        
pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))
ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))
#spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)))
#sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)))

print('done pmi')

pmi_use = ppmi_mat
embedding_size = 50
uu, ss, vv = linalg.svds(pmi_use, embedding_size)

uu.shape

unorm = uu / np.sqrt(np.sum(uu*uu, axis=1, keepdims=True))
vnorm = vv / np.sqrt(np.sum(vv*vv, axis=0, keepdims=True))
#word_vecs = unorm
#word_vecs = vnorm.T
word_vecs = uu + vv.T
word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs*word_vecs, axis=1, keepdims=True))

print("word vec done")

print(word_vecs.shape)

def ww_sim(word, mat, topn=10):
    """Calculate topn most similar words to word"""
    indx = tok2indx[word]
    if isinstance(mat, sparse.csr_matrix):
        v1 = mat.getrow(indx)
    else:
        v1 = mat[indx:indx+1, :]
    print(v1.shape)
    sims = cosine_similarity(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    print(sim_word_scores)
    return sim_word_scores

ww_sim('anarchism', word_vecs)
print(len(tok2indx))
print(len(indx2tok))
