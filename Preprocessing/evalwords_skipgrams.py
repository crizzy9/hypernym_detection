import glob
import nltk
import string
from collections import Counter
import itertools
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import pickle
import sys
from scipy import sparse
from scipy.sparse import linalg 
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords



punct = set(string.punctuation)
stopwords_set = set(stopwords.words('english'))


corpus = glob.glob("corpus" + "/*.xml")


evalwords = []
with open("test_words.txt", "r") as fp:
    text = fp.readlines()

for word in text:
    evalwords.append(word.strip().lower())



data = []
for text_file in corpus:
    with open(text_file, "r", errors="ignore", encoding="utf-8") as fp:

        line = []

        for i in fp:
            
            line_flag = False
            token = i.strip().split("\t")[0]
            if token == "<s>":
                line = []
            elif token == "</s>":
                final_line = line
                line_flag = True
            elif token in punct or token in stopwords_set:
                pass
            else:
                line.append(i.strip().split("\t")[0])


            if line_flag:
                flag = False
                for word in final_line:
                    for w in evalwords:
                        if w.lower() == word.lower():
                            flag = True
                        else:
                            pass
                    else:
                        pass

                if flag and len(final_line) > 5:
                    data.append(final_line)
                else:
                    pass
            else:
                pass



tok2indx = dict()
unigram_counts = Counter()
for ii, line in enumerate(data):
    for token in line:
        #if token.lower in evalwords:
        if True:
            unigram_counts[token] += 1
            if token not in tok2indx:
                tok2indx[token] = len(tok2indx)
        else:
            pass

indx2tok = {indx:tok for tok,indx in tok2indx.items()}

pickle.dump(indx2tok, open( "indx2tok.pkl", "wb" ))
pickle.dump(tok2indx, open( "tok2indx.pkl", "wb" ))



#print('done')
#print('vocabulary size: {}'.format(len(unigram_counts)))
#print('most common: {}'.format(unigram_counts.most_common(10)))


back_window = 2
front_window = 2
skipgram_counts = Counter()
for iline, line in enumerate(data):
    for ifw, fw in enumerate(line):

        if fw.lower() in evalwords:
            icw_min = max(0, ifw - back_window)
            icw_max = min(len(line) - 1, ifw + front_window)
            icws = [ii for ii in range(icw_min, icw_max + 1) if ii != ifw]
            for icw in icws:
                skipgram = (line[ifw], line[icw])
                skipgram_counts[skipgram] += 1
        else:
            pass    

pickle.dump(skipgram, open( "skipgram.pkl", "wb" ))
pickle.dump(skipgram_counts, open( "skipgram_counts.pkl", "wb" ))


#print('done skipgram')
#print('number of skipgrams: {}'.format(len(skipgram_counts)))
#print('most common: {}'.format(skipgram_counts.most_common(10)))