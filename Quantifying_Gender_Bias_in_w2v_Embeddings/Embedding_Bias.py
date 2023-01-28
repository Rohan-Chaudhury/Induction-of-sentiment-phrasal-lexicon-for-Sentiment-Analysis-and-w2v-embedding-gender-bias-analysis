#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import pprint
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def loadData():
    
    with open ('wordpairs.txt') as f:
        wordpairs=json.load(f)
        
    with open ('occupations.txt') as f:
        occupations=json.load(f)
    
    vecs = []
    words = []
    with open('w2v_gnews_small.txt', "r", encoding='utf8') as f:
        for line in f:
            s = line.split()
            v = np.array([float(x) for x in s[1:]])
            if len(vecs) and vecs[-1].shape!=v.shape:
                print("Got weird line", line)
                continue
            words.append(s[0])
            vecs.append(v)
    
    return wordpairs, occupations, words, vecs

def genderDicrection(wordpairs,words, vecs):
    # implement your code here
    word_embeddings={k:v for k,v in zip(words,vecs)}
    g_vecs=[]
    for i in range(len(wordpairs)):
        mid_vec=(word_embeddings[wordpairs[i][0]]+word_embeddings[wordpairs[i][1]])/2
        # vec_diff=word_embeddings[wordpairs[i][0]]-word_embeddings[wordpairs[i][1]]
        vec_diff1=word_embeddings[wordpairs[i][0]]-mid_vec
        vec_diff2=word_embeddings[wordpairs[i][1]]-mid_vec
        # normalized_vec_diff=vec_diff/np.linalg.norm(vec_diff) if np.linalg.norm(vec_diff)!=0 else vec_diff
        normalized_vec_diff1=vec_diff1/np.linalg.norm(vec_diff1) if np.linalg.norm(vec_diff1)!=0 else vec_diff1
        normalized_vec_diff2=vec_diff2/np.linalg.norm(vec_diff2) if np.linalg.norm(vec_diff2)!=0 else vec_diff2
        # g_vecs.append(normalized_vec_diff)
        g_vecs.append(normalized_vec_diff1)
        g_vecs.append(normalized_vec_diff2)

    pca = PCA(n_components=1)
    pca.fit_transform(g_vecs)
    return pca.components_

    
    # return np.zeros((300,))
    

def directBias(g, occupations, words, vecs):  
    # implement your code here
    dict_occupation={}
    for occupation in occupations:
        occupation_vec=vecs[words.index(occupation)]
        dict_occupation[occupation]=np.dot(occupation_vec,g.T)/(np.linalg.norm(occupation_vec)*np.linalg.norm(g))

    return dict_occupation


def indirectBias(g, occupations, words, vecs):  
    # implement your code here
    
    return [[],[]]

wordpairs, occupations, words, vecs=loadData()

g=genderDicrection(wordpairs,words, vecs)
# print(g)
dval=directBias(g, occupations, words, vecs)
dval_sorted=sorted(dval.items(), key=lambda x: abs(x[1]), reverse=False)
print("5 occupations with lowest direct bias:", dval_sorted[:5])
print("\n")
dval_sorted_reverse=sorted(dval.items(), key=lambda x: abs(x[1]), reverse=True)
print("5 occupations with highest direct bias:", dval_sorted_reverse[:5])
print("\n")
dval_sorted_reverse_man=sorted(dval.items(), key=lambda x: x[1], reverse=False)
print("5 occupations with higest direct bias towards man:", dval_sorted_reverse_man[:5])
print("\n")
dval_sorted_reverse_woman=sorted(dval.items(), key=lambda x: x[1], reverse=True)
print("5 occupations with highest direct bias towards woman:", dval_sorted_reverse_woman[:5])
print("\n")
# pp = pprint.PrettyPrinter()
# print('Direct Bias (%f): ' % dval)
# pp.pprint(dval)

# toplists=indirectBias(g, occupations, words, vecs)
# print('Top10 Softball Extreme Words:')
# pp.pprint(toplists[0])

# print('Top10 Football Extreme Words:')
# pp.pprint(toplists[1])


