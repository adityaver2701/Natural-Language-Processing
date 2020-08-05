# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:38:14 2020

@author: kavya
"""

#!pip install pyLDAvis
import pyLDAvis.sklearn

import pickle
import seaborn as sns
import pandas as pd
import itertools
import numpy as np
import re, nltk, spacy, gensim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_score, make_scorer, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import LinearSVC
from matplotlib.collections import LineCollection
from nltk.tokenize import ToktokTokenizer
from nltk.stem import wordnet
from nltk.corpus import stopwords
from string import punctuation
import pickle
import joblib

token = ToktokTokenizer()
punct = punctuation

stop_words = set(stopwords.words("english"))
def stopWordsRemove(text):
    ''' Removing all the english stop words from a corpus

    Parameter:

    text: document to remove stop words from it
    '''

    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    
    return ' '.join(map(str, filtered))

def clean_text(text):
    ''' Lowering text and removing undesirable marks

    Parameter:
    
    text: document to be cleaned    
    '''
    
    text = text.lower()
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text) # matches all whitespace characters
    text = text.strip(' ')
    return text

def strip_list_noempty(mylist):
    
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    
    return [item for item in newlist if item != '']
def clean_punct(text): 
    ''' Remove all the punctuation from text, unless it's part of an important 
    tag (ex: c++, c#, etc)

    Parameter:
    
    text: document to remove punctuation from it
    '''

    words = token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    
    for w in words:
        #if w in top_tags:
        #    punctuation_filtered.append(w)
        #else:
        w = re.sub('^[0-9]*', " ", w)
        punctuation_filtered.append(regex.sub('', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))

stop_words = set(stopwords.words("english"))
def stopWordsRemove(text):
    ''' Removing all the english stop words from a corpus

    Parameter:

    text: document to remove stop words from it
    '''

    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    
    return ' '.join(map(str, filtered))


#model_test = joblib.load('C:\\Users\\kavya\\Desktop\\Kavya\\Tag Recommendation\\Model\\lda_model.jl')
#lda_components = model_test.components_ / model_test.components_.sum(axis=1)[:, np.newaxis] # normalization

#vectorizer = pickle.load(open('C:\\Users\\kavya\\Desktop\\Kavya\\Tag Recommendation\\Model\\vectorizer.pk','rb'))


def recommended_tags(text):
    
    model_test = joblib.load('resources/lda_model.jl')
    lda_components = model_test.components_ / model_test.components_.sum(axis=1)[:, np.newaxis]
    vectorizer = pickle.load(open('resources/vectorizer.pk','rb'))
    text = stopWordsRemove(text) 
    
    
    text = str(text)
    text = clean_text(text)
    text = clean_punct(text)
    text = stopWordsRemove(text)
    
    n_topics = 10
    threshold = 0.001
    list_scores = []
    list_words = []
    used = set()
    
    text_tfidf = vectorizer.transform([text])
    text_projection = model_test.transform(text_tfidf)
    feature_names = vectorizer.get_feature_names()
    for topic in range(n_topics):
        topic_score = text_projection[0][topic]
    
        for (word_idx, word_score) in zip(lda_components[topic].argsort()[:-5:-1], sorted(lda_components[topic])[:-5:-1]):
            score = topic_score*word_score
    
            if score >= threshold:
                list_scores.append(score)
                list_words.append(feature_names[word_idx])
                used.add(feature_names[word_idx])
                
    results = [tag for (y,tag) in sorted(zip(list_scores,list_words), key=lambda pair: pair[0], reverse=True)]
    unique_results = [x for x in results if x not in used] # get only unique tags
    tags = " ".join(results[:5])
    return tags

recommended_tags('where to learn sql from')
