import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import nltk
import spacy
spacy.load("en_core_web_sm")
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
import unicodedata
from string import punctuation
from nltk.tokenize.toktok import ToktokTokenizer
from collections import defaultdict

def get_topics(topics, feature_names, sorting, topics_per_chunk=5,
                 n_words=5):
    response = defaultdict(list)
    for i in range(0, len(topics), topics_per_chunk):
        # for each chunk:
        these_topics = topics[i: i + topics_per_chunk]
        # maybe we have less than topics_per_chunk left
        len_this_chunk = len(these_topics)
        # print top n_words frequent words
        for i in range(n_words):
            try:
                topic_name = (("{:<5}" * len_this_chunk).format(*feature_names[sorting[these_topics, i]]))
                response['topic'+str(i)].append(topic_name)
               
            except:
                pass
        print("\n")

    return response


def detect(text):
    print('Printing Text Typed:')
    text = re.sub("[^a-zA-Z]+", " ", str(text))
    print(text)
    print(type(text))
    
    
    vect=CountVectorizer(ngram_range=(1,3),stop_words='english')
    dtm=vect.fit_transform([text])
    print(dtm)
    dtm.toarray()
    
    columns=vect.get_feature_names()
    lda=LatentDirichletAllocation(n_components=5)
    lda_test = lda.fit_transform(dtm)
    print("printing LDA values")
    print(lda_test)
    sorting=np.argsort(lda.components_)[:,::-1]
    features=np.array(columns)
    print("topics are forming")

    
    output = get_topics(topics=range(5), feature_names=features,
                               sorting=sorting, topics_per_chunk=5, n_words=10)
    print("printing Output")
    print(output)
    print(type(output))
    
   
    # pyLDAvis.save_html(zit, 'lda.html')
    # zit=pyLDAvis.sklearn.prepare(lda,dtm,vect)
    # pyLDAvis.show(zit)
    
    return output
   
