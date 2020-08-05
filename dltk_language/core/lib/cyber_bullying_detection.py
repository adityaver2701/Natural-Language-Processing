import nltk
import pickle
nltk.download('stopwords')
from core.lib.lib import sentiment
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import numpy as np
import os
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

def clean_text(text):

    ## Remove puncuation
    text = text.translate(string.punctuation)

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)



    return text

def detect(text):
    maxlen = 50
    embed_dim = 100
    max_words = 20000
    print('inside detect function before cleaning')
    text=clean_text(str(text))
    print(text)
    print(type(text))
    with open('resources/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    ls=[]
    ls.append(text)
    data = tokenizer.texts_to_sequences(ls)
    print('tokenized data-',data)
    data = pad_sequences(data, maxlen=maxlen, padding='post')
    print('padded-seq',data)
    model=load_model('resources/Cyber_bullying-LSTM-multi-class')
    params={}
    params['text']=text
    res=sentiment(params)
    comment_state=res['emotion']
    tagged_words=res['text'].split(':')[1]
    sample='no discernable type'
    if res['emotion']=='NEGATIVE':
        print(res)
        pred=model.predict(data)
        print(pred)
    #print(model.summary())
        #dt=tokenizer.word_index
        print(pred)
        print(np.argmax(pred))
        sample=''
        if np.argmax(pred)==1:
            sample='racism'
        elif np.argmax(pred)==2:
            sample='sexism'
        else:
            print("not detected")
    dt={}
    dt['state']=comment_state
    dt['tagged_words']=tagged_words
    dt['cyber_bullying_type']=sample
    return dt
    #print(dt['screw'])
