import nltk
import spacy
spacy.load("en_core_web_sm")
import pickle
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import keras
import numpy as np
import os
import string
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import unicodedata
from string import punctuation
from keras.preprocessing import text, sequence
from nltk.tokenize.toktok import ToktokTokenizer


def clean_text(text):

    text = BeautifulSoup(text, "html.parser")
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'http\S+', '', text)
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
    num_words=35000
    maxlen = 20
    print('Printing Text Typed:')
    text=clean_text(str(text))
    print(text)
    print(type(text))
    with open('resources/sarcasm_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    ls = []
    ls.append(text)
    data = tokenizer.fit_on_texts(ls)
    data = tokenizer.texts_to_sequences(ls)
    print('tokenized data-',data)
    data = sequence.pad_sequences(data, maxlen)
    print('pad sequences data-',data)
    model= tf.keras.models.load_model('resources/sarcasm_word2vec.h5')
    pred=model.predict(data)
    print('Printing Predictions')
    print(pred)
    sample='nothing detected'
    if pred>0.5:
        sample='Sarcastic'
        print("Sarcastic")
    elif pred<0.5:
        sample='No Sarcastic'
        print("No Sarcastic")
    return sample
