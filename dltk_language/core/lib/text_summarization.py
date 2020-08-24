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
import pandas as pd 
import os
import string

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import unicodedata
from string import punctuation
from nltk.tokenize.toktok import ToktokTokenizer

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
#Attention mechanism
import tensorflow as tf
import os
from keras.models import load_model
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
   
K.clear_session()
from keras.utils import CustomObjectScope
warnings.filterwarnings("ignore")

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
    tot_cnt = 0
    cnt = 0
    num_words = tot_cnt-cnt
    max_text_len=40
    max_summary_len=13
    print('Printing Text Typed:')
    text=clean_text(str(text))
    print(text)
    print(type(text))
    with open('resources/text_sum_y_tokenizer.pickle', 'rb') as handle:
        y_tokenizer = pickle.load(handle)
    ls = []
    ls.append(text)
    y_data = y_tokenizer.fit_on_texts(ls)
    y_data = y_tokenizer.texts_to_sequences(ls)
    print('y_tokenized data-',y_data)
    y_tr_data = pad_sequences(y_data, maxlen = max_text_len, padding='post')
    print('y_pad sequences data-',y_tr_data)
    
    with open('resources/text_sum_x_tokenizer.pickle', 'rb') as handle:
        x_tokenizer = pickle.load(handle)
    x_data = x_tokenizer.fit_on_texts(ls)
    x_data = x_tokenizer.texts_to_sequences(ls)
    print('x_tokenized data-',x_data)
    x_tr_data = pad_sequences(x_data, maxlen = max_text_len, padding='post')
    print('x_pad sequences data-',x_tr_data)
    
    #implimenting Class AttentionLayer
    class AttentionLayer(Layer):

        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)
    
        def build(self, input_shape):
            assert isinstance(input_shape, list)
            # Create a trainable weight variable for this layer.
    
            self.W_a = self.add_weight(name='W_a',
                                       shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                       initializer='uniform',
                                       trainable=True)
            self.U_a = self.add_weight(name='U_a',
                                       shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                       initializer='uniform',
                                       trainable=True)
            self.V_a = self.add_weight(name='V_a',
                                       shape=tf.TensorShape((input_shape[0][2], 1)),
                                       initializer='uniform',
                                       trainable=True)
    
            super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end
    
        def call(self, inputs, verbose=False):
            """
            inputs: [encoder_output_sequence, decoder_output_sequence]
            """
            assert type(inputs) == list
            encoder_out_seq, decoder_out_seq = inputs
            if verbose:
                print('encoder_out_seq>', encoder_out_seq.shape)
                print('decoder_out_seq>', decoder_out_seq.shape)
    
            def energy_step(inputs, states):
                """ Step function for computing energy for a single decoder state
                inputs: (batchsize * 1 * de_in_dim)
                states: (batchsize * 1 * de_latent_dim)
                """
    
                assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
                assert isinstance(states, list) or isinstance(states, tuple), assert_msg
    
                """ Some parameters required for shaping tensors"""
                en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
                de_hidden = inputs.shape[-1]
    
                """ Computing S.Wa where S=[s0, s1, ..., si]"""
                # <= batch size * en_seq_len * latent_dim
                W_a_dot_s = K.dot(encoder_out_seq, self.W_a)
    
                """ Computing hj.Ua """
                U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
                if verbose:
                    print('Ua.h>', U_a_dot_h.shape)
    
                """ tanh(S.Wa + hj.Ua) """
                # <= batch_size*en_seq_len, latent_dim
                Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
                if verbose:
                    print('Ws+Uh>', Ws_plus_Uh.shape)
    
                """ softmax(va.tanh(S.Wa + hj.Ua)) """
                # <= batch_size, en_seq_len
                e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
                # <= batch_size, en_seq_len
                e_i = K.softmax(e_i)
    
                if verbose:
                    print('ei>', e_i.shape)
    
                return e_i, [e_i]
    
            def context_step(inputs, states):
                """ Step function for computing ci using ei """
    
                assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
                assert isinstance(states, list) or isinstance(states, tuple), assert_msg
    
                # <= batch_size, hidden_size
                c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
                if verbose:
                    print('ci>', c_i.shape)
                return c_i, [c_i]
    
            fake_state_c = K.sum(encoder_out_seq, axis=1)
            fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim
    
            """ Computing energy outputs """
            # e_outputs => (batch_size, de_seq_len, en_seq_len)
            last_out, e_outputs, _ = K.rnn(
                energy_step, decoder_out_seq, [fake_state_e],
            )
    
            """ Computing context vectors """
            last_out, c_outputs, _ = K.rnn(
                context_step, e_outputs, [fake_state_c],
            )
    
            return c_outputs, e_outputs
    
        def compute_output_shape(self, input_shape):
            """ Outputs produced by the layer """
            return [
                tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
                tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
            ]
    
    
    
    
    #size of vocabulary ( +1 for padding token)
    x_voc =  x_tokenizer.num_words + 1
    y_voc =  y_tokenizer.num_words + 1

    latent_dim = 300
    embedding_dim=100
    
    # Encoder
    encoder_inputs = Input(shape=(max_text_len,))
    
    #embedding layer
    enc_emb = Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)
    
    #encoder lstm 1
    encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)
    
    #encoder lstm 2
    encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)
    
    #encoder lstm 3
    encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
    encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    
    #embedding layer
    dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
    decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])
    
    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    
    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
    
    #dense layer
    decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    model.summary()
    
    #Next, let’s build the dictionary to convert the index to word for target and source vocabulary:
    reverse_target_word_index=y_tokenizer.index_word
    reverse_source_word_index=x_tokenizer.index_word
    target_word_index=y_tokenizer.word_index
    
    #Inference
    #Set up the inference for the encoder and decoder:
    # Encode the input sequence to get the feature vector
    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])


    # Decoder setup
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))
    
    # Get the embeddings of the decoder sequence
    dec_emb2= dec_emb_layer(decoder_inputs) 
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
    
    #attention inference
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_inf_concat) 

    # Final decoder model
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])
    
    
    #We are defining a function below which is the implementation of the inference process.
    def decode_sequence(input_seq):
        e_out, e_h, e_c = encoder_model.predict(input_seq)
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = target_word_index['sostok']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]
        
            if(sampled_token!='eostok'):
                decoded_sentence += ' '+sampled_token

       
            if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c

        return decoded_sentence
    
    
    #Let us define the functions to convert an integer sequence to a word sequence for summary as well as the reviews:
    def seq2summary(input_seq):
        newString=''
        for i in input_seq:
            if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
                newString=newString+reverse_target_word_index[i]+' '
        return newString

    def seq2text(input_seq):
        newString=''
        for i in input_seq:
            if(i!=0):
                newString=newString+reverse_source_word_index[i]+' '
        return newString

    
    
    #Here are a few summaries generated by the model:
    #returing response
    response ={
        "Text": seq2text(x_tr_data[0]),
        "Original Summary" : seq2summary(y_tr_data[0]),
        "Predicted Summary": decode_sequence(x_tr_data[0].reshape(1,max_text_len))
        }
    
    return response

