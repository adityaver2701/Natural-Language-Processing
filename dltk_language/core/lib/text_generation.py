# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:07:55 2020

@author: kavya
"""

#%%
import pickle
import logging
from simpletransformers.language_generation import LanguageGenerationModel

def text_generator(text):
    model = LanguageGenerationModel("gpt2", "gpt2", args={"length": 256},use_cuda=False)
    generated = model.generate(text, args={"max_length":300})
    return(generated[0].replace('\n',''))

#text_generator('travelling to hawaii is the best idea because there are a lot of places to visit')

#def text_generator(text):
#    model = pickle.load(open('resources/text_generation/textgen_gpt2.pk','rb'))
#    generated = model.generate(text, args={"max_length":300})
#    return(generated[0])

#text_generator('travelling to hawaii is the best idea because there are a lot of places to visit')