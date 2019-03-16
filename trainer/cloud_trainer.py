#dependencies

import json #files
import numpy as np #vectorization
import random #generating random text
import tensorflow #ML
import datetime #clock training time

text = open('issa_haikus_english').read()
print('text length in number of characters', len(text))
#print('head of text')
#print(text[:1000])

#print out characters and sort
chars = sorted(list(set(text)))
char_size = len(chars)
print('number of characters', char_size)
print(chars)

#char2id = dict((c, i) for i, c in en)