#dependencies

import numpy as np #vectorization
import random #generating text
#import tensorflow #ML
import datetime #clock training 
import json

text = open('issa-haikus-english.json').read()
data = json.loads(text)
print(data)