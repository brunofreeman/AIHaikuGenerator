#https://www.youtube.com/watch?v=ZGU5kIG7b2I
#dependencies

import json #files
import numpy as np #vectorization
import random #generating random text
import tensorflow as tf #ML
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

char2id = dict((c, i) for i, c in enumerate(chars))
id2char = dict((i, c) for i, c in enumerate(chars))

#generate probability of each next character
def sample(prediction):
	r = random.uniform(0,1)
	#store prediction char
	s = 0
	char_id = len(prediction) - 1
	#for each char prediction probability
	for i in range(len(prediction)):
		s += prediction[i]
		#generated threshold
		if s >= r:
			char_id = i
			break

	#one hot = know difference between values without ranking
	char_one_hot = np.zeros(shape[char_size])
	char_one_hot[char_id] = 1.0
	return char_one_hot

#vectorize data to feed into model
len_per_section = 5
skip = 2
sections = []
next_chars = []

for i in range(0, len(text) - len_per_section, skip):
	sections.append(text[i: i + len_per_section])
	next_chars.append(text[i + len_per_section])

#vectorize
#matrix of section length by num of characters
X = np.zeros((len(sections), len_per_section, char_size))
#label column for all the character id's, still zero
y = np.zeros((len(sections), char_size))
#for each char in each section, convert each char to an ID
#for each section convert the labels to id's
for i, section in enumerate(sections):
	for j, char in enumerate(section):
		X[i, j, char2id[char]] = 1
	y[i, char2id[next_chars[i]]] = 1
print(y)

#machine learning time
batch_size = 512
#number of iterations
#below values 1/10 of tutorial
max_steps = 7000
log_every = 10
save_every = 600
#needs to be set to avoid under and over fitting
hidden_nodes = 1024
test_start = 'the'
#save model
checkpoint_directory = 'ckpt'

#create checkpoint directory
if tf.gfile.Exists(checkpoint_directory):
	tf.gfile.DeleteRecursively(checkpoint_directory)
tf.gfile.MakeDirs(checkpoint_directory)

print('training data size:', len(X))
print('approximate steps per epoch:', int(len(X)/batch_size))

#build model
graph = tf.Graph()
with graph.as_default():

	global_step = tf.Variable(0)

	data = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size])
	labels = tf.placeholder(tf.float32, [batch_size, char_size])

	#input gate, output gate, forget gate, internal state
	#calculated in vacuums

	#input gate - weights for input, weights for previous output, bias
	w_ii = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
	w_io = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
	b_i = tf.Variable(tf.zeros([1, hidden_nodes]))

	#forget gate
	w_fi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
	w_fo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
	b_f = tf.Variable(tf.zeros([1, hidden_nodes]))

	#output gate
	w_ii = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
	w_oo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
	b_o = tf.Variable(tf.zeros([1, hidden_nodes]))

	#memory cell
	w_ci = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
	w_co = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
	b_c = tf.Variable(tf.zeros([1, hidden_nodes]))

	def lstm(i, o, state):

		#calculated sperately, no overlap, until...
		#(input * input weights) + (output * weights for previous output) + bias
		input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
		
		#(input * forget weights) + (output * weights for previous output) + bias
		forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
		
		#(input * output weights) + (output * weights for previous output) + bias
		output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
		
		#(input * internal state weights) + (output * weights for previous output) + bias
		memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)

		#... now! multiply forget gate * given state + input gate * hidden state
		state = forget_gate * state + input_gate * memory_cell
		#squash that state with tanh nonlin (computes hyperbolic tangent of x element-wise)
		#multiply by output
		output = output_gate * tf.tanh(state)
		#return
		return output, state


















