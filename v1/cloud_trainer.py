#https://www.youtube.com/watch?v=ZGU5kIG7b2I
#dependencies

import numpy as np #vectorization
import random #generating random text
import tensorflow as tf #ML
import datetime #clock training time

bucket = "gs://aihaikugenerator_storage/"

text = open(bucket + 'issa_haikus_english').read()

#sort characters
chars = sorted(list(set(text)))
char_size = len(chars)

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
	char_one_hot = np.zeros(shape=[char_size])
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

#machine learning time
batch_size = 512
#number of iterations
max_steps = 25001
log_every = 50
save_every = 500
#needs to be set to avoid under and over fitting
hidden_nodes = 1024
#save model
checkpoint_directory = bucket + 'ckpt'

#create checkpoint directory
'''
if tf.gfile.Exists(checkpoint_directory):
	tf.gfile.DeleteRecursively(checkpoint_directory)
tf.gfile.MakeDirs(checkpoint_directory)
'''

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
	w_oi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
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

	############
	#operation
	############
	#LSTM
	#both start off as empty, LSTM will calculate this
	output = tf.zeros([batch_size, hidden_nodes])
	state = tf.zeros([batch_size, hidden_nodes])

	#unrolled LSTM loop
	#for each input set
	for i in range(len_per_section):
		#calculate state and output from LSTM
		output, state = lstm(data[:, i, :], output, state)
		#to start, 
		if i == 0:
			#store initial output and labels
			outputs_all_i = output
			labels_all_i = data[:, i+1, :]
		#for each new set, concat outputs and labels
		elif i != len_per_section - 1:
			#concatenates (combines) vectors along a dimension axis, not multiply
			outputs_all_i = tf.concat([outputs_all_i, output], 0)
			labels_all_i = tf.concat([labels_all_i, data[:, i+1, :]], 0)
		else:
			#final store
			outputs_all_i = tf.concat([outputs_all_i, output], 0)
			labels_all_i = tf.concat([labels_all_i, labels], 0)

	#Classifier
	#The Classifier will only run after saved_output and saved_state were assigned.

	#calculate weight and bias values for the network
	#generated randomly given a size and distribution
	w = tf.Variable(tf.truncated_normal([hidden_nodes, char_size], -0.1, 0.1))
	b = tf.Variable(tf.zeros([char_size]))
	#Logits simply means that the function operates on the unscaled output 
	#of earlier layers and that the relative scale to understand the units 
	#is linear. It means, in particular, the sum of the inputs may not equal 1, 
	#that the values are not probabilities (you might have an input of 5).
	logits = tf.matmul(outputs_all_i, w) + b

	#logits is our prediction outputs, lets compare it with our labels
	#cross entropy since multiclass classification
	#computes the cost for a softmax layer
	#then Computes the mean of elements across dimensions of a tensor.
	#average loss across all values
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_all_i))

	#Optimizer
	#minimize loss with graident descent, learning rate 10,keep track of batches
	optimizer = tf.train.GradientDescentOptimizer(10.).minimize(loss, global_step=global_step)

	###########
	#Test
	###########
	test_data = tf.placeholder(tf.float32, shape=[1, char_size])
	test_output = tf.Variable(tf.zeros([1, hidden_nodes]))
	test_state = tf.Variable(tf.zeros([1, hidden_nodes]))

	#Reset at the beginning of each test
	reset_test_state = tf.group(test_output.assign(tf.zeros([1, hidden_nodes])), test_state.assign(tf.zeros([1, hidden_nodes])))

	#LSTM
	test_output, test_state = lstm(test_data, test_output, test_state)
	test_prediction = tf.nn.softmax(tf.matmul(test_output, w) + b)

#time to train the model, initialize a session with a graph
with tf.Session(graph=graph) as sess:
	#standard init step
	tf.global_variables_initializer().run()
	try:
		model = tf.train.latest_checkpoint(checkpoint_directory)
		saver = tf.train.Saver()
		saver.restore(sess, model)
		file = open(bucket + "ckpt/step.txt", "r")
		step_start = int(file.read()) + 1
		file.close()
	except:
		saver = tf.train.Saver()
		step_start = 0;
	offset = 0
	
	#for each training step
	for step in range(step_start, max_steps):
		#starts off as 0
		offset = offset % len(X)
		
		#calculate batch data and labels to feed model iteratively
		if offset <= (len(X) - batch_size):
				#first part
				batch_data = X[offset: offset + batch_size]
				batch_labels = y[offset: offset + batch_size]
				offset += batch_size
		#until when offset= batch size, then we 
		else:
				#last part
				to_add = batch_size - (len(X) - offset)
				batch_data = np.concatenate((X[offset: len(X)], X[0: to_add]))
				batch_labels = np.concatenate((y[offset: len(X)], y[0: to_add]))
				offset = to_add
		
		#optimize!!
		_, training_loss = sess.run([optimizer, loss], feed_dict={data: batch_data, labels: batch_labels})
		
		if step % log_every == 0:
				print('training loss at step %d: %.2f (%s)' % (step, training_loss, datetime.datetime.now()))

		if step % save_every == 0:
				saver.save(sess, checkpoint_directory + '/model', global_step=step)
				file = open(bucket + "ckpt/step.txt", "w")
				file.write(str(step))
				file.close()