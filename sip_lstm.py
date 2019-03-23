import datetime
import json
import numpy as np
import random
import re
import tensorflow as tf

json_data = json.load(open('sip_lstm_config.json'))

text = open('issa_haiku').read()

word_regex = '(?:[A-Za-z\']*(?:(?<!-)-(?!-))*[A-Za-z\']+)+'
regex = word_regex + '|--|\\.{3}|\\n| |"|,|!|\\?'
words = re.findall(regex, text)
word_list = sorted(list(set(words)))
num_words = len(word_list)

word_to_id = dict((w, i) for i, w in enumerate(word_list))
id_to_word = dict((i, w) for i, w in enumerate(word_list))

def sample(prediction):
	r = random.uniform(0,1)
	s = 0
	word_id = len(prediction) - 1
	for i in range(len(prediction)):
		s += prediction[i]
		if s >= r:
			word_id = i
			break
	word_one_hot = np.zeros(shape=[num_words])
	word_one_hot[word_id] = 1.0
	return word_one_hot

len_per_section = json_data['len_per_section']
skip = json_data['skip']
sections = []
next_words = []

for i in range(0, len(words) - len_per_section, skip):
	sections.append(words[i: i + len_per_section])
	next_words.append(words[i + len_per_section])

X = []

def X_section(row_start, row_end):
	generated_X = np.zeros((row_end - row_start, len_per_section, num_words))
	for non_zero in X:
		if row_start <= non_zero[0] and non_zero[0] < row_end:
			generated_X[non_zero[0] - row_start, non_zero[1], non_zero[2]] = 1
	return generated_X

y = np.zeros((len(sections), num_words))

for i, section in enumerate(sections):
	for j, word in enumerate(section):
		X.append([i, j, word_to_id[word]])
	y[i, word_to_id[next_words[i]]] = 1

batch_size = json_data['batch_size']
hidden_nodes = json_data['hidden_nodes']

checkpoint_directory = json_data['checkpoint_directory']
current_step_path = checkpoint_directory + '/' + json_data['current_step_name']

graph = tf.Graph()
with graph.as_default():
	global_step = tf.Variable(0)
	data = tf.placeholder(tf.float32, [batch_size, len_per_section, num_words])
	labels = tf.placeholder(tf.float32, [batch_size, num_words])

	w_ii = tf.Variable(tf.truncated_normal([num_words, hidden_nodes], -0.1, 0.1))
	w_io = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
	b_i = tf.Variable(tf.zeros([1, hidden_nodes]))

	w_fi = tf.Variable(tf.truncated_normal([num_words, hidden_nodes], -0.1, 0.1))
	w_fo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
	b_f = tf.Variable(tf.zeros([1, hidden_nodes]))

	w_oi = tf.Variable(tf.truncated_normal([num_words, hidden_nodes], -0.1, 0.1))
	w_oo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
	b_o = tf.Variable(tf.zeros([1, hidden_nodes]))

	w_ci = tf.Variable(tf.truncated_normal([num_words, hidden_nodes], -0.1, 0.1))
	w_co = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
	b_c = tf.Variable(tf.zeros([1, hidden_nodes]))

	def lstm(i, o, state):
		input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
		forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
		output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
		memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)
		state = forget_gate * state + input_gate * memory_cell
		output = output_gate * tf.tanh(state)
		return output, state

	output = tf.zeros([batch_size, hidden_nodes])
	state = tf.zeros([batch_size, hidden_nodes])

	for i in range(len_per_section):
		output, state = lstm(data[:, i, :], output, state)
		if i == 0:
			outputs_all_i = output
			labels_all_i = data[:, i+1, :]
		elif i != len_per_section - 1:
			outputs_all_i = tf.concat([outputs_all_i, output], 0)
			labels_all_i = tf.concat([labels_all_i, data[:, i+1, :]], 0)
		else:
			outputs_all_i = tf.concat([outputs_all_i, output], 0)
			labels_all_i = tf.concat([labels_all_i, labels], 0)

	w = tf.Variable(tf.truncated_normal([hidden_nodes, num_words], -0.1, 0.1))
	b = tf.Variable(tf.zeros([num_words]))

	logits = tf.matmul(outputs_all_i, w) + b
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_all_i))
	optimizer = tf.train.GradientDescentOptimizer(10.).minimize(loss, global_step=global_step)

	test_data = tf.placeholder(tf.float32, shape=[1, num_words])
	test_output = tf.Variable(tf.zeros([1, hidden_nodes]))
	test_state = tf.Variable(tf.zeros([1, hidden_nodes]))

	reset_test_state = tf.group(test_output.assign(tf.zeros([1, hidden_nodes])), test_state.assign(tf.zeros([1, hidden_nodes])))

	test_output, test_state = lstm(test_data, test_output, test_state)
	test_prediction = tf.nn.softmax(tf.matmul(test_output, w) + b)

def train_lstm(max_steps, log_every, save_every):
	with tf.Session(graph=graph) as sess:
		tf.global_variables_initializer().run()
		try:
			model = tf.train.latest_checkpoint(checkpoint_directory)
			saver = tf.train.Saver()
			saver.restore(sess, model)
			file = open(current_step_path, 'r')
			step_start = int(file.read()) + 1
			file.close()
		except:
			saver = tf.train.Saver()
			step_start = 0

		offset = 0

		if step_start >= max_steps + 1:
			print('training has already reached target step %d (%s)' % (max_steps, datetime.datetime.now()))

		for step in range(step_start, max_steps + 1):
			offset = offset % len(sections)
			if offset <= (len(sections) - batch_size):
				batch_data = X_section(offset, offset + batch_size)
				batch_labels = y[offset: offset + batch_size]
				offset += batch_size
			else:
				to_add = batch_size - (len(sections) - offset)
				batch_data = np.concatenate((X_section(offset, len(sections)), X_section(0, to_add)))
				batch_labels = np.concatenate((y[offset: len(sections)], y[0: to_add]))
				offset = to_add

			_, training_loss = sess.run([optimizer, loss], feed_dict={data: batch_data, labels: batch_labels})

			if step % log_every == 0:
				print('training loss at step %d: %.3f (%s)' % (step, training_loss, datetime.datetime.now()))

			if step % save_every == 0:
				saver.save(sess, checkpoint_directory + '/model', global_step=step)
				file = open(current_step_path, 'w')
				file.write(str(step))
				file.close()

			if step == max_steps:
				print('training has reached target step %d (%s)' % (max_steps, datetime.datetime.now()))

def test_lstm(start):
	if start != ['']:
		test_start = start
	else:
		test_start = [random.choice(words)]
		while test_start[0] in ['\n', ' ', '!', ',', '--', '...', '?']:
			test_start = [random.choice(words)]

	with tf.Session(graph=graph) as sess:
		tf.global_variables_initializer().run()
		model = tf.train.latest_checkpoint(checkpoint_directory)
		saver = tf.train.Saver()
		saver.restore(sess, model)
		reset_test_state.run()
		test_generated = ''.join(test_start)

		for i in range(len(test_start) - 1):
			test_X = np.zeros((1, num_words))
			test_X[0, word_to_id[test_start[i]]] = 1.
			_ = sess.run(test_prediction, feed_dict={test_data: test_X})

		test_X = np.zeros((1, num_words))
		test_X[0, word_to_id[test_start[-1]]] = 1.

		for i in range(json_data['num_generate_on_test']):
			prediction = test_prediction.eval({test_data: test_X})[0]
			next_word_one_hot = sample(prediction)
			next_word = id_to_word[np.argmax(next_word_one_hot)]
			test_generated += next_word
			test_X = next_word_one_hot.reshape((1, num_words))

	return test_generated

def generate_sip(start):
	test_generated = test_lstm(start)
	sips = test_generated.split('\n\n')

	for i in reversed(range(len(sips))):
		sips[i] = sips[i].strip()
		if sips[i] == '':
			del sips[i]

	try:
		return random.choice(sips[2: len(sips) - 2])
	except:
		try:
			return random.choice(sips)
		except:
			return '***no data generated***'