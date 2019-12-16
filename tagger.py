import numpy as np

from util import accuracy
from hmm import HMM

def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	state_dict = {}
	obs_dict = {}

	state_counter = 0
	obs_counter = 0

	for tag in tags:
		state_dict[tag] = state_counter
		state_counter = state_counter + 1

	for line in train_data:
		for word in line.words:
			if word not in obs_dict:
				obs_dict[word] = obs_counter
				obs_counter = obs_counter + 1

	S = len(state_dict.keys())
	L = len(obs_dict.keys())
	A = np.zeros([S, S])
	B = np.zeros([S, L])
	pi = np.zeros([S])

	for line in train_data:
		for w in range(len(line.words)):
			B[state_dict[line.tags[w]], obs_dict[line.words[w]]] = B[state_dict[line.tags[w]], obs_dict[line.words[w]]] + 1

	for i in range(S):
		sums = np.sum(B[i, :])
		for j in range(L):
			B[i, j] = B[i, j] / sums

	for line in train_data:
		for i in range(len(line.tags)-1):
			#print(i, ",", i+1)
			A[state_dict[line.tags[i]], state_dict[line.tags[i+1]]] = A[state_dict[line.tags[i]], state_dict[line.tags[i+1]]] + 1

	for i in range(S):
		sums = np.sum(A[i, :])
		for j in range(S):
			A[i, j] = A[i, j] / sums

	for line in train_data:
		pi[state_dict[line.tags[0]]] = pi[state_dict[line.tags[0]]] + 1
	pi = pi / np.sum(pi)

	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model

def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	S = len(model.state_dict.keys())
	L = len(model.obs_dict.keys())
	#Make sure ever word/observation for all the test_data is present in the obs_dict, else update obs_dict and B
	for line in test_data:
		for word in line.words:
			if word not in model.obs_dict:
				b = np.ones([S, 1]) * 0.000001
				model.B = np.append(model.B, b, axis = 1)
				model.obs_dict[word] = len(model.obs_dict.keys())

	#Call viterbi algorithm with the updated model and the given sequence of words/observations
	for line in test_data:
		tagg = model.viterbi(line.words)
		tagging.append(tagg)
	###################################################
	return tagging
