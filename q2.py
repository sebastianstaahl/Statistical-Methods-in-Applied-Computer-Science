from q2generator import *
import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pdb
import random

def make_deepcopy( Switches, Graph_sz ):
	""" Creates a deep copy of all switch settiings within a graph"""
	return [[Switches[y][x] for x in range(Graph_sz)] for y in range(Graph_sz)]

def mh_w_gibbs(o, G, num_iter, error_prob=0.1):
	s = [] # store samples for the start positions
	X = [] # store switch states
	accepted=0
	proposals=0

	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]

	# Gibbs for the startposition
	for n in range(num_iter):
		log_likelihoods=[]
		for r in range(G.lattice_size):
			for c in range(G.lattice_size):
				switches_prev_it = X[-1]
				logL = cond_L(o, G, G.get_node(r,c), switches_prev_it ,error_prob) #+math.log(1/n_pos)
				log_likelihoods.append( logL )

		# 'Unlogging'the likelihoods
		likelihoods = np.exp(log_likelihoods - np.max(log_likelihoods))
		# Normalization
		probabilities = likelihoods / np.sum( likelihoods)

		#Categorical sampling of new start position
		s_new = np.argmax(np.random.multinomial(1,probabilities))

		# we extract a node for the index for the sampled start position and add it to our list of smaples
		s.append( extract_start_pos(len(X[0]), s_new, G))

		# Metropolis hastings for the switchsettings

		# First we make a deepcopy of the switchsettinsg smapled from the previous iteration
		last_sampled_switch=X[-1]
		X_previous = make_deepcopy(last_sampled_switch, G.lattice_size )

		#We create a vector to store our new accepted switches
		x_new = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

		#We make a copy of the last accepted swittchsettings so that we can condition
		#on the rest of the nodes
		X_proposal = make_deepcopy(last_sampled_switch, G.lattice_size )


		for row in range(G.lattice_size):
			for col in range(G.lattice_size):
				s_last_it=s[-1] # we want to condition on s1 from the last iteration

				#For every node we sample a new switchsetting once from uniform distribution.
				x_proposal =np.random.randint(1,4)
				X_proposal[row][col] = x_proposal

				x_previous = X_previous[row][col] # we keep the previous value in case the proposal is rejected

				# Compute the loglikelihoods for the proposal and the previous iteration
				logL_previous = cond_L(o, G, s_last_it, X_previous, error_prob)
				logL_proposal = cond_L(o, G, s_last_it, X_proposal, error_prob)

				u = np.random.rand()

				#We compute the acceptence prob
				acceptance_prob = min( math.exp(logL_proposal - logL_previous) , 1)

				if u < acceptance_prob:
				# If the acceptence probability is larger than an uniform sample we accpt the proposal
					accepted+=1
					proposals+=1
					X_previous = make_deepcopy(X_proposal, G.lattice_size )
					x_new[row][col] = x_proposal

				else:
					X_proposal = make_deepcopy( X_previous , G.lattice_size )
					x_new[row][col] = x_previous
					proposals+=1

		X.append(x_new)

	print('Acceptance rate ',accepted / proposals)
	return s, X

def extract_start_pos(sz,new_s1,G):
	""" Extraxt a new sampled startposition  node from the graph"""
	c=np.mod(new_s1,sz)
	r=(int)(new_s1/sz)
	return G.get_node(r,c)

def gibbs(o, G, num_iter, prob_err=0.1):
	"""A gibbs sampler"""
	acceptancerate=0
	proposals=0

	s = [] # Start position samples
	X = [] # Switch setting samples
	graph_sz=len(G.G)
	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[ 0 , 0 ]

	for n in range(num_iter):
		log_likelihoods=[]
		#For the gibbs sampler we want to go thorugh every possible start position
		#for s1 in the graph.
		for row in range(graph_sz):
			for col in range(graph_sz):
				#We compute the loged likelihood for every startposition
				start_pos=G.get_node(row,col)
				last_switchsetting=X[-1]
				log_likelihoods.append(cond_L(o, G,start_pos,last_switchsetting,prob_err))

		Likelihoods=np.exp(log_likelihoods-np.max(log_likelihoods)) #"unlogging" the logged likelihoods
		probabilities=Likelihoods/np.sum(Likelihoods) #Normalization

		#Sampling from a Categorical
		samples=np.random.multinomial(1,probabilities)
		new_s1 = np.argmax(samples)

		s.append(extract_start_pos(len(X[0]),new_s1,G)) #Extracting the node for the sampled start position
		X_last_sample= make_deepcopy(X[-1], G.lattice_size) #Making a deepcopy of the last sampled X... python is not nice sometimes

		temp=[]

		for row in range(graph_sz):
			for col in range(graph_sz):
				log_likelihoods=[]

				for switch_setting in range(1,4):
				#For every node in the current Graoh we want to go through every
				#possible switch setting, i.e. 1,2 and 3
					X_last_sample[row][col]=switch_setting
					last_start_pos=s[-1]
					namnare_temp= cond_L( o, G,last_start_pos, X_last_sample, prob_err) + math.log(1 / 3) # the prior is given in the assignment as 1/3
					log_likelihoods.append( namnare_temp )

				unlogedL=np.exp(log_likelihoods-np.max(log_likelihoods)) #unlogging
				normL=unlogedL/np.sum(unlogedL) #normalized liklihoods

				#categorical sampling of new switch setting
				samples=np.random.multinomial(1,normL)
				temp.append(np.max(samples))
				x_val_new = np.argmax(samples)
				X_last_sample[row][col] = 1 + x_val_new

		X.append(X_last_sample)
	return s, X


def convert_array_to_matrix(array_list):
	"""Converts an array of size 9 into a 3x3 matrix
	"""
	return [array_list[0:3], array_list[3:6], array_list[6:9]]


def convert_matrix_to_array( matrix ):
	"""converts a matrix into an array
	"""
	return [val for row in matrix for val in row]


def block_gibbs(o, G, num_iter, error_prob=0.1):
	s = [] # store samples for the start positions
	X = [] # store switch states
	X.append(sample_switch_states(G.lattice_size)) # generate initial switch state
	s.append(sample_start_pos(G)) # set the initial start position as the one at G[0][0]

	for n in range(num_iter):
		log_likelihoods=[]

		for row in range( G.lattice_size ):
			for col in range( G.lattice_size ):
				last_switch_setting = X[-1]

				# We go thorugh every possible value the start pos
				s1=G.get_node( row, col )

				logL = cond_L( o, G, s1, last_switch_setting ,error_prob)
				log_likelihoods.append( logL )

		Likleihood = np.exp( log_likelihoods - np.max( log_likelihoods ))
		#normalize the likelihoods
		probs = Likleihood / np.sum(Likleihood)
		# categorical sampling of new startposition
		s_new = np.argmax(np.random.multinomial(1,probs))
		# we extract a node for the index for the sampled start position and add it to our list of smaples
		s.append( extract_start_pos( len(X[0]), s_new, G))



		# A list with the indicies of the 3 blocks as suggested by Seyong during
		#the help session

		block_1 = [0,2,4]
		block_2 = [3,5,7]
		block_3 = [1,6,8]

		block_indicies = [block_1, block_2, block_3]
		#block_indicies = [[0,2,4], [3,5,7], [3,5,7]]


		# We create an array of the switches in the last samplex switch setting(s) X
		last_switches=X[-1]
		X_tmp=make_deepcopy( last_switches, G.lattice_size )
		X_array = convert_matrix_to_array(X_tmp )


		for block in block_indicies:
			log_likelihoods = []
			# So for the three blocks we want to go though all possible swithc setting for each position
			# Also note that we need three nested loops since we want to compute all possible cobintaions of switch settiings
			#for the three blocks
			for sw_b1 in range(1, 4):
				for sw_b2 in range(1,4):
					for sw_b3 in range(1,4):

						ind_b1 = block[0]
						X_array[ind_b1] = sw_b1

						ind_b2 = block[1]
						X_array[ind_b2] = sw_b2

						ind_b3 = block[2]
						X_array[ind_b3] = sw_b3

						# We convert the array into a matrix again so taht we can send it into cond_L()
						X_array =convert_array_to_matrix( X_array )

						prev_start_pos = s[-1]
						logL = cond_L( o, G, prev_start_pos, X_array, error_prob)+math.log(1/3)
						log_likelihoods.append(logL)
						X_array = convert_matrix_to_array(X_array)

			unLogL =np.exp(log_likelihoods-np.max(log_likelihoods)) # unlogging
			probs = unLogL / np.sum(unLogL) #normalization
			# Categorical resmapling
			x_new = np.argmax(np.random.multinomial( 1, probs ))

			# We extract the new indicies for the new sampled x SEBBE??
			X_NEW=extracter(x_new,sw_b2,sw_b3,X_array,block)

		X_new =convert_array_to_matrix( X_NEW )
		X.append(X_new)

	return s, X
def extracter(x,sw_b2,sw_b3,X,b):
	
	X[b[2]] = np.mod(x, sw_b3 ) + 1
	X[b[1]] = np.mod(np.floor_divide(x , sw_b3), sw_b2) + 1
	X[b[0]] =  np.floor_divide(x, (sw_b2*sw_b3)) + 1
	return X

def cond_L(o,G,start,X,p):
	"""Computes the conditional likelihood
	"""
	logL=0
	O_len=len(o)

	#Initialisation: We start by getting the next node/state from the starting node
	st_new=G.get_next_node(start,0, X)[0]
	curr_node=G.get_node(start.row,start.col)
	prev_dir=G.get_entry_direction(curr_node,st_new)

	for t in range (1,O_len):
		if prev_dir!=0:
			# if the previous direction was not 0, then we know that the next correct
			#observation should be zero
			if o[t]==0:
				#Therefore we add log(1-p) i.e 0.9 if the next observation is 0. Since
				#we did not enter through zero and now have to exit through another switch
				prob=1-p
				logP=math.log(prob)
				logL+=logP
			else:
				#if the observations is not zero it is incorrect. Therefore we add log(p)
				prob=p
				logP=math.log(prob)
				logL+=logP
		else:
			#We know that the previous observation was zero. Therefore we if the switch setting is correct
			#we should exit through the switch setting
			true_switch_setting=X[st_new.row][st_new.col]
			if o[t]==true_switch_setting:
				prob=1-p
				logP=math.log(prob)
				logL+=logP
			else:
				prob=p
				logP=math.log(prob)
				logL+=logP
		start, st_new, prev_dir= next_state(st_new,G,start,X)
	return logL

def next_state(st_n,G,start,X):
	st_hold=st_n
	entry_dir=G.get_entry_direction(start,st_n)
	st_n=G.get_next_node(st_n,entry_dir,X)
	start=st_hold
	st_n=st_n[0]
	node_prev=G.get_node(start.row,start.col)
	node_new=G.get_node(st_n.row,st_n.col)
	prev_dir=G.get_entry_direction(node_prev,node_new)
	return start, st_n, prev_dir

def calc_acc(burn_in,lag,s,s_truth):
	"""Calculates the accuracy of a start position sequence
	"""
	s_b=s[burn_in:-1]
	s_lag=s_b[0::lag]
	s_str=convert_node_to_string(s_lag)

	cnt = Counter(s_str)
	tmp1=cnt.most_common(9)
	occurances_most_common=tmp1[0][0][1]

	s1_truth_str_rep=str(convert_node_to_string([s_truth])[0])


	print('The most common sample was correct! It was: ',s1_truth_str_rep)
	print('Accuracy: ',int(occurances_most_common)/len(s_lag))
	print()


def convert_node_to_string(sequence):
	""" Converts a node to astring representation"""
	str_list=[]
	for s in sequence:
		str_list.append(str(s.row)+' '+str(s.col))
	return str_list

def convergence_histogram_plotter(burn_in,lag,s,s_truth):
	""""Computes histograms for 3 chains for all algorithms for s1 """
	print()
	true_s1_str=str(convert_node_to_string([s_truth])[0])

	s_mhg=s[0]
	sg=s[1]
	sbg=s[2]

	s_mhg=s_mhg[burn_in:-1]
	s_mhg=s_mhg[0::lag]

	sg=sg[burn_in:-1]
	sg=sg[0::lag]

	sbg=sbg[burn_in:-1]
	sbg=sbg[0::lag]
	#
	sbg_str_rep=convert_node_to_string(sbg)
	sg_str_rep=convert_node_to_string(sg)
	s_mhg_str_rep=convert_node_to_string(s_mhg)


	height0 = [sbg_str_rep.count('0 0'), sbg_str_rep.count('0 1'), sbg_str_rep.count('0 2'), sbg_str_rep.count('1 0'),sbg_str_rep.count('1 1'),sbg_str_rep.count('1 2'), sbg_str_rep.count('2 0'), sbg_str_rep.count('2 1'), sbg_str_rep.count('2 2')]
	bars0 = ['(0,0)','(0,1)', '(0,2)', '(1,0)', '(1,1)', '(1,2)','(2,0)','(2,1)','(2,2)']
	y0_pos = np.arange(len(bars0))
	plt.bar(y0_pos, height0, color = 'r')
	title_str='Start Positions - Blocked Gibbs - last half of samples - True s1='+str(true_s1_str)
	plt.title(title_str)
	plt.xlabel('Start positions')
	plt.ylabel('Occurances')
	plt.xticks(y0_pos, bars0)
	plt.show()

	height0 = [sg_str_rep.count('0 0'), sg_str_rep.count('0 1'), sg_str_rep.count('0 2'), sg_str_rep.count('1 0'),sg_str_rep.count('1 1'),sg_str_rep.count('1 2'), sg_str_rep.count('2 0'), sg_str_rep.count('2 1'), sg_str_rep.count('2 2')]
	bars0 = ['(0,0)','(0,1)', '(0,2)', '(1,0)', '(1,1)', '(1,2)','(2,0)','(2,1)','(2,2)']
	y0_pos = np.arange(len(bars0))
	plt.bar(y0_pos, height0, color = 'r')
	title_str='Start Positions - Gibbs - last half of samples - True s1='+str(true_s1_str)
	plt.title(title_str)
	plt.xlabel('Start positions')
	plt.ylabel('Occurances')
	plt.xticks(y0_pos, bars0)
	plt.show()

	height0 = [s_mhg_str_rep.count('0 0'), s_mhg_str_rep.count('0 1'), s_mhg_str_rep.count('0 2'), s_mhg_str_rep.count('1 0'),s_mhg_str_rep.count('1 1'),s_mhg_str_rep.count('1 2'), s_mhg_str_rep.count('2 0'), s_mhg_str_rep.count('2 1'), s_mhg_str_rep.count('2 2')]
	bars0 = ['(0,0)','(0,1)', '(0,2)', '(1,0)', '(1,1)', '(1,2)','(2,0)','(2,1)','(2,2)']
	y0_pos = np.arange(len(bars0))
	plt.bar(y0_pos, height0, color = 'r')
	title_str='Start Positions - MH within Gibbs - last half of samples - True s1='+str(true_s1_str)
	plt.title(title_str)
	plt.xlabel('Start positions')
	plt.ylabel('Occurances')
	plt.xticks(y0_pos, bars0)
	plt.show()


def main():

	data_seed = 9 #A seed to generate data
	n_lattice = 3
	T = 100
	p = 0.1
	G, X_truth, s_truth, o = generate_data(data_seed, n_lattice, T, p)
	num_iter = 1000

	seeds_chains=[225]#,11,2222] # Different seeds used to run different chains
	for seed in seeds_chains:
		np.random.seed(seed)

		#Running the samplers
		sbg, X=block_gibbs(o, G, num_iter)
		smhg, X = mh_w_gibbs(o, G, num_iter, p)
		sg, X = gibbs(o, G, num_iter, p)

		s_list=[smhg,sg,sbg]
		s_list_check_convergence=[smhg[0:200],sg[0:200],sbg[0:200]]
		s_list_check_convergence_second_half=[smhg[500:-1],sg[500:-1],sbg[500:-1]]

		burn_in=100
		lag=5

		# Computing accuracy for the algorithms
		# for s_s in s_list:
		#      calc_acc(burn_in,lag,s_s,s_truth[0])

		# Creating histograms for the startpositions
		#convergence_histogram_plotter(burn_in,lag,s_list,s_truth[0])

		#convergence_histogram_plotter(burn_in,lag,s_list_check_convergence,s_truth[0])
		convergence_histogram_plotter(0,lag,s_list_check_convergence_second_half,s_truth[0])


if __name__ == '__main__':
	main()
