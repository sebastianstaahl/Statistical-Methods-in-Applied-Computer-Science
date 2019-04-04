import numpy as np
import math
import matplotlib.pyplot as plt

alphabet=[1,2,3,4]

def generator(seed, N, M, K, W, alpha_bg, alpha_mw):
	# Data generator.
	# Input: seed: int, N: int, M: int, K: int, W: int, alpha_bg: numpy array with shape(K), alpha_mw: numpy array with shape(K)
	# Output: D: numpy array with shape (N,M), R_truth: numpy array with shape(N), theta_bg: numpy array with shape (K), theta_mw: numpy array with shape (W,K)

	np.random.seed(seed)        # Set the seed

	D = np.zeros((N,M))         # Sequence matrix of size NxM
	R_truth = np.zeros(N)       # Start position of magic word of each sequence

	theta_bg = np.zeros(K)      # Categorical distribution parameter of background distribution
	theta_mw = np.zeros((W,K))  # Categorical distribution parameter of magic word distribution

	#Generating the Data


	# YOUR CODE:
	# Generate D, R_truth, theta_bg, theta_mw. Please use the specified data types and dimensions.

	R_truth = np.random.randint(0,M-W+1,N) #between 0 and 6 for our values

	theta_bg= np.random.dirichlet(alpha_bg)
	theta_mw=np.random.dirichlet(alpha_mw,W)
	#print(theta_mw)
	#D
	for n in range(N):
		#print('N ', N)
		it=0
		for m in range(M):
			if m in range(R_truth[n],R_truth[n]+W):
				#print(m)
				D[n,m]=alphabet[np.argmax(np.random.multinomial(1,theta_mw[it]))]
				it+=1
			else:
				D[n,m]=alphabet[np.argmax(np.random.multinomial(1,theta_bg))]

	return D, R_truth, theta_bg, theta_mw

def cond_prob_mw(alpha_mw,N_vec_j,N): #where j is the current position in the word
	num1=math.gamma(np.sum(alpha_mw))
	denom1=math.gamma(N+np.sum(alpha_mw))
	ratio1=math.log(num1)-math.log(denom1) #division becomes subtraction with log
	#print('alpha ',alpha_mw)
	#print()
	prod_sums=alpha_mw+N_vec_j
	#print('sum ',prod_sums)
	#input()

	for k in range(len(alpha_mw)):
		prod_sums[k]=math.log(math.gamma(prod_sums[k]))-math.log(math.gamma(alpha_mw[k])) #division becomes subtraction with log
		#print('prod ',prod_sums[k])
		#input()

	ratio2=np.sum(prod_sums) #prod becomes sum with log
	#print(ratio2)
	#input('svar ovan')
	prob=ratio1+ratio2


	return prob

def cond_prob_bg(alpha_bg,B_vec,B):
	num1=math.gamma(np.sum(alpha_bg))
	denom1=math.gamma(B+np.sum(alpha_bg))
	ratio1=math.log(num1)-math.log(denom1) #division becomes subtraction with log

	prod_sums=alpha_bg+B_vec

	for k in range(len(alpha_bg)):
		prod_sums[k]=math.log(math.gamma(prod_sums[k]))-math.log(math.gamma(alpha_bg[k])) #division becomes subtraction with log

	ratio2=np.sum(prod_sums) #prod becomes sum with log
	prob=ratio1+ratio2


	return prob

def posterior_full_cond(D, alpha_bg, alpha_mw, W,R_prev,seq,N,M,K,B,N_mw):  #rename remove some input varibles

	#Drilling through different values of rn while R_n and D is Set

	pos_range_mw=M-W+1
	probs_for_start_pos=[]
	for r_val in range(pos_range_mw):
		R_curr =R_prev# going through all possible start positions for the current sequence
		R_curr[seq]=r_val

		N_vec= np.zeros((W,K))
		B_vec=np.zeros(K)

		cond_probs_mw=[]
		#print('seq ',seq)
		#print("rval ", r_val)
		#print('D ',D)
		for row in range(D.shape[0]):
			index_mw=0
			for col in range(D.shape[1]):
				if col in range(int(R_curr[row]),int(R_curr[row])+W):
					k=int(D[row,col])-1
					N_vec[index_mw,k]=N_vec[index_mw,k]+1
					index_mw+=1
				else:
					k=int(D[row,col])-1
					B_vec[k]=B_vec[k]+1
		#print()
		#print('N_vec ',N_vec)
		#input()


		#Get the log likelihoods
		for j in range(W): #where j is a certain column in the magic words
			prob_pos_j_mw=cond_prob_mw(alpha_mw,N_vec[j],N_mw)
			cond_probs_mw.append(prob_pos_j_mw)

		cond_prob_bg_val=cond_prob_bg(alpha_bg,B_vec,B)

		res_prob=cond_prob_bg_val+sum(cond_probs_mw)
		probs_for_start_pos.append(res_prob)

	return probs_for_start_pos

def gibbs(D, alpha_bg, alpha_mw, num_iter):
	# Gibbs sampler.
	# Input: D: numpy array with shape (N,M),  alpha_bg: numpy array with shape(K), alpha_mw: numpy array with shape(K), num_iter: int
	# Output: R: numpy array with shape(num_iter, N)

	N = D.shape[0]
	R = np.zeros(( num_iter,N)) # Store samples for start positions of magic word of each sequence
	#print('R', R.shape)
	# YOUR CODE:
	# Implement gibbs sampler for start positions.
	W = 5
	M = D.shape[1]
	K=len(alphabet)
	#p.162 murphy, eqv 5.26

	#first we randomly generate start values for the r:s from an arbitary distribution
	R_start = np.random.randint(0,M-W+1,N)
	R_start=(R_start)
	R=np.insert(R, 0, R_start, 0)

	print('R_start ',R_start)
	#now we wanna compute N and B
	B=N*(M-W)
	N_mw=N*W

	probs_res=[]
	for it in range(num_iter):
		R_curr=np.copy(R[it])
		prob_it=[]
		for seq in range(N):
			prob_pos=posterior_full_cond(D, alpha_bg, alpha_mw, W,R_curr,seq,N,M,K,B,N_mw) #latest sample R[it] check that ok
			#print(prob_pos)
			#input()
			prob_pos=np.exp(prob_pos-np.max(prob_pos))
			prob_pos=prob_pos/np.sum(prob_pos) #normalize to get categorical

			#print(prob_pos)
			#input()
			index_rand=np.random.multinomial(1,prob_pos)
			prob_it.append(np.max(prob_pos))
			#print(prob_it[0])
			#input()
			#print(temp)
			r_val_new = np.argmax(index_rand) #categorical sampling


			R_curr[seq]=r_val_new
		R[it+1]=R_curr

		probs_res.append(prob_it)
		# if it>100:
		#     print(R[it+1])
		#     input()

	return R,probs_res

def main():
	seed = 123

	N = 20
	#N=5
	M = 10
	K = 4
	W = 5
	alpha_bg =[7,13,1,5]#[12,7,3,1]
	#alpha_mw = np.ones(K) * 0.9
	alpha_mw= np.ones(K)# [12,7,3,1]

	num_iter =1000

	print("Parameters: ", seed, N, M, K, W, num_iter)
	print(alpha_bg)
	print(alpha_mw)

	# Generate synthetic data.
	D, R_truth, theta_bg, theta_mw = generator(seed, N, M, K, W, alpha_bg, alpha_mw)
	print("\nSequences: ")
	print(D)
	print("\nStart positions (truth): ")
	print(R_truth)

	seed = 321
	# Use D, alpha_bg and alpha_mw to infer the start positions of magic words.
	R,probs_res = gibbs(D, alpha_bg, alpha_mw, num_iter)

	r0=R[0,:]


	#input()
	print("\nStart positions (sampled): ")
	print(R[0,:])
	print(R[1,:])
	print(R[-1,:])

	for i in range(20):
		print(R_truth[i])
		print(R[-1,i])

	# YOUR CODE:
	# Analyze the results. Check for the convergence.

	norm=num_iter/N

	plot_r_0=[]
	plot_r_1=[]
	plot_r_2=[]
	plot_r_3=[]

	plot_r0_val=[]
	plot_r1_val=[]
	plot_r2_val=[]
	plot_r3_val=[]
	plot_r4_val=[]
	plot_r5_val=[]
	plot_r6_val=[]
	plot_r7_val=[]
	plot_r8_val=[]
	plot_r9_val=[]

	for it in range(len(R[:,0])):
		plot_r0_val.append(R[it,0])
		plot_r1_val.append(R[it,1])
		plot_r2_val.append(R[it,2])
		plot_r3_val.append(R[it,3])
		plot_r4_val.append(R[it,4])
		plot_r5_val.append(R[it,5])
		plot_r6_val.append(R[it,6])
		plot_r7_val.append(R[it,7])
		plot_r8_val.append(R[it,8])
		plot_r9_val.append(R[it,9])



	r0_accuracy = plot_r0_val.count(R_truth[0])/len(plot_r0_val)
	print('Accuracy r0: ',r0_accuracy)
	burn_in=100
	plot_r0_val=plot_r0_val[burn_in:-1]
	plot_r1_val=plot_r1_val[burn_in:-1]
	plot_r2_val=plot_r2_val[burn_in:-1]
	plot_r3_val=plot_r3_val[burn_in:-1]
	plot_r4_val=plot_r4_val[burn_in:-1]
	plot_r5_val=plot_r5_val[burn_in:-1]
	plot_r6_val=plot_r6_val[burn_in:-1]
	plot_r7_val=plot_r7_val[burn_in:-1]
	plot_r8_val=plot_r8_val[burn_in:-1]
	plot_r9_val=plot_r9_val[burn_in:-1]

	lag=2
	plot_r0_val=plot_r0_val[0::lag]
	plot_r1_val=plot_r1_val[0::lag]
	plot_r2_val=plot_r2_val[0::lag]
	plot_r3_val=plot_r3_val[0::lag]
	plot_r4_val=plot_r4_val[0::lag]
	plot_r5_val=plot_r5_val[0::lag]
	plot_r6_val=plot_r6_val[0::lag]
	plot_r7_val=plot_r7_val[0::lag]
	plot_r8_val=plot_r8_val[0::lag]
	plot_r9_val=plot_r9_val[0::lag]

################################################################################
	height0 = [plot_r0_val.count(0), plot_r0_val.count(1), plot_r0_val.count(2), plot_r0_val.count(3), plot_r0_val.count(4), plot_r0_val.count(5)]
	bars0 = ['0','1', '2', '3', '4', '5']
	y0_pos = np.arange(len(bars0))
	plt.bar(y0_pos, height0, color = 'r')
	title_str='True r0 ='+str(R_truth[0])
	plt.title(title_str)
	plt.xlabel('Start positions')
	plt.ylabel('Occurances')
	plt.xticks(y0_pos, bars0)
	plt.show()
	r0_accuracy = plot_r0_val.count(R_truth[0])/len(plot_r0_val)
	print('Accuracy r0: ',r0_accuracy)
################################################################################
	height1 = [plot_r1_val.count(0), plot_r1_val.count(1), plot_r1_val.count(2), plot_r1_val.count(3), plot_r1_val.count(4), plot_r1_val.count(5)]
	bars1 = ['0','1', '2', '3', '4', '5']
	y1_pos = np.arange(len(bars1))
	plt.bar(y1_pos, height1, color = 'r')
	title_str='True r1 ='+str(R_truth[1])
	plt.title(title_str)
	plt.xlabel('Start positions')
	plt.ylabel('Occurances')
	plt.xticks(y1_pos, bars1)
	plt.show()
	r1_accuracy = plot_r1_val.count(R_truth[1])/len(plot_r1_val)
	print('Accuracy r1: ',r1_accuracy)
################################################################################
	height2 = [plot_r2_val.count(0), plot_r2_val.count(1), plot_r2_val.count(2), plot_r2_val.count(3), plot_r2_val.count(4), plot_r2_val.count(5)]
	bars2 = ['0','1', '2', '3', '4', '5']
	y2_pos = np.arange(len(bars2))
	plt.bar(y2_pos, height1, color = 'r')
	title_str='True r2 ='+str(R_truth[2])
	plt.title(title_str)
	plt.xlabel('Start positions')
	plt.ylabel('Occurances')
	plt.xticks(y2_pos, bars2)
	plt.show()
	r2_accuracy = plot_r2_val.count(R_truth[2])/len(plot_r2_val)
	print('Accuracy r2: ',r2_accuracy)
################################################################################
	height2 = [plot_r3_val.count(0), plot_r3_val.count(1), plot_r3_val.count(2), plot_r3_val.count(3), plot_r3_val.count(4), plot_r3_val.count(5)]
	bars2 = ['0','1', '2', '3', '4', '5']
	y3_pos = np.arange(len(bars2))
	plt.bar(y3_pos, height1, color = 'r')
	title_str='True r3 ='+str(R_truth[3])
	plt.title(title_str)
	plt.xlabel('Start positions')
	plt.ylabel('Occurances')
	plt.xticks(y2_pos, bars2)
	plt.show()

	r3_accuracy = plot_r3_val.count(R_truth[3])/len(plot_r3_val)
	print('Accuracy r3: ',r3_accuracy)

	r4_accuracy = plot_r4_val.count(R_truth[4])/len(plot_r4_val)
	print('Accuracy r4: ',r4_accuracy)

	r5_accuracy = plot_r5_val.count(R_truth[5])/len(plot_r5_val)
	print('Accuracy r5: ',r5_accuracy)

	r6_accuracy = plot_r6_val.count(R_truth[6])/len(plot_r6_val)
	print('Accuracy r6: ',r6_accuracy)

	r7_accuracy = plot_r7_val.count(R_truth[7])/len(plot_r7_val)
	print('Accuracy r7: ',r7_accuracy)


	r8_accuracy = plot_r8_val.count(R_truth[8])/len(plot_r8_val)
	print('Accuracy r8: ',r8_accuracy)

	r9_accuracy = plot_r9_val.count(R_truth[9])/len(plot_r9_val)
	print('Accuracy r9: ',r9_accuracy)

	# r plot over sequence
	# plt.plot(plot_r0_val)
	# text0='value r0, r0_truth:'+str(R_truth[0])
	# plt.title(text0)
	# plt.show()
	#
	# plt.plot(plot_r1_val)
	# text1='value r1, r1_truth:'+str(R_truth[1])
	# plt.title(text1)
	# plt.show()
	#
	# plt.plot(plot_r2_val)
	# text2='value r2, r2_truth:'+str(R_truth[2])
	# plt.title(text2)
	# plt.show()
	#
	# plt.plot(plot_r3_val)
	# text3='value r3, r3_truth:'+str(R_truth[3])
	# plt.title(text3)
	# plt.show()
	#
	#
	# #probability plots
	# plt.plot(plot_r_0)
	# #title('prob max(p(rn|R))')
	# plt.show()
	#
	# plt.plot(plot_r_1)
	# plt.show()
	#
	# plt.plot(plot_r_2)
	# plt.show()
	#
	# plt.plot(plot_r_3)
	# plt.show()
	#
	# # plt.plot(plot_r_2)
	# # plt.show()
	# #
	# # plt.plot(plot_r_3)
	# # plt.show()


if __name__ == '__main__':
	main()
