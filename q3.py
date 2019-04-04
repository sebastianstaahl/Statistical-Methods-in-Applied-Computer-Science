#FINAL
import numpy as np
import pdb
from scipy.stats import norm
import scipy.stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

class SVParams:
	def __init__(self, phi, sigma, beta):
		self.phi = phi
		self.sigma = sigma
		self.beta = beta

	def __repr__(self):
		params= 'sigma :', self.sigma, ' beta:',self.beta
		return str(params)


def generator(T, sv_params):
	x = np.zeros(T)
	y = np.zeros(T)
	x[0] = np.random.normal(0, sv_params.sigma)
	y[0] = np.random.normal(0, math.sqrt(np.power(sv_params.beta, 2) * np.exp(x[0])))

	for t in range(1, T):
		x[t] = np.random.normal(sv_params.phi * x[t-1], sv_params.sigma)
		y[t] = np.random.normal(0, math.sqrt(np.power(sv_params.beta, 2) * np.exp(x[t])))

	return x, y



def gaussian_pdf(x, obs, beta):
   return norm.pdf(obs, loc = 0, scale = math.sqrt(np.power(beta,2)*np.exp(x)))

def sis(obs, num_particles, sv_params):
	T = len(obs)
	x = np.zeros([T, num_particles])
	w = np.zeros([T, num_particles])
	for t in range(T):
		for n in range(num_particles):
			if t==0:

				x[t,n]=np.random.normal(0, sv_params.sigma)
				alpha= gaussian_pdf(x[t,n],obs[t],sv_params.beta)#np.random.normal(0,np.power(sv_params.beta,2)*np.exp(x[t,n]))
				#alpha=weight_update(x[t,n],obs[t],sv_params)
				w[t,n]=alpha #1/num_particles initially we do not have any other weights therefore w0 is equal to g
			else:
				x[t,n]=np.random.normal(sv_params.phi * x[t-1,n], sv_params.sigma)
				#alpha= gaussian_pdf(x[t,n],0,math.sqrt(np.power(sv_params.beta,2)*np.exp(x[t,n])))#np.random.normal(0, np.power(sv_params.beta, 2) * np.exp(x[t,n]))
				alpha= gaussian_pdf(x[t,n],obs[t],sv_params.beta)
				w[t,n]=w[t-1,n]*alpha

	return w, x


def smc_multinomial(obs, num_particles, sv_params):
	T = len(obs)
	x = np.zeros([T, num_particles])
	w = np.zeros([T, num_particles])
	#w_norm = np.zeros([T, num_particles])
	for t in range(T):
		for n in range(num_particles):
			if t==0:
				x[t,n]=np.random.normal(0, sv_params.sigma)
				#alpha= gaussian_pdf(x[t,n],0,np.power(sv_params.beta,2)*np.exp(x[t,n]))
				alpha= gaussian_pdf(x[t,n],obs[t],sv_params.beta)
				w[t,n]=alpha #1/num_particles initially we do not have any other weights therefore w0 is equal to g
			else:
				index_re_samp=np.argmax(np.random.multinomial(1,w[t-1,:])) #resampling particles based on normalized weighs. categorical since 1 particle at a time
				x[t,n]=np.random.normal(sv_params.phi * x[t-1,index_re_samp], sv_params.sigma)
				#alpha= gaussian_pdf(x[t,n],0,np.power(sv_params.beta,2)*np.exp(x[t,n]))#np.random.normal(0, np.power(sv_params.beta, 2) * np.exp(x[t,n]))
				alpha= gaussian_pdf(x[t,n],obs[t],sv_params.beta)
				w[t,n]=w[t-1,index_re_samp]*alpha
				#w[t,n]=w[t,n]/np.sum(w[t,:])
		w[t,:]=w[t,:]/np.sum(w[t,:])

	return w, x


def smc_stratified(obs, num_particles, sv_params):
	T = len(obs)
	x = np.zeros([T, num_particles])
	w = np.zeros([T, num_particles])


	u  = np.zeros(num_particles)
	u[0] = np.random.uniform(0,1/num_particles)

	for j in range(1,num_particles):
		u[j] = (u[j-1])+(1/num_particles)

	for t in range(T):
		for n in range(num_particles):
			if t==0:
				x[t,n]=np.random.normal(0, sv_params.sigma)
				alpha= gaussian_pdf(x[t,n],obs[t],sv_params.beta)
				w[t,n]=alpha
			else:
				x[t,n]=np.random.normal(sv_params.phi * x[t-1,n], sv_params.sigma)
				alpha= gaussian_pdf(x[t,n],obs[t],sv_params.beta)
				w[t,n]=w[t-1,n]*alpha

		w[t,:]=w[t,:]/np.sum(w[t,:])

		x_copy=np.copy(x)
		for parent in range(num_particles):
			if parent ==0:
				lower_bound = 0
			else:
				lower_bound = np.sum(w[t,:parent])
			upper_bound = np.sum(w[t,:parent+1])
			boolean_offsprings = [lower_bound <= uv <= upper_bound for uv in u ]

			indicies_offsprings=[i for i, k in enumerate(boolean_offsprings) if k]

			count=len(indicies_offsprings)
			parent_list=list()
			ind_list=list()

			if count > 0:
				for ind in indicies_offsprings:
					new_parent=x_copy[t,parent]
					x[t,ind]=new_parent

		w[t,:] = 1/num_particles

	return w, x


def compute_pointestimate_x_sis(xT,wT):
	w_normT=wT/np.sum(wT)
	weighted_xT=np.multiply(xT,w_normT) #here we use the normalized weights
	return np.sum(weighted_xT)


def compute_emperical_variance(wT):
	w_normT=wT/np.sum(wT)
	return np.var(w_normT)


def compute_mean_squared_error(sampler_function,params,T):
	num_particles_first=100
	for i in range(1,9):
		seed = 11111
		np.random.seed(seed)
		x_truth, y = generator(T, params) #q9 data
		num_particles=num_particles_first*i

		west, xest=sampler_function(y, num_particles, params)
		#print(west)
		#print()
		#print(west[-1])
		#input()
		#MSE_vec=[]
		wnorm=west[-1]/np.sum(west[-1])
		#print('norm weights ',wnorm[-1])
		#input()
		MSE=0
		#for t in range(len(x_truth)):
			#x_truth_vec=np.ones(len(xest[t]))*x_truth[t]
			#MSE=mean_squared_error(x_truth_vec, xest[t])
		#xT_point_estimate = compute_pointestimate_x_sis(xest[-1],west[-1])
		xT=xest[-1]
		#wT=west[-1]
		for p in range(num_particles):
			#print('diff ',x_truth[-1]-xT[p])
			#print('weight ',wnorm[p])
			#print('res: ',wnorm[p]*np.power((x_truth[-1]-xT[p]),2))
			MSE += wnorm[p]*np.power((x_truth[-1]-xT[p]),2)
			#print('MSE sum',MSE)
		print('MSE for num_particles=',num_particles, 'MSE ',MSE)
			#MSE_vec.append(MSE)
		#plt.figure()
		#plt.plot(MSE_vec, '-r', label='MSE num_particles=%d'+repr(num_particles))
		#plt.legend(loc='upper left')
		#plt.title(function.__name__)
	#plt.show()


def plot_weighted_latent_varibles(truth, x, weights, no_obs):


	x_vectors=[]

	for t in range(no_obs):
		wn = weights[t] / np.sum(weights[t])
		x_vectors.append(np.sum(x[t]*wn))

	plt.plot(x_vectors, '-r', label='mean for latent vars')
	plt.legend(loc='upper left')
	plt.plot(truth, label='ground truth')
	plt.legend(loc='upper left')
	plt.show()


def main():
	seed = 57832
	np.random.seed(seed)
	T = 1000
	#params = SVParams(0.6, 0.7, 0.9) #q9: params
	params =SVParams(1,0.16,0.64) #According to assignment description

	x, y = generator(T, params) #q9 data

	#print(len(x[0]))

	print('True beginning ',x[0:5])
	print('True ending ',x[-5:T])

	num_particles = 100
	# YOUR CODE
	#################SIS#######################
	print('Sequential importance sampling')
	west, xest=sis(y, num_particles, params)
	plot_weighted_latent_varibles(x, xest, west, T)
	xT_point_estimate = compute_pointestimate_x_sis(xest[-1],west[-1])
	print('Point estimate, last time step T, for SIS :',xT_point_estimate, 'x truth ',x[-1])
	var=compute_emperical_variance(west[-1])
	print('variance: ',var)
	compute_mean_squared_error(sis,params,T)

	print('###############################################################################')
	#
	# # # # print('Estimates')
	# # # print(compute_pointestimate_x_sis(xest[-1],west[-1]))
	# # # print('hej')
	# # #
	# # # print('Ending')
	# # # for i in range(5):
	# # # 	xT_point_estimate = compute_pointestimate_x_sis(xest[-i+1],west[-i+1])
	# # # 	print('point estimate ',xT_point_estimate)
	# # # print('Beginning')
	# # # for i in range(5):
	# # # 	xT_point_estimate = compute_pointestimate_x_sis(xest[i],west[i])
	# # # 	print('point estimate ',xT_point_estimate)
	# #
	# # # print('point estimate of xT: ',xT_point_estimate)
	# #
	# # # compute_mean_squared_error(x[-1],y,params, lambda: sis())
	# # # variance= compute_emperical_variance(west[-1])
	# # # print('Emperical variance normalized weights, T=100, num particles =100:', variance)
	# # ###################SIS###########################
	# #
	# #

	# # ##############  SIS with multinomial resampling ##########################
	#
	print()
	print('SIS with multinomial resampling')
	west_sir,xest_sir=smc_multinomial(y, num_particles, params)
	plot_weighted_latent_varibles(x, xest_sir, west_sir, T)
	xT_point_estimate = compute_pointestimate_x_sis(xest_sir[-1],west_sir[-1])
	print('Point estimate, last time step T, stratified  resampling :',xT_point_estimate, 'x truth ',x[-1])
	var=compute_emperical_variance(west_sir[-1])
	compute_mean_squared_error(smc_multinomial,params,T)
	print('variance: ',var)
	# print('###############################################################################')
	# #
	# #
	# # #
	# # # var=compute_emperical_variance(west_sir)
	# # # print('var ',var)
	# # #
	# # #
	# # # print('Ending')
	# # # for i in range(5):
	# # # 	xT_point_estimate = compute_pointestimate_x_sis(xest_sir[-(i+1)],west_sir[-(i+1)])
	# # # 	print('point estimate for index',-(i+1),'x val : ',xT_point_estimate)
	# # # print('Beginning')
	# # # for i in range(5):
	# # # 	xT_point_estimate = compute_pointestimate_x_sis(xest_sir[i],west_sir[i])
	# # # 	print('point estimate for index',(i),'x val : ',xT_point_estimate)
	# #

	# # ##############  SIS with multinomial resampling ##########################
	print()
	print('SIS -stratified ')
	w_strat, x_strat =smc_stratified(y, num_particles, params)
	plot_weighted_latent_varibles(x, x_strat, w_strat, T)

	xT_point_estimate = compute_pointestimate_x_sis(x_strat[-1],w_strat[-1])
	print('Point estimate, last time step T, stratified  resampling :',xT_point_estimate, 'x truth ',x[-1])
	var=compute_emperical_variance(w_strat[-2])
	print('variance: ',var)
	print('###############################################################################')

	# # # print('Ending')
	# # # for i in range(5):
	# # # 	xT_point_estimate = compute_pointestimate_x_sis(xest_sir[-(i+1)],west_sir[-(i+1)])
	# print('point estimate for index',-(i+1),'x val : ',xT_point_estimate)
	# print('Beginning')
	# for i in range(5):
	#  	xT_point_estimate = compute_pointestimate_x_sis(xest_sir[i],west_sir[i])
	#  	print('point estimate for index',(i),'x val : ',xT_point_estimate)



if __name__ == '__main__':
	main()
