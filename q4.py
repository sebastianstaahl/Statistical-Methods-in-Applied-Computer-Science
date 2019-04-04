import numpy as np
import pdb
from scipy.stats import norm
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

class SVParams:
    def __init__(self, phi, sigma, beta):
        self.phi = phi
        self.sigma = sigma
        self.beta = beta


def generator(T, sv_params):
    x = np.zeros(T)
    y = np.zeros(T)
    x[0] = np.random.normal(0, sv_params.sigma)
    y[0] = np.random.normal(0, math.sqrt(np.power(sv_params.beta, 2) * math.sqrt(np.exp(x[0]))))

    for t in range(1, T):
        x[t] = np.random.normal(sv_params.phi * x[t-1], sv_params.sigma)
        y[t] = np.random.normal(0, math.sqrt(np.power(sv_params.beta, 2) * math.sqrt(np.exp(x[t]))))

    return x, y



def Q15(pre_comp=False):
    if pre_comp==False:
        T = 1000 #2000 originally
        num_particles = 100
        sv_truth=SVParams(1,0.16,0.64)
        x,y=generator(T, sv_truth)
        num_mcmc_iter=10000
        num_par_samplesize = [5,10,15,20,25,35,45,70,90,140,200]
        var_num_par = []
        var_ts=[]
        time_steps = [10,40,80,120,140,180,255,280,320,350,400]
        it=0

        samples1=[]

        for it in range(len(num_par_samplesize)):
            logLik_ts=[]
            logLik_np=[]
            for i in range(50):
                logLik_np.append(SMC(num_par_samplesize[it],y, sv_truth))
                obs =y[0:time_steps[it]]
                logLik_ts.append(SMC(num_particles, obs, sv_truth))
            var_np=np.var(logLik_np)
            var_num_par.append(var_np)

            var_t=np.var(logLik_ts)
            var_ts.append(var_t)
        npt=var_num_par
        tst=var_ts
    else:

        npt = [870.0895939161729, 227.80268917749112, 83.99338813288774, 40.821355668176096, 23.93967918514418, 20.135667806622937, 15.628814043413488, 9.604761762521012, 6.859806529446114, 4.382434063818218, 2.9859405498410814]
        tst = [0.00913427787309596, 0.08101709027811596, 0.22053044295982643, 0.43435550182201454, 0.5979187429794074, 0.7346394705101653, 1.3338263091728089, 1.2245095556960293, 2.2210955197300617, 1.2816747412055955, 1.3581309432131345]

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(num_par_samplesize, npt,color='g')
    axes[1].plot(time_steps, tst, color='g')
    axes[1].set_xlabel('Number of time steps')
    axes[0].set_xlabel('Number of particles')

    axes[0].set_ylabel('Variance')
    axes[1].set_ylabel('Variance')
    #ax.plot(ns, variances, 'o-')
    plt.show()


def Q14(obs,num_particles):
    """A function that creates a grid containg the log marginal likelihoods
    p,theta(y1:T) for different combinations of sigma and beta"""
    T=len(obs)
    sigmas=np.arange(0.25, 2, 0.25)
    betas=np.arange(0.25, 2, 0.25)
    parameter_grid_log=np.zeros((len(betas),len(sigmas)))
    parameter_grid=np.zeros((len(betas),len(sigmas)))
    best_theta=[0,0]
    max_like=-1000000

    for sigma in range(len(sigmas)):
        for beta in range(len(betas)):
            Lik_list=[]
            for i in range(10):
                sv_params = SVParams(1,sigmas[sigma],betas[beta])
                log_p_hat,x= SMC(num_particles,obs,sv_params)
                Lik_list.append(log_p_hat)
            parameter_grid[sigma,beta]=np.exp(np.mean(Lik_list))
            parameter_grid_log[sigma,beta]=log_p_hat
            if log_p_hat > max_like:
                max_like=log_p_hat
                best_theta[0]=sigmas[sigma]
                best_theta[1]=betas[beta]


    print('Marginal likelihoods ptheta(y1:T) for kombinations of beta and sigma in the individual rangea 0.25:2')
    print('Likelihoods')
    print(parameter_grid)

    print('Loged likelihoods')
    print(parameter_grid_log)

    print('Best hyperparameters')
    print('sigma :',best_theta[0],' ','beta :', best_theta[1])



def extract_one_param(params):
    sigmas=[]
    betas=[]
    for p in params:
        sigmas.append(p.sigma)
        betas.append(p.beta)
    return sigmas, betas


def gaussian_new(obs,x,params):
    x_vec=np.sqrt((params.beta**2)*np.exp(x))
    return norm.logpdf(obs, loc = 0, scale = x_vec)



def SMC(num_particles, obs,sv_params):

    T = len(obs)
    x = np.zeros([T, num_particles])
    a = np.zeros([T, num_particles])
    w = np.zeros([T, num_particles])
    w_norm = np.zeros([T, num_particles])
    logL=0
    for t in range(T):
        if t==0:
            x[0]= stats.norm.rvs(0, sv_params.sigma,num_particles)
            w[0]= stats.norm.logpdf(obs[t], loc=0,scale=sv_params.beta * np.sqrt(np.exp(x[0])))
            w_norm[0]=np.exp(w[0])/np.sum(np.exp(w[0]))
        else:
            inds_resamp = np.argmax(np.random.multinomial(1,w_norm[t-1],size=num_particles),axis=1)
            a[t]=inds_resamp
            x[t]=np.random.normal(sv_params.phi * x[t-1,inds_resamp], sv_params.sigma)

            w[t] = stats.norm.logpdf(obs[t], loc=0,scale=sv_params.beta * np.sqrt(np.exp(x[t]))) #+ w[t-1]

            w_max= np.max(w[t])
            w[t] = w[t] - w_max
            logL += w_max + np.log(np.sum(np.exp(w[t])))-np.log(num_particles)
            w_norm[t]=np.exp(w[t])/np.sum(np.exp(w[t]))

    b =  np.argmax(np.random.multinomial(1,w_norm[T-1,:]))
    x_v = np.zeros([T])
    for t in range(T-1,-1,-1):
        x_v=x[t,b.astype(int)]
        b=a[t,b.astype(int)]

    return logL, x_v




def pmmh(obs, num_particles=200, num_mcmc_iter=2000):
    acc=0
    rej=0
    T = len(obs)

    X = np.zeros([num_mcmc_iter, T])
    log_likelihoods=np.zeros([num_mcmc_iter])
    params = [] # list of SV_params
    # YOUR CODE

    #sv_init=SVParams(1,1,0.7)
    sv_init=SVParams(1,1,0.2)

    params.append(sv_init)

    logLinit, x_v=SMC(num_particles, obs, sv_init)

    log_likelihoods[0]=logLinit

    for it in range(1,num_mcmc_iter):


        old_params=[params[it-1].sigma,params[it-1].beta]
        id_mat=np.eye(2)
        id_mat[0][0]=0.03**2
        id_mat[1][1]=0.32**2


        #Make a new Gaussian Normal Random Walk proposal
        prop_params=stats.multivariate_normal.rvs( mean=old_params, cov=id_mat, size=1 )
        if (prop_params[0]>0 and prop_params[1]>0):

            sv_prop=SVParams(1,prop_params[0],prop_params[1])
            logLprop, x_v=SMC(num_particles, obs, sv_prop)

            #Using
            prior=stats.invgamma.logpdf(prop_params[0], 0.01, scale=0.01)
            prior+=stats.invgamma.logpdf(prop_params[1], 0.01, scale=0.01)

            prior-=stats.invgamma.logpdf(params[it-1].sigma, 0.01, scale=0.01)
            prior-=stats.invgamma.logpdf(params[it-1].beta, 0.01, scale=0.01)

            ##############


            log_diff=logLprop-log_likelihoods[it-1]
            tmp=np.exp(log_diff)

            tot_rhs=log_diff + prior
            tot_rhs_unlog=np.exp(tot_rhs)
            a=min(1, tot_rhs_unlog)
            u = stats.uniform.rvs()

            if u <= a:
                print('ACEPTED')
                acc+=1
                rej=0
                #print('accrate ',acc)
                #print('reject ',rej)
                params.append(SVParams(1,prop_params[0],prop_params[1]))
                log_likelihoods[it]=logLprop
                print(prop_params)
            else:
                #print('REJECTED!')
                rej+=1
                acc=0
                #print('accrate ',acc)
                #print('reject ',rej)
                params.append(params[-1])
                log_likelihoods[it]=log_likelihoods[it-1]

                #print(params[-1].sigma,' ',params[-1].beta)
        else:
            #print('less than zero..',prop_params[0],' ',prop_params[1])
            rej+=1
            #print('accrate ',acc/(acc+rej))
            params.append(params[-1])
            log_likelihoods[it]=log_likelihoods[it-1]

            #print(params[-1].sigma,' ',params[-1].beta)
        #input()
    return X, params

def main():
    seed=4812 #57832
    np.random.seed(seed)
    T = 100 #2000 originally
    num_particles = 100
    sv_truth=SVParams(1,0.16,0.64)
    x,y=generator(T, sv_truth)
    num_mcmc_iter=20000

    ##### Compute results for question 15 #####
    # print('Running code for Q15')
    #pre_computed_res=False
    #Q15(pre_computed_res)

    ### Compute results for question 14 #####
    # print('Running code for Q14')
    #Q14(y,num_particles)



    X, params_res = pmmh(y,100,num_mcmc_iter)



main()
