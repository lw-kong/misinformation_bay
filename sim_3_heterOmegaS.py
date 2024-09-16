# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 22:40:52 2024

@author: lkong


to plot accuracy vs omega_s, but with a heter distribution of omega_s

"""



import numpy as np
import matplotlib.pyplot as plt
import time
import winsound


from sim_funcs import get_flip_times, get_flip_lag_list, get_minority_error

# config
num_agents = 100  # number of agents
k = 8 # number of neighbours each agent has
T = 1500 # time of simulation
dt = 1.0  # time step
omega_g = 0.1
noise_sigma = 1.0  # sigma, amplitude of noise


std_omega_s = 0.2 # diversity in omega_s
para_set = np.arange(0.5, 0.60 + 0.01, 0.01) # range of omega s to sweep
repeat_num = 10 # number of repeating the simulation

env_half_period = 250

def G(t, half_period = env_half_period, high_value=1.0, low_value=-1.0):
    return np.where((t % (2 * half_period)) < half_period, high_value, low_value)

def drift(g_i, G_t):
    return -omega_g * (g_i - G_t)

def main_sim(avg_omega_s,std_omega_s):
    len_time_steps = int(T/dt)  # number of time steps
    g = np.zeros((num_agents, len_time_steps)) # personal estimates
    u = np.zeros((num_agents, len_time_steps)) # actions
    acc = np.zeros((num_agents, len_time_steps-1)) # accuracy 
    
    omega_s_set = np.random.normal(avg_omega_s, std_omega_s, num_agents)

    # simulate the Ornstein-Uhlenbeck process with the 2nd order Heun method    
    u[:,0] = np.random.choice([-1, 1], size = num_agents)
    # g[:,0]
    for t in range(1,len_time_steps):
        G_t = G(t * dt)  # current enviromental factor
        dW = np.random.normal(0, np.sqrt(dt), num_agents)  # Gaussian noise
    
        for i in range(num_agents):
            # Predictor step
            g_pred = g[i, t - 1] + drift(g[i, t - 1], G_t) * dt + noise_sigma * dW[i]
           
            # Corrector step
            G_t_next = G((t + 1) * dt)
            g[i, t] = g[i, t - 1] + 0.5 * (drift(g[i, t - 1], G_t) + drift(g_pred, G_t_next)) * dt + noise_sigma * dW[i]
            
            temp_e = np.exp(  4*omega_g * g[i, t] / (noise_sigma**2) )
            
            #sum_n_u = np.sum( u[:,t] ) - u[i,t] # fully connected    
            available_indices = np.delete(np.arange(num_agents), i)
            selected_indices = np.random.choice(available_indices, size=k, replace=False) # neighbours           
            
            omega_s = omega_s_set[i]
            
            # Bayesian update to have agent i's final decision at time step t
            sum_n_u = np.sum(u[selected_indices, t - 1])            
            u[i,t] = np.sign(-1 + temp_e* (omega_s/(1-omega_s))**sum_n_u)
            
            acc[i,t-1] = 1.0 - np.abs(G_t - u[i,t])/2.0
    
    average_u = np.average(u,axis=0)
    flips = get_flip_times(average_u)
    lag_times = get_flip_lag_list(average_u,flips,env_half_period)
    minority_error_list = get_minority_error(average_u,flips,env_half_period)
    
    return np.average(acc), np.average(lag_times), np.average(minority_error_list)

result_set = np.zeros([len(para_set),repeat_num,3])
tic = time.time()
for para_i in range(len(para_set)):
    avg_omega_s = para_set[para_i]
    
    for repeat_i in range(repeat_num):
        acc, lag, minor = main_sim(avg_omega_s,std_omega_s)
        
        result_set[para_i,repeat_i,0] = acc
        result_set[para_i,repeat_i,1] = lag
        result_set[para_i,repeat_i,2] = minor
    
    print(f"average omega s = {avg_omega_s}, average accuracy = {np.average(result_set[para_i,:])}")
    toc = time.time() - tic
    print(f"= run time {toc:.2f}s")      


plt.figure(figsize=(10, 6))
plt.plot(para_set, np.average(result_set[:,:,0],axis=1), 'o-')
plt.xlabel('average omega s')
plt.ylabel('accuracy')
plt.title(f"std omega s = {std_omega_s}")
plt.ylim(0.39, 0.94)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()

np.average(result_set[:,:,:],axis=1)

winsound.Beep(1000, 500)

result_std0 = np.array([[6.60909940e-01, 7.60000000e+00, 6.66921218e-01],
       [7.15414276e-01, 7.04000000e+00, 5.55522758e-01],
       [7.84108072e-01, 8.18000000e+00, 4.08847161e-01],
       [8.65981321e-01, 9.90000000e+00, 2.27367316e-01],
       [9.19848566e-01, 1.22600000e+01, 9.64521300e-02],
       [9.18699800e-01, 2.14600000e+01, 3.98659658e-02],
       [8.82110073e-01, 3.63400000e+01, 1.81574186e-02],
       [5.25136758e-01, 1.26313333e+02, 3.40537148e-02],
       [5.04725150e-01,            np.nan,            np.nan],
       [5.01196131e-01,            np.nan,            np.nan],
       [5.00320213e-01,            np.nan,            np.nan]])
# 5 min

result_std00125 = np.array([[6.59027352e-01, 1.09566667e+01, 6.73622220e-01],
       [7.08380254e-01, 7.30000000e+00, 5.69341487e-01],
       [7.74449633e-01, 7.90000000e+00, 4.30589070e-01],
       [8.48697131e-01, 9.46000000e+00, 2.65997809e-01],
       [9.00939293e-01, 1.24400000e+01, 1.37625365e-01],
       [9.19374249e-01, 1.80000000e+01, 6.16123738e-02],
       [8.95845897e-01, 2.98800000e+01, 2.71042987e-02],
       [6.15138092e-01, 1.18706667e+02, 3.21836643e-02],
       [5.21767178e-01,            np.nan,            np.nan],
       [5.02462975e-01,            np.nan,            np.nan],
       [5.00875917e-01,            np.nan,            np.nan]])


result_std0025 = np.array([[6.51491661e-01, 8.10000000e+00, 6.84924798e-01],
       [6.88891261e-01, 6.80000000e+00, 6.08820304e-01],
       [7.37734490e-01, 7.52000000e+00, 5.07499874e-01],
       [8.07953302e-01, 8.66000000e+00, 3.55243792e-01],
       [8.55721815e-01, 1.14800000e+01, 2.41510496e-01],
       [8.99542362e-01, 1.56800000e+01, 1.20422410e-01],
       [8.97398266e-01, 2.50000000e+01, 6.30779892e-02],
       [8.40290193e-01, 4.63200000e+01, 3.90361823e-02],
       [5.74565043e-01,            np.nan,            np.nan],
       [5.08983322e-01,            np.nan,            np.nan],
       [5.04476985e-01,            np.nan,            np.nan]])

result_std005 = np.array([[ 0.63379787, 11.54333333,  0.72295272],
       [ 0.66016611,  8.16      ,  0.66733518],
       [ 0.70156771,  8.1       ,  0.57738212],
       [ 0.72783856,  8.42      ,  0.52465267],
       [ 0.77821147,  9.88      ,  0.41352255],
       [ 0.82090327, 11.72      ,  0.31465129],
       [ 0.8411481 , 12.5       ,  0.26427725],
       [ 0.86869913, 22.14      ,  0.15088793],
       [ 0.85883055, 28.44      ,  0.13669785],
       [ 0.77540694, 60.38666667,  0.10160151],
       [ 0.60578986,         np.nan,         np.nan]])

result_std01 = np.array([[ 0.61386791, 26.12      ,  0.76820557],
       [ 0.62208272, 16.89333333,  0.74491675],
       [ 0.64927352, 12.21333333,  0.69030622],
       [ 0.65447899, 12.37      ,  0.6761908 ],
       [ 0.66925684, 12.10333333,  0.64563789],
       [ 0.71079186,  8.06      ,  0.56119565],
       [ 0.71648432,  8.7       ,  0.54633344],
       [ 0.76544163, 11.24      ,  0.4332824 ],
       [ 0.78494063, 15.02      ,  0.37377601],
       [ 0.79232422, 15.42      ,  0.35766519],
       [ 0.82412141, 22.38      ,  0.25402518]])


result_std02 = np.array([[ 0.58097732, 35.66666667,  0.82925894],
       [ 0.58678786, 28.11      ,  0.81924837],
       [ 0.58878319, 30.44333333,  0.8153354 ],
       [ 0.60568179, 19.82      ,  0.77884073],
       [ 0.61616144, 28.23666667,  0.75750574],
       [ 0.62183122, 23.07666667,  0.74500131],
       [ 0.63653169, 15.46333333,  0.71349978],
       [ 0.63181788, 13.12333333,  0.72445649],
       [ 0.64579987, 11.59333333,  0.6973072 ],
       [ 0.67584656, 10.12      ,  0.62855899],
       [ 0.70058105, 10.86      ,  0.57261088]])

plot_mkr_size = 10

plt.figure(figsize=(10, 6))
plt.plot(para_set, result_std0[:,0], 'o-', label='std 0.000', markersize=plot_mkr_size)
plt.plot(para_set, result_std00125[:,0], '^-', label='std 0.0125', markersize=plot_mkr_size)
plt.plot(para_set, result_std0025[:,0], '+-', label='std 0.025', markersize=plot_mkr_size)
plt.plot(para_set, result_std005[:,0], 'x-', label='std 0.05', markersize=plot_mkr_size)
plt.plot(para_set, result_std01[:,0], 's-', label='std 0.1', markersize=plot_mkr_size)
plt.plot(para_set, result_std02[:,0], 'v-', label='std 0.2', markersize=plot_mkr_size)
plt.xlabel('average omega s')
plt.ylabel('accuracy')
plt.xlim(0.498, 0.602)
plt.ylim(0.39, 0.94)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(para_set, result_std0[:,1], 'o-', label='std 0.000', markersize=plot_mkr_size)
plt.plot(para_set, result_std00125[:,1], '^-', label='std 0.0125', markersize=plot_mkr_size)
plt.plot(para_set, result_std0025[:,1], '+-', label='std 0.025', markersize=plot_mkr_size)
plt.plot(para_set, result_std005[:,1], 'x-', label='std 0.05', markersize=plot_mkr_size)
plt.plot(para_set, result_std01[:,1], 's-', label='std 0.1', markersize=plot_mkr_size)
plt.plot(para_set, result_std02[:,1], 'v-', label='std 0.2', markersize=plot_mkr_size)
plt.xlabel('average omega s')
plt.ylabel('average lag in flippings')
plt.xlim(0.498, 0.602)
#plt.ylim(0.39, 0.94)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(para_set, result_std0[:,2], 'o-', label='std 0.000', markersize=plot_mkr_size)
plt.plot(para_set, result_std00125[:,2], '^-', label='std 0.0125', markersize=plot_mkr_size)
plt.plot(para_set, result_std0025[:,2], '+-', label='std 0.025', markersize=plot_mkr_size)
plt.plot(para_set, result_std005[:,2], 'x-', label='std 0.05', markersize=plot_mkr_size)
plt.plot(para_set, result_std01[:,2], 's-', label='std 0.1', markersize=plot_mkr_size)
plt.plot(para_set, result_std02[:,2], 'v-', label='std 0.2', markersize=plot_mkr_size)
plt.xlabel('average omega s')
plt.ylabel('average minority error')
plt.xlim(0.498, 0.602)
#plt.ylim(0.39, 0.94)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(loc='upper right')
plt.show()
