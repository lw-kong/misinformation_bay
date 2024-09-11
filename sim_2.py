# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 00:33:59 2024

@author: lkong

Torney 2014 model

to plot Fig.1 (c)

"""


import numpy as np
import matplotlib.pyplot as plt
import time


# 定义参数
num_agents = 100  # number of agents
k = 8 # number of neighbours each agent has
T = 1500 # time of simulation
dt = 1.0  # time step
omega_g = 0.1
noise_sigma = 1.0  # sigma, amplitude of noise


para_set = np.arange(0.5, 0.65 + 0.01, 0.01) # range of omega s to sweep
repeat_num = 5 # number of repeating the simulation


def G(t, half_period=250, high_value=1.0, low_value=-1.0):
    return np.where((t % (2 * half_period)) < half_period, high_value, low_value)

def drift(g_i, G_t):
    return -omega_g * (g_i - G_t)

def main_sim(omega_s):
    len_time_steps = int(T/dt)  # number of time steps
    g = np.zeros((num_agents, len_time_steps)) # personal estimates
    u = np.zeros((num_agents, len_time_steps)) # actions
    acc = np.zeros((num_agents, len_time_steps-1)) # accuracy 

    # simulate the Ornstein-Uhlenbeck process with the 2nd order Heun method    
    u[:,0] = np.random.choice([-1, 1], size=num_agents)
    # g[:,0]
    for t in range(1,len_time_steps):
        G_t = G(t * dt)  # 当前环境参数
        dW = np.random.normal(0, np.sqrt(dt), num_agents)  # 高斯噪声项
    
        for i in range(num_agents):
            # Predictor step
            g_pred = g[i, t - 1] + drift(g[i, t - 1], G_t) * dt + noise_sigma * dW[i]
           
            # Corrector step
            G_t_next = G((t + 1) * dt)  # 计算下一步的环境参数
            g[i, t] = g[i, t - 1] + 0.5 * (drift(g[i, t - 1], G_t) + drift(g_pred, G_t_next)) * dt + noise_sigma * dW[i]
            
            temp_e = np.exp(  4*omega_g * g[i, t] / (noise_sigma**2) )
            
            #sum_n_u = np.sum( u[:,t] ) - u[i,t] # fully connected    
            available_indices = np.delete(np.arange(num_agents), i)
            selected_indices = np.random.choice(available_indices, size=k, replace=False)
            sum_n_u = np.sum(u[selected_indices, t - 1])
            
            u[i,t] = np.sign(-1 + temp_e* (omega_s/(1-omega_s))**sum_n_u)
            acc[i,t-1] = 1.0 - np.abs(G_t - u[i,t])/2.0
    
    return np.average(acc)

result_set = np.zeros([len(para_set),repeat_num])
tic = time.time()
for para_i in range(len(para_set)):
    omega_s = para_set[para_i]
    
    for repeat_i in range(repeat_num):
        result_set[para_i,repeat_i] = main_sim(omega_s)
    
    print(f"omega s = {omega_s}")
    toc = time.time() - tic
    print(f"= run time {toc:.2f}s")      


plt.figure(figsize=(10, 6))
plt.plot(para_set, np.average(result_set,axis=1), 'o-')
plt.xlabel('omega s')
plt.ylabel('accuracy')
plt.ylim(0.39, 0.94)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()

