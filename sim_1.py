# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 23:18:06 2024

@author: lkong

Torney 2014 model

"""



import numpy as np
import matplotlib.pyplot as plt
import time


# config
num_agents = 100  # number of agents
k = 8 # number of neighbours each agent has
T = 1500 # time of simulation
dt = 1.0  # time step
omega_g = 0.1
noise_sigma = 1.0  # sigma, amplitude of noise

omega_s = 0.57

def G(t, half_period=250, high_value=1.0, low_value=-1.0):
    return np.where((t % (2 * half_period)) < half_period, high_value, low_value)


len_time_steps = int(T/dt)  # number of time steps
g = np.zeros((num_agents, len_time_steps))  # personal estimates
time_steps = np.arange(len_time_steps) * dt
u = np.zeros((num_agents, len_time_steps))  # actions
acc = np.zeros((num_agents, len_time_steps-1))

# 定义漂移项f
def drift(g_i, G_t):
    return -omega_g * (g_i - G_t)

# simulate the Ornstein-Uhlenbeck process with the 2nd order Heun method
tic = time.time()
u[:,0] = np.random.choice([-1, 1], size=num_agents)
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
        selected_indices = np.random.choice(available_indices, size=k, replace=False)
        sum_n_u = np.sum(u[selected_indices, t - 1])
        
        u[i,t] = np.sign(-1 + temp_e* (omega_s/(1-omega_s))**sum_n_u)
        acc[i,t-1] = 1.0 - np.abs(G_t - u[i,t])/2.0

average_u = np.average(u,axis=0)
        
toc = time.time() - tic
print(f"= run time {toc:.2f}s")      


# plot the results
plt.figure(figsize=(10, 6))
for i in range(num_agents):
    plt.plot(time_steps, g[i, :], label=f'Agent {i+1}')
plt.plot(time_steps, G(time_steps), 'k--', label='Environment G(t)', linewidth=2)
plt.xlabel('time_steps')
plt.ylabel('Estimate g_i(t)')
plt.legend(loc='upper right')
plt.title('Ornstein-Uhlenbeck Process for Multiple Agents (Heun Method)')
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(time_steps, average_u, label='average U')
plt.plot(time_steps, G(time_steps), 'k--', label='Environment G(t)', linewidth=2)
plt.xlabel('time_steps')
plt.ylabel('Averag U')
plt.legend(loc='upper right')
plt.title(f"omega s = {omega_s}")
plt.show()