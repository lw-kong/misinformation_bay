# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:02:40 2024

@author: lkong
"""

import numpy as np


def get_flip_times(average_u):
    # 参数设置
    min_stay_time = 5  # 至少停留的时间步数
    skip_init = 200
    
    # 初始化 flip 时间列表
    flips = []
    
    # 遍历时间序列，从第一个时间步到倒数第 5 个时间步（确保至少有 5 个时间步可以检查）
    for t in range(skip_init,len(average_u) - min_stay_time):
        # 当前值和下一个值的符号
        current_sign = np.sign(average_u[t])
        next_sign = np.sign(average_u[t + 1])
        
        # 检查是否越过了 0（即符号改变）
        if current_sign != next_sign and next_sign != 0:
            # 检查接下来 min_stay_time 时间步内是否保持在新的一侧
            if all(np.sign(average_u[t + 1:t + 1 + min_stay_time]) == next_sign):
                flips.append(t + 1)  # 记录发生 flip 的时间点
    return flips


# 示例：假设环境 G 的初始状态为正，从正到负 flip，然后每 half_period 时间步 flip 一次
def G_flip_direction(t, half_period):
    """
    根据时间 t 和 half_period 计算环境 G 的 flip 方向。
    返回 1 表示从负到正，-1 表示从正到负。
    """
    # 每 half_period 时间环境 G flip 一次，flip 方向每次相反
    return 1 if (t // half_period) % 2 == 0 else -1

# 计算 average_u 的 flip 方向
def average_u_flip_direction(average_u, flip_time):
    """
    根据时间序列 average_u 和 flip 发生的时间点 flip_time 计算 flip 方向。
    返回 1 表示从负到正，-1 表示从正到负。
    """
    if flip_time == 0 or flip_time >= len(average_u):
        return None  # 越界保护
    if average_u[flip_time + 2] > 0:
        bool_dir = 1
    elif average_u[flip_time + 2] < 0:
        bool_dir = -1
    else:
        return None
    return bool_dir

def get_flip_lag_list(average_u,flips,half_period):
    
    # 初始化一个列表来存储滞后时间
    lag_times = []
    
    # 记录已处理的 half_period
    processed_half_periods = set()
    
    # 遍历 flips 列表
    for flip_time in flips:
        # 找到当前 flip_time 所属的 half_period
        current_half_period = (flip_time // half_period) * half_period
        
        # 检查这个 half_period 是否已经处理过
        if current_half_period in processed_half_periods:
            continue  # 如果已经处理过，跳过
    
        # 计算环境 G 的 flip 方向
        g_flip_dir = G_flip_direction(flip_time, half_period)
        
        # 计算 average_u 的 flip 方向
        u_flip_dir = average_u_flip_direction(average_u, flip_time)
        
        # 只有在两个 flip 方向一致的情况下才计算滞后时间
        if g_flip_dir == u_flip_dir:
            # 找到最近的 G flip 时间点
            closest_g_flip_time = round(flip_time / half_period) * half_period
            
            # 计算滞后时间
            lag_time = flip_time - closest_g_flip_time
            
            # 如果滞后时间为负，调整为正数（越过最近的 G flip）
            if lag_time < 0:
                lag_time += half_period
            
            # 滞后时间不能超过 half_period
            lag_time = min(lag_time, half_period)
            
            # 存储结果
            lag_times.append(lag_time)
            
            # 标记该 half_period 已处理
            processed_half_periods.add(current_half_period)
    
    return lag_times

def get_minority_error(average_u,flips,half_period):

    
    # 初始化一个列表来存储每个 half_period 的平均差值
    average_diff_per_half_period = []
    
    # 记录已处理的 half_period
    processed_half_periods = set()
    
    # 遍历 flips 列表
    for flip_time in flips:
        # 找到当前 flip_time 所属的 half_period
        # 这里用的是 half_period 的开始时间点
        current_half_period = (flip_time // half_period) * half_period
        
        # 检查这个 half_period 是否已经处理过
        if current_half_period in processed_half_periods:
            continue  # 如果已经处理过，跳过
    
        # 计算环境 G 的 flip 方向
        g_flip_dir = G_flip_direction(flip_time, half_period)
        
        # 计算 average_u 的 flip 方向
        u_flip_dir = average_u_flip_direction(average_u, flip_time)
        
        # 只有在两个 flip 方向一致的情况下才继续计算
        if g_flip_dir == u_flip_dir:
            # 找到当前 half_period 的结束时间
            end_of_half_period = current_half_period + half_period
            
            # 确保索引不越界
            if end_of_half_period > len(average_u):
                end_of_half_period = len(average_u)
            
            # 计算从 flip_time 到 end_of_half_period 的 average_u 的平均值
            segment_mean = np.mean(average_u[flip_time:end_of_half_period])
            
            # 计算该 half_period 内的环境 G 的真实值（1 或 -1）
            true_G_value = g_flip_dir  # 1 或 -1
            
            # 计算差值的平均值
            mean_diff = np.abs(segment_mean - true_G_value)
            
            # 存储结果
            average_diff_per_half_period.append(mean_diff)
            
            # 标记该 half_period 已处理
            processed_half_periods.add(current_half_period)
    
    # 输出每个 half_period 的平均差值
    #for i, diff in enumerate(average_diff_per_half_period):
    #    print(f"第 {i+1} 个 half_period 的平均差值：{diff:.4f}")
    return average_diff_per_half_period