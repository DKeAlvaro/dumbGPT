import os
import random
import torch
import torch.nn.functional as F
import sys
import torch

def read_observations(filename):
    numbers = []

    with open(filename, 'r') as file:
        for line in file:
            number = float(line.strip())
            numbers.append(number)
    # The fact that we return the reversed list makes sense but cant explain now 
    # So it wont make sense no
    return numbers[::-1]  # Return reversed list
    
def transform_outliers(numbers, upper_bound):
    return [num if num <= upper_bound else upper_bound for num in numbers]

def save_to_lists(directory = "inputs", upper_bound = 10):
    full_obs = []
    for i, file in enumerate(os.listdir(directory)):
        if file.endswith(".txt"):
            curr_obs = read_observations(f'{directory}/obs_{i+1}.txt')
            curr_filtered_obs = transform_outliers(curr_obs, upper_bound)
            full_obs.append(curr_filtered_obs)
    return full_obs

def divide_lists(full_obs, train_size):
    random.shuffle(full_obs)  # Shuffle the list in place
    train_obs = full_obs[:train_size]
    test_obs = full_obs[train_size:]
    return train_obs, test_obs
