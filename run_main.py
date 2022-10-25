import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math, copy, time
from torch.autograd import Variable
import random
import os
import pandas


#load training functions
from run_training import run_training_function

#parameters
num_patient_noise_files = 1
num_control_noise_files = 1
num_repetitions = 10
min_frequency = 0.0
max_frequency = None

for index in range(num_repetitions):
    output_filename = 'Experimental_results/exp_' + str(num_patient_noise_files) + '_' + str(num_control_noise_files) + '_' + str(index) + '.txt'
    model_save_path = 'model_test_mfcc_' + str(num_patient_noise_files) + '_' + str(num_control_noise_files) + '_' + str(index) + '.ckpt'
    run_training_function(output_filename, model_save_path, num_patient_noise_files, num_control_noise_files, min_frequency, max_frequency)


