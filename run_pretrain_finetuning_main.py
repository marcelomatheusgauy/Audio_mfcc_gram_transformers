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
from run_pretraining import run_pretraining_function
from run_finetuning import run_finetuning_function

#parameters
num_patient_noise_files = 1
num_control_noise_files = 1
num_repetitions = 10
extract_coeffs = 'mfcc'
mask_proportion = 0.0
mask_frequency_proportion = 0.0
random_noise_proportion = 0.1
pretrain = 'ri'

#run pretraining function
output_filename = 'Experimental_results/exp_pretraining_' + str(num_patient_noise_files) + '_' + str(num_control_noise_files) + '.txt'
model_save_path = 'model_test_pretrain_noise_mfcc_' + str(num_patient_noise_files) + '_' + str(num_control_noise_files) + '.ckpt'
run_pretraining_function(output_filename, model_save_path, num_patient_noise_files, num_control_noise_files, extract_coeffs, mask_proportion, mask_frequency_proportion, random_noise_proportion)

#run finetuning function
model_load_path = model_save_path
for index in range(num_repetitions):
    output_filename = 'Experimental_results/exp_pretrained_' + str(num_patient_noise_files) + '_' + str(num_control_noise_files) + '_' + str(index) + '.txt'
    model_save_path = 'model_pretrained_mfcc_' + str(num_patient_noise_files) + '_' + str(num_control_noise_files) + '_' + str(index) + '.ckpt'
    run_finetuning_function(output_filename, model_load_path, model_save_path, num_patient_noise_files, num_control_noise_files, extract_coeffs, pretrain)


