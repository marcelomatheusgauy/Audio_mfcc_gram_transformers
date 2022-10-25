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


#load transformers and training functions
from transformer_encoder import make_transformer_encoder_model
from train_utils import run_epoch, LossCompute, NoamOpt, build_data_paths_with_index

def run_pretraining_function(output_filename, model_save_path, num_patient_noise_files, num_control_noise_files, extract_coeffs, mask_proportion, mask_frequency_proportion, random_noise_proportion):

    output_file = open(output_filename, 'w')
    #generate list of filenames for the audio
    folder = 'dados/coral/'
    data_paths = []
    for file in os.listdir(folder):
        data_path = folder+file
        build_data_paths_with_index(data_paths, data_path)
    
    folder = 'dados/sp/'
    #data_paths = []
    for file in os.listdir(folder):
        data_path = folder+file
        build_data_paths_with_index(data_paths, data_path)

    folder = 'dados/ALIP_Corpus/data/' 

    for file in os.listdir(folder):
        data_path = folder+file
        build_data_paths_with_index(data_paths, data_path)

    folder = 'dados/NURC_RE/'
    for folder1 in os.listdir(folder):
        folderjoin = os.path.join(folder, folder1)
        for folder2 in os.listdir(folderjoin):
            join_folder = os.path.join(folderjoin, folder2)
            for file in os.listdir(join_folder):
                if (file.find('wav') != -1):
                    data_path = os.path.join(join_folder, file)
                    build_data_paths_with_index(data_paths, data_path)

    
    #folder = 'dados/nurcsp/1_FASE/'
    #for file in os.listdir(folder):
    #    data_path = folder+file
    #    build_data_paths_with_index(data_paths, data_path)

    #folder = 'dados/nurcsp/2_FASE/'
    #for file in os.listdir(folder):
    #    data_path = folder+file
    #    build_data_paths_with_index(data_paths, data_path)

    #folder = 'dados/nurcsp/3_FASE/'
    #for file in os.listdir(folder):
    #    data_path = folder+file
    #    build_data_paths_with_index(data_paths, data_path)

    random.shuffle(data_paths)
    
    #
    noise_file_paths = []
    noise_folder = 'SPIRA_Dataset_V2/Ruidos-Hospitalares_V1/Ruidos-Hospitalares/Ruidos/Hospitalares-validados/'

    for file in os.listdir(noise_folder):
        if file.find(".wav") != -1:
            data_path = noise_folder+file
            noise_file_paths.append(data_path)

    ##############################################################################################

    #run training

    V = 128
    pretrain = True
    d_model = 512
    model = make_transformer_encoder_model(V, out_coeffs=2, N=3, d_model=d_model, d_ff=2048)
    model_opt = NoamOpt(d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    avg_loss=0
    best_val_acc = 0
    #model_path = 'model_test_pretrain_mfcc_noise.ckpt'

    for epoch in range(5):
        model.train()
        loss, avg_loss, _ = run_epoch(model, 
              LossCompute(model.generator, model.softgenerator, model_opt, pretrain),
              data_paths, noise_file_paths, output_file, 
              num_patient_noise_files=num_patient_noise_files, 
              num_control_noise_files=num_control_noise_files, pretrain=pretrain, avg_loss=avg_loss,  
              extract_coeffs=extract_coeffs, number_coeffs=128, mask_proportion=mask_proportion, 
              mask_frequency_proportion=mask_frequency_proportion, 
              random_noise_proportion=random_noise_proportion)

        print(avg_loss, file=output_file)

    print('Saving model', file=output_file)
    torch.save({
        'model_state_dict': model.state_dict()
    }, model_save_path)
    

