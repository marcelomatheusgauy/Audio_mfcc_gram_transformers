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
from train_utils import run_epoch, LossCompute, NoamOpt,build_data_paths_with_index




def run_training_function(output_filename, model_save_path, num_patient_noise_files, num_control_noise_files, min_frequency, max_frequency):

    output_file = open(output_filename, 'w')

    #
    data_paths_train = []
    folder = 'SPIRA_Dataset_V2/'
    training_csv_file = 'SPIRA_Dataset_V2/metadata_train.csv'
    train_csv = pandas.read_csv(training_csv_file)

    for file_path in train_csv['file_path']:
        data_path = folder+file_path
        build_data_paths_with_index(data_paths_train, data_path)

    #shuffle data
    random.shuffle(data_paths_train)

    data_paths_valid = []
    validation_csv_file = folder + 'metadata_eval.csv'
    validation_csv = pandas.read_csv(validation_csv_file)

    for file_path in validation_csv['file_path']:
        data_path = folder+file_path
        build_data_paths_with_index(data_paths_valid, data_path)

    
    data_paths_test = []
    test_csv_file = folder + 'metadata_test.csv'
    test_csv = pandas.read_csv(test_csv_file)

    for file_path in test_csv['file_path']:
        data_path = folder+file_path
        build_data_paths_with_index(data_paths_test, data_path)


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
    pretrain = False
    d_model = 512
    model = make_transformer_encoder_model(V, out_coeffs=2, N=3, d_model=d_model, d_ff=2048)
    model_opt = NoamOpt(d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    avg_loss=0
    best_val_acc = 0
    #model_path = 'model_test_mfcc.ckpt'

    for epoch in range(20):
        model.train()
        loss, avg_loss, _ = run_epoch(model, 
              LossCompute(model.generator, model.softgenerator, model_opt, pretrain),
              data_paths_train, noise_file_paths, output_file, 
              num_patient_noise_files=num_patient_noise_files, 
              num_control_noise_files=num_control_noise_files, pretrain=pretrain, avg_loss=avg_loss,
              extract_coeffs='mfcc', min_frequency=min_frequency, max_frequency=max_frequency, 
              number_coeffs=128)
        model.eval()
        loss, _, val_acc = run_epoch(model, 
                    LossCompute(model.generator, model.softgenerator, None, pretrain=pretrain),
                    data_paths_valid, noise_file_paths, output_file, 
                    num_patient_noise_files=num_patient_noise_files, 
                    num_control_noise_files=num_control_noise_files, pretrain=pretrain, 
                    extract_coeffs='mfcc', min_frequency=min_frequency, max_frequency=max_frequency,
                    number_coeffs=128)
        print(val_acc, file=output_file)
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            print('Saving model', file=output_file)
            torch.save({
                'model_state_dict': model.state_dict()
                }, model_save_path)
        

    #run test
    #model_path = 'model_test_mfcc.ckpt'
    checkpoint = torch.load(model_save_path)
    V=128
    model = make_transformer_encoder_model(V, out_coeffs=2, N=3, d_model=512, d_ff=2048)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('End training. Starting test with noise added', file=output_file)
    print(run_epoch(model, LossCompute(model.generator, model.softgenerator, None, pretrain=False),
                    data_paths_test, noise_file_paths, output_file, 
                    num_patient_noise_files=num_patient_noise_files, 
                    num_control_noise_files=num_control_noise_files, pretrain=False, 
                    extract_coeffs='mfcc',  min_frequency=min_frequency, max_frequency=max_frequency, 
                    number_coeffs=128), file=output_file)

    print('End test with noise. Starting test without noise added', file=output_file)
    print(run_epoch(model, LossCompute(model.generator, model.softgenerator, None, pretrain=False),
                    data_paths_test, noise_file_paths, output_file, num_patient_noise_files=0, num_control_noise_files=0, pretrain=False, 
                    extract_coeffs='mfcc',  min_frequency=min_frequency, max_frequency=max_frequency,
                    number_coeffs=128), file=output_file)

