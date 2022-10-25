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

#extract data path and index from data path with index

def build_data_paths_with_index(data_paths, data_path):
    data_info = torchaudio.info(data_path)
    total_audio_length = data_info.num_frames
    sample_rate = data_info.sample_rate
    number_audio_files = 1
    audio_length = 4
    if total_audio_length > audio_length*sample_rate:
        number_audio_files = int(math.ceil((total_audio_length/sample_rate)-audio_length+1))
    for index in range(number_audio_files):
        data_path_with_index = data_path + '_' + str(index)
        data_paths.append(data_path_with_index)

def extract_index_from_path(data_path_with_index):
    if (data_path_with_index.find('wav') != -1):
        _position_in_path = data_path_with_index.find('wav')+3
    elif (data_path_with_index.find('mp3') != -1):
        _position_in_path = data_path_with_index.find('mp3')+3
    else:
        _position_in_path = data_path_with_index.find('ogg')+3
    data_path = data_path_with_index[:_position_in_path]
    index = int(data_path_with_index[_position_in_path+1:])
    return data_path, index

#build function to process data in batches
def process_batches(data_paths, noise_file_paths, num_patient_noise_files, num_control_noise_files, number_coeffs, extract_coeffs, min_frequency, max_frequency, mask_proportion, mask_consecutive_frames, mask_frequency_proportion, random_noise_proportion, batch_size, pretrain, path_index):
    
    #parameters below maybe should be defined elsewhere
    #set audio length in seconds - this is max length of audios
    audio_length = 4
    device = 'cuda'
    
    
    #noise parameters
    noise_max_amp = 0.19233719
    noise_min_amp = 0.033474047
    
    ################################
    
    data_batch = []
    
    #in case we are doing supervised training we also need to store whether the file comes from healthy/unhealthy - is always computed but only used for supervised training
    data_target_list = []
    while len(data_batch) < batch_size and path_index < len(data_paths):
        data_path_with_index = data_paths[path_index]
        data_path, index = extract_index_from_path(data_path_with_index)
        sample_rate = torchaudio.info(data_path).sample_rate

        data_elem, sample_rate = torchaudio.load(data_path, frame_offset=index*sample_rate, num_frames = audio_length*sample_rate)
        data_elem = data_elem[0]

        
        #if doing supervised training we add noise to the file
        if not pretrain:
            #add the noise the corresponding number of times
            if data_path.find("PTT") != -1:
                for noise_index in range(num_patient_noise_files):
                    #select random file
                    random_noise_index = random.randint(0, len(noise_file_paths)-1)
                    noise_path = noise_file_paths[random_noise_index]
                    #load audio file
                    noise_file = torchaudio.load(noise_path)[0]
                    audio_len = data_elem.size(0)
                    noise_len = noise_file[0].size(0)
                    noise_start = random.randint(0,noise_len-(audio_len+1))
                    noise_file = noise_file[0][noise_start:noise_start+audio_len]
                    #should I play with the amplitude of the noise file
                    max_amp = random.uniform(noise_min_amp, noise_max_amp)
                    reduct_factor = max_amp/float(noise_file.max().numpy())
                    noise_wav = noise_file*reduct_factor
                    data_elem = data_elem + noise_file
            else:
                for noise_index in range(num_control_noise_files):
                    #select random file
                    random_noise_index = random.randint(0, len(noise_file_paths)-1)
                    noise_path = noise_file_paths[random_noise_index]
                    #load audio file
                    noise_file = torchaudio.load(noise_path)[0]
                    audio_len = data_elem.size(0)
                    noise_len = noise_file[0].size(0)
                    noise_start = random.randint(0,noise_len-(audio_len+1))
                    noise_file = noise_file[0][noise_start:noise_start+audio_len]
                    #max_amp = random.uniform(noise_min_amp, noise_max_amp)
                    #reduct_factor = max_amp/float(noise_file.max().numpy())
                    #noise_wav = noise_file*reduct_factor
                    data_elem = data_elem + noise_file
        
        
        data_batch.append(data_elem)
        
        #for supervised training we store whether the file comes from patient/healthy group
        if data_path.find("PTT") != -1:#maybe use another method to determine whether it comes from the patient group
            data_target_list.append(1)
        else:
            data_target_list.append(0)
        #######################
        

        path_index+=1
        
    #convert list to torch tensor (pads different audio lengths to same size)
    data_batch = nn.utils.rnn.pad_sequence(data_batch, batch_first=True)
    
    data_batch = data_batch.to(device)
    
    #for supervised training
    data_target_list = torch.FloatTensor(data_target_list)
    data_target_list = data_target_list.to(device)
    ###########################
    
    if extract_coeffs == 'mfcc':#extract MFCC from data
        data_batch = torchaudio.transforms.MFCC(sample_rate, number_coeffs, melkwargs={"f_min":min_frequency, "f_max":max_frequency}).to(device)(data_batch)
    elif extract_coeffs == 'mel':#extract MelSpectrogram otherwise
        data_batch = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=number_coeffs, f_min=min_frequency, f_max=max_frequency).to(device)(data_batch)
    else:
        data_batch_1 = torchaudio.transforms.MFCC(sample_rate, number_coeffs, melkwargs={"f_min":min_frequency, "f_max":max_frequency}).to(device)(data_batch)
        data_batch_2 = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=number_coeffs, f_min=min_frequency, f_max=max_frequency).to(device)(data_batch)
        data_batch = torch.cat([data_batch_1, data_batch_2], dim = 1)

    #blocks size - we block multiple audio frames into one
    block_size = 1
    time_steps = data_batch.shape[2]
    time_blocks = int(math.floor(time_steps/block_size))
    data_batch = data_batch[:,:,:block_size*time_blocks]
    data_batch = torch.reshape(data_batch, (data_batch.shape[0], block_size*data_batch.shape[1], time_blocks))
    
    #permute so we have batch_size, time, n_coeffs
    data_batch = data_batch.permute(0,2,1)
    #print(data_batch.shape)

    #mask data
    data_batch_masked, mask_label = mask_audio_data(data_batch, mask_proportion, mask_consecutive_frames, mask_frequency_proportion, random_noise_proportion)
    
    return data_batch_masked, mask_label, data_batch, data_target_list, path_index

#this function is used to mask along multiple consecutive frames - see https://github.com/s3prl/s3prl/blob/master/pretrain/mockingjay/task.py
def _starts_to_intervals(starts, consecutive):
    tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
    offset = torch.arange(consecutive).expand_as(tiled)
    intervals = tiled + offset
    return intervals.view(-1)

#function to mask spectograms/mfcc. -> it should not make a difference whether we are trying to mask mfcc/spectograms
def mask_audio_data(data_batch, mask_proportion, mask_consecutive_frames, mask_frequency_proportion, random_noise_proportion):
    #check https://github.com/s3prl/s3prl/blob/master/pretrain/mockingjay/task.py
    
    #extract sizes
    batch_size = data_batch.shape[0]
    time_len = data_batch.shape[1]
    n_coeffs = data_batch.shape[2]

    #store masked values
    data_batch_masked = copy.deepcopy(data_batch)
    
    #do I need to keep a label on which elements I masked?
    mask_label = torch.zeros_like(data_batch, dtype=torch.uint8)
    
    for idx in range(batch_size):

        #mask along time
        #select starts to mask
        proportion = round(time_len*mask_proportion/mask_consecutive_frames)
        valid_start_max = max(time_len - mask_consecutive_frames - 1, 0)
        chosen_starts = torch.randperm(valid_start_max + 1)[:proportion]

        #select intervals where we mask
        chosen_intervals = _starts_to_intervals(chosen_starts, mask_consecutive_frames)

        # determine whether to mask / random / or do nothing to the frame
        dice = random.random()
        # mask to zero
        if dice < 0.8:
            data_batch_masked[idx, chosen_intervals, :] = 0
        # replace to random frames
        elif dice >= 0.8 and dice < 0.9:
            random_starts = torch.randperm(valid_start_max + 1)[:proportion]
            random_intervals = _starts_to_intervals(random_starts, mask_consecutive_frames)
            data_batch_masked[idx, chosen_intervals, :] = data_batch_masked[idx, random_intervals, :]
        # do nothing
        else:
            pass
        
        #store label of masked indices
        mask_label[idx, chosen_intervals, :] = 1
        
        #mask along frequency/quefrency axis
        if mask_frequency_proportion>0.:
            rand_bandwidth = random.randint(0, int(round(n_coeffs*mask_frequency_proportion)))
            chosen_starts = torch.randperm(n_coeffs-rand_bandwidth)[:1]
            chosen_intervals = _starts_to_intervals(chosen_starts, rand_bandwidth)
            data_batch_masked[idx, :, chosen_intervals] = 0
            
            #store label of masked indices
            mask_label[idx, :, chosen_intervals] = 1
        
    # noise augmentation
    if random_noise_proportion > 0.:
        dice = random.random()
        if dice < random_noise_proportion:
            noise_sampler = torch.distributions.Normal(0, 0.2)
            data_batch_masked += noise_sampler.sample(data_batch_masked.shape).to(device=data_batch_masked.device)
        
    return data_batch_masked, mask_label


#function to train model
def run_epoch(model, loss_compute, data_paths, noise_file_paths, output_file, num_patient_noise_files=1, num_control_noise_files=1, avg_loss=0, pretrain=True, batch_size=16, extract_coeffs='both', min_frequency=0.0, max_frequency=None, number_coeffs=128, mask_proportion=0., mask_consecutive_frames=7, mask_frequency_proportion=0., random_noise_proportion=0.0):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    train_acc_avg = 0
    
    number_elements = len(data_paths)
    #number_steps = int(math.ceil(number_elements/batch_size))
    
    #path index is the index of the audio file in the filenames list
    path_index = 0
    #step index stores the amount of steps taken by the algorithm so far
    step_index = 0
    while path_index < number_elements:
        step_index +=1
        #load the data and mask it
        data_batch_masked, mask_label, data_batch, data_target_list, path_index = process_batches(data_paths, noise_file_paths, num_patient_noise_files, num_control_noise_files, number_coeffs, extract_coeffs, min_frequency, max_frequency, mask_proportion, mask_consecutive_frames, mask_frequency_proportion, random_noise_proportion, batch_size, pretrain, path_index)
        b_size = data_batch_masked.shape[0]

        src_mask = (data_batch[:,:,0] != 0).unsqueeze(1)
        #pass data through transformer
        out = model.forward(data_batch_masked, src_mask)
        #compute loss
        #print('out', out.shape)
        #print('data_batch', data_batch.shape)
        if pretrain:
            #print('data_batch')
            loss, train_acc = loss_compute(out, data_batch, mask_label)
        else:
            #print('data_target_list')
            loss, train_acc = loss_compute(out, data_target_list, mask_label)
        
        total_loss += loss
        avg_loss = avg_loss*0.99 + loss*0.01
        train_acc_avg = (train_acc_avg*(step_index-1)+train_acc)/(step_index)
        total_tokens += b_size
        tokens += b_size
        
        #if path_index > 10:
        #    break
        
        if step_index % 5 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f Train_acc: %f" %
                    (step_index, avg_loss, tokens / elapsed, train_acc_avg), file=output_file)
            start = time.time()
            tokens = 0
    return total_loss / (total_tokens), avg_loss, train_acc_avg


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, softgenerator, opt=None, pretrain=True):
        self.generator = generator
        self.softgenerator = softgenerator
        self.opt = opt
        self.pretrain = pretrain
        
    def __call__(self, x, y, mask_label):
        train_acc = 0
        if self.pretrain:
            x = self.generator(x)
            L1_loss = nn.L1Loss()
            #print('x', x.shape)
            #print('mask_label', mask_label.shape)
            #loss = L1_loss(x*mask_label, y*mask_label)
            loss = L1_loss(x, y)
        else:
            x = self.softgenerator(x)
            cross_entropy_loss = nn.BCELoss()
            #check the indices for cross entropy loss
            #print('x', x.shape)
            y = y.unsqueeze(1).unsqueeze(1)
            #print('y', y.shape)
            loss = cross_entropy_loss(x[:,0:1,1:2], y)
            
            #compute accuracy
            _, predicted = torch.max(x[:, 0:1], 2)
            #print('data_target_list', y[:,0,0])
            #print('predicted', predicted[:,0])
            train_acc = torch.sum(predicted == y[:, 0])/y.shape[0]
            
            
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        #print(loss.data.shape)
        return loss.data.item(), train_acc
    

