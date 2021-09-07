# # Training of models for AudioSet animal sound database 

# ### Code and models are inspired by https://github.com/qiuqiangkong/audioset_tagging_cnn and *[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019)*

# * #### *dataset* is a folder that contains the test (eval.h5), validation (balanced_train.h5) set, and all parts of the training set
# * #### *indices* is a folder that contains infromation on the classes, and the differents sets of data

pip install torchlibrosa


# Importing packages
import os
import sys
import numpy as np
import argparse
import time
import logging
import torch
import torch.nn as nn # Basic building block for graphs
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data
import logging
import h5py
import soundfile
import librosa
import pandas as pd
from scipy import stats 
import datetime
import pickle
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import csv
from sklearn import metrics    


from utilities import (create_folder, get_filename, create_logging, Mixup, 
    StatisticsContainer)
from models import (Cnn14, Cnn14_no_specaug, Cnn14_no_dropout, 
    Cnn6, Cnn10, ResNet22, ResNet38, ResNet54, Cnn14_emb512, Cnn14_emb128, 
    Cnn14_emb32, MobileNetV1, MobileNetV2, LeeNet11, LeeNet24, DaiNet19, 
    Res1dNet31, Res1dNet51, Wavegram_Cnn14, Wavegram_Logmel_Cnn14, 
    Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_8k, Cnn14_mel32, Cnn14_mel128, 
    Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt)
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup)
from data_generator import (AudioSetDataset, TrainSampler, BalancedTrainSampler, 
    AlternateTrainSampler, EvaluateSampler, collate_fn)
from evaluate import Evaluator
import config
from losses import get_loss_func




def train(workspace,sample_rate,window_size,hop_size,mel_bins,fmin,fmax,model_type,loss_type,balanced,augmentation,
      batch_size,learning_rate,early_stop,cuda,clip_samples,classes_num):
    """Train AudioSet tagging model. 

    Args:
      dataset_dir: str
      workspace: str
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    # Arguments & parameters
    num_workers = 0
    clip_samples = clip_samples
    classes_num = classes_num
    loss_func = get_loss_func(loss_type)
    resume_iteration = 0
    data_type='full_train'

    # Paths
    black_list_csv = None
    
    train_indexes_hdf5_path = r'/your_path/input/indices/train_100_idx.h5'

    eval_bal_indexes_hdf5_path = os.path.join(workspace,
                                              'balanced_train_idx.h5')

    eval_test_indexes_hdf5_path = os.path.join(workspace, 
        'eval_idx.h5')

    checkpoints_dir = os.path.join('./', 'checkpoints', 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    
    statistics_path = os.path.join('./', 'statistics', 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join('./', 'logs',  
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    
    if cuda==True:
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
     
    params_num = count_parameters(model)
     logging.info('Parameters num: {}'.format(params_num))
    
    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    dataset = AudioSetDataset(sample_rate=sample_rate)

    # Train sampler
    if balanced == 'none':
        Sampler = TrainSampler
    elif balanced == 'balanced':
        Sampler = BalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = AlternateTrainSampler
     
    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
        black_list_csv=black_list_csv)
    
    # Evaluate sampler
    eval_bal_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size)

    eval_test_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler = train_sampler , collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=False)
    
    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_bal_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=False)

    eval_test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=False)

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    # Evaluator
    evaluator = Evaluator(model=model)
        
    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    train_bgn_time = time.time()
    
    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join('./', 'checkpoints', 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    print('OK:', train_loader)
     
    time1 = time.time()
    
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        # Evaluate
        if (iteration % 2000 == 0 and iteration > resume_iteration) or (iteration == 0): # Evaluer chaque 2000 iterations
            train_fin_time = time.time()

            bal_statistics = evaluator.evaluate(eval_bal_loader)
            test_statistics = evaluator.evaluate(eval_test_loader)
                            
            logging.info('Validate bal mAP: {:.3f}'.format(
                np.mean(bal_statistics['average_precision'])))

            logging.info('Validate test mAP: {:.3f}'.format(
                np.mean(test_statistics['average_precision'])))

            statistics_container.append(iteration, bal_statistics, data_type='bal')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()
        
        # Save model
        if iteration % 15000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        # Mixup lambda
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Forward
        model.train()

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'], 
                batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'], 
                batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

        # Loss
        loss = loss_func(batch_output_dict, batch_target_dict)

        # Backward
        loss.backward()
        print(loss)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if iteration % 10 == 0:
            print('--- Iteration: {}, train time: {:.3f} s / 10 iterations ---'                .format(iteration, time.time() - time1))
            time1 = time.time()
        
        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1




workspace=r'/your_path/input/dataset' 
sample_rate=32000 # default=32000
window_size=1024 # default=1024
hop_size=320 # default=320
mel_bins=64 # default=64
fmin=50 # default=50
fmax=14000 # default=14000 
model_type = 'CNN14'  
loss_type='clip_bce'
balanced='balanced'  # default='balanced', choices=['none', 'balanced', 'alternate'])
augmentation='mixup'  # default='mixup', choices=['none', 'mixup'])
batch_size=32 # default=32
learning_rate=1e-3  # default=1e-3
early_stop=15000 # default=20000
cuda = True # default=False


train(workspace,sample_rate,window_size,hop_size,mel_bins,fmin,fmax,model_type,loss_type,balanced,augmentation,batch_size,learning_rate,early_stop,cuda,clip_samples,classes_num)

