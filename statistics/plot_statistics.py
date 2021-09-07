import os
import sys
import numpy as np
import argparse
import h5py
import time
import pickle
import matplotlib.pyplot as plt
import csv
from sklearn import metrics

from utilities import (create_folder, get_filename, d_prime)
import config


def load_statistics(statistics_path):
    statistics_dict = pickle.load(open(statistics_path, 'rb'))

    bal_map = np.array([statistics['average_precision'] for statistics in statistics_dict['bal']])    # (N, classes_num)
    bal_map = np.mean(bal_map, axis=-1)
    test_map = np.array([statistics['average_precision'] for statistics in statistics_dict['test']])    # (N, classes_num)
    test_map = np.mean(test_map, axis=-1)

    return bal_map, test_map


def crop_label(label):
    max_len = 16
    if len(label) <= max_len:
        return label
    else:
        words = label.split(' ')
        cropped_label = ''
        for w in words:
            if len(cropped_label + ' ' + w) > max_len:
                break
            else:
                cropped_label += ' {}'.format(w)
    return cropped_label


def add_comma(integer):
    """E.g., 1234567 -> 1,234,567
    """
    integer = int(integer)
    if integer >= 1000:
        return str(integer // 1000) + ',' + str(integer % 1000)
    else:
        return str(integer)


def plot_classwise_iteration_map(args):
    
    # Paths
    save_out_path = 'path_to_save.pdf'
 #   create_folder(os.path.dirname(save_out_path))

    # Load statistics
    statistics_dict = pickle.load(open(r'path_to_model.pkl', 'rb'))
    for i in range(377):
        statistics_dict['bal'][i]['average_precision'] = statistics_dict['bal'][i]['average_precision'][72:138] 
        statistics_dict['bal'][i]['auc'] = statistics_dict['bal'][i]['auc'][72:138]
        statistics_dict['test'][i]['average_precision'] = statistics_dict['test'][i]['average_precision'][72:138]
        statistics_dict['test'][i]['auc'] = statistics_dict['test'][i]['auc'][72:138]

    mAP_mat = np.array([e['average_precision'] for e in statistics_dict['test']])
    mAP_mat = mAP_mat[0 : 300, :]   # 300 * 2000 = 600k iterations
    sorted_indexes = np.argsort(config.full_samples_per_class)[::-1]
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    ranges = [np.arange(1, 11), np.arange(11, 22), np.arange(22, 33)]
    axs[0].set_ylabel('AP')

    for col in range(0, 3):
        axs[col].set_ylim(0, 1.)
        axs[col].set_xlim(0, 301)
        axs[col].set_xlabel('Iterations')
        axs[col].set_ylabel('AP')
        axs[col].xaxis.set_ticks(np.arange(0, 301, 100))
        axs[col].xaxis.set_ticklabels(['0', '200k', '400k', '600k'])
        lines = []
        for _ix in ranges[col]:
            _label = crop_label(config.labels[sorted_indexes[_ix]]) + \
                ' ({})'.format(add_comma(config.full_samples_per_class[sorted_indexes[_ix]]))
            line, = axs[col].plot(mAP_mat[:, sorted_indexes[_ix]], label=_label)
            lines.append(line)
        box = axs[col].get_position()
        axs[col].set_position([box.x0, box.y0, box.width * 1., box.height])
        axs[col].legend(handles=lines, bbox_to_anchor=(1., 1.))
        axs[col].yaxis.grid(color='k', linestyle='solid', alpha=0.3, linewidth=0.3)
 
    plt.tight_layout(pad=4, w_pad=1, h_pad=1)
    plt.savefig(save_out_path)
    print(save_out_path)

plot_classwise_iteration_map('plot_classwise_iteration_map')


def plot_six_figures(args):
    
    # Arguments & parameters
    classes_num = config.classes_num
    labels = config.labels
    max_plot_iteration = 15000
    iterations = np.arange(0, max_plot_iteration, 2000)

    # Paths
    class_labels_indices_path = r'F:/audioset_tagging_cnn/metadata/class_labels_indices_clean.csv'
    save_out_path = r"F:\audioset_data\results\two_figures_2.pdf"
#    create_folder(os.path.dirname(save_out_path))
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    bal_alpha = 0.3
    test_alpha = 1.0
    linewidth = 1.

    if True:
        lines = []

        (bal_map, test_map) = load_statistics(r"F:\audioset_data\CNN14 balanced mixup 100 percent 14750 iter\statistics\sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000\balanced=balanced\augmentation=mixup\batch_size=32\statistics.pkl")
        line, = ax[1].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1].plot(test_map, label='CNN14 balanced mixup, batch size=32', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        (bal_map, test_map) = load_statistics(r"F:\audioset_data\CNN14 balanced mixup batch16 15000 iter\statistics\sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000\batch_size=16\statistics.pkl")
        line, = ax[1].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[1].plot(test_map, label='CNN14 balanced mixup, batch size=16', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[1].legend(handles=lines, loc=2)
        ax[1].set_title('(d) Comparison of batch size')

    # (d) Comparison of amount of training data
    if True:
        lines = []

        # 100% of full training data
        (bal_map, test_map) = load_statistics(r"F:\audioset_data\CNN14 balanced mixup 100 percent 14750 iter\statistics\sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000\balanced=balanced\augmentation=mixup\batch_size=32\statistics.pkl")
        line, = ax[0].plot(bal_map, color='r', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0].plot(test_map, label='CNN14 (100% full)', color='r', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)


        # 50% of full training data
        (bal_map, test_map) = load_statistics(r"F:\audioset_data\CNN14 50 percent 20000 iter\statistics\sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000\batch_size=32\statistics.pkl")
        line, = ax[0].plot(bal_map, color='g', alpha=bal_alpha, linewidth=linewidth)
        line, = ax[0].plot(test_map, label='CNN14 (50% full)', color='g', alpha=test_alpha, linewidth=linewidth)
        lines.append(line)

        ax[0].legend(handles=lines, loc=2)
        ax[0].set_title('(c) Amount of training data comparison')


    for i in range(2):
            ax[i].set_ylim(0, 0.8)
            ax[i].set_xlim(0, len(iterations))
            ax[i].set_xlabel('Iterations')
            ax[i].set_ylabel('mAP')
            ax[i].xaxis.set_ticks(np.arange(0, len(iterations), 2))
            ax[i].xaxis.set_ticklabels(['0', '4k', '8k', '12k', '20k', '25k'])
            ax[i].yaxis.set_ticks(np.arange(0, 0.81, 0.05))
            ax[i].yaxis.set_ticklabels(['0', '', '0.1', '', '0.2', '', '0.3', 
                '', '0.4', '', '0.5', '', '0.6', '', '0.7', '', '0.8'])
            ax[i].yaxis.grid(color='k', linestyle='solid', alpha=0.3, linewidth=0.3)
            ax[i].xaxis.grid(color='k', linestyle='solid', alpha=0.3, linewidth=0.3)

    plt.tight_layout(0, 1, 0)
    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))

plot_six_figures('plot_six_figures')
