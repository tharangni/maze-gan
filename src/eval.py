import numpy as np
import torch

from helpers.logger import Logger
import helpers.evaluation as ev
import argparse
import glob
import os

from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- EVALUATION OPTIONS -- #
    parser.add_argument('-m', '--model', type=str, help='the model to use. should reference folder and python file')
    parser.add_argument('-r', '--run', type=str, help='the run id, a datetime')
    parser.add_argument('-d', '--dataset', type=str, help='the data set to use. possible values: mnist, mazes', choices=['mnist', 'mazes'])
    parser.add_argument('-a', '--action', type=str, help='what to do. possible values: check, draw', choices=['draw','check_ind','check_avg', "check_and_draw"])
    opt = parser.parse_args()

    module_path = os.path.abspath(os.path.join('models', opt.model))
    samples_path = os.path.abspath(os.path.join('models', opt.model, 'samples', opt.run, '*.sample.tar'))
    sample_files = glob.glob(samples_path)
    sample_files.sort()

    if opt.action == 'draw':
        logger = Logger(module_path, opt.run, opt)
        ev.draw(sample_files, logger)
    if opt.dataset != 'mazes':
        print("Invalid dataset")
    else:
        if len(sample_files) > 0:
            if opt.action == 'check_ind':
               _ = ev.check_ind(sample_files)
            if opt.action == 'check_avg':
                ev.check_avg(sample_files)
            if opt.action == 'check_and_draw':
                ev.check_and_draw(sample_files)
        else:
            print("No sample files")