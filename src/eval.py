import helpers.evaluation as ev
import argparse
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- EVALUATION OPTIONS -- #
    parser.add_argument('-m', '--model', type=str, help='the model to use. should reference folder and python file')
    parser.add_argument('-r', '--run', type=str, help='the run id, a datetime')
    parser.add_argument('-d', '--dataset', type=str, help='the data set to use. possible values: mnist, mazes')
    parser.add_argument('-a', '--action', type=str, help='what to do. possible values: check, draw')
    opt = parser.parse_args()
    print(opt)

    path = os.path.abspath(os.path.join('models', 'mazes', opt.model, 'samples', opt.run, '*.sample.tar'))
    sample_files = glob.glob(path)
    sample_files.sort()

    if opt.action == 'check_ind':
        ev.check_ind(sample_files)
    if opt.action == 'check_avg':
        ev.check_avg(sample_files)
    if opt.action == 'check_and_draw':
        ev.check_and_draw(sample_files)
