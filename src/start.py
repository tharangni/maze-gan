import os
import torch
import pickle
import codecs
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from maze_gen import check_maze, draw_maze
from gan import GAN
from scipy.interpolate import spline


def_dir = 'maze_results'


def visualise_results(dir, eg_no):
    # path = os.path.join(dir, 'real_mazes.pickle')
    path = os.path.join(dir, 'fake_{}.pickle'.format(eg_no))
    print('Visualising sample from {}'.format(path))
    # visualise sample from final results
    mazes = torch.load(open(path, 'rb'))
    # print(mazes)
    # takes sample and plot
    test_results(dir, eg_no)
    maze_u = []
    for maze in mazes:
        # maze[maze < 0.5] = 0
        # maze[maze > 0.5] = 1
        # is it a valid maze?
        if torch.cuda.is_available(): maze = maze.cpu()
        maze = maze.detach().numpy()
        check = check_maze(maze)
        if (check):
            maze_u.append(maze)

    # uniqueness
    unique_maze = np.unique(np.array(maze_u), axis = 0)
    print("{}/{} are unique mazes".format(len(unique_maze), len(maze_u)))
    for maze in unique_maze:
        print(maze)
        draw_maze(maze)


def test_results(dir, eg_no):
    path = os.path.join(dir, 'fake_{}.pickle'.format(eg_no))
    print('Testing results from {}'.format(path))
    mazes = torch.load(open(path, 'rb'))
    # print(mazes)
    r = np.array([])
    for maze in mazes:
        # maze[maze < 0.5] = 0
        # maze[maze > 0.5] = 1
        if torch.cuda.is_available(): maze = maze.cpu()
        maze = maze.detach().numpy()
        r = np.append(r, check_maze(maze))

    print(r.sum(), " out of ", len(r), "  ", r.sum() / len(r) * 100, "%")


def get_results(dir, print_flag=True):
    length = len(os.listdir(dir))
    len_b = length*400
    arr = np.arange(0, len_b, 400)
    all_counts = []
    for i in arr:
        path = os.path.join(dir, 'fake_{}.pickle'.format(i))
        with codecs.open(path, 'rb') as mazes:
            mazes = torch.load(mazes)
            r = np.array([])
            for maze in mazes:
                # maze[maze < 0.5] = 0
                # maze[maze > 0.5] = 1
                if torch.cuda.is_available(): maze = maze.cpu()
                maze = maze.detach().numpy()
                r = np.append(r, check_maze(maze))
            all_counts.append(r.sum())


            # For analysis per pickle file
            # print('Testing results from fake maze-{} : {} out of {}: {:.2f} %'.format(i+1, r.sum(), len(r), r.sum()/len(r) * 100))
        if i % 100 == 0 and print_flag:
            print("Completed {}/{}...".format(i, len_b))
    return all_counts, r


def all_results(dir):
    '''
    Function for an overall summary of results for the entire run
    NOTE: Make the model first before executing this
    '''
    print("Starting to read all files in directory: {}".format(dir))

    length = len(os.listdir(dir))
    len_b = length*400
    arr = np.arange(0, len_b, 400)
    all_counts, r = get_results(dir)
    max_count = max(all_counts)
    if max_count == len(r):
        all_counts.remove(max(all_counts))
        max_count = max(all_counts)
        arr = arr[1:]
    if max_count > 0:
        for i, c in enumerate(arr):
            if all_counts[i] == max_count:
                print('Most number of correct fake mazes found at file {} with {}/{}.'.format(c, max_count, len(r)))
                visualise_results(dir, c)
                visualise_loss("images/files", dir)
    else:
        print('No correct mazes were found')


def visualise_loss(m_dir, r_dir):
    epoch_file = pd.read_csv(os.path.join(m_dir, 'epoch.csv'))
    fig = plt.figure(figsize=(20, 10))

    # plt.subplot(1, 2, 1)

    plt.plot(epoch_file['epoch_no'], epoch_file['d_loss'], color='g', label="d_loss", lw = 0.5)
    plt.plot(epoch_file['epoch_no'], epoch_file['g_loss'], color='orange', label="g_loss", lw = 0.5)
    plt.plot(epoch_file['epoch_no'], epoch_file['D(x)'], color='r', label="D(X)", lw = 0.5)
    plt.plot(epoch_file['epoch_no'], epoch_file['D(G(X))'], color='b', label="D(G(X))", lw = 0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('GAN Loss')
    plt.legend()

    # plt.subplot(1, 2, 2)
    # maze_results, r = get_results(r_dir, print_flag=False)
    # plt.plot(np.unique(epoch_file['epoch_no']), maze_results, color='g', label="No. of correct mazes generated")
    # plt.title("Correct mazes")
    # plt.legend()
    plt.show()


def start():
    # look for cmd arguments here

    parser = argparse.ArgumentParser(description='Run GAN or visualise maze.')
    parser.add_argument('--v', '--visualise', action='store', nargs=2, help='Visualise a sample of fake results')
    parser.add_argument('--t', '--test', action='store', nargs=2, help='Test fake results for a sample')
    parser.add_argument('--a', '--all', action='store', nargs=1, help='Print results of the complete sample')
    parser.add_argument('--vl', action='store', nargs=2, help='Visualise loss on GAN')
    # ------ Have to check which are rows and columns -------#
    parser.add_argument('--mx', help='No. columns in maze', type=int, default=4)
    parser.add_argument('--my', help='No. rows in maze', type=int, default=4)
    parser.add_argument('--N', help='No. of traning examples to generate', type=int, default=200)
    # -------------------------------------------------------#
    parser.add_argument('--input_size', help='No. inputs for generator', type=int, default=4)
    parser.add_argument('--hidden_size', help='No. of hidden neurons', type=int, default=8)
    parser.add_argument('--num_epochs', help='No. of epochs', type=int,
                        default=200)  # i.e. number of fake mazes to generate
    parser.add_argument('--batch_size', help='Size of batch to use (Must be compatible with N)', type=int, default=100)
    parser.add_argument('--maze_dir', help='Directory results are stored in', type=str, default=def_dir)

    # TODO check batch size is appropoate for the N given

    args = parser.parse_args()

    if args.v:
        visualise_results(args.v[0], args.v[1])
    elif args.t:
        test_results(args.t[0], args.t[1])
    elif args.a:
        all_results(args.a[0])
    elif args.vl:
        visualise_loss(args.vl[0], args.vl[1])
    else:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(args.maze_dir):
            os.makedirs(args.maze_dir)

        writer = SummaryWriter()
        gan = GAN(device,
                  args.input_size,
                  args.hidden_size,
                  args.num_epochs,
                  args.batch_size,
                  args.mx,
                  args.my,
                  args.N,
                  args.maze_dir,
                  writer)

        # train
        gan.train()
        # writer.export_scalars_to_json("./tensorboard_data.json") # use this istead of pickle??
        writer.close()

    # save gan


if __name__ == '__main__':
    start()
