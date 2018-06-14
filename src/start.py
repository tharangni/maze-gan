import os
import torch
import pickle
import codecs
import argparse
import numpy as np
from matplotlib import pyplot as plt

from maze_gen import check_maze, draw_maze
from gan import GAN

def_dir='maze_results'


def visualise_results(dir, eg_no):
    # path = os.path.join(dir, 'real_mazes.pickle')
    path = os.path.join(dir, 'fake_mazes-{}.pickle'.format(eg_no))
    print('Visualising sample from {}'.format(path))
    #visualise sample from final results
    mazes = pickle.load(open(path, 'rb'))
    #print(mazes)
    #takes sample and plot
    for maze in mazes[:10]:
        print(maze)
        maze[maze<0.5] = 0
        maze[maze>0.5] = 1
        # is it a valid maze?
        if torch.cuda.is_available(): maze = maze.cpu()
        maze = maze.detach().numpy()
        check = check_maze(maze)
        print(check)
        # draw_maze(maze)
    test_results(dir, eg_no)


def test_results(dir, eg_no):
    path =  os.path.join(dir, 'fake_mazes-{}.pickle'.format(eg_no))
    print('Testing results from {}'.format(path) )
    mazes = pickle.load(open(path, 'rb'))
    # print(mazes)
    r = np.array([])
    for maze in mazes:
        maze[maze < 0.5] = 0
        maze[maze > 0.5] = 1
        if torch.cuda.is_available(): maze = maze.cpu()
        maze = maze.detach().numpy()
        r = np.append(r, check_maze(maze))

    print(r.sum(), " out of ", len(r), "  ", r.sum()/len(r) * 100, "%")


def all_results(dir):
    '''
    Function for an overall summary of results for the entire run
    NOTE: Make the model first before executing this
    '''
    length = len(os.listdir(dir)) - 1
    all_counts = []
    print("Starting to read all files in directory: {}".format(dir))
    for i in range(length):
        path = os.path.join(dir, 'fake_mazes-{}.pickle'.format(i+1))
        with codecs.open(path, 'rb') as mazes :
            mazes = pickle.load(mazes)
            r = np.array([])
            for maze in mazes:
                maze[maze < 0.5] = 0
                maze[maze > 0.5] = 1
                if torch.cuda.is_available(): maze = maze.cpu()
                maze = maze.detach().numpy()
                r = np.append(r, check_maze(maze))
            all_counts.append(r.sum())
            # For analysis per pickle file
            # print('Testing results from fake maze-{} : {} out of {}: {:.2f} %'.format(i+1, r.sum(), len(r), r.sum()/len(r) * 100))
        if i%100 == 0:
            print("Completed {}/{}...".format(i, length))
    max_count = max(all_counts)
    for i in range(len(all_counts)):
        if all_counts[i]==max_count:
            print('Most number of correct fake mazes found at file {} with {}/{}.'.format(i+1, max_count, len(r)))


def start():
    #look for cmd arguments here

    parser = argparse.ArgumentParser(description='Run GAN or visualise maze.')
    parser.add_argument('--v', '--visualise', action='store',nargs=2, help='Visualise a sample of fake results')
    parser.add_argument('--t', '--test', action='store', nargs=2, help='Test fake results for a sample')
    parser.add_argument('--a', '--all', action='store', nargs=1, help='Print results of the complete sample')
    #------ Have to check which are rows and columns -------#
    parser.add_argument('--mx', help='No. columns in maze', type=int, default=4)
    parser.add_argument('--my', help='No. rows in maze', type=int, default=4)
    parser.add_argument('--N', help='No. of traning examples to generate', type=int, default=200)
    #-------------------------------------------------------#
    parser.add_argument('--input_size', help='No. inputs for generator', type=int, default=4)
    parser.add_argument('--hidden_size', help='No. of hidden neurons', type=int, default=8)
    parser.add_argument('--num_epochs', help='No. of epochs', type=int, default=200) #i.e. number of fake mazes to generate
    parser.add_argument('--batch_size', help='Size of batch to use (Must be compatible with N)', type=int, default=100)
    parser.add_argument('--maze_dir', help='Directory results are stored in', type=str, default=def_dir)

    #TODO check batch size is appropoate for the N given

    args = parser.parse_args()

    if args.v:
        visualise_results(args.v[0], args.v[1])
    elif args.t:
        test_results(args.t[0], args.t[1])
    elif args.a:
        all_results(args.a[0])
    else:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(args.maze_dir):
            os.makedirs(args.maze_dir)

        gan = GAN(device,
                  args.input_size,
                  args.hidden_size,
                  args.num_epochs,
                  args.batch_size,
                  args.mx,
                  args.my,
                  args.N,
                  args.maze_dir)

        #train
        gan.train()

    #save gan
if __name__ == '__main__':
    start()
