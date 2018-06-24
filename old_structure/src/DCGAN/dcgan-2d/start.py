import argparse
import os
import torch
import pickle
from maze_gen import check_maze, draw_maze
from gan import GAN
import numpy as np
from matplotlib import pyplot as plt

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
        maze = maze.squeeze()
        maze[maze<0.5] = 0
        maze[maze>0.5] = 1
        # is it a valid maze?
        if torch.cuda.is_available(): maze = maze.cpu()
        maze = maze.detach().numpy()
        check = check_maze(maze)
        if check:
            print(check)
            draw_maze(maze)
        else:
            print(check)
            draw_maze(maze)
    correct = 0
    for maze in mazes:
        maze = maze.squeeze()
        maze[maze<0.5] = 0
        maze[maze>0.5] = 1
        if torch.cuda.is_available(): maze = maze.cpu()
        maze = maze.detach().numpy()
        check = check_maze(maze)
        if check:
            correct += 1
            draw_maze(maze)
    print(correct, ' correct out of ', len(mazes))


def test_results(dir, eg_no):
    tot_r = 0
    for i in range(1, int(eg_no)+1):
        path =  os.path.join(dir, 'fake_mazes-{}.pickle'.format(i))
        mazes = pickle.load(open(path, 'rb'))
        # print(mazes)
        r = []
        for maze in mazes:
            maze = maze.squeeze()
            maze[maze<0.5] = 0
            maze[maze>0.5] = 1
            r.append(check_maze(maze))
        tot_r += sum(r)
        if i % 100 == 0:
            print('average of ', i-100, '-', i)
            print(tot_r/100, ' of ', len(r))
            tot_r = 0



def start():
    #look for cmd arguments here

    parser = argparse.ArgumentParser(description='Run GAN or visualise maze.')
    parser.add_argument('--v', '--visualise', action='store',nargs=2, help='Visualise a sample of fake results')
    parser.add_argument('--t', '--test', action='store', nargs=2, help='Test fake results')
    #------ Have to check which are rows and columns -------#
    parser.add_argument('--mx', help='No. columns in maze', type=int, default=8)
    parser.add_argument('--my', help='No. rows in maze', type=int, default=8)
    parser.add_argument('--N', help='No. of traning examples to generate', type=int, default=200)
    #-------------------------------------------------------#
    parser.add_argument('--input_size', help='No. inputs for generator', type=int, default=20)
    parser.add_argument('--hidden_size', help='No. of hidden neurons', type=int, default=10)
    parser.add_argument('--num_epochs', help='No. of epochs', type=int, default=200) #i.e. number of fake mazes to generate
    parser.add_argument('--batch_size', help='Size of batch to use (Must be compatible with N)', type=int, default=100)
    parser.add_argument('--maze_dir', help='Directory results are stored in', type=str, default=def_dir)

    #TODO check batch size is appropoate for the N given

    args = parser.parse_args()

    if args.v:
        visualise_results(args.v[0], args.v[1])
    elif args.t:
        test_results(args.t[0], args.t[1])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
