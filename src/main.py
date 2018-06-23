import argparse
import glob
import os
import torch
from helpers.MazeGenerator import check_maze, draw_maze

from models.continuous_vanilla.GeneralAdversarialNetwork import GeneralAdversarialNetwork
from models.discrete_recurrent.DiscreteRecurrentAdversarialNetwork import DiscreteRecurrentAdversarialNetwork
from models.discrete_vanilla.DiscreteAdversarialNetwork import DiscreteAdversarialNetwork
from models.discrete_boundary_seeking.BoundarySeekingAdversarialNetwork import BoundarySeekingAdversarialNetwork

def_dir = 'maze_results'


def visualise_results(path, eg_no):
    # path = os.path.join(dir, 'real_mazes.pickle')
    path = os.path.join(path, 'fake_mazes_{}.pickle'.format(eg_no))
    print('Visualising sample from {}'.format(path))
    # visualise sample from final results
    mazes = torch.load(path, map_location='cpu').detach().numpy()

    # print(mazes)
    # takes sample and plot
    for maze in mazes[:10]:
        print(maze)
        check = check_maze(maze)
        if check:
            print(check)
            draw_maze(maze)
        else:
            print(check)
            draw_maze(maze)
    correct = 0
    for maze in mazes:
        maze = maze
        check = check_maze(maze)
        if check:
            correct += 1
    print(correct, ' correct out of ', len(mazes))


def visualise_training_set(path):
    # path = os.path.join(dir, 'real_mazes.pickle')
    path = os.path.join(path, 'real_mazes.pickle')
    print('Visualising real mazes from {}'.format(path))
    # visualise sample from final results
    mazes = torch.load(path, map_location='cpu').detach()
    correct = 0
    for maze in mazes:
        maze = maze.detach().numpy()
        check = check_maze(maze)
        if check:
            correct += 1
            draw_maze(maze)
    print(correct, ' correct out of ', len(mazes))


def test_results(directory, eg_no):
    path = os.path.join(directory, 'fake_mazes_{}.pickle'.format(eg_no))
    print('Testing results from {}'.format(path))
    mazes = torch.load(path, map_location='cpu').detach()
    # print(mazes)
    r = []
    correct = 0
    for each_maze in mazes:
        t = check_maze(each_maze)
        r.append(t)
        if t:
            correct += 1
    print(correct, '/', len(r))


def test_all_results_average(directory):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    files = glob.glob(directory + '/fake_*.pickle')
    files.sort()
    for idx, chunk in enumerate(chunks(files, 100)):
        correct = 0
        total = 0
        for file in chunk:
            mazes = torch.load(file, map_location='cpu').detach()
            total += mazes.size(0)
            for each_maze in mazes:
                if check_maze(each_maze): correct += 1
        print(idx + 1, ':', correct / total)


def test_all_results_ind(directory):
    files = glob.glob(directory + '/fake_*.pickle')
    files.sort()
    for idx, file in enumerate(files):
        correct = 0
        total = 0
        mazes = torch.load(file, map_location=lambda storage, loc: storage).detach()
        total += mazes.size(0)
        for each_maze in mazes:
            if check_maze(each_maze): correct += 1
        print(file, ':', correct, "/", total)


def start():
    # look for cmd arguments here

    parser = argparse.ArgumentParser(description='Run GAN or visualise maze.')
    parser.add_argument('--v', '--visualise', action='store', nargs=2, help='Visualise a sample of fake results')
    parser.add_argument('--r', '--real', action='store', nargs=1, help='Visualise real results')
    parser.add_argument('--t', '--test', action='store', nargs=2, help='Test fake results')
    parser.add_argument('--test_all_average', action='store', nargs=1,
                        help='Test all fake results averaging over 100 epochs')
    parser.add_argument('--test_all_ind', action='store', nargs=1, help='Test all fake results individually')

    # ------ Have to check which are rows and columns -------#
    parser.add_argument('--mx', help='No. columns in maze', type=int, default=28)
    parser.add_argument('--my', help='No. rows in maze', type=int, default=28)
    parser.add_argument('--N', help='No. of training examples to generate', type=int, default=1000)
    # -------------------------------------------------------#
    parser.add_argument('--latent_size', help='No. inputs for generator', type=int, default=64)
    parser.add_argument('--hidden_size', help='No. of hidden neurons', type=int, default=256)
    parser.add_argument('--num_epochs', help='No. of epochs', type=int, default=200)
    parser.add_argument('--batch_size', help='Size of batch to use (Must be compatible with N)', type=int, default=100)

    parser.add_argument('--g_lr', help='Generator learning rate', type=float, default=0.0002)
    parser.add_argument('--d_lr', help='Discriminator learning rate', type=float, default=0.0002)
    parser.add_argument('--temp', help='Temperature for gumbel-softmax-trick', type=float, default=0.8)

    parser.add_argument('--model', help='Which model to train', type=str, default='mnist')
    parser.add_argument('--resume', help='Whether to resume or start fresh', type=bool, default=False)

    args = parser.parse_args()

    if args.v:
        visualise_results(args.v[0], args.v[1])
    elif args.r:
        visualise_training_set(args.r[0])
    elif args.t:
        test_results(args.t[0], args.t[1])
    elif args.test_all_average:
        test_all_results_average(args.test_all_average[0])
    elif args.test_all_ind:
        print(args)
        test_all_results_ind(args.test_all_ind[0])
    else:
        if args.model == 'mnist':
            gan = GeneralAdversarialNetwork(args)
            gan.train()
        elif args.model == 'vanilla_maze':
            gan = DiscreteAdversarialNetwork(args)
            gan.train()
        elif args.model == 'boundary_seeking_maze':
            gan = BoundarySeekingAdversarialNetwork(args)
            gan.train()
        elif args.model == 'recurrent_maze':
            gan = DiscreteRecurrentAdversarialNetwork(args)
            gan.train()


if __name__ == '__main__':
    start()
