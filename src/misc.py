import argparse

from helpers import maze_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- MODEL OPTIONS -- #
    parser.add_argument('-a', '--action', type=str, help='the action to take: create.')
    parser.add_argument('-n', '--n_examples', type=int, help='the data set to use. possible values: mnist, mazes.')
    parser.add_argument('-s', '--size', type=int, help='size of the maze to create.')
    args = parser.parse_args()

    print(args)

    if args.action == 'create':
        maze_utils.gen_maze_data(args.n_examples, args.size, args.size)