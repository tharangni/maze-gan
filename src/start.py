import argparse
import os
import torch
from gan import GAN


def start():
    #look for cmd arguments here
    parser = argparse.ArgumentParser(description='Run GAN or visualise maze.')
    parser.add_argument("--v", "--visualise", help="Visualise a sample of fake results", action="store_true", default=None)
    parser.add_argument("--t", "--test", help="Test fake results", action="store_true", default=None)
    #------ Have to check which are rows and columns -------#
    parser.add_argument("--mx", help="No. columns in maze", type=int, default=4)
    parser.add_argument("--my", help="No. rows in maze", type=int, default=4)
    #-------------------------------------------------------#
    parser.add_argument("--N", help="No. of traning examples to generate", type=int, default=200)
    parser.add_argument("--input_size", help="No. inputs for generator", type=int, default=4)
    parser.add_argument("--hidden_size", help="No. of hidden neurons", type=int, default=8)
    parser.add_argument("--num_epochs", help="No. of epochs", type=int, default=200)
    parser.add_argument("--batch_size", help="Size of batch to use (Must be compatible with N)", type=int, default=100)
    parser.add_argument("--maze_dir", help="Directory results are stored in", type=str, default="maze_results")

    #TODO check batch size is appropoate for the N given


    #parser.parse_args('t') #test fake results
    args = parser.parse_args()
    print(args)
    if args.v:
        print("Visualising sample")
    elif args.t:
        print("Testing fake results")

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


def test_results():
    print(test_results)