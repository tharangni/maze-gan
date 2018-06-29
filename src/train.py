import argparse
import importlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- MODEL OPTIONS -- #
    parser.add_argument('-m', '--model', type=str, help='the model to use. should reference folder and python file')
    parser.add_argument('-d', '--dataset', type=str, help='the data set to use. possible values: mnist, mazes')
    parser.add_argument('-r', '--resume', type=bool, help='whether to resume training')

    # -- LOGGING OPTIONS -- #
    parser.add_argument('-l', '--log_details', type=bool, default=True,
                        help='whether to log parameter, gradient data and epochs')

    # -- HYPER PARAMS -- #
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--n_examples', type=int, default=60000, help='training examples to load (used only for mazes)')
    parser.add_argument('--batch_size', type=int, default=60, help='size of the batches')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='adam: learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='adam: learning rate for generator')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval between image samples')
    parser.add_argument('--img_size', type=int, default=28,
                        help='size of each MNIST image dimension (Discriminator input)')
    parser.add_argument('--maze_size', type=int, default=8,
                        help='size of each maze dimension (Discriminator input)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='The dimension of the latent space (Generator input)')
    parser.add_argument('--temp', type=float, default=0.2, help='temperature to use for gumbel-softmax quantization')
    opt = parser.parse_args()

    print(opt)

    model = importlib.import_module('.'.join(['models', opt.model, opt.model]))
    model.run(opt)
