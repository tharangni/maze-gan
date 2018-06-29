from typing import Union

from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch
import os
import csv
import json


# noinspection PyMethodMayBeStatic
class Logger:
    def __init__(self, module_path: str, run: Union[str, None], opt):
        """Instantiate a logger to handle console and disk output.

        Args:
            module_path: The path to the module that is currently being executed.
            run: An id of the current run. Should match the Tensorboard run id. Usually a datetime.
        """

        self.sample_path = os.path.join(module_path, 'samples', run)
        self.image_path = os.path.join(module_path, 'images', run)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(module_path, 'runs', run))
        path = os.path.join(module_path, 'runs', run, )
        self.log_hyper_parameters(os.path.join(path, "model_params.txt"), opt)
        self.csv_file = open(os.path.join(path, "epoch.csv"), 'w+', newline='')
        self.csv_writer = csv.writer(self.csv_file, delimiter=',')  # for looging results for graphing.
        self.csv_writer.writerow(['epoch_no', 'batch_no', 'd_loss', 'g_loss', 'D(x)', 'D(G(X))'])

    def log_batch_statistics(self, epoch: int, epochs: int, batch: int, batches: int,
                             d_loss: Variable, g_loss: Variable,
                             real_scores: Variable, fake_scores: Variable) -> None:
        """ Print statistics of current batch to console.

        Args:
            epoch: The current epoch.
            epochs: Total number of epochs.
            batch: The current batch.
            batches: Total number of batches in an epoch.
            d_loss: The discriminator loss. A Tensor of size 1 x 1.
            g_loss: The generator loss. A Tensor of size 1 x 1.
            real_scores: Discriminator scores on real images. A Tensor of size batch_size x 1.
            fake_scores: Discriminator scores on fake images. A Tensor of size batch_size x 1.
        """
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [D(x): %.2f] [D(G(z)): %.2f]" %
              (epoch + 1, epochs, batch, batches, d_loss.item(), g_loss.item(),
               real_scores.detach().mean().item(), fake_scores.detach().mean().item()))
        self.csv_writer.writerow([epoch + 1, batch + 1, d_loss.item(), g_loss.item(), real_scores.mean().item(),
                                  fake_scores.mean().item()])

    def save_image_grid(self, real_imgs, fake_imgs, step) -> None:
        """Save a  5 x 5 grid of images, real and generated. Does not do any upscaling on the images,
        so small mazes of e.g. 8 x 8 will not show well. Accepts batches of images using
        PyTorch batch representation batch_size x 1 x *image_dimensions

        Args:
            real_imgs: The training images. A Tensor of size batch_size x 1 x ...
            fake_imgs: The generated images. A Tensor of size batch_size x 1 x ...
            step: The current global step.
        """
        real_path = os.path.join(self.image_path, 'real_{0:0=8d}.png').format(step)
        fake_path = os.path.join(self.image_path, 'fake_{0:0=8d}.png').format(step)
        if real_imgs is not None:
            size = real_imgs.size()
            save_image(real_imgs.view(size[0], 1, size[1], size[1]).data[:25], real_path, nrow=5, normalize=True)
        if fake_imgs is not None:
            size = fake_imgs.size()
            save_image(fake_imgs.view(size[0], 1, size[1], size[1]).data[:25], fake_path, nrow=5, normalize=True)

    def log_tensorboard_basic_data(self, g_loss: Variable, d_loss: Variable, real_scores: Variable,
                                   fake_scores: Variable, step: int) -> None:
        """ Log basic data to show plots of generator and discriminator losses and the mean scores of the
        discriminator on real and generated images. Should be called with some frequency in any training procedure.

        Args:
            step: The current global step.
            g_loss: The generator loss. A Tensor of size 1 x 1.
            d_loss: The discriminator loss. A Tensor of size 1 x 1.
            real_scores: Discriminator scores on real images. A Tensor of size batch_size x 1.
            fake_scores: Discriminator scores on fake images. A Tensor of size batch_size x 1.
        """
        self.writer.add_scalar('Generator/loss', g_loss.item(), step)
        self.writer.add_scalar('Discriminator/loss', d_loss.item(), step)
        self.writer.add_scalar('D(x)', real_scores.detach().mean().item(), step)
        self.writer.add_scalar('D(G(z))', fake_scores.detach().mean().item(), step)

    def log_tensorboard_parameter_data(self, generator: torch.nn.Module,
                                       discriminator: torch.nn.Module, step: int) -> None:
        """ Log advanced parameter and gradient data. Should only be called when debugging for performance reasons.

        Args:
            step: The current global step.
            generator: The generator module.
            discriminator: The discriminator module.
        """
        for name, param in generator.named_parameters():
            name = name.replace('.', '/')
            self.writer.add_histogram("Generator/" + name, param.detach().data.cpu().numpy(),
                                      step, bins='auto')
            # DISABLED BECAUSE UNSTABLE
            # self.writer.add_histogram("Generator/" + name + '/grad', param.grad.detach().data.cpu().numpy(),
            #                      step, bins='auto')

        for name, param in discriminator.named_parameters():
            name = name.replace('.', '/')
            self.writer.add_histogram("Discriminator/" + name, param.detach().data.cpu().numpy(),
                                      step, bins='auto')
            # DISABLED BECAUSE UNSTABLE
            # self.writer.add_histogram("Discriminator/" + name + '/grad', param.grad.detach().data.cpu().numpy(),
            #                      step, bins='auto')

    def log_generated_sample(self, data: Variable, step: int) -> None:
        input_size = data.size(-1)
        path = os.path.join(self.sample_path, 'fake_{0:0=8d}.sample.tar'.format(step))
        torch.save(data.detach().cpu().view(-1, input_size, input_size), path)

    def log_hyper_parameters(self, path, hyperparameters):
        exDict = {'hyperparameters': vars(hyperparameters)}

        with open(path, 'w+') as file:
            file.write(json.dumps(exDict))  # use `json.loads` to do the reverse

    def close_writers(self):
        self.writer.close()
        self.csv_file.close()
