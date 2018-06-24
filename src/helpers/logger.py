import os

from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import torch


def log_batch_statistics(epoch: int, epochs: int, batch: int, batches: int,
                         d_loss: torch.Tensor, g_loss: torch.Tensor,
                         real_scores: torch.Tensor, fake_scores: torch.Tensor) -> None:
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


def save_image_grid(module_path, run_id, real_imgs, fake_imgs, step) -> None:
    """Save a  5 x 5 grid of images, real and generated. Does not do any upscaling on the images,
    so small mazes of e.g. 8 x 8 will not show well. Accepts batches of images using
    PyTorch batch representation batch_size x 1 x *image_dimensions

    Args:
        module_path: The path to the module that is currently being executed.
        run_id: An id of the current run. Should match the Tensorboard run id. Usually a datetime.
        real_imgs: The training images. A Tensor of size batch_size x 1 x ...
        fake_imgs: The generated images. A Tensor of size batch_size x 1 x ...
        step: The current global step.
    """
    path = os.path.join(module_path, 'images', run_id)
    os.makedirs(path, exist_ok=True)
    save_image(real_imgs.data[:25], os.path.join(path, 'real_%d.png') % step, nrow=5, normalize=True)
    save_image(fake_imgs.data[:25], os.path.join(path, 'fake_%d.png') % step, nrow=5, normalize=True)


def log_tensorboard_basic_data(writer: SummaryWriter, step: int,
                               g_loss: torch.Tensor, d_loss: torch.Tensor,
                               real_scores: torch.Tensor, fake_scores: torch.Tensor) -> None:
    """ Log basic data to show plots of generator and discriminator losses and the mean scores of the
    discriminator on real and generated images. Should be called with some frequency in any training procedure.

    Args:
        writer: A tensorboardX SummaryWriter, which should log to the module directory.
        step: The current global step.
        g_loss: The generator loss. A Tensor of size 1 x 1.
        d_loss: The discriminator loss. A Tensor of size 1 x 1.
        real_scores: Discriminator scores on real images. A Tensor of size batch_size x 1.
        fake_scores: Discriminator scores on fake images. A Tensor of size batch_size x 1.
    """
    writer.add_scalar('Generator/loss', g_loss.item(), step)
    writer.add_scalar('Discriminator/loss', d_loss.item(), step)
    writer.add_scalar('D(x)', real_scores.detach().mean().item(), step)
    writer.add_scalar('D(G(z))', fake_scores.detach().mean().item(), step)


def log_tensorboard_parameter_data(writer: SummaryWriter, step: int,
                                   generator: torch.nn.Module, discriminator: torch.nn.Module) -> None:
    """ Log advanced parameter and gradient data. Should only be called when debugging for performance reasons.

    Args:
        writer: A tensorboardX SummaryWriter, which should log to the module directory.
        step: The current global step.
        generator: The generator module.
        discriminator: The discriminator module.
    """
    for name, param in generator.named_parameters():
        name = name.replace('.', '/')
        writer.add_histogram("Generator/" + name, param.detach().data.cpu().numpy(),
                             step, bins='auto')
        # DISABLED BECAUSE UNSTABLE
        # writer.add_histogram("Generator/" + name + '/grad', param.grad.detach().data.cpu().numpy(),
        #                      step, bins='auto')

    for name, param in discriminator.named_parameters():
        name = name.replace('.', '/')
        writer.add_histogram("Discriminator/" + name, param.detach().data.cpu().numpy(),
                             step, bins='auto')
        # DISABLED BECAUSE UNSTABLE
        # writer.add_histogram("Discriminator/" + name + '/grad', param.grad.detach().data.cpu().numpy(),
        #                      step, bins='auto')
