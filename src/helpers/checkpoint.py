import os
from typing import Tuple

import torch


class Checkpoint:
    def __init__(self, module_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Register a path to save state checkpoints. Will create a `checkpoints` folder and remember its path.

        Args:
            module_path: The path to the module that is being saved. Retrievable with `os.path.abspath(__file__)`
            model: The model to save / load (Generator or Discriminator)
            optimizer: The optimizer to save / load
        """

        checkpoint_dir = os.path.join(module_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.path = os.path.join(checkpoint_dir, '{}.checkpoint.pth.tar'.format(type(model).__name__.lower()))

    def save(self, run: str, epoch: int) -> None:
        """Persist the current state of the model and optimizer to disk. Stores:
            * Model parameters.
            * Optimizer parameters.
            * Starting epoch.

        Args:
            run: The id of the current run. Typically a datetime.
            epoch: The current epoch.
        """
        state = {
            'run': run,
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, self.path)

    def load(self) -> Tuple[str, int]:
        """loads the previously saved states into the model and optimizer and returns the last epoch trained.

        Returns:
            The last epoch trained.
        """
        checkpoint = torch.load(self.path)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['run'], checkpoint['epoch']
