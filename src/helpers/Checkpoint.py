import os
import torch


class Checkpoint:

    def __init__(self, state, gan_type, model_name):
        self.state = state
        self.path = os.path.join('checkpoints', gan_type, model_name + '.checkpoint.pth.tar')
        if not os.path.exists(os.path.join('checkpoints', gan_type)):
            os.makedirs(os.path.join('checkpoints', gan_type))

    def save(self):
        if self.state is None:
            raise ValueError('No state provided.')
        torch.save(self.state, self.path)

    def load(self, model):
        checkpoint = torch.load(self.path)
        start_epoch = checkpoint.epoch
        model.model.load_state_dict(checkpoint.model)
        model.optimizer.load_state_dict(checkpoint.optimizer)

        return start_epoch
