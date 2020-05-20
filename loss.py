import torch


def loss_fn(outputs, inputs):
    mse = ((inputs - outputs) ** 2).mean(-1).mean(-1).squeeze()
    loss = torch.sqrt((torch.sqrt(mse) ** 2).mean())
    return loss
