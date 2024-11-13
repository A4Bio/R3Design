
from torch import optim

def get_optim_scheduler(lr, epoch, model, steps_per_epoch, anneal_base=0.95):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # lam = lambda epoch: anneal_base ** epoch
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epoch * steps_per_epoch)
    return optimizer, scheduler