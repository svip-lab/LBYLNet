from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb

def make_scheduler(optimizer, system_config, last_epoch=-1, logger=None):
    warm_up = system_config._configs['warm_up']
    max_epoch =  system_config._configs['nb_epoch'] 
    warm_up_epoch = system_config._configs["warm_up_epoch"] if warm_up else 0

    if system_config._configs["lr_scheduler"] == "step_lr":
        lr_scheduler = StepLR(optimizer, 
                            step_size=system_config._configs["stepsize"],
                            gamma=system_config._configs["gamma"], 
                            last_epoch=last_epoch)
    elif system_config._configs["lr_scheduler"] == "multi_step_lr":
        lr_scheduler = MultiStepLR(optimizer, 
                                   milestones= system_config._configs["milestones"],
                                   gamma= system_config._configs["gamma"], 
                                   last_epoch=last_epoch)
    elif system_config._configs["lr_scheduler"] == "poly_lr":
        lr_scheduler = PolyScheduler(optimizer,
                                     max_epoch= system_config._configs["nb_epoch"],
                                     power=0.9, 
                                     last_epoch=last_epoch)
    elif system_config._configs["lr_scheduler"] == "cosin_lr":
        max_epoch = max_epoch - warm_up_epoch
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch)

    if system_config._configs['warm_up']:
        warm_up_lr = system_config._configs["warm_up_from_lr"]
        target_lr  = system_config._configs["learning_rate"]
        warm_up_epoch = system_config._configs["warm_up_epoch"]
        multiplier = target_lr / warm_up_lr
        warm_up_scheduler =  GradualWarmupScheduler(optimizer, 
                                                    multiplier,
                                                    warm_up_epoch, 
                                                    after_scheduler=lr_scheduler, 
                                                    last_epoch=last_epoch)
        lr_scheduler = warm_up_scheduler
        info = 'Warm up learning from {:.9f} to {:.9f} in the first {} epoches'\
                                            .format(warm_up_lr,
                                            target_lr, 
                                            warm_up_epoch)
        print(info)
        info = "after that {} are applied.".format(system_config._configs["lr_scheduler"])
        print(info)
        
    return lr_scheduler


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, max_epoch, power, last_epoch=-1):
        self.max_epoch = max_epoch
        self.power = power
        super(PolyScheduler, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch > self.max_epoch:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [base_lr*(1-self.last_epoch/self.max_epoch)**self.power for base_lr in self.base_lrs]


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)