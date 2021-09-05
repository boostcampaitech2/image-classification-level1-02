import torch

class Optim:
    def __init__(self, opt_name, model_parameters, lr, momentum = 0.99, weight_decay = 0):
        
        if opt_name in [ "Adam", "adam", "ADAM"]:
            self.opt = torch.optim.Adam(
                model_parameters,
                lr = lr,
                weight_decay = weight_decay
            )
        elif opt_name in [ "SGD", "sgd", "Sgd"]:
            self.opt = torch.optim.SGD(
                model_parameters,
                lr = lr,
                momentum = momentum,
                weight_decay = weight_decay
            )
        else:
            raise ValueError("Invalid Optimizer Name.")
    def set_scheduler(
        self,
        name,
        last_epoch = -1,
        verbose=False,
        lr_lambda = None,
        gamma = 0.1,
        
    ):
        
        if name in ["lambda", "Lambda", "lambda_lr", "lambdaLR", "LambdaLR"]:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt,
                lr_lambda,
                last_epoch=last_epoch,
                verbose=verbose
            )
        elif name in ["exp","exponential","exponential_lr","ExponentialLR"]:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.opt,
                gamma = gamma,
                last_epoch=last_epoch,
                verbose=verbose
            )
        
        # ToDo :: append more scheduler
        return self.scheduler
    
    def zero_grad(self):
        self.opt.zero_grad()
    def step(self):
        self.opt.step()