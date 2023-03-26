from loss.loss_zoo.mse_loss import LrLoss
from mindspore.ops import value_and_grad
from mindspore import nn, ops
from mindspore import ParameterTuple

mse_loss = LrLoss()


def train_epoch(engine, pretrain=True):
    engine.model.set_train()

    def forward_fn(views, miss_matrixs):
        out, fusion2recon, latents, fusion_latent = engine.model(views, miss_matrixs)
        output = {'preds': out, 
                  'reals': views, 
                  'fusion2recon': fusion2recon, 
                  'miss_matrixs': miss_matrixs,
                  'latents': latents,
                  'fusion_latent': fusion_latent}
        loss_dict = mse_loss(**output)
        if not pretrain:
            loss_dict = engine.loss_func(output)

        loss = 0
        for k, v in loss_dict.items():
            loss += v
        return loss, output

    grad_fn = value_and_grad(forward_fn, None, engine.optimizer.parameters, has_aux=True)

    def train_step(views, miss_matrixs):
        (loss, output), grads = grad_fn(views, miss_matrixs)
        engine.optimizer(grads)
        return loss, output
    

    for batch, data in enumerate(engine.dataloader):
        inputs = [data[0], data[1]]
        targets = data[2]
        miss_matrixs = [data[3], data[4]]
        loss, output = train_step(inputs, miss_matrixs)
        
        return loss, output, targets
        