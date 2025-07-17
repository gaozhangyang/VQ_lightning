import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch.optim.lr_scheduler as lrs
import inspect


class MInterface_base(pl.LightningModule):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)
        
    def forward(self, input):
        pass
    
    
    def training_step(self, batch, batch_idx, **kwargs):
        pass


    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
    
    def get_schedular(self, optimizer, lr_scheduler='onecycle'):
        if lr_scheduler == 'step':
            scheduler = lrs.StepLR(optimizer,
                                    step_size=self.hparams.lr_decay_steps,
                                    gamma=self.hparams.lr_decay_rate)
        elif lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(optimizer,
                                                T_max=self.hparams.lr_decay_steps)
        elif lr_scheduler == 'onecycle':
            scheduler = lrs.OneCycleLR(optimizer, max_lr=self.hparams.lr, T_max=self.hparams.lr_decay_steps, three_phase=False)
        elif lr_scheduler == 'plateau':
            scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        else:
            raise ValueError('Invalid lr_scheduler type!')

        return scheduler

    def configure_optimizers(self):
        # 获取超参数，使用 getattr 以防止未设置时报错
        weight_decay_g = getattr(self.hparams, "weight_decay", 0.0)
        weight_decay_d = getattr(self.hparams, "disc_weight_decay", 0.0)
        lr_g = getattr(self.hparams, "lr", 1e-4)
        lr_d = getattr(self.hparams, "disc_lr", 1e-4)
        beta1 = getattr(self.hparams, "beta1", 0.9)
        beta2 = getattr(self.hparams, "beta2", 0.999)
        # scheduler_cfg = getattr(self.hparams, "lr_scheduler", None)
        
        # lr_g = lr_g * self.hparams.batch_size / 128
        # lr_d = lr_d * self.hparams.batch_size / 128

        # 定义优化器
        optimizer_g = torch.optim.AdamW(
            self.model.parameters(), lr=lr_g,
            betas=(beta1, beta2), weight_decay=weight_decay_g
        )
        optimizer_d = torch.optim.AdamW(
            self.vq_loss.discriminator.parameters(), lr=lr_d,
            betas=(beta1, beta2), weight_decay=weight_decay_d
        )

        # # 定义 scheduler
        # scheduler_g = self.get_schedular(optimizer_g, scheduler_cfg)
        # scheduler_d = self.get_schedular(optimizer_d, scheduler_cfg)
        
        # from timm.scheduler import create_scheduler_v2 as create_scheduler
        
        # self.vqvae_lr_scheduler, _ = create_scheduler(
        #     sched=self.hparams.lr_scheduler,
        #     optimizer=optimizer_g,
        #     patience_epochs=0,
        #     step_on_epochs=True,
        #     updates_per_epoch=self.hparams.steps_per_epoch,
        #     num_epochs=self.hparams.epochs,
        #     warmup_epochs=1,
        #     min_lr=5e-5,
        # )
        # self.disc_lr_scheduler, _ = create_scheduler(
        #     sched=self.hparams.lr_scheduler,
        #     optimizer=optimizer_d,
        #     patience_epochs=0,
        #     step_on_epochs=True,
        #     updates_per_epoch=self.hparams.steps_per_epoch,
        #     num_epochs=self.hparams.epochs - self.hparams.disc_epoch_start,
        #     warmup_epochs=int(0.02 * self.hparams.epochs),
        #     min_lr=5e-5,
        # )
        return [optimizer_g, optimizer_d]

        # # 返回优化器与调度器
        # return  (
        #             [optimizer_g, optimizer_d],
        #             [
        #                 {"scheduler": scheduler_g, "interval": "step"},
        #                 {"scheduler": scheduler_d, "interval": "step"},
        #             ],
        #         )

        
        
    def lr_scheduler_step(self, *args, **kwargs):
        scheduler = self.lr_schedulers()
        if self.hparams.lr_scheduler != 'plateau':
            scheduler.step()
        
    
    def configure_devices(self):
        self.device = torch.device(self.hparams.device)

    def configure_loss(self):
        self.loss_function = nn.CrossEntropyLoss(reduction='none')
        
    def load_model(self):
        self.model = None

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
