import os
import numpy as np
import torch
import torch.nn.functional as F
from src.interface.model_interface import MInterface_base
from src.model.xqgan_model import VQ_models
from src.model.vq_loss import VQLoss
from torchvision.utils import make_grid
from PIL import Image
import wandb
import torch.distributed as dist
from tqdm import tqdm
from src.model.evaluator_torch import PytorchFIDEvaluator
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.load_model()
        self._context = {
            "validation": {
                "real_feats": [],
                "fake_feats": [],
                "real_imgs": None,
                "fake_imgs": None,
            }
        }
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)
        self.fid_extractor = FrechetInceptionDistance(feature=2048, normalize=False, feature_extractor_weights_path='/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/BSQ/data/fid_weight/weights-inception-2015-12-05-6726825d.pth')

    def forward(self, batch, optimizer_idx):
        x, y = batch
        global_step = self.global_step
        if self.hparams.aug_fade_steps >= 0:
            fade_blur_schedule = 0 if global_step < self.hparams.disc_start else min(1.0, (global_step - self.hparams.disc_start) / (self.hparams.aug_fade_steps + 1))
            fade_blur_schedule = 1 - fade_blur_schedule
        else:
            fade_blur_schedule = 0
            
        results = self.model(x, self.current_epoch)
        
        
        loss_gen = self.vq_loss(results, 
                                results['sem_loss'], 
                                results['detail_loss'], 
                                results['dependency_loss'], 
                                x,       
                                results['recons_imgs'], 
                                optimizer_idx=optimizer_idx, global_step=global_step,
                                last_layer=self.model.decoder.last_layer, 
                                log_every=self.hparams.log_every, fade_blur_schedule=fade_blur_schedule,
                                log_data_func = self.log_data_func)
        return {'loss': loss_gen, 'recons_imgs': results['recons_imgs'], 'gt': x, 'codebook_usage': results['codebook_usage'], 'codebook_usage_var': results['codebook_usage_var'],'semantic_loss': results['sem_loss'], 'detail_loss': results['detail_loss'], 'dependency_loss': results['dependency_loss']}

    def training_step(self, batch, batch_idx, **kwargs):
        optimizer_g, optimizer_d = self.optimizers()
        
        # =============train generator===========
        self.toggle_optimizer(optimizer_g)
        g_results = self(batch, optimizer_idx=0)
        g_loss = g_results['loss']
        
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        
        
        # =============train discriminator===========
        self.toggle_optimizer(optimizer_d)
        d_results  = self(batch, optimizer_idx=1)
        d_loss = d_results['loss']
        
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.validation_batch = batch
        ret = self(batch, optimizer_idx=0)
        loss = ret['loss']
        sample = ret['recons_imgs']
        x = ret['gt']
        sample = (((torch.clamp(sample, -1,1)+1)/2)*255).byte()
        x = (((torch.clamp(x, -1,1)+1)/2)*255).byte()
        
        transform = transforms.Resize((299, 299))
        real_feats = self.fid_extractor.inception(transform(sample))   # [B, 2048]
        fake_feats = self.fid_extractor.inception(transform(x))  
                            
        self._context["validation"]["real_feats"].append(real_feats)
        self._context["validation"]["fake_feats"].append(fake_feats)
        self._context["validation"]["real_imgs"] = x
        self._context["validation"]["fake_imgs"] = sample
        
        log_dict = {'val_loss': loss}
        self.log_dict(log_dict, rank_zero_only=True)
        return self.log_dict
    
    @rank_zero_only
    def log_image(self, image, step=None):
        self.logger.experiment.log({"image": wandb.Image(image)}, step=step)
    
    @rank_zero_only
    def log_data_func(self, data, step=None):
        self.logger.experiment.log(data)

    @torch.no_grad()
    def on_validation_epoch_end(self):
        # ======== Log validation images =========
        gt, samples = self._context["validation"]["real_imgs"][:4], self._context["validation"]["fake_imgs"][:4]
        image = torch.cat([samples, gt], dim=0) #/255
        grid = make_grid(image, nrow=4, padding=0, pad_value=255)
        # 转为 [H, W, C] 并强制类型为 uint8
        grid = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # image = image.mul_(255).cpu().numpy()
        image = Image.fromarray(grid)

        self.log_image(image, step=self.global_step)

        
        real_feats = torch.cat(self._context["validation"]["real_feats"], dim=0)
        fake_feats = torch.cat(self._context["validation"]["fake_feats"], dim=0)
        real_feats = torch.cat(dist.nn.all_gather(real_feats), dim=0)
        fake_feats = torch.cat(dist.nn.all_gather(fake_feats), dim=0)
        dist.barrier()
        FID = calculate_fid(real_feats, fake_feats)

        self.log_dict({"val_FID": FID}, prog_bar=True, rank_zero_only=True)
        
        self._context["validation"]["real_imgs"] = None
        self._context["validation"]["fake_imgs"] = None
        self._context["validation"]["real_feats"] = []
        self._context["validation"]["fake_feats"] = []

        
        
    def load_model(self):
        self.model = VQ_models[self.hparams.vq_model](
            codebook_size=self.hparams.codebook_size,
            codebook_embed_dim=self.hparams.codebook_embed_dim,
            commit_loss_beta=self.hparams.commit_loss_beta,
            entropy_loss_ratio=self.hparams.entropy_loss_ratio,
            dropout_p=self.hparams.dropout_p,
            v_patch_nums=self.hparams.v_patch_nums,
            enc_type=self.hparams.enc_type,
            encoder_model=self.hparams.encoder_model,
            dec_type=self.hparams.dec_type,
            decoder_model=self.hparams.decoder_model,
            semantic_guide=self.hparams.semantic_guide,
            detail_guide=self.hparams.detail_guide,
            num_latent_tokens=self.hparams.num_latent_tokens,
            abs_pos_embed=self.hparams.abs_pos_embed,
            share_quant_resi=self.hparams.share_quant_resi,
            product_quant=self.hparams.product_quant,
            codebook_drop=self.hparams.codebook_drop,
            half_sem=self.hparams.half_sem,
            start_drop=self.hparams.start_drop,
            sem_loss_weight=self.hparams.sem_loss_weight,
            detail_loss_weight=self.hparams.detail_loss_weight,
            clip_norm=self.hparams.clip_norm,
            sem_loss_scale=self.hparams.sem_loss_scale,
            detail_loss_scale=self.hparams.detail_loss_scale,
            guide_type_1=self.hparams.guide_type_1,
            guide_type_2=self.hparams.guide_type_2,
            vq_type=self.hparams.vq_type,
            image_size=self.hparams.image_size,
        )
        
        
        
        
        self.vq_loss = VQLoss(
            disc_start=self.hparams.disc_start, 
            disc_weight=self.hparams.disc_weight,
            disc_type=self.hparams.disc_type,
            disc_loss=self.hparams.disc_loss,
            gen_adv_loss=self.hparams.gen_loss,
            image_size=self.hparams.image_size,
            perceptual_weight=self.hparams.perceptual_weight,
            reconstruction_weight=self.hparams.reconstruction_weight,
            reconstruction_loss=self.hparams.reconstruction_loss,
            codebook_weight=self.hparams.codebook_weight,
            lecam_loss_weight=self.hparams.lecam_loss_weight,
            disc_adaptive_weight=self.hparams.disc_adaptive_weight,
            norm_type=self.hparams.norm_type,
            aug_prob=self.hparams.aug_prob,
        )
    


def calculate_fid(real_feats: torch.Tensor,
                  fake_feats: torch.Tensor,
                  eps: float = 1e-6) -> float:
    """
    Compute Frechet Inception Distance between two sets of features.
    real_feats, fake_feats: [N, D], [M, D] float32/64 Tensors on the same device.
    """

    # 1) 直接计算均值
    #    unsqueeze(0) 保持和原来代码兼容的 [1, D] 维度：
    mean_real = real_feats.mean(dim=0, keepdim=True)   # [1, D]
    mean_fake = fake_feats.mean(dim=0, keepdim=True)   # [1, D]

    # 2) 计算协方差：（与原来除以 N-1 保持一致 = 无偏估计）
    N = real_feats.shape[0]
    M = fake_feats.shape[0]

    # 去中心化
    real_centered = real_feats - mean_real    # [N, D]
    fake_centered = fake_feats - mean_fake    # [M, D]

    # 样本协方差矩阵
    cov_real = (real_centered.T @ real_centered) / (N - 1)   # [D, D]
    cov_fake = (fake_centered.T @ fake_centered) / (M - 1)   # [D, D]
    fid = _compute_fid(
        mean_real.squeeze(0).float(),
        cov_real.float(),
        mean_fake.squeeze(0).float(),
        cov_fake.float()
    )
    return fid

def _compute_fid(mu1, sigma1, mu2, sigma2):
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c

