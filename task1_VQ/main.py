import datetime
import os
import sys; sys.path.append("/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark")
os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0" 
# os.environ["WANDB_API_KEY"] = "d4aae42367c9842a7ddfdf29258565305fdd5496" 

import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
from pytorch_lightning.trainer import Trainer
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from model_interface import MInterface
from data_interface import DInterface
import pytorch_lightning.loggers as plog
from omegaconf import DictConfig, OmegaConf
import hydra
from src.utils.logger import SetupCallback
from src.utils.utils import flatten_dict
from pytorch_lightning.strategies import DDPStrategy
import math
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from hydra import initialize, compose

def eval_resolver(expr: str):
    return eval(expr, {}, {})

OmegaConf.register_new_resolver("eval", eval_resolver, use_cache=False)

def create_parser():
    parser = argparse.ArgumentParser()
    
    # Set-up parameters
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument('--offline', default=1, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seq_len', default=1022, type=int)
    parser.add_argument('--gpus_per_node', default=2, type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    
    # Training parameters
    parser.add_argument('--epoch', default=50, type=int, help='end epoch')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--disc-weight-decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument('--lr_scheduler', default='cosine')
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--v-patch-nums", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16], nargs='+', help="number of patch numbers of each scale")
    parser.add_argument("--enc_type", type=str, default="cnn")
    parser.add_argument("--dec_type", type=str, default="cnn")
    parser.add_argument("--encoder_model", type=str, default='vit_small_patch14_dinov2.lvd142m',
                        help='encoder model name')
    parser.add_argument("--decoder_model", type=str, default='vit_small_patch14_dinov2.lvd142m',
                        help='encoder model name')
    parser.add_argument("--semantic_guide", type=str, default="none")
    parser.add_argument("--detail_guide", type=str, default="none")
    parser.add_argument("--num_latent_tokens", type=int, default=256)
    parser.add_argument("--abs_pos_embed", type=bool, default=False)
    parser.add_argument("--product_quant", type=int, default=1)
    parser.add_argument("--share_quant_resi", type=int, default=4)
    parser.add_argument("--codebook_drop", type=float, default=0.0)
    parser.add_argument("--half_sem", type=bool, default=False)
    parser.add_argument("--start_drop", type=int, default=1)
    parser.add_argument("--lecam_loss_weight", type=float, default=None)
    parser.add_argument("--sem_loss_weight", type=float, default=0.1)
    parser.add_argument("--detail_loss_weight", type=float, default=0.1)
    parser.add_argument("--clip_norm", type=bool, default=False)
    parser.add_argument("--sem_loss_scale", type=float, default=1.0)
    parser.add_argument("--detail_loss_scale", type=float, default=1.0)
    parser.add_argument("--guide_type_1", type=str, default='class', choices=["patch", "class"])
    parser.add_argument("--guide_type_2", type=str, default='class', choices=["patch", "class"])
    parser.add_argument("--lfq", action='store_true', default=False, help="if use LFQ")
    
    parser.add_argument("--disc-start", type=int, default=0, help="iteration to start discriminator training and loss") 
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")   
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--disc_adaptive_weight", type=bool, default=False)
    parser.add_argument("--norm_type", type=str, default='bn')
    parser.add_argument("--aug_prob", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--aug_fade_steps", type=int, default=0)

    
    # Model parameters
    parser.add_argument("--config_name", type=str, default='VQ-4096', help="Name of the Hydra config to use")
    parser.add_argument("--vq_type", type=str, default='vvq')
    
    # ----------------------------------------------------------------------------
    # 1) 先不看命令行，拿到纯“parser 默认值”：
    defaults = parser.parse_args([])
    defaults_dict = vars(defaults)

    # 2) 真正去解析一次命令行（CLI + 默认）：
    args = parser.parse_args()
    args_dict = vars(args)

    # 3) 再读你的 Hydra config，平展开成普通 dict：
    with initialize(config_path="configs"):
        cfg = compose(config_name=args.config_name)
    config_dict = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # ----------------------------------------------------------------------------
    # 4) 挖出哪些 key 的值是“真由用户在命令行里指定”的：
    passed = set()
    for tok in sys.argv[1:]:
        if not tok.startswith('--'):
            continue
        # 支持 --foo=bar 和 --foo bar 两种写法
        key = tok.lstrip('-').split('=')[0].replace('-', '_')
        passed.add(key)

    # 5) 最终合并：CLI > config_file > parser_default
    merged = {}
    for key in set(list(defaults_dict.keys())+list(config_dict.keys())):
        if key in passed:
            # 用户显式传进来的
            merged[key] = args_dict[key]
        elif key in config_dict:
            # config 文件里有，且用户没在 CLI 指定，就用它
            merged[key] = config_dict[key]
        else:
            # 都没有指定，就退回 parser 默认
            merged[key] = defaults_dict[key]

    # 用合并后的结果更新 args Namespace
    args.__dict__.update(merged)

    print(args)
    return args


def load_callbacks(args):
    callbacks = []
    
    logdir = str(os.path.join(args.res_dir, args.ex_name))
    
    ckptdir = os.path.join(logdir, "checkpoints")
    

    metric = "val_FID"
    sv_filename = 'best-{epoch:02d}-{val_FID:.4f}'
    callbacks.append(plc.ModelCheckpoint(
        monitor=metric,
        filename=sv_filename,
        save_top_k=5,
        mode='min',
        save_last=True,
        dirpath = ckptdir,
        verbose = True,
        # every_n_train_steps=args.check_val_every_n_step
        every_n_epochs = args.check_val_every_n_epoch,
    ))

    
    now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    cfgdir = os.path.join(logdir, "configs")
    callbacks.append(
        SetupCallback(
                now = now,
                logdir = logdir,
                ckptdir = ckptdir,
                cfgdir = cfgdir,
                config = args.__dict__,
                argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],)
    )
    
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))
    return callbacks

@rank_zero_only
def init_wandb(args):
    os.makedirs(os.path.join(args.res_dir, args.ex_name), exist_ok=True)
    # 你想要的 group 名，可以是实验名也可以自定义
    run = wandb.init(
        project='VQ',
        entity='biomap_ai',
        group=args.ex_name,     # ← 所有卡都能看到同一个 group
        dir=os.path.join(args.res_dir, args.ex_name),
        config=vars(args),
        reinit=True,
        # id=args.ex_name,
        name=args.ex_name,
    )

    # 把同一个 run 传给 Lightning 的 logger
    return WandbLogger(
        project='VQ',
        name=args.ex_name,
        group=args.ex_name,     # ← Lightning 端也打同样的 group
        save_dir=os.path.join(args.res_dir, args.ex_name),
        dir=os.path.join(args.res_dir, args.ex_name),
        offline=args.offline,
        entity='biomap_ai',
        log_model=False,         # 可选
        # id=args.ex_name,
    )

def main():
    args = create_parser()
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb_logger = init_wandb(args)
    
    
    pl.seed_everything(args.seed)
    data_module = DInterface(**vars(args))
    data_module.data_setup()
    gpu_count = torch.cuda.device_count()
    steps_per_epoch = len(data_module.train_dataloader())
    args.steps_per_epoch = steps_per_epoch
    # steps_per_epoch = math.ceil(len(data_module.train_set)/args.batch_size/gpu_count)
    # args.lr_decay_steps =  steps_per_epoch*args.epoch
    
    model = MInterface(**vars(args))

    data_module.MInterface = model
    callbacks = load_callbacks(args)
    trainer_config = {
        "accelerator": "gpu",
        'devices': gpu_count,  # Use all available GPUs
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': args.num_nodes,  # Number of nodes to use for distributed training
        "strategy": DDPStrategy(find_unused_parameters=True), # 'ddp', 'deepspeed_stage_2
        "precision": 'bf16', # "bf16", 16
        'accelerator': 'gpu',  # Use distributed data parallel
        'callbacks': callbacks,
        'logger': wandb_logger,
    }

    trainer_opt = argparse.Namespace(**trainer_config)
    
    trainer = Trainer(**vars(trainer_opt))

    trainer.fit(model, data_module)

    # ============================
    # 4. 评估最佳模型
    # ============================
    checkpoint_callback = callbacks[0]
    print(f"Best model path: {checkpoint_callback.best_model_path}")

    # 载入最佳模型
    model_state_path = os.path.join(checkpoint_callback.best_model_path, "checkpoint", "mp_rank_00_model_states.pt")
    state = torch.load(model_state_path, map_location="cuda:0")
    model.load_state_dict(state['module'])

    # 进行测试
    results = trainer.test(model, datamodule=data_module)
    # 打印测试结果
    print(f"Test Results: {results}")


if __name__ == "__main__":
    main()
    
