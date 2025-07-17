import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from scipy import linalg

class NpzDataset(Dataset):
    def __init__(self, source, arr_name='arr_0', transform=None):
        if isinstance(source, str):
            obj = np.load(source)
            self.data = obj[arr_name]
        else:
            self.data = source
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.uint8)  # shape: (H, W, 3), uint8
        if self.transform:
            img = self.transform(img)          # -> (3, 299, 299), float32
        return img

class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

class PytorchFIDEvaluator:
    def __init__(self, device='cuda', batch_size=64, num_workers=4, weights_path=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size  = batch_size
        self.num_workers = num_workers

        # 1) 加载 InceptionV3（aux_logits=True 用于匹配权重结构）
        if weights_path:
            inception = inception_v3(pretrained=False, aux_logits=True)
            state = torch.load(weights_path, map_location='cpu')
            inception.load_state_dict(state)
        else:
            weights   = Inception_V3_Weights.IMAGENET1K_V1
            inception = inception_v3(weights=weights, aux_logits=True)

        inception.to(self.device).eval()

        # 2) 创建特征提取器，只要 avgpool 层（输出 [N, 2048, 1, 1]）
        self.extractor = create_feature_extractor(
            inception, return_nodes={'avgpool': 'feat'}
        ).to(self.device).eval()

        # 3) 预处理：NHWC [0,255] -> CHW [0,1], resize + normalize
        self.transform = transforms.Compose([
            transforms.ToTensor(),                 # (H,W,3,uint8) -> (3,H,W,float)
            transforms.Resize((299, 299)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    def _get_activations(self, loader: DataLoader) -> np.ndarray:
        acts = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)         # shape [B,3,299,299]
                out   = self.extractor(batch)['feat'] # [B,2048,1,1]
                feat  = out.view(out.size(0), -1)    # [B,2048]
                acts.append(feat.cpu().numpy())
        return np.concatenate(acts, axis=0)

    def read_activations(self, source) -> np.ndarray:
        """
        source: .npz 文件路径（含 arr_name='arr_0'）或 numpy 数组，形状 (N,H,W,3)
        返回: 特征矩阵，shape (N, 2048)
        """
        ds = NpzDataset(source, transform=self.transform)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return self._get_activations(loader)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu    = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)
