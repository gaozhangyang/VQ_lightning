import torch
from torch import distributed as tdist, nn as nn
import numpy as np

def random_rotation_bf16_householder(d, batch_size=1, device=None):
    # 随机向量
    v = torch.randn(batch_size, d, 1, dtype=torch.bfloat16, device=device)
    # 归一化
    v_norm = torch.linalg.norm(v, dim=1, keepdim=True)
    u = v / v_norm
    # 构造 I
    I = torch.eye(d, dtype=torch.bfloat16, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    # Householder 反射矩阵 H = I - 2 u u^T
    H = I - 2 * (u @ u.transpose(1, 2))
    # H 本身即是正交矩阵 det(H)=−1；如需 det=+1，可再乘以一个 180° 反射
    # 这里为了示例直接返回 H
    return H

def random_rotation_matrices(d: int,
                             batch_size: int = 1,
                             *,
                             dtype: torch.dtype = torch.float32,
                             device: torch.device = None) -> torch.Tensor:
    """
    生成批量的 d×d 随机旋转矩阵（正交且 det=+1）。

    参数:
        d (int): 空间维度。
        batch_size (int): 要生成的矩阵个数 B。
        dtype, device: 返回张量的数据类型和设备。

    返回:
        Tensor: 形状 (B, d, d) 的旋转矩阵批。
    """
    # 1. 随机矩阵
    A = torch.randn(batch_size, d, d, dtype=dtype, device=device).float()
    # 2. QR 分解
    #    torch.linalg.qr 默认返回 Q, R
    Q, R = torch.linalg.qr(A)
    # 3. 修正行列式符号，保证 det(Q') == +1
    #    计算每个 Q 的行列式符号
    det = torch.linalg.det(Q)
    #    det.sign() 形状 (B,), 要扩展到 (B,1,1)
    sign = det.sign().view(batch_size, 1, 1)
    Q = Q * sign
    return Q.to(dtype)

def round_ste(z):
    """Round with straight through gradients."""
    # R = random_rotation_matrices(z.shape[-1], z.shape[0], dtype=z.dtype, device=z.device)
    # z_ = (z[:,None]@R)[:,0]
    # zhat = torch.round(z_)
    # zhat = (zhat[:,None]@R.transpose(1, 2))[:,0] 
    zhat = torch.round(z)
    return z + (zhat - z).detach()

class RotFSQ(nn.Module):
    """Quantizer."""

    def __init__(self, vocab_size=4096, embedding_dim=6, eps: float = 1e-3):
        super(RotFSQ, self).__init__()
        self.vocab_size = vocab_size
        if vocab_size == 4096:
            levels=[8,8,8,8]
        vq_dim = len(levels)
        self.vq_dim = vq_dim
        self.proj_in = nn.Linear(embedding_dim, vq_dim)
        self.proj_out = nn.Linear(vq_dim, embedding_dim)
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(([1], np.cumprod(self._levels_np[:-1]))).astype(np.int64)  # 修改此处为 np.int64
        self._implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size, dtype=torch.int64))
        self.register_buffer("ema_vocab_hit_SV", torch.full((self.vocab_size,), fill_value=0.0))
        self.record_hit = 0

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return np.prod(self._levels)

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = np.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = np.tan(offset / half_l)

        half_l = torch.tensor(half_l, dtype=z.dtype, device=z.device)
        shift = torch.tensor(shift, dtype=z.dtype, device=z.device)
        offset = torch.tensor(offset, dtype=z.dtype, device=z.device)
        h = torch.tanh(z + shift) * half_l - offset
        return h

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        z = self.bound(z)

        quantized = round_ste(z)
        self.latentf = quantized.reshape(-1, quantized.shape[-1])

        vq_loss = torch.mean((quantized - z) ** 2)
        

        # Renormalize to [-1, 1].
        half_width = self._levels_np // 2
        return quantized / torch.tensor(half_width, dtype=z.dtype, device=z.device), vq_loss

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * torch.tensor(half_width, dtype=zhat_normalized.dtype,
                                               device=zhat_normalized.device)) + torch.tensor(half_width,
                                                                                              dtype=zhat_normalized.dtype,
                                                                                              device=zhat_normalized.device)

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)) / torch.tensor(half_width,
                                                                                                      dtype=zhat.dtype,
                                                                                                      device=zhat.device)

    def codes_to_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * torch.tensor(self._basis, dtype=zhat.dtype, device=zhat.device)).sum(dim=-1).type(
            torch.int64)  # 修改此处为 torch.int64

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `codes_to_indexes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = np.mod(
            np.floor_divide(indices.cpu().numpy(), self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(
            torch.tensor(codes_non_centered, dtype=indices.dtype, device=indices.device))

    def forward(self, h_in, ret_usages=True, dropout=None):
        B, D, W, H = h_in.shape
        h_in = h_in.permute(0, 2, 3, 1).reshape(B * W * H, D)
        vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=h_in.device)
        h = self.proj_in(h_in)
        z_q, vq_loss = self.quantize(h)
        min_encoding_indices = self.codes_to_indexes(z_q)
        z_q = self.proj_out(z_q)
        
        codebook_usage, codebook_usage_var = 0, 0
        if ret_usages and self.training:
            min_encoding_indices = torch.clamp(min_encoding_indices, 0, self.vocab_size - 1).long()
            hit_V = min_encoding_indices.bincount(minlength=self.vocab_size).float()
            handler = tdist.all_reduce(hit_V, async_op=True)
            handler.wait()
            if self.record_hit == 0:
                self.ema_vocab_hit_SV.copy_(hit_V)
            elif self.record_hit < 100:
                self.ema_vocab_hit_SV.mul_(0.9).add_(hit_V.mul(0.1))
            else:
                self.ema_vocab_hit_SV.mul_(0.99).add_(hit_V.mul(0.01))
            self.record_hit += 1
            vocab_hit_V.add_(hit_V)

            margin = tdist.get_world_size() * (z_q.numel() / self.vq_dim) / self.vocab_size * 0.08

            codebook_usage = (self.ema_vocab_hit_SV >= margin).float().mean().item() * 100
            codebook_usage_var = torch.var(self.ema_vocab_hit_SV/self.ema_vocab_hit_SV.sum(), unbiased=False).item()

        z_q = z_q.reshape(B, W, H, D).permute(0, 3, 1, 2)
        return z_q, codebook_usage, codebook_usage_var, vq_loss, 0.25*vq_loss, 0.0

if __name__ == "__main__":
    # 测试代码
    levels = [8, 8, 8, 8]
    embedding_dim = 6
    quantizer = RotFSQ(levels=levels, embedding_dim=embedding_dim)

    # 随机输入
    z = torch.randn(2, embedding_dim)
    z_q, vq_loss, min_encoding_indices = quantizer(z)

    print("Quantized Output:", z_q)
    print("VQ Loss:", vq_loss)
    print("Min Encoding Indices:", min_encoding_indices)