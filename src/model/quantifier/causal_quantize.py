import torch
import torch.nn as nn
import numpy as np
import torch.distributed as tdist

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper to rotate tensor halves for RoPE:
    Split last dim into (d/2, 2), then swap and negate.
    """
    *shape, d = x.shape
    x = x.view(*shape, d // 2, 2)
    # x: [..., d/2, 2]
    x = torch.stack((-x[..., 1], x[..., 0]), dim=-1)  # [..., d/2, 2]
    return x.flatten(-2)


class RotaryEmbedding(nn.Module):
    """
    Fixed rotary positional embeddings (RoPE), batch-first:
    Input shape: [batch, seq_len, dim]
    """
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.einsum("n,d->nd", positions, inv_freq)
        cos = torch.cos(sinusoid_inp)
        sin = torch.sin(sinusoid_inp)
        # Interleave to shape [max_seq_len, dim]
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        # Register buffers
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to tensor x of shape [batch, seq_len, dim].
        """
        batch, seq_len, dim = x.shape
        # [seq_len, dim] -> [1, seq_len, dim]
        cos = self.cos[:seq_len].unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0)
        # Broadcast and apply
        return x * cos + rotate_half(x) * sin


class CausalTransformerRoPE(nn.Module):
    """
    Simple causal transformer with RoPE, batch-first, input/output dim=2.
    """
    def __init__(
        self,
        d_model: int = 2,
        nhead: int = 1,
        num_layers: int = 1,
        dim_feedforward: int = 8,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        # Rotary positional embedding
        self.rotary = RotaryEmbedding(d_model, max_seq_len)
        # Attention + FFN stacks
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model),
            )
            for _ in range(num_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        # Final projection to output dim=2
        self.in_proj = nn.Linear(2, d_model)
        self.out_proj = nn.Linear(d_model, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [batch, seq_len, 2]
        returns: Tensor of shape [batch, seq_len, 2]
        """
        x = self.in_proj(x)
        batch, seq_len, _ = x.size()
        # Build causal mask: shape [seq_len, seq_len]
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )
        h = x
        for attn, ffn, ln1, ln2 in zip(
            self.attn_layers,
            self.ffn_layers,
            self.norm1,
            self.norm2
        ):
            # Apply RoPE to Q and K
            rot_h = self.rotary(h)
            attn_out, _ = attn(rot_h, rot_h, h, attn_mask=mask)
            h = ln1(h + attn_out)
            ffn_out = ffn(h)
            h = ln2(h + ffn_out)
        # Project back to output dimension
        return self.out_proj(h)

def compute_bin_indices(x, boundaries):
    """
    x: [N, 1]
    boundaries: [N, K-1] (not necessarily sorted)
    Returns: bin_idx: [N], value in 0..K-1, or -1 if x doesn't fall into any valid bin
    """
    device = x.device
    N, k_minus_1 = boundaries.shape
    K = k_minus_1 + 1
    MIN, MAX = x.min(), x.max()
    ONES = torch.ones(N, 1, device=device, dtype=x.dtype)
    x_quant = torch.zeros_like(x)

    # Step 1: build left and right edges for all K bins
    left = torch.cat([MIN*ONES-0.01, boundaries], dim=1)  # shape [N, K]
    right = torch.cat([boundaries, MAX*ONES+0.01], dim=1)  # shape [N, K]
    mid = (left + right) / 2  # shape [N, K]

    # Step 2: create a mask of shape [N, K] indicating where x falls into a valid bin
    # x in [left, right), and right > left (valid bin)
    x_expanded = x.expand(-1, K)  # [N, K]
    in_bin = (x_expanded >= left) & (x_expanded < right)
    valid_bin = right > left
    final_mask = in_bin & valid_bin  # [N, K]

    # Step 3: get the index of the bin
    bin_idx = torch.full((N,), -1, dtype=torch.long, device=device)  # default: -1 for no match
    matched = final_mask.float().argmax(dim=1)  # get first matching bin
    has_match = final_mask.any(dim=1)
    bin_idx[has_match] = matched[has_match]
    x_quant[has_match,0] = mid[has_match, matched[has_match]]  # assign mid value to x_quant

    return x_quant, bin_idx

def group_bincount(bin_ids: torch.Tensor, group_ids: torch.Tensor):
    unique_groups = torch.unique(group_ids)
    group_keys = []
    result_keys = []
    result_counts = []

    for g in unique_groups:
        mask = (group_ids == g)
        bins = bin_ids[mask]
        keys, counts = torch.unique(bins, return_counts=True)
        group_keys.append(g*torch.ones_like(keys))
        result_keys.append(keys)
        result_counts.append(counts)

    return torch.cat(group_keys).long(), torch.cat(result_keys).long(), torch.cat(result_counts).float()


def calculate_quantization_ratios(x, thresholds, prev_global_idx, ratios):
    """
    计算 x 中的数据在每个量化区间的占比。

    参数:
        x (torch.Tensor): 一维张量，形状 (N,) 表示数据。
        thresholds (torch.Tensor): 量化区间边界，形状 (T,) 表示 T 个阈值。

    返回:
        ratios (torch.Tensor): 各量化区间数据的占比 (T+1,)。
        counts (torch.Tensor): 各量化区间的数据数量。
    """
    
    x_quant, bin_ids = compute_bin_indices(x, thresholds)
    group_keys, value_keys, counts = group_bincount(bin_ids, prev_global_idx)
    ratios[group_keys, value_keys] = counts
    return x_quant, bin_ids[:,None], ratios


class CQ(nn.Module):
    def __init__(self, vocab_size, z_channels=32):
        super(CQ, self).__init__()
        
        if vocab_size==4096:
            levels = [8,8,8,8]
        
        self.vocab_size = vocab_size 
        self.z_channels = z_channels
        self.code_channels = len(levels)
        self._levels = levels
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(([1], np.cumprod(self._levels_np[:-1]))).astype(np.int64)
        self.condition_model = CausalTransformerRoPE(d_model = 256,
                                                    nhead = 8,
                                                    num_layers = 3,
                                                    dim_feedforward = 512,
                                                    max_seq_len = 16)
        self.proj_prev = nn.Linear(z_channels, self.code_channels)
        self.proj_post = nn.Linear(self.code_channels, z_channels)
        self.register_buffer("ema_vocab_hit_SV", torch.zeros(4096))  
        self.register_buffer("ratios", torch.zeros(torch.prod(torch.tensor(levels)), 8)) 
        self.record_hit = 0
            

    def forward(self, x, ret_usages=True, dropout=0.0):
        B, _, H, W = x.shape
        x = self.proj_prev(x.permute(0, 2, 3, 1).reshape(B,H*W,-1))  # B C H W -> B H W C
        uniform_loss = 0
        B, L, C = x.shape  # 16 10 128

        x = x.reshape(-1, C)  # 160 128  B*h*w C
        x=torch.tanh(x)
        self.latent = x.reshape(-1, x.shape[-1])
        x_list = torch.split(x, 1, dim=1)
        x_historical = []
        z_list = []
        q_list = []
        code_list = []
        code_i_prev = None
        for i, xi in enumerate(x_list):
            if code_i_prev is None:
                code_i_prev = torch.zeros_like(xi)
                code_list.append(code_i_prev)
            x_historical.append( torch.cat([code_i_prev, xi], dim=-1) )
            z_out = self.condition_model(torch.stack(x_historical, dim=1))
            z_pred = z_out[:,-1, 0:1]
            
            
            # ===========update thresholds===========
            boundary = z_out[:,-1, 1:self._levels[i]]
            prev_sub_index = torch.cat(code_list,dim=-1)
            num_prev = prev_sub_index.shape[1]
            prev_levels = [1]+self._levels
            prev_global_idx = self.subspace_to_joint_index(prev_sub_index, prev_levels[:num_prev])
            q_i, code_i_prev, ratios = calculate_quantization_ratios(z_pred, boundary, prev_global_idx, self.ratios)
            loss_order = -torch.clamp(boundary.diff(dim=-1), -9999999, 0.1).mean()
            gradient = -torch.diff(ratios[prev_global_idx, :self._levels[i]],dim=-1)
            gradient = gradient/torch.norm(gradient, dim=-1, keepdim=True)
            uniform_loss += (boundary * gradient).sum(dim=-1).mean()+loss_order
            # =======================================
            
            z_list.append(z_pred)
            q_list.append(q_i)
            code_list.append(code_i_prev)
       
        z = torch.cat(z_list, dim=1)
        q = torch.cat(q_list, dim=1)
        code = torch.cat(code_list, dim=1)[:,1:]
        out = q.detach() + z - z.detach()
        min_encoding_indices = self.subspace_to_joint_index(code, self._levels)
        
        vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=z.device)
        codebook_usage, codebook_usage_var = 0, 0
        if ret_usages and self.training:
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

            margin = tdist.get_world_size() * (z.numel() / self.z_channels) / self.vocab_size * 0.08

            codebook_usage = (self.ema_vocab_hit_SV >= margin).float().mean().item() * 100
            codebook_usage_var = torch.var(self.ema_vocab_hit_SV/self.ema_vocab_hit_SV.sum(), unbiased=False).item()
            
        vq_loss = torch.mean((q - z) ** 2)
        
        out = out.reshape(B, L, -1)
        out = self.proj_post(out)
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)  # B C L -> B L C -> B C H W
        z_q=out

        return z_q, codebook_usage, codebook_usage_var, vq_loss, uniform_loss, 0

    def subspace_to_joint_index(self, subspace_idx: torch.Tensor, dims: list[int]) -> torch.Tensor:
        """
        将 k 维子空间索引转换为联合索引（flattened index）。

        参数：
            subspace_idx: shape [..., k] 的张量，每一行是一个 k 维的子空间索引。
            dims: 每个维度的状态数量列表，例如 [n1, n2, ..., nk]。

        返回：
            联合索引（flattened index），shape 为 subspace_idx.shape[:-1]
        """
        dims = torch.tensor(dims, dtype=torch.long, device=subspace_idx.device)
        strides = torch.cumprod(torch.cat([torch.tensor([1], device=subspace_idx.device), dims[:-1]]), dim=0)
        joint_idx = (subspace_idx * strides).sum(dim=-1)
        return joint_idx.long()
    
    def joint_to_subspace_index(self, joint_idx: torch.Tensor, dims: list[int]) -> torch.Tensor:
        """
        将联合索引（flattened index）转换为 k 维子空间索引。

        参数：
            joint_idx: shape [...] 的张量，每个元素是 flatten 后的索引。
            dims: 每个维度的状态数量列表，例如 [n1, n2, ..., nk]。

        返回：
            shape [..., k] 的子空间索引张量。
        """
        joint_idx = joint_idx.clone()
        dims = torch.tensor(dims, dtype=torch.long, device=joint_idx.device)
        strides = torch.cumprod(torch.cat([torch.tensor([1], device=joint_idx.device), dims[:-1]]), dim=0)

        subspace_idx = []
        for stride, dim in zip(reversed(strides), reversed(dims)):
            val = (joint_idx // stride) % dim
            subspace_idx.append(val)
        subspace_idx = torch.stack(list(reversed(subspace_idx)), dim=-1)
        return subspace_idx.long()

