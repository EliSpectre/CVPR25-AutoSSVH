import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import math
import warnings

import numpy as np

from functools import partial
from einops import rearrange

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

	
def __call_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

		
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class Round3(Function):
    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        output = torch.round(input)
        ctx.input = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask = ~(ctx.input==0)
        mask = Variable(mask).cuda().float()
        grad_output = grad_output * mask
        return grad_output, None, None


class DropPath(nn.Module):
    '''
    drop paths (stochastic depth) per sample, (when applied in main path of residual blocks)
    '''
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p = {}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, 
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias = False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                    drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
    
    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


# Scoring neural network, assigning a score to each frame.
class ImportantScore(nn.Module):
    def __init__(self,
                 nhead: int,
                 n_dimension: int):
        super().__init__()
        self.importance_score = nn.Sequential(
            nn.Linear(n_dimension, n_dimension),
            nn.GELU(),
            nn.Linear(n_dimension, n_dimension)
        )
        self.norm = nn.LayerNorm(n_dimension)

        # nhead = 1
        self.transf = nn.Sequential(nn.Linear(n_dimension, nhead),
                                    nn.Sigmoid())

    def forward(self,x: torch.Tensor):
        r"""
            input: [b,n,c]
            output: [b,n,1]
        """
        b, n = x.shape[0], x.shape[1]
        # C = x.shape[2]
        # C = 256
        ret = self.transf(x + self.norm(self.importance_score(x))) 

        return ret
        

class SoftChooseDenseGrad(Function):
    @staticmethod
    # def forward(ctx, imp_A, imp_B, q_top, q_bot, temperature, **kwargs):
    def forward(ctx, ret, q, temperature, **kwargs):
        # q: (b,n,c)
        # ret:(b,n)
        # selected_tokens: (b,n,c)
        # selected_tokens_flat: (b,n,c)
        selected_tokens = q * ret.float().unsqueeze(-1)
        b , n = ret.shape
        ctx.save_for_backward(ret, q, temperature)  # Save for backward
        return selected_tokens

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output:(b,n,c)
        ret, q, temperature = ctx.saved_tensors

        b, n = ret.shape  
        grad_ret = torch.zeros_like(ret)
        grad_q = torch.zeros_like(q) # (b,n,c)
        
        grad_ret = (grad_output * q).sum(dim=-1)

        grad_q = grad_output * ret.float().unsqueeze(-1)

        # Gradient reversal.
        alpha = - 1.0
        grad_ret = alpha * grad_ret
        
        # Return the gradient for each input.
        return grad_ret, grad_q, None



# Implement the generation of Gumbel noise to make the sampling process differentiable.
def get_gumbel(tensor):
    # Generate the original Gumbel noise.
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(tensor) + 1e-20) + 1e-20)

    
    """Create a sufficiently large Gaussian noise to ensure 
    the distinctiveness of the frame sequences from the two views"""
    normal_scale = 10
    normal_noise = torch.randn_like(tensor) * normal_scale

    result = tensor + gumbel_noise + normal_noise

    # The results can be standardized to make the results closer to a Gaussian distribution
    mean = result.mean()
    std = result.std()
    result = (result - mean) / std
    # print(result)
    return result


# Sampler:Sampling network
class SampleNet(nn.Module):
    def __init__(self, 
                 n_sampled_token: int,
                 n_dimension_x: int,
                 epoch: int,
                 temperature=0.1,
                 nhead=1):
        super().__init__()
        # nhead = 1
        self.nhead = nhead
        
        self.n_sampled_token = n_sampled_token
        self.importance_score = ImportantScore(nhead=nhead,n_dimension=n_dimension_x)

        self.temperature = nn.Parameter(torch.ones(1, nhead, 1, 1) * temperature, requires_grad=True)

        self.alpha = 1
        self.last_epoch = 0

    def forward(self, x, adv, epoch, hard = True):
        b,n,c = x.shape
        
        importance_scores = self.importance_score(x)
        if adv:
            importance_scores = importance_scores*-1
        
        q = x.reshape(b,1,n,c)

        if self.training :
            importance_scores = importance_scores.squeeze(-1)
            importance_scores_gumbel = get_gumbel(importance_scores)
            importance_scores_gumbel = torch.softmax(importance_scores_gumbel/0.5, dim=-1).squeeze(0)  # 归一化
            _, indices = torch.topk(importance_scores_gumbel, self.n_sampled_token, dim=1, largest=True, sorted=True)
            first_k_indices = indices[:, :self.n_sampled_token]

            # Sort indices within each part to maintain the original order
            _, sorted_first_k_indices = torch.sort(first_k_indices, dim=1)

            # Gather scores and indices back in their original order
            first_k_indices = first_k_indices.gather(1, sorted_first_k_indices)
            
            if hard:
                mask_shape = (b,n)
                mask_shape = torch.zeros(mask_shape)

                devices = importance_scores_gumbel.device
                mask_shape = mask_shape.to(devices)
                
                y_hard = torch.zeros_like(mask_shape,memory_format=torch.legacy_contiguous_format).scatter_(1,first_k_indices,1.0)
                ret = y_hard - importance_scores_gumbel.detach() + importance_scores_gumbel     

            q = q.squeeze(1)
            q_top = SoftChooseDenseGrad.apply(ret, q, self.temperature)

            q_top = q_top.masked_select((ret.unsqueeze(-1)).bool()).reshape(b, self.n_sampled_token,-1)
            q_top = q_top.unsqueeze(1)

        else:
            _, indices_top = torch.topk(importance_scores, self.n_sampled_token, dim=1, largest=True, sorted=False)
            indices_top = indices_top.transpose(1, 2).unsqueeze(-1)
            q_top = torch.take_along_dim(q, indices=indices_top, dim=2)

        q_top = rearrange(q_top,'b nhead n c -> b n (nhead c)',c=c,nhead=self.nhead)

        if self.training:
            return q_top.contiguous(), first_k_indices
        else:
            return q_top.contiguous(), indices_top


class Encoder(nn.Module):
    def __init__(self, feature_num=4096, embed_dim=256, max_frame=25, nbits=64, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                    qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                    norm_layer=nn.LayerNorm, init_values=None, use_learnable_pos_emb=False,mask_prob=0.5,epoch = 1):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        num_patches = max_frame
        self.num_patches = num_patches
        self.nbits = nbits

        self.first_k_indices = None

        self.patch_embed = nn.Linear(feature_num, embed_dim)
        self.sample_token = int(max_frame*(1-mask_prob)) 
        self.sample_net = SampleNet(n_dimension_x = embed_dim,n_sampled_token = self.sample_token,epoch = 1)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, mask=None,adv=False,epoch=1):
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, _, C = x.shape
        if mask is not None:
            x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        else:
            # Sampler:Sampling network
            x_vis, self.first_k_indices = self.sample_net(x,adv,epoch)


        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask,adv=False,epoch=1):
        x = self.forward_features(x, mask,adv,epoch)
        x = self.head(x)
        return x, self.first_k_indices


class Decoder(nn.Module):
    def __init__(self, feature_num=4096, embed_dim=256, max_frame=25, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        num_patches = max_frame
        self.num_patches = num_patches

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, feature_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))
        return x


class AutoSSVH_model(nn.Module):
    def __init__(self,
                 cfg,
                 feature_num=4096, 
                 encoder_embed_dim=256, 
                 max_frame=25,
                 nbits=64,
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_embed_dim=256, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 mask_prob = 0.5,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super(AutoSSVH_model, self).__init__()
        self.cfg = cfg

        self.data_aug = nn.Dropout(p=cfg.data_drop_rate) if cfg.data_drop_rate > 1e-2 else None
        self.encoder = Encoder(
            feature_num=feature_num, 
            embed_dim=encoder_embed_dim, 
            max_frame=max_frame,
            nbits=nbits,
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            mask_prob = mask_prob)

        self.decoder = Decoder(
            feature_num=feature_num, 
            embed_dim=decoder_embed_dim, 
            max_frame=max_frame,
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)

        
        self.encoder_to_decoder = nn.Linear(self.encoder.nbits, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.max_frame = max_frame

        self.pos_embed = nn.Parameter(torch.zeros(1, self.encoder.num_patches, decoder_embed_dim)) #可学习的位置编码

        self.binary = nn.Linear(self.encoder.num_features, self.encoder.nbits)
        self.ln = nn.LayerNorm(self.encoder.nbits)
        self.activation = self.binary_tanh_unit

        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def binary_tanh_unit(self, x):
        y = self.hard_sigmoid(x)
        out = 2. * Round3.apply(y) - 1.
        return out
    
    def hard_sigmoid(self, x):
        y = (x + 1.) / 2.
        y[y > 1] = 1
        y[y < 0] = 0
        return y

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask=None,adv=False,epoch=1):
        if self.training and self.data_aug is not None:
            x = self.data_aug(x)
        x_feat, first_k_indices = self.encoder(x, mask,adv,epoch = epoch)

        hash_code = self.binary(x_feat)
        hash_code = self.ln(hash_code)
        hash_code = self.activation(hash_code)

        x_vis = self.encoder_to_decoder(hash_code)

        B, N, C = x_vis.shape
        expand_pos_embed = self.pos_embed.expand(B, -1, -1)

        if mask  is not None:
            pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
            pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
            x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
            x = self.decoder(x_full, pos_emd_mask.shape[1])
        else:
            x_full = torch.cat([x_vis, self.mask_token.repeat(x.shape[0], x.shape[1]  - x_vis.shape[1], 1)], dim=1)+expand_pos_embed
            x = self.decoder(x_full,0)

        return x, hash_code, first_k_indices
    def get_features(self,x):
        x = self.inference(x)
        ## Hash Voting
        x = torch.mean(x,1)
        # x = torch.sign(x) 
        x_norm = nn.functional.normalize(x, dim=1)
        return x,x_norm
    def inference(self, x):
        batch_size = x.size(0)
        frame_num = x.size(1)
        device = x.device
        mask = torch.zeros((batch_size, frame_num), dtype=torch.bool, device=device)

        x, _ = self.encoder(x, mask)

        x = self.binary(x)
        x = self.ln(x)
        x = self.activation(x)

        return x
    def get_encoder_param_count(self):
        return sum(p.numel() for p in self.encoder.parameters())


def AutoSSVH(cfg):
    assert cfg.AutoSSVH_type in ['small', 'base', 'large', 'mini']
    if cfg.AutoSSVH_type == 'small':
        model = AutoSSVH_model(
            cfg=cfg,
            feature_num=cfg.feature_size, 
            encoder_embed_dim=cfg.hidden_size, 
            max_frame=cfg.max_frames,
            nbits=cfg.nbits,
            encoder_depth=12,
            encoder_num_heads=6,
            decoder_embed_dim=192,
            decoder_depth=2,
            decoder_num_heads=3,
            mlp_ratio=4, 
            qkv_bias=True,
            mask_prob = cfg.mask_prob,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif cfg.AutoSSVH_type == 'base':
        model = AutoSSVH_model(
            cfg=cfg,
            feature_num=cfg.feature_size, 
            encoder_embed_dim=cfg.hidden_size, 
            max_frame=cfg.max_frames,
            nbits=cfg.nbits,
            encoder_depth=12, 
            encoder_num_heads=12,
            decoder_embed_dim=192,
            decoder_depth=2,
            decoder_num_heads=3,
            mlp_ratio=4, 
            qkv_bias=True,
            mask_prob = cfg.mask_prob,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif cfg.AutoSSVH_type == 'large':
        model = AutoSSVH_model(
            cfg=cfg,
            feature_num=cfg.feature_size, 
            encoder_embed_dim=cfg.hidden_size, 
            max_frame=cfg.max_frames,
            nbits=cfg.nbits,
            encoder_depth=24, 
            encoder_num_heads=16,
            decoder_embed_dim=192,
            decoder_depth=2,
            decoder_num_heads=3,
            mlp_ratio=4, 
            qkv_bias=True,
            mask_prob = cfg.mask_prob,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif cfg.AutoSSVH_type == 'mini':
        model = AutoSSVH_model(
            cfg=cfg,
            feature_num=cfg.feature_size, 
            encoder_embed_dim=cfg.hidden_size, 
            max_frame=cfg.max_frames,
            nbits=cfg.nbits,
            encoder_depth=1, 
            encoder_num_heads=1,
            decoder_embed_dim=192,
            decoder_depth=2,
            decoder_num_heads=3,
            mlp_ratio=4, 
            qkv_bias=True,
            mask_prob = cfg.mask_prob,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

if __name__=='__main__':
    x = torch.randn(1,25,4096)
    model = SampleNet(n_dimension_x=4096,n_sampled_token=10)
    print(model(x).shape)