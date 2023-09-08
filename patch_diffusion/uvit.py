import torch
import torch.nn as nn
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
import numpy as np
from .fp16_util import convert_module_to_f16_new, convert_module_to_f32

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, n_patches, patch_size, channels=3):
    # patch_size = int((x.shape[2] // channels) ** 0.5)
    # h = w = int(x.shape[1] ** .5)
    # assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    h, w = n_patches
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size[0], p2=patch_size[1])
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class PatchEmbedNew(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, image_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        padding1 = math.ceil(image_size[0] % patch_size[0]) * patch_size[0] // 2
        padding2 = math.ceil(image_size[1] % patch_size[1]) * patch_size[1] // 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=(padding1, padding2), padding_mode='zeros')

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class UViT(nn.Module):
    def __init__(self, img_size=(640, 641), patch_size=(16, 16), in_chans=1, out_chans=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.patch_embed = PatchEmbedNew(image_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (math.ceil(img_size[0] / patch_size[0]), math.ceil(img_size[1] / patch_size[1]) )
        num_patches = math.ceil(img_size[0] / patch_size[0]) * math.ceil(img_size[1] / patch_size[1]) 

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size[0] * patch_size[1] * self.out_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.out_chans, self.out_chans, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        pass
        # self.patch_embed.proj.apply(convert_module_to_f16_new)
        # self.in_blocks.apply(convert_module_to_f16_new)
        # self.mid_block.apply(convert_module_to_f16_new)
        # self.out_blocks.apply(convert_module_to_f16_new)
        # self.decoder_pred.apply(convert_module_to_f16_new)
        # self.final_layer.apply(convert_module_to_f16_new)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        pass
        # self.patch_embed.apply(convert_module_to_f32)
        # self.in_blocks.apply(convert_module_to_f32)
        # self.mid_block.apply(convert_module_to_f32)
        # self.out_blocks.apply(convert_module_to_f32)
        # self.final_layer.apply(convert_module_to_f32)
        # self.decoder_pred.apply(convert_module_to_f32)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.n_patches, self.patch_size, self.out_chans)
        x = self.final_layer(x)
        x = x[:, :, math.floor((x.shape[-2]-self.img_size[0]) / 2): x.shape[-2]-math.ceil((x.shape[-2]-self.img_size[0]) / 2),
              math.floor((x.shape[-1]-self.img_size[1]) / 2): x.shape[-1]-math.ceil((x.shape[-1]-self.img_size[1]) / 2)]
        return x
    
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class UViTSino(nn.Module):
    def __init__(self, img_size=(640, 641), patch_size=641, in_chans=1, out_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.img_size = img_size

        # self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed = MLP(in_dim=img_size[1], out_dim=embed_dim, hidden_list=[256, 256])
        num_patches = img_size[0] # (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))
        # self.tpos_embed = nn.Parameter(torch.zeros(1, self.extras, embed_dim)).cuda()
        # views = []
        # for i in range(img_size[0]):
        #     views.append(np.array((np.sin(i/img_size[0]*2*np.pi), np.cos(i/img_size[0]*2*np.pi))))
        # view = np.stack(views, axis=0)
        # if embed_dim % 2 == 0:
        #     self.view_embed = np.concatenate([view]*(embed_dim//2), axis=-1)[None, ...]
        # else:
        #     self.view_embed = np.concatenate([view]*(embed_dim//2)+[view[:, 0, None]], axis=-1)[None, ...]
        # self.view_embed = nn.Parameter(torch.FloatTensor(self.view_embed), requires_grad=False).cuda()
        # self.pos_embed = torch.cat([self.tpos_embed, self.view_embed], dim=1)
        
        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = img_size[1] # patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim * self.out_chans // self.in_chans, bias=True)
        self.final_layer = nn.Conv2d(self.out_chans, self.out_chans, 3, padding=1) if conv else nn.Identity()
        # trunc_normal_(self.tpos_embed, std=.02)
        # trunc_normal_(self.view_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        pass

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        pass

    def forward(self, x, timesteps, y=None):
        x_ = x.clone()
        x = self.patch_embed(x.squeeze(1))
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        # x[:, :self.extras, :] = x[:, :self.extras, :] + self.tpos_embed
        # x[:, self.extras:, :] = x[:, self.extras:, :] + self.view_embed
        # x = x + torch.cat([self.tpos_embed, self.view_embed], dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = x.view(B, self.out_chans, self.img_size[0], self.img_size[1])
        # x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        x[:, 0, ...] = x[:, 0, ...] + x_[:, 0, ...]
        return x
    
class UViTSinoSlice(nn.Module):
    def __init__(self, img_size=641, patch_size=4, in_chans=1, out_chans=1, embed_dim=192, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=640,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.img_size = img_size
        self.patch_size = patch_size
        if img_size % patch_size == 0:
            padding = 0
        else:
            padding=(patch_size - img_size % patch_size) // 2 + 1
        self.patch_embed = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, 
                                     padding=padding, padding_mode='zeros')
        # self.patch_embed = MLP(in_dim=img_size[1], out_dim=embed_dim, hidden_list=[256, 256])
        self.num_patches = img_size // patch_size + 1 # (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            # self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.label_emb = nn.Linear(2, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.num_patches, embed_dim))

        
        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size # patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim * self.out_chans // self.in_chans, bias=True)
        # self.final_layer = nn.Conv1d(self.out_chans, self.out_chans, 3, padding=1) if conv else nn.Identity()
        # trunc_normal_(self.tpos_embed, std=.02)
        # trunc_normal_(self.view_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        pass

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        pass

    def forward(self, x, timesteps, view=None):
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if view is not None:
            label_emb = self.label_emb(view)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        # x[:, :self.extras, :] = x[:, :self.extras, :] + self.tpos_embed
        # x[:, self.extras:, :] = x[:, self.extras:, :] + self.view_embed
        # x = x + torch.cat([self.tpos_embed, self.view_embed], dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = x.view(B, self.out_chans, self.num_patches * self.patch_size)
        x = x[:, :, (self.num_patches * self.patch_size - self.img_size) // 2: (self.num_patches * self.patch_size - self.img_size) // 2 + self.img_size]
        # x = unpatchify(x, self.in_chans)
        # x = self.final_layer(x)
        # x[:, 0, ...] = x[:, 0, ...] + x_[:, 0, ...]
        return x