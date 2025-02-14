import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import BEVEmbedding, generate_grid
from einops import rearrange, repeat

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
            self,
            x: torch.FloatTensor,
            bev: BEVEmbedding,
            feature: torch.FloatTensor,
            I_inv: torch.FloatTensor,
            E_inv: torch.FloatTensor,
        ):
            """
            x: (b*t, c, H, W)
            feature: (b*t, cam_n, dim_in, h, w)
            I_inv: (b*t, cam_n, 3, 3)
            E_inv: (b*t, cam_n, 4, 4)

            Returns: (b*t, d, H, W)
            """
            b_t, n, _, _, _ = feature.shape

            pixel = self.image_plane                                                # b n 3 h w
            _, _, _, h, w = pixel.shape
            c = E_inv[..., -1:]                                                     # b n 4 1 torch.Size([20, 4, 1])
            c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
            c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

            pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
            cam = I_inv @ pixel_flat                                                # b n 3 (h w)
            cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
            d = E_inv @ cam                                                         # b n 4 (h w)
            d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
            d_embed = self.img_embed(d_flat)                                        # (b n) d h w

            img_embed = d_embed - c_embed                                           # (b n) d h w
            img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

            world = bev.grid[:2]                                                    # 2 H W
            w_embed = self.bev_embed(world[None])                                   # 1 d H W
            bev_embed = w_embed - c_embed                                           # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b_t, n=n)      # b n d H W

            feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

            if self.feature_proj is not None:
                key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
            else:
                key_flat = img_embed                                                # (b n) d h w

            val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

            # Expand + refine the BEV embedding
            query = query_pos + x[:, None]                                          # b n d H W
            key = rearrange(key_flat, '(b n) ... -> b n ...', b=b_t, n=n)             # b n d h w
            val = rearrange(val_flat, '(b n) ... -> b n ...', b=b_t, n=n)             # b n d h w

            return self.cross_attend(query, key, val, skip=x if self.skip else None)
        
        
class FeatureFusionAttention(nn.Module):
    def __init__(self, feature_dim, bev_dim, time_steps, spatial_dim, pooled_dim=16):
        """
        Args:
            feature_dim (int): Dimension of the front-view and ego features (e.g., 128).
            bev_dim (int): Dimension of the BEV feature channels (e.g., 128).
            time_steps (int): Number of time steps in the BEV and ego features (e.g., 2).
            spatial_dim (int): Height and width of the BEV spatial feature map (e.g., 32).
            pooled_dim (int): Dimension to which the BEV feature spatial size is pooled.
        """
        super(FeatureFusionAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.bev_dim = bev_dim
        self.time_steps = time_steps
        self.spatial_dim = spatial_dim
        self.pooled_dim = pooled_dim

        # Adaptive Average Pooling to reduce BEV feature spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pooled_dim, pooled_dim))

        # Attention mechanism for front-view feature and ego feature
        self.front_ego_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        
        # Attention mechanism for BEV features
        self.bev_attention = nn.MultiheadAttention(embed_dim=bev_dim * pooled_dim * pooled_dim, num_heads=4, batch_first=True)

        # Fully connected layer to project attention output
        self.fc = nn.Linear(feature_dim + bev_dim, feature_dim)

    def forward(self, front_feature, bev_feature, ego_feature):
        """
        Args:
            front_feature (torch.Tensor): Front-view feature [batch, feature_dim].
            bev_feature (torch.Tensor): BEV features [batch, time, bev_dim, height, width].
            ego_feature (torch.Tensor): Ego features [batch, time, feature_dim].
        
        Returns:
            torch.Tensor: Fused feature with time dimension [batch, time, feature_dim].
        """
        batch_size, time_steps, bev_dim, height, width = bev_feature.size()

        # Apply Adaptive Average Pooling to BEV features
        bev_feature = self.adaptive_pool(bev_feature.view(-1, bev_dim, height, width))  # [batch * time, bev_dim, pooled_dim, pooled_dim]
        bev_feature = bev_feature.view(batch_size, time_steps, bev_dim, self.pooled_dim, self.pooled_dim)  # [batch, time, bev_dim, pooled_dim, pooled_dim]

        # Reshape BEV features for attention
        bev_feature = bev_feature.view(batch_size, time_steps, -1)  # [batch, time, bev_dim * pooled_dim * pooled_dim]

        # Apply attention between BEV features
        bev_out, _ = self.bev_attention(bev_feature, bev_feature, bev_feature)  # [batch, time, bev_dim * pooled_dim * pooled_dim]

        # Reshape back to separate dimensions for concatenation
        bev_out = bev_out.view(batch_size, time_steps, bev_dim, self.pooled_dim, self.pooled_dim)  # [batch, time, bev_dim, pooled_dim, pooled_dim]

        # Reshape front-view feature for concatenation
        front_feature = front_feature.unsqueeze(1).expand(-1, time_steps, -1)  # [batch, time, feature_dim]

        # Apply attention between front-view and ego features
        fused_ego, _ = self.front_ego_attention(ego_feature, front_feature, front_feature)  # [batch, time, feature_dim]

        # Concatenate BEV and fused ego features
        fused_feature = torch.cat([fused_ego, bev_out.mean(dim=[3, 4])], dim=-1)  # [batch, time, feature_dim + bev_dim]

        # Final projection
        output = self.fc(fused_feature)  # [batch, time, feature_dim]

        return output