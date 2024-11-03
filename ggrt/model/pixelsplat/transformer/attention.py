# MIT License

# Copyright (c) 2022 Karl Stelzner

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file comes from https://github.com/stelzner/srt

import torch
from einops import rearrange
from torch import nn





class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, selfatt=True, kv_dim=None
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        if selfatt:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, z=None):

        if z is None:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
        # if x.shape[1]==140 or x.shape[0]==4480:#裁剪的情况 selfattention需要重新设计
        #     if z is None:
        #         qkv = self.to_qkv(x).chunk(3, dim=-1)
        #     else:
        #         q = self.to_q(x)
        #         k, v = self.to_kv(z).chunk(2, dim=-1)
        #         qkv = (q, k, v)

        #     q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        #     dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        #     attn = self.attend(dots)

        #     out = torch.matmul(attn, v)
        #     out = rearrange(out, "b h n d -> b n (h d)")
        # else:  #全图
        #     if z is None:  #q 全图作selfattention 保存四次attn
        #         qkv = self.to_qkv(x).chunk(3, dim=-1)
        #         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        #         _,_,hw,_=q.shape
        #         q_1=rearrange(q, "b n (h w) d -> b n h w d",h=20,w=28)
        #         k_1=rearrange(k, "b n (h w) d -> b n h w d",h=20,w=28)
        #         v_1=rearrange(v, "b n (h w) d -> b n h w d",h=20,w=28)
        #         out=torch.zeros([2,4,20,28,128]).cuda()
        #         for i in range(2):
        #             for j in range(2):
        #                  q=rearrange(q_1[:,:,10*i:10*(i+1),14*j:14*(j+1),:],"b n h w d -> b n (h w) d")
        #                  k=rearrange(k_1[:,:,10*i:10*(i+1),14*j:14*(j+1),:],"b n h w d -> b n (h w) d")
        #                  v=rearrange(v_1[:,:,10*i:10*(i+1),14*j:14*(j+1),:],"b n h w d -> b n (h w) d")    #140 128
        #                  dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  #140 140 
        #                  attn_crop = self.attend(dots)  #140 140 
        #                 #  attn_crop=rearrange(attn_crop, "b n (h w) d -> b n h w d",h=10,w=14)   #1   10 14 140
        #                  out_crop = torch.matmul(attn_crop, v)  #140 128
        #                  out_crop=rearrange(out_crop, "b n (h w) d -> b n h w d",h=10,w=14)  
        #                  out[:,:,10*i:10*(i+1),14*j:14*(j+1),:]=out_crop           #20 28 128
        #         out = rearrange(out, "b h n w d -> b h (n w) d")  #560 128
        #         out = rearrange(out, "b h n d -> b n (h d)")

        #     else:
        #         q = self.to_q(x)
        #         k, v = self.to_kv(z).chunk(2, dim=-1)
        #         qkv = (q, k, v)

        #         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        #         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        #         attn = self.attend(dots)

        #         out = torch.matmul(attn, v)
        #         out = rearrange(out, "b h n d -> b n (h d)")
        # return self.to_out(out)
