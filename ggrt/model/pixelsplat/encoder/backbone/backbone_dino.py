from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from ggrt.dataset.types import BatchedViews
from .backbone import Backbone
from .backbone_resnet import BackboneResnet, BackboneResnetCfg


@dataclass
class BackboneDinoCfg:
    name: Literal["dino"]
    model: Literal["dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8"]
    d_out: int


class BackboneDino(Backbone[BackboneDinoCfg]):
    def __init__(self, cfg: BackboneDinoCfg, d_in: int) -> None:
        super().__init__(cfg)
        assert d_in == 3
        self.resnet_backbone = BackboneResnet(
            BackboneResnetCfg("resnet", "dino_resnet50", 4, False, cfg.d_out),
            d_in,
        )

    def forward(
        self,
        context: BatchedViews,
    ) -> Float[Tensor, "batch view d_out height width"]:
        # Compute features from the DINO-pretrained resnet50.
        resnet_features = self.resnet_backbone(context)

        return resnet_features.to(torch.float)
        # return resnet_features + local_tokens + global_token

    @property
    def patch_size(self) -> int:
        return int("".join(filter(str.isdigit, self.cfg.model)))

    @property
    def d_out(self) -> int:
        return self.cfg.d_out
