from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    
    def to(self, type=torch.bfloat16) -> "Gaussians":
        return Gaussians(means=self.means.to(type), covariances=self.covariances.to(type), harmonics=self.harmonics.to(type), opacities=self.opacities.to(type))

    def detach(self) -> "Gaussians":
        return Gaussians(means=self.means.detach(), covariances=self.covariances.detach(), harmonics=self.harmonics.detach(), opacities=self.opacities.detach())