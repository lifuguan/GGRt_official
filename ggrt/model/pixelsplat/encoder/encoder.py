from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import Tensor,nn

from ....dataset.types import BatchedViews, DataShim
from ..types import Gaussians

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        features: Tensor,
        clip_h: int,
        clip_w: int,
        deterministic: bool,
        just_return_future: bool = False,
        crop_size = None,
    ) -> Gaussians:
        pass

def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
