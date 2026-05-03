from typing import List, Optional, Union

import open_clip
import torch
import torch.nn as nn
from PIL import Image


class OpenClipImageEncoder(nn.Module):
    """
    OpenCLIP image encoder wrapper.
    Accepts a list of image paths and returns raw image features: [B, clip_dim]
    """

    def __init__(self, model_name: str, pretrained: str, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        dev = str(device) if isinstance(device, torch.device) else (device or "cpu")
        self.clip, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=dev,
        )
        self.preprocess = self.preprocess_val or self.preprocess_train
        self.clip_dim = self._infer_dim()

    def _infer_dim(self) -> int:
        self.clip.eval()
        with torch.no_grad():
            image = torch.zeros(1, 3, 224, 224, device=next(self.clip.parameters()).device)
            out = self.clip.encode_image(image)
            return int(out.shape[-1])

    def _load_image(self, image_path: str) -> Image.Image:
        with Image.open(image_path) as img:
            return img.convert("RGB")

    def encode_raw(self, image_paths: List[str]) -> torch.Tensor:
        images = [self.preprocess(self._load_image(path)) for path in image_paths]
        image_tensor = torch.stack(images, dim=0).to(next(self.clip.parameters()).device)
        return self.clip.encode_image(image_tensor)

    def forward(self, image_paths: List[str]) -> torch.Tensor:
        return self.encode_raw(image_paths)

    def _visual_params(self):
        params = []

        if hasattr(self.clip, "visual") and isinstance(self.clip.visual, nn.Module):
            params.extend(list(self.clip.visual.parameters()))

        if hasattr(self.clip, "image_projection"):
            image_projection = getattr(self.clip, "image_projection")
            if isinstance(image_projection, nn.Module):
                params.extend(list(image_projection.parameters()))
            elif isinstance(image_projection, nn.Parameter):
                params.append(image_projection)

        seen = set()
        uniq = []
        for param in params:
            if id(param) not in seen:
                uniq.append(param)
                seen.add(id(param))
        return uniq

    def freeze_image(self, freeze_image_projection: bool = True):
        for param in self._visual_params():
            param.requires_grad = False

        if not freeze_image_projection and hasattr(self.clip, "image_projection"):
            image_projection = getattr(self.clip, "image_projection")
            if isinstance(image_projection, nn.Module):
                for param in image_projection.parameters():
                    param.requires_grad = True
            elif isinstance(image_projection, nn.Parameter):
                image_projection.requires_grad = True

    def unfreeze_image(self, train_image_projection: bool = False):
        for param in self._visual_params():
            param.requires_grad = True

        if hasattr(self.clip, "image_projection"):
            image_projection = getattr(self.clip, "image_projection")
            if isinstance(image_projection, nn.Module):
                for param in image_projection.parameters():
                    param.requires_grad = bool(train_image_projection)
            elif isinstance(image_projection, nn.Parameter):
                image_projection.requires_grad = bool(train_image_projection)
