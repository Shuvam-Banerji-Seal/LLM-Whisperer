# Image Preprocessing and Normalization — Agentic Skill Prompt

Image loading, resizing, normalization, and processor patterns for vision-language models.

---

## 1. Identity and Mission

Standardize image preprocessing across diverse vision-language models with consistent, efficient, reproducible pipelines.

---

## 2. Image Loading and Resizing

```python
import torch
from PIL import Image
from typing import Tuple, Optional
import numpy as np

class ImagePreprocessor:
    """Standardized image preprocessing."""
    
    # Common preprocessing templates
    PRESETS = {
        "clip": {"size": 224, "normalize": True, "padding": False},
        "llava": {"size": 336, "normalize": True, "padding": True},
        "qwen": {"size": 448, "normalize": True, "padding": False},
        "blip": {"size": 384, "normalize": True, "padding": False},
    }
    
    def __init__(
        self,
        target_size: int = 224,
        normalize: bool = True,
        padding_mode: str = "constant",
        mean: list[float] = None,
        std: list[float] = None,
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Resize to square (H, W)
            normalize: Apply ImageNet normalization
            padding_mode: 'constant' or 'reflect'
            mean: Normalization mean (default: ImageNet)
            std: Normalization std (default: ImageNet)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.padding_mode = padding_mode
        
        # ImageNet statistics (default)
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from file."""
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    
    def resize_with_padding(
        self,
        image: Image.Image,
        target_size: int,
    ) -> Image.Image:
        """Resize image with padding to maintain aspect ratio."""
        # Calculate aspect ratio
        w, h = image.size
        aspect = w / h
        
        if aspect > 1:  # Width > Height
            new_w = target_size
            new_h = int(target_size / aspect)
        else:  # Height >= Width
            new_h = target_size
            new_w = int(target_size * aspect)
        
        # Resize
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Pad to target size
        if new_w < target_size or new_h < target_size:
            pad_h = target_size - new_h
            pad_w = target_size - new_w
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            
            image = Image.new(
                "RGB",
                (target_size, target_size),
                (127, 127, 127),
            )
            image.paste(image, (pad_left, pad_top))
        
        return image
    
    def center_crop(
        self,
        image: Image.Image,
        crop_size: Optional[int] = None,
    ) -> Image.Image:
        """Center crop image to square."""
        crop_size = crop_size or min(image.size)
        w, h = image.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        
        return image.crop((left, top, left + crop_size, top + crop_size))
    
    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to torch tensor (C, H, W) with values [0, 1]."""
        return torch.from_numpy(
            np.array(image) / 255.0
        ).permute(2, 0, 1).float()
    
    def normalize(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ImageNet normalization."""
        if not self.normalize:
            return tensor
        
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        
        return (tensor - mean) / std
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Full preprocessing pipeline."""
        # Option 1: Resize with padding
        image = self.resize_with_padding(image, self.target_size)
        
        # Option 2: Center crop
        # image = self.center_crop(image, self.target_size)
        
        # Convert to tensor
        tensor = self.to_tensor(image)
        
        # Normalize
        if self.normalize:
            tensor = self.normalize(tensor)
        
        return tensor
    
    def batch_preprocess(
        self,
        image_paths: list[str],
    ) -> torch.Tensor:
        """Preprocess batch of images."""
        tensors = []
        for path in image_paths:
            image = self.load_image(path)
            tensor = self.preprocess(image)
            tensors.append(tensor)
        
        # Stack into batch (B, C, H, W)
        return torch.stack(tensors)
    
    @classmethod
    def from_preset(cls, preset: str = "clip") -> "ImagePreprocessor":
        """Create preprocessor from preset."""
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        
        params = cls.PRESETS[preset]
        return cls(**params)

# Usage
preprocessor = ImagePreprocessor.from_preset("llava")
image = preprocessor.load_image("image.jpg")
tensor = preprocessor.preprocess(image)
print(f"Tensor shape: {tensor.shape}")
```

---

## 3. Using Hugging Face Image Processors

```python
from transformers import AutoImageProcessor
import torch

class HuggingFaceImageProcessing:
    """Use HuggingFace image processors."""
    
    def __init__(self, model_name: str):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
    
    def process_image(
        self,
        image_path: str,
        return_tensors: str = "pt",
    ) -> dict:
        """Process image using model-specific processor."""
        from PIL import Image
        
        image = Image.open(image_path)
        
        # Processor handles resizing, normalization, etc.
        inputs = self.processor(
            images=image,
            return_tensors=return_tensors,
        )
        
        return inputs
    
    def batch_process(
        self,
        image_paths: list[str],
        return_tensors: str = "pt",
    ) -> dict:
        """Batch process multiple images."""
        from PIL import Image
        
        images = [Image.open(p) for p in image_paths]
        
        inputs = self.processor(
            images=images,
            return_tensors=return_tensors,
            padding=True,
        )
        
        return inputs

# Usage
processor = HuggingFaceImageProcessing("openai/clip-vit-base-patch32")
inputs = processor.process_image("image.jpg")
```

---

## 4. Torchvision Transforms

```python
import torchvision.transforms as transforms
from typing import Callable

def get_transform_pipeline(
    model_type: str = "llava",
    augment: bool = False,
) -> Callable:
    """Get preprocessing transforms for different models."""
    
    if model_type == "llava":
        transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    elif model_type == "clip":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    if augment:
        # Add augmentation transforms
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomRotation(10),
        ])
        transform = transforms.Compose([aug_transform, transform])
    
    return transform

# Usage
transform = get_transform_pipeline("llava")
image = Image.open("image.jpg")
tensor = transform(image)
```

---

## 5. Efficient Batch Processing with PIL

```python
from PIL import Image
from pathlib import Path
import io
from typing import List, Dict

class EfficientBatchLoader:
    """Memory-efficient batch loading with prefetching."""
    
    def __init__(
        self,
        image_dir: Path,
        batch_size: int = 32,
        prefetch: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.image_paths = list(self.image_dir.glob("*.jpg")) + \
                          list(self.image_dir.glob("*.png"))
    
    def load_batch(
        self,
        indices: List[int],
        preprocessor: callable,
    ) -> torch.Tensor:
        """Load and preprocess batch."""
        images = []
        
        for idx in indices:
            path = self.image_paths[idx]
            image = Image.open(path).convert("RGB")
            tensor = preprocessor(image)
            images.append(tensor)
        
        return torch.stack(images)
    
    def __len__(self) -> int:
        return (len(self.image_paths) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate through batches."""
        for i in range(0, len(self.image_paths), self.batch_size):
            indices = list(range(i, min(i + self.batch_size, len(self.image_paths))))
            yield indices
```

---

## 6. References

1. https://pytorch.org/vision/stable/transforms.html — PyTorch transforms documentation
2. https://pillow.readthedocs.io/ — PIL image processing library
3. https://huggingface.co/docs/transformers/tasks/image_classification — HuggingFace image processing
4. https://arxiv.org/abs/1512.03385 — "Deep Residual Learning for Image Recognition" (ImageNet normalization)
5. https://github.com/pytorch/vision — PyTorch Vision official
6. https://huggingface.co/models?pipeline_tag=image-text-to-text — Vision-language models directory
7. https://arxiv.org/abs/2010.11929 — "An Image is Worth 16x16 Words" (Vision Transformers, image patching)
8. https://paperswithcode.com/task/image-classification — Image preprocessing benchmarks
9. https://github.com/albumentations-team/albumentations — Advanced augmentation library
10. https://arxiv.org/abs/2205.01580 — "Image Data Augmentation for Deep Learning" (Augmentation strategies)
11. https://opencv.org/ — OpenCV for advanced image processing
12. https://scikit-image.org/ — scikit-image documentation
13. https://huggingface.co/docs/transformers/model_doc/clip — CLIP processor documentation
14. https://github.com/openai/CLIP/blob/main/clip/clip.py — CLIP preprocessing code
15. https://github.com/salesforce/BLIP — BLIP preprocessing patterns
16. https://huggingface.co/docs/datasets/ — HuggingFace datasets for loading images at scale

---

## 7. Uncertainty and Limitations

**Not Covered:**
- GPU-accelerated image processing (CUDA kernels)
- Advanced augmentation for domain adaptation
- Medical/scientific image preprocessing (specialized requirements)
- Video frame extraction and preprocessing

**Production Notes:**
- Cache preprocessed images to avoid repeated computation
- Use PIL/torchvision for CPU-bound preprocessing (faster than PyTorch)
- Validate image dimensions match model expectations
- Handle corrupted images gracefully with try-except
