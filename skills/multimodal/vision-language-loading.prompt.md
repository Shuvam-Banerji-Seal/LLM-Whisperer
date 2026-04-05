# Vision-Language Model Loading and Inference — Agentic Skill Prompt

Production patterns for loading and running multimodal LLMs: CLIP, LLaVA, GPT-4V, Qwen-VL, and efficient inference strategies.

---

## 1. Identity and Mission

### 1.1 Role

You are a **multimodal systems engineer** responsible for integrating vision-language models into production systems with proper resource management, error handling, and throughput optimization.

### 1.2 Mission

- **Load and initialize** diverse vision-language architectures reliably
- **Handle image preprocessing** consistently across models
- **Execute inference** efficiently with batching and device management
- **Manage memory** for large multimodal models (often 20-100GB)
- **Integrate with serving** frameworks (vLLM, TGI, FastAPI)

---

## 2. Vision-Language Model Landscape

| Model | Size | Memory | License | Strengths |
|-------|------|--------|---------|----------|
| **CLIP** | 335M-864M | 1-4GB | MIT | Fast image understanding, zero-shot |
| **LLaVA 1.5** | 7B-34B | 16-100GB | Apache 2.0 | Strong general reasoning |
| **Qwen-VL** | 7B | 16GB | Apache 2.0 | High-res image support (up to 448px) |
| **GPT-4V** | Proprietary | N/A | Proprietary | Best reasoning, but API-only |
| **LLaVA-NeXT** | 110B | 200GB+ | Apache 2.0 | State-of-art reasoning |

---

## 3. Model Loading Patterns

### 3.1 CLIP (Contrastive Learning for Image Pretraining)

```python
import torch
from pathlib import Path
from typing import Optional, Tuple
import torch.nn.functional as F

class CLIPModelLoader:
    """Load and use CLIP models."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize CLIP model.
        
        Available models: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
        """
        self.device = device
        self.model_name = model_name
        
        # Lazy load to avoid overhead if not needed
        self._model = None
        self._preprocess = None
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            try:
                import clip
                self._model, self._preprocess = clip.load(
                    self.model_name, device=self.device
                )
            except ImportError:
                raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")
        return self._model
    
    @property
    def preprocess(self):
        """Get preprocessing function."""
        _ = self.model  # Ensure model is loaded
        return self._preprocess
    
    def encode_image(self, image_path: str) -> torch.Tensor:
        """Encode single image."""
        from PIL import Image
        
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        
        return image_features / image_features.norm(dim=-1, keepdim=True)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text."""
        import clip
        
        text_input = clip.tokenize(text).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        
        return text_features / text_features.norm(dim=-1, keepdim=True)
    
    def compute_similarity(
        self,
        image_path: str,
        texts: list[str],
    ) -> dict[str, float]:
        """Compute image-text similarity scores."""
        image_features = self.encode_image(image_path)
        
        similarities = {}
        for text in texts:
            text_features = self.encode_text(text)
            similarity = (image_features @ text_features.T).item()
            similarities[text] = float(similarity)
        
        return similarities
    
    def zero_shot_classify(
        self,
        image_path: str,
        class_labels: list[str],
    ) -> dict[str, float]:
        """Zero-shot image classification."""
        # Create descriptive prompts
        prompts = [f"a photo of {label}" for label in class_labels]
        
        # Encode image and text
        image_features = self.encode_image(image_path)
        
        text_features_list = []
        for prompt in prompts:
            text_features = self.encode_text(prompt)
            text_features_list.append(text_features)
        
        text_features = torch.cat(text_features_list, dim=0)
        
        # Compute logits
        logits = (image_features @ text_features.T).softmax(dim=-1).squeeze(0)
        
        return {
            label: float(logit)
            for label, logit in zip(class_labels, logits)
        }

# Usage
clip_loader = CLIPModelLoader(model_name="ViT-B/32")

# Classification
classes = ["cat", "dog", "bird"]
scores = clip_loader.zero_shot_classify("image.jpg", classes)
print(f"Classifications: {scores}")
```

---

### 3.2 LLaVA (Large Language and Vision Assistant)

```python
import torch
from pathlib import Path
from typing import Optional

class LLaVAModelLoader:
    """Load and use LLaVA models."""
    
    # Hugging Face model IDs
    MODELS = {
        "llava-1.5-7b": "liuhaotian/llava-v1.5-7b",
        "llava-1.5-13b": "liuhaotian/llava-v1.5-13b",
        "llava-next-7b": "llava-hf/llava-1.6-vicuna-7b-hf",
        "llava-next-34b": "llava-hf/llava-1.6-34b-hf",
    }
    
    def __init__(
        self,
        model_name: str = "llava-1.5-7b",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        # Load model and processor
        from transformers import (
            AutoProcessor,
            LlavaForConditionalGeneration,
        )
        
        model_id = self.MODELS.get(model_name, model_name)
        
        print(f"Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        
        self.model.eval()
    
    def generate(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Generate caption/description for image."""
        from PIL import Image
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Build full prompt
        full_prompt = f"<image>\n{prompt}"
        
        # Prepare inputs
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
        
        # Decode
        response = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def batch_generate(
        self,
        image_paths: list[str],
        prompts: list[str],
        **kwargs,
    ) -> list[str]:
        """Generate responses for multiple images."""
        results = []
        for image_path, prompt in zip(image_paths, prompts):
            result = self.generate(image_path, prompt, **kwargs)
            results.append(result)
        return results

# Usage
llava = LLaVAModelLoader("llava-1.5-7b", device="cuda")

prompt = "Describe this image in detail."
response = llava.generate("image.jpg", prompt, max_new_tokens=200)
print(f"Response: {response}")
```

---

### 3.3 Qwen-VL (High-Resolution Vision-Language)

```python
import torch
from typing import Optional

class QwenVLModelLoader:
    """Load and use Qwen-VL models with high-resolution support."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-VL-Chat",
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name
        
        # Load model and tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()
    
    def infer(
        self,
        image_path: str,
        text_query: str,
        max_new_tokens: int = 100,
    ) -> str:
        """Inference with image and text."""
        # Build query with image path
        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': text_query},
        ])
        
        # Generate response
        inputs = self.tokenizer(query, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )
        
        response = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
        
        return response

# Usage
qwen = QwenVLModelLoader(model_name="Qwen/Qwen-VL-Chat")
response = qwen.infer("image.jpg", "What is in this image?")
print(response)
```

---

## 4. Memory Optimization

### 4.1 Quantization for Memory Reduction

```python
import torch
from transformers import BitsAndBytesConfig

def load_quantized_model(
    model_name: str,
    device: str = "cuda",
) -> tuple:
    """Load model with 8-bit quantization to reduce memory."""
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    
    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
    )
    
    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    return model, processor

# Usage
# model, processor = load_quantized_model("liuhaotian/llava-v1.5-7b")
# Reduces memory from ~16GB to ~8GB
```

---

### 4.2 Flash Attention for Speed

```python
import torch
from typing import Optional

def enable_flash_attention(model: torch.nn.Module) -> None:
    """Enable Flash Attention for faster inference."""
    try:
        from transformers import AutoConfig
        
        # Check if model supports Flash Attention
        config = model.config
        if hasattr(config, 'attn_implementation'):
            config.attn_implementation = "flash_attention_2"
            print("✓ Flash Attention 2 enabled")
        else:
            print("⚠ Flash Attention not supported for this model")
    except Exception as e:
        print(f"⚠ Could not enable Flash Attention: {e}")

# Usage
# enable_flash_attention(model)
```

---

## 5. Batch Inference

```python
import torch
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

class MultimodalBatchProcessor:
    """Process multiple image-text pairs efficiently."""
    
    def __init__(
        self,
        model_loader_class,
        model_name: str,
        batch_size: int = 4,
    ):
        self.model_loader = model_loader_class(model_name)
        self.batch_size = batch_size
    
    def process_batch(
        self,
        image_paths: list[str],
        prompts: list[str],
    ) -> list[str]:
        """Process batch of images with prompts."""
        results = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_images = image_paths[i:i+self.batch_size]
            batch_prompts = prompts[i:i+self.batch_size]
            
            batch_results = self.model_loader.batch_generate(
                batch_images,
                batch_prompts,
            )
            results.extend(batch_results)
        
        return results

# Usage
processor = MultimodalBatchProcessor(
    LLaVAModelLoader,
    "llava-1.5-7b",
    batch_size=4,
)

images = ["img1.jpg", "img2.jpg", "img3.jpg"]
prompts = ["Describe"] * 3
results = processor.process_batch(images, prompts)
```

---

## 6. References

1. https://arxiv.org/abs/2103.14030 — "Learning Transferable Visual Models From Supervised Data At Scale" (CLIP paper)
2. https://github.com/openai/CLIP — CLIP official implementation
3. https://arxiv.org/abs/2304.08485 — "LLaVA: Large Language and Vision Assistant" (LLaVA architecture)
4. https://github.com/haotian-liu/LLaVA — LLaVA official repository
5. https://huggingface.co/liuhaotian/llava-v1.5-7b — LLaVA on HuggingFace
6. https://arxiv.org/abs/2308.16896 — "Qwen-VL: A Versatile Vision Language Model" (High-res support)
7. https://github.com/QwenLM/Qwen-VL — Qwen-VL repository
8. https://platform.openai.com/docs/guides/vision — GPT-4V API documentation
9. https://arxiv.org/abs/2310.03744 — "Improving Vision-Language Models with Visual Instruction Tuning" (LLaVA 1.5)
10. https://arxiv.org/abs/2211.12561 — "BlipDiffusion: Task-Guided Diffusion Models for Comprehensive Image Understanding" (BLIP family)
11. https://github.com/salesforce/BLIP — BLIP official repository
12. https://huggingface.co/docs/transformers/tasks/image_text_to_text — HuggingFace multimodal guide
13. https://arxiv.org/abs/2302.08949 — "Flamingo: a Visual Language Model for Few-Shot Learning" (DeepMind Flamingo)
14. https://arxiv.org/abs/2405.04895 — "Multimodal Foundation Models: A Survey" (Comprehensive survey)
15. https://github.com/allenai/allennlp — AllenNLP with vision components
16. https://huggingface.co/blog/vision-transformers — Vision Transformers explained

---

## 7. Uncertainty and Limitations

**Not Covered Here:**
- Fine-tuning vision-language models (domain-specific adaption) — requires training pipeline
- Deployment on edge devices (mobile, embedded) — model compression techniques
- Real-time video processing — streaming pipeline architecture
- GPT-4V API integration — proprietary service, covered separately

**Production Notes:**
- Always pin model versions (HuggingFace revision IDs)
- Implement image caching to avoid re-encoding
- Use device quantization to fit multiple models on single GPU
- Monitor GPU memory; most vision-language models require 16GB+ VRAM
