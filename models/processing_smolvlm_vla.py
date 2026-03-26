"""
SmolVLM-VLA Processor

Unified multimodal processor for SmolVLM-VLA models.
Handles 512x512 image inputs required by SmolVLM-500M.

Optimized version with:
- Fast GPU-based image preprocessing
- Cached normalization parameters
- Minimal CPU-GPU transfers
"""

from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor
from typing import List, Union, Dict, Any, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import logging


class SmolVLMVLAProcessor:
    """
    SmolVLMVLAProcessor: Unified multimodal processor for SmolVLM-VLA models.

    Features:
      - Uses 512x512 image resolution (SmolVLM-500M requirement)
      - Compatible with SmolVLM's tokenizer and image processor
      - All views processed together

    Note: This class does NOT inherit from ProcessorMixin to avoid tokenizer 
    type checking issues.

    Attributes
    ----------
    num_views : int, default=3
        Expected number of image views per sample.
    image_size : int, default=384
        Target image size for SmolVLM (384x384 or 512x512).
    language_max_length : int, default=50
        Maximum token length for text encoding.
    """

    num_views: int = 3
    image_size: int = 384
    language_max_length: int = 50

    def __init__(
        self, 
        image_processor=None, 
        tokenizer=None,
        smolvlm_model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
    ):
        """
        Initialize SmolVLMVLAProcessor.

        Parameters
        ----------
        image_processor : PreTrainedImageProcessor, optional
            The image processor for SmolVLM.
        tokenizer : PreTrainedTokenizer, optional
            The tokenizer for SmolVLM.
        smolvlm_model_path : str
            Path to SmolVLM model for loading default processor.
        """
        self.smolvlm_model_path = smolvlm_model_path
        
        # Load SmolVLM processor
        self._smolvlm_processor = AutoProcessor.from_pretrained(
            smolvlm_model_path,
            trust_remote_code=True,
        )
        
        # Use provided or load from SmolVLM
        self.image_processor = image_processor or self._smolvlm_processor.image_processor
        self.tokenizer = tokenizer or self._smolvlm_processor.tokenizer
        
        # Get actual image size from image processor
        if hasattr(self.image_processor, 'size'):
            size = self.image_processor.size
            if isinstance(size, dict):
                self.image_size = size.get('height', size.get('shortest_edge', 384))
            elif isinstance(size, int):
                self.image_size = size
        
        # ============ OPTIMIZATION: Cache normalization parameters ============
        # Extract mean/std from image_processor config
        self._image_mean = None
        self._image_std = None
        if hasattr(self.image_processor, 'image_mean'):
            self._image_mean = torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1)
        if hasattr(self.image_processor, 'image_std'):
            self._image_std = torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1)
        
        # Default to ImageNet normalization if not found
        if self._image_mean is None:
            self._image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        if self._image_std is None:
            self._image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        logging.info(f"[SmolVLMVLAProcessor] Initialized with image_size={self.image_size}, "
                     f"mean={self._image_mean.squeeze().tolist()}, std={self._image_std.squeeze().tolist()}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load processor from a pretrained SmolVLM model or local path.
        """
        smolvlm_path = kwargs.pop('smolvlm_model_path', pretrained_model_name_or_path)
        
        try:
            return cls(smolvlm_model_path=smolvlm_path)
        except Exception as e:
            print(f"Warning: Could not load from {smolvlm_path}: {e}")
            print("Using default SmolVLM-500M-Instruct")
            return cls(smolvlm_model_path="HuggingFaceTB/SmolVLM-500M-Instruct")

    # ================== LANGUAGE ENCODING ==================
    def encode_language(self, language_instruction: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize language instructions using SmolVLM tokenizer.

        Parameters
        ----------
        language_instruction : str or List[str]
            A single instruction or a batch of instructions.

        Returns
        -------
        Dict[str, torch.Tensor]
            {"input_ids": tensor of shape [B, L]}
        """
        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]

        inputs = self.tokenizer(
            language_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.language_max_length,
            truncation=True,
        )
        return {"input_ids": inputs["input_ids"]}

    # ================== OPTIMIZED IMAGE ENCODING ==================
    def encode_image(
        self,
        images: Union[List, List[List]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess multi-view images for SmolVLM (384x384 or 512x512).
        
        OPTIMIZED VERSION:
        - Uses torch operations instead of PIL for speed
        - Batches all images together for efficient processing
        - Avoids repeated CPU-GPU transfers

        Parameters
        ----------
        images : List or List[List]
            Single sample: [img1, img2, ...]
            Batch: [[img1a, img1b], [img2a, img2b, img2c], ...]
            Each image may be a PIL.Image, NumPy array, or torch.Tensor.

        Returns
        -------
        Dict[str, torch.Tensor]
            {
              "image_input": tensor [B, num_views, C, H, W],
              "image_mask": tensor [B, num_views]
            }
        """
        # Normalize to batch form
        if not isinstance(images[0], (list, tuple)):
            images = [images]

        batch_imgs, batch_masks = [], []

        for sample_imgs in images:
            processed_tensors = []
            
            for img in sample_imgs:
                # Convert to torch tensor [C, H, W] in float32, range [0, 1]
                if isinstance(img, np.ndarray):
                    # numpy array [H, W, C] -> torch [C, H, W]
                    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                elif isinstance(img, torch.Tensor):
                    if img.dim() == 3 and img.shape[0] != 3:
                        # [H, W, C] -> [C, H, W]
                        tensor = img.permute(2, 0, 1).float()
                    else:
                        tensor = img.float()
                    # Normalize to [0, 1] if needed
                    if tensor.max() > 1.0:
                        tensor = tensor / 255.0
                elif isinstance(img, Image.Image):
                    # PIL Image -> numpy -> torch
                    np_img = np.array(img)
                    tensor = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
                
                # Fast resize using torch (bicubic interpolation)
                _, h, w = tensor.shape
                if h != self.image_size or w != self.image_size:
                    tensor = F.interpolate(
                        tensor.unsqueeze(0),  # [1, C, H, W]
                        size=(self.image_size, self.image_size),
                        mode='bicubic',
                        align_corners=False,
                        antialias=True,
                    ).squeeze(0)  # [C, H, W]
                
                # Normalize with cached mean/std
                tensor = (tensor - self._image_mean.squeeze(0)) / self._image_std.squeeze(0)
                
                processed_tensors.append(tensor)
            
            # Stack views for this sample
            V_exist = len(processed_tensors)
            if V_exist > 0:
                processed = torch.stack(processed_tensors, dim=0)  # [V, C, H, W]
            else:
                processed = torch.zeros(0, 3, self.image_size, self.image_size)

            # Pad to num_views
            if V_exist < self.num_views:
                processed = torch.cat(
                    [processed,
                     processed.new_zeros(self.num_views - V_exist, *processed.shape[1:])],
                    dim=0,
                )

            # Mask: True for valid slots
            image_mask = torch.zeros(self.num_views, dtype=torch.bool)
            image_mask[:V_exist] = True

            batch_imgs.append(processed)
            batch_masks.append(image_mask)

        image_input = torch.stack(batch_imgs, dim=0)  # [B, num_views, C, H, W]
        image_mask = torch.stack(batch_masks, dim=0)  # [B, num_views]

        return {"image_input": image_input, "image_mask": image_mask}

    # ================== LEGACY (SLOW) IMAGE ENCODING ==================
    def encode_image_legacy(
        self,
        images: Union[List, List[List]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        LEGACY VERSION - Kept for compatibility testing.
        Uses HuggingFace image_processor (slower but guaranteed compatible).
        """
        # Normalize to batch form
        if not isinstance(images[0], (list, tuple)):
            images = [images]

        batch_imgs, batch_masks = [], []

        for sample_imgs in images:
            # Convert to PIL if needed and resize to target size
            processed_imgs = []
            for img in sample_imgs:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                elif isinstance(img, torch.Tensor):
                    if img.dim() == 3:
                        img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    else:
                        img = Image.fromarray(img.cpu().numpy().astype(np.uint8))
                
                # Resize to target size if needed
                if img.size != (self.image_size, self.image_size):
                    img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
                processed_imgs.append(img)
            
            # Use SmolVLM image processor
            processed = self.image_processor(
                processed_imgs, 
                return_tensors="pt",
                **kwargs
            )["pixel_values"]
            
            V_exist = processed.size(0)

            # Pad to num_views
            if V_exist < self.num_views:
                processed = torch.cat(
                    [processed,
                     processed.new_zeros(self.num_views - V_exist, *processed.shape[1:])],
                    dim=0,
                )

            # Mask: True for valid slots
            image_mask = torch.zeros(self.num_views, dtype=torch.bool, device=processed.device)
            image_mask[:V_exist] = True

            batch_imgs.append(processed)
            batch_masks.append(image_mask)

        image_input = torch.stack(batch_imgs, dim=0)  # [B, num_views, C, H, W]
        image_mask = torch.stack(batch_masks, dim=0)  # [B, num_views]

        return {"image_input": image_input, "image_mask": image_mask}

    # ================== COMBINED CALL ==================
    def __call__(
        self,
        images: Optional[Union[List, List[List]]] = None,
        language_instruction: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Combine image and text encoding into unified multimodal input.

        Parameters
        ----------
        images : List or List[List], optional
            Single-sample or batched multi-view images.
        language_instruction : str or List[str], optional
            Corresponding text instructions.

        Returns
        -------
        Dict[str, torch.Tensor]
            {
              "input_ids": [B, L],
              "image_input": [B, num_views, C, H, W],
              "image_mask": [B, num_views]
            }
        """
        outputs: Dict[str, Any] = {}

        if language_instruction is not None:
            outputs.update(self.encode_language(language_instruction))

        if images is not None:
            outputs.update(self.encode_image(images, **kwargs))

        # Sanity check
        if "input_ids" in outputs and "image_input" in outputs:
            assert outputs["input_ids"].size(0) == outputs["image_input"].size(0), (
                f"Batch mismatch: text batch {outputs['input_ids'].size(0)} "
                f"!= image batch {outputs['image_input'].size(0)}"
            )
        return outputs

    def apply_chat_template(
        self,
        images: List[Image.Image],
        text: str,
        add_generation_prompt: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply SmolVLM chat template for multi-image input.
        
        This is useful for inference when you want to use SmolVLM's
        native chat template format.
        
        Parameters
        ----------
        images : List[Image.Image]
            List of PIL images.
        text : str
            Text prompt.
        add_generation_prompt : bool
            Whether to add generation prompt.
            
        Returns
        -------
        Dict with input_ids, attention_mask, pixel_values, etc.
        """
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})
        
        messages = [{"role": "user", "content": content}]
        
        inputs = self._smolvlm_processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
