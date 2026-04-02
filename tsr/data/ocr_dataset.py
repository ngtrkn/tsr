"""
OCR Dataset for character-level text recognition.

Reads a pipe-delimited text file:
    image_path|ocr_text

Produces sequences compatible with the TSR collate_fn so that OCR and
TSR datasets can be mixed in the same DataLoader via ConcatDataset.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .serialization import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, LINESEP_TOKEN
from tsr.utils.vocab import extend_vocab_from_texts


class OCRDataset(Dataset):
    """
    Dataset for OCR text recognition from cropped images.

    Input format — a text file with one sample per line:
        relative/path/to/image.png|Ground Truth Text

    The pipe ``|`` is split only on the *first* occurrence so the GT text
    may itself contain pipes.

    Sequence layout:
        input_ids:  <BOS>  c1 c2 ... cN
        token_ids:   c1    c2 c3 ... <EOS>

    All character tokens are marked as content; <BOS>/<EOS> as structure.
    No bounding-box fields are emitted (the collate_fn fills zeros when
    a mixed batch contains TSR items that do have bboxes).
    """

    def __init__(
        self,
        data_path: str,
        vocab: Dict[str, int],
        image_size: Tuple[int, int] = (512, 640),
        augment: bool = False,
        max_samples: Optional[int] = None,
        base_dir: Optional[str] = None,
    ):
        """
        Args:
            data_path: Path to the pipe-delimited text file.
            vocab:     Shared vocabulary dict (will be **extended in-place**
                       with any new characters found in the OCR texts).
            image_size: (W, H) to resize images to.
            augment:   Reserved for future augmentation.
            max_samples: Cap the number of samples (for debugging).
            base_dir:  Root directory for resolving relative image paths.
                       Defaults to the parent of *data_path*.
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.augment = augment
        self.vocab = vocab

        if base_dir is not None:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = self.data_path.parent

        self.samples: List[Tuple[str, str]] = []
        self._load_samples()

        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        # Extend vocab with characters from all GT texts
        texts = [text for _, text in self.samples]
        added = extend_vocab_from_texts(self.vocab, texts)
        if added > 0:
            print(f"[OCRDataset] Extended vocab by {added} new tokens "
                  f"(total {len(self.vocab)})")

        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def _load_samples(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\n\r")
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) != 2:
                    print(f"[OCRDataset] Skipping malformed line {lineno}: {line!r}")
                    continue
                img_rel, text = parts
                img_path = str(self.base_dir / img_rel) if not Path(img_rel).is_absolute() else img_rel
                self.samples.append((img_path, text))

        print(f"[OCRDataset] Loaded {len(self.samples)} samples from {self.data_path}")

    def _load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (image - mean) / std

    def _tokenize(self, text: str) -> List[str]:
        """Character-level tokenization with <LineSep> for newlines."""
        tokens = []
        for ch in text:
            if ch == "\n":
                tokens.append(LINESEP_TOKEN)
            else:
                tokens.append(ch)
        return tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, text = self.samples[idx]

        image = self._load_image(img_path)

        content_tokens = self._tokenize(text)
        target_tokens = content_tokens + [EOS_TOKEN]
        input_tokens = [BOS_TOKEN] + content_tokens

        pad_id = self.vocab[PAD_TOKEN]
        input_ids = torch.tensor(
            [self.vocab.get(t, pad_id) for t in input_tokens], dtype=torch.long)
        token_ids = torch.tensor(
            [self.vocab.get(t, pad_id) for t in target_tokens], dtype=torch.long)

        seq_len = len(input_tokens)
        struct_mask = torch.zeros(seq_len, dtype=torch.bool)
        cont_mask = torch.zeros(seq_len, dtype=torch.bool)

        # BOS is structural; content chars are content
        struct_mask[0] = True
        cont_mask[1:] = True

        return {
            "image": image,
            "input_ids": input_ids,
            "token_ids": token_ids,
            "structure_mask": struct_mask,
            "content_mask": cont_mask,
        }
