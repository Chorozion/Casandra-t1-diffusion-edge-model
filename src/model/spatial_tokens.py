"""
Sophia T1 — Spatial Token System

Instead of encoding bounding boxes as text strings like "0.35, 0.72, 0.15, 0.20",
we add dedicated coordinate tokens to the vocabulary. This lets the model learn
spatial relationships as first-class concepts, not text parsing.

Vocab layout (32768 total):
  0-31999:      Standard BPE text tokens
  32000-32099:  X coordinate bins (0.00 to 0.99 in 1% increments)
  32100-32199:  Y coordinate bins
  32200-32299:  Width bins
  32300-32399:  Height bins
  32400-32409:  Depth tokens (foreground, midground, background, etc.)
  32410-32419:  Spatial relation tokens (above, below, left_of, etc.)
  32420-32499:  Reserved spatial tokens
  32500-32599:  OCR region tokens (header, body, footer, etc.)
  32600-32699:  Scene type tokens
  32700-32759:  Special tokens (BBOX_START, BBOX_END, OBJECT_START, etc.)
  32760-32766:  Reserved
  32767:        PAD token
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json


# Token ranges
X_START = 32000
Y_START = 32100
W_START = 32200
H_START = 32300
DEPTH_START = 32400
RELATION_START = 32410
OCR_REGION_START = 32500
SCENE_TYPE_START = 32600
SPECIAL_START = 32700

# Special tokens
BBOX_START = 32700
BBOX_END = 32701
OBJECT_START = 32702
OBJECT_END = 32703
SCENE_START = 32704
SCENE_END = 32705
OCR_START = 32706
OCR_END = 32707
LABEL_START = 32708
LABEL_END = 32709
CONFIDENCE_HIGH = 32710  # > 0.9
CONFIDENCE_MED = 32711   # 0.7-0.9
CONFIDENCE_LOW = 32712   # < 0.7

# Depth tokens
DEPTH_TOKENS = {
    "foreground": DEPTH_START,
    "midground": DEPTH_START + 1,
    "background": DEPTH_START + 2,
    "overhead": DEPTH_START + 3,
    "underground": DEPTH_START + 4,
}

# Spatial relation tokens
RELATION_TOKENS = {
    "above": RELATION_START,
    "below": RELATION_START + 1,
    "left_of": RELATION_START + 2,
    "right_of": RELATION_START + 3,
    "inside": RELATION_START + 4,
    "overlapping": RELATION_START + 5,
    "near": RELATION_START + 6,
    "behind": RELATION_START + 7,
    "in_front_of": RELATION_START + 8,
}

# OCR region tokens
OCR_REGIONS = {
    "header": OCR_REGION_START,
    "body": OCR_REGION_START + 1,
    "footer": OCR_REGION_START + 2,
    "sidebar": OCR_REGION_START + 3,
    "caption": OCR_REGION_START + 4,
    "title": OCR_REGION_START + 5,
    "table": OCR_REGION_START + 6,
    "signature": OCR_REGION_START + 7,
    "total": OCR_REGION_START + 8,
    "date": OCR_REGION_START + 9,
}

# Scene type tokens
SCENE_TYPES = {
    "indoor": SCENE_TYPE_START,
    "outdoor": SCENE_TYPE_START + 1,
    "aerial": SCENE_TYPE_START + 2,
    "underwater": SCENE_TYPE_START + 3,
    "medical": SCENE_TYPE_START + 4,
    "document": SCENE_TYPE_START + 5,
    "satellite": SCENE_TYPE_START + 6,
    "microscope": SCENE_TYPE_START + 7,
}


def coord_to_token(value: float, offset: int) -> int:
    """Convert a 0.0-1.0 coordinate to a token ID."""
    bin_idx = max(0, min(99, int(value * 100)))
    return offset + bin_idx


def token_to_coord(token_id: int, offset: int) -> float:
    """Convert a token ID back to a 0.0-1.0 coordinate."""
    bin_idx = token_id - offset
    return bin_idx / 100.0


def encode_bbox(x: float, y: float, w: float, h: float) -> List[int]:
    """Encode a bounding box as spatial tokens."""
    return [
        BBOX_START,
        coord_to_token(x, X_START),
        coord_to_token(y, Y_START),
        coord_to_token(w, W_START),
        coord_to_token(h, H_START),
        BBOX_END,
    ]


def decode_bbox(tokens: List[int]) -> Optional[Tuple[float, float, float, float]]:
    """Decode spatial tokens back to a bounding box."""
    if len(tokens) < 6 or tokens[0] != BBOX_START or tokens[5] != BBOX_END:
        return None
    return (
        token_to_coord(tokens[1], X_START),
        token_to_coord(tokens[2], Y_START),
        token_to_coord(tokens[3], W_START),
        token_to_coord(tokens[4], H_START),
    )


def encode_object(label_tokens: List[int], bbox: List[float],
                   depth: str = "midground", confidence: float = 0.9) -> List[int]:
    """Encode a detected object as a token sequence."""
    tokens = [OBJECT_START]
    tokens.append(LABEL_START)
    tokens.extend(label_tokens)  # BPE tokens for the label text
    tokens.append(LABEL_END)
    tokens.extend(encode_bbox(*bbox))

    # Depth
    if depth in DEPTH_TOKENS:
        tokens.append(DEPTH_TOKENS[depth])

    # Confidence
    if confidence > 0.9:
        tokens.append(CONFIDENCE_HIGH)
    elif confidence > 0.7:
        tokens.append(CONFIDENCE_MED)
    else:
        tokens.append(CONFIDENCE_LOW)

    tokens.append(OBJECT_END)
    return tokens


def encode_spatial_scene(scene_data: dict) -> List[int]:
    """
    Convert a spatial scene JSON to a token sequence.

    Input format:
    {
        "scene": "office desk with...",
        "objects": [{"label": "cup", "bbox": [0.3, 0.5, 0.1, 0.1], "depth": "foreground", "confidence": 0.95}],
        "spatial_relationships": ["cup is left of monitor"],
        "scene_type": "indoor"
    }
    """
    tokens = [SCENE_START]

    # Scene type
    scene_type = scene_data.get("scene_type", "indoor")
    if scene_type in SCENE_TYPES:
        tokens.append(SCENE_TYPES[scene_type])

    # Objects
    for obj in scene_data.get("objects", []):
        label = obj.get("label", "object")
        # Convert label to character tokens for now (BPE tokenizer will replace this)
        label_tokens = [min(ord(c), 31999) for c in label]
        bbox = obj.get("bbox", [0.5, 0.5, 0.1, 0.1])
        depth = obj.get("depth", "midground")
        conf = obj.get("confidence", 0.9)
        tokens.extend(encode_object(label_tokens, bbox, depth, conf))

    # Spatial relationships
    for rel_str in scene_data.get("spatial_relationships", []):
        for rel_name, rel_token in RELATION_TOKENS.items():
            if rel_name.replace("_", " ") in rel_str:
                tokens.append(rel_token)
                break

    tokens.append(SCENE_END)
    return tokens


if __name__ == "__main__":
    # Test encoding/decoding
    bbox = encode_bbox(0.35, 0.72, 0.15, 0.20)
    print(f"Bbox tokens: {bbox}")
    decoded = decode_bbox(bbox)
    print(f"Decoded: {decoded}")

    # Test scene encoding
    scene = {
        "scene": "office desk",
        "objects": [
            {"label": "cup", "bbox": [0.3, 0.5, 0.1, 0.1], "depth": "foreground", "confidence": 0.95},
            {"label": "monitor", "bbox": [0.1, 0.2, 0.4, 0.3], "depth": "midground", "confidence": 0.92},
        ],
        "spatial_relationships": ["cup is left of monitor"],
        "scene_type": "indoor",
    }
    tokens = encode_spatial_scene(scene)
    print(f"\nScene tokens ({len(tokens)} tokens): {tokens[:20]}...")

    # Show spatial token stats
    spatial_tokens = sum(1 for t in tokens if t >= 32000)
    text_tokens = sum(1 for t in tokens if t < 32000)
    print(f"Spatial tokens: {spatial_tokens}, Text tokens: {text_tokens}")
