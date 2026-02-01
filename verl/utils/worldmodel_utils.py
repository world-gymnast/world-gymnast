"""Utils for world model integration."""

import numpy as np
import torch
from PIL import Image
from pathlib import Path

try:
    from world_model_eval.world_model import WorldModel
    from world_model_eval.utils import rescale_bridge_action, predict
except ImportError as e:
    print(f"Warning: can't import world_model_eval: {e}")


def load_png_to_tensor(png_path, target_size=256):
    """
    Load PNG file and convert to tensor format expected by world model.

    Args:
        png_path: Path to PNG file
        target_size: Target image size (default 256x256)

    Returns:
        Tensor of shape (H, W, C) with values in [0, 1] range
    """
    img = Image.open(png_path).convert("RGB")
    img = img.resize((target_size, target_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_array)


def worldmodel_frame_to_vla_input(frame):
    """
    Convert world model output frame to VLA input format.

    Args:
        frame: Tensor of shape (1, 1, H, W, C) from world model, values in [0, 1]

    Returns:
        Dictionary with 'full_image' key
    """
    frame_np = frame[0, 0].cpu().numpy()
    frame_uint8 = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
    return {
        "full_image": frame_uint8,
    }


def pad_and_rescale_action(action, target_dim=10):
    """
    Pad 7D action to 10D and rescale for world model.

    Args:
        action: Action tensor of shape (batch_size, num_chunks, action_dim), typically 7D from VLA
        target_dim: Target action dimension (default 10)

    Returns:
        Rescaled action tensor of shape (batch_size, num_chunks, target_dim)
    """
    batch_size, num_chunks, action_dim = action.shape
    if action_dim < target_dim:
        pad_size = target_dim - action_dim
        padding = torch.zeros(
            (batch_size, num_chunks, pad_size),
            dtype=action.dtype,
            device=action.device,
        )
        action = torch.cat([action, padding], dim=-1)
    action_flat = action.reshape(-1, target_dim)
    action_rescaled = torch.stack([rescale_bridge_action(a) for a in action_flat])
    action = action_rescaled.reshape(batch_size, num_chunks, target_dim)
    return action

def load_world_model(config, rank):
    """
    Load and initialize world model.

    Args:
        config: Configuration for the world model

    Returns:
        Initialized WorldModel instance
    """
    world_model = WorldModel(
        checkpoint_path=config.checkpoint,
        use_pixel_rope=config.use_pixel_rope,
        default_cfg=config.default_cfg,
        rank=rank,
    )

    return world_model