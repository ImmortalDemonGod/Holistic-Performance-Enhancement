from dataclasses import dataclass
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class ContextPair:
    context_input: torch.Tensor
    context_output: torch.Tensor

    def __post_init__(self):
        expected_shape = (30, 30)  # Adjust based on your specific requirements
        if self.context_input.shape != expected_shape:
            logger.error(f"Context input shape {self.context_input.shape} does not match expected shape {expected_shape}.")
            raise ValueError(f"Context input must have shape {expected_shape}, got {self.context_input.shape}")
        if self.context_output.shape != expected_shape:
            logger.error(f"Context output shape {self.context_output.shape} does not match expected shape {expected_shape}.")
            raise ValueError(f"Context output must have shape {expected_shape}, got {self.context_output.shape}")
