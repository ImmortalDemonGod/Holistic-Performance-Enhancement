# JARC-Reactor/Utils/context_data.py
from dataclasses import dataclass
from typing import Optional
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class ContextPair:
    context_input: Optional[torch.Tensor] = None
    context_output: Optional[torch.Tensor] = None

    def __post_init__(self):
        expected_shape = (30, 30)  # Adjust based on your specific requirements
        if self.context_input is not None:
            if self.context_input.shape != expected_shape:
                raise ValueError(f"Context input must have shape {expected_shape}, got {self.context_input.shape}")
        else:
            logger.warning("context_input is None. Using default tensor.")
            self.context_input = torch.zeros(expected_shape)

        if self.context_output is not None:
            if self.context_output.shape != expected_shape:
                raise ValueError(f"Context output must have shape {expected_shape}, got {self.context_output.shape}")
        else:
            logger.warning("context_output is None. Using default tensor.")
            self.context_output = torch.zeros(expected_shape)
