import logging
from pathlib import Path
import torch
import json
from datetime import datetime

class DebugLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / 'debug_logs'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logger
        self.logger = logging.getLogger('evaluation_debug')
        self.logger.setLevel(logging.DEBUG)
        
        # Create unique log file for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler('jarc_reactor/logs' / f'debug_{timestamp}.log')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)
    
    def log_tensor(self, name: str, tensor: torch.Tensor):
        """Log tensor shape, type, and first few values"""
        info = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'first_values': tensor.flatten()[:5].tolist(),
            'min': float(tensor.min()),
            'max': float(tensor.max()),
            'mean': float(tensor.mean()),
            'has_nan': bool(torch.isnan(tensor).any()),
            'has_inf': bool(torch.isinf(tensor).any())
        }
        self.logger.debug(f"{name} tensor info: {json.dumps(info, indent=2)}")
    
    def log_batch(self, batch, prefix: str = 'batch'):
        """Log information about a complete batch"""
        self.logger.debug(f"\n{'='*50}\nProcessing {prefix}")
        for i, item in enumerate(batch):
            if isinstance(item, torch.Tensor):
                self.log_tensor(f'{prefix}_item_{i}', item)
            else:
                self.logger.debug(f'{prefix}_item_{i} type: {type(item)}')
