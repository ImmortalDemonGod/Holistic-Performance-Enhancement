# jarc_reactor/evaluation/logger.py
import logging
from pathlib import Path
from datetime import datetime

class EvaluationLogger:
    @staticmethod
    def setup_logging(log_dir='cultivation/systems/arc_reactor/logs/evaluation', logger_name='evaluation'):
        """Setup detailed logging configuration."""
        Path(log_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(log_dir) / f'evaluation_{timestamp}.log'
        
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Prevent adding multiple handlers if logger already has handlers
        if not logger.handlers:
            # File handler for debugging
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(fh)
            
            # Console handler for info
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(ch)
        
        return logger