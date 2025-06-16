# test_script.py
import sys
import os
import logging

# Set up logging
from cultivation.utils.logging_config import setup_logging
setup_logging(log_file='submission_debug.log')

logger = logging.getLogger(__name__)

def main():
    # Print basic environment info
    logger.info("Script started")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    try:
        # Try importing key dependencies
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        import kagglehub
        logger.info("kagglehub imported successfully")
        
        # Print Kaggle environment info if available
        if os.path.exists('/kaggle'):
            logger.info("Running in Kaggle environment")
            logger.info(f"Kaggle working dir contents: {os.listdir('/kaggle/working')}")
        else:
            logger.info("Not running in Kaggle environment")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error in script: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())