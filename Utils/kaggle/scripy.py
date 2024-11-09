# script.py
import sys
import os
import kagglehub
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Ensure we're in the Kaggle environment
        if not os.path.exists('/kaggle'):
            raise RuntimeError("This script must be run in a Kaggle environment")
            
        logger.info("Running submission script...")
        
        # Install required packages
        logger.info("Installing dependencies...")
        os.system('pip install pytorch-lightning orjson tqdm')
        
        # Import and run submission
        from kaggle_submission import load_model_and_create_submission
        load_model_and_create_submission()
        
        logger.info("Submission completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error running submission: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())