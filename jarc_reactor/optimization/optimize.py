# Utils/optuna/optimize.py
import optuna                                                                                          
import logging                                                                                         
import torch                                                                                           
import sys                                                                                             
from pathlib import Path                                                                               
                                                                                                    
# Add project root to path to ensure modules can be imported correctly                                 
project_root = Path(__file__).resolve().parents[2]  # Navigate two levels up to root
sys.path.append(str(project_root))                                                                     
                                                                                                    
from jarc_reactor.config import Config
from jarc_reactor.utils.train import TransformerTrainer
from pytorch_lightning import Trainer
from jarc_reactor.optimization.objective import create_objective
from jarc_reactor.data.data_preparation import prepare_data
from pytorch_lightning.callbacks import EarlyStopping
                                                                                                    
# Setup logging                                                                                        
logging.basicConfig(                                                                                   
    level=logging.DEBUG,                                                                               
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',                                     
    handlers=[                                                                                         
        logging.FileHandler('optuna_optimization.log'),                                                
        logging.StreamHandler()                                                                        
    ]                                                                                                  
)                                                                                                      
logger = logging.getLogger(__name__)                                                                   
                                                                                                    
def run_optimization(config, delete_study=False):
    """Main optimization entry point"""
    try:
        logger.info("Starting optimization")
        logger.debug(f"Using config: {vars(config.optuna)}")

        # Enable synthetic data for optimization
        config.training.include_synthetic_training_data = True
        logger.info("Using synthetic data for hyperparameter optimization")

        # Optionally delete existing study
        if delete_study:
            logger.info(f"Deleting existing study '{config.optuna.study_name}'")
            optuna.delete_study(study_name=config.optuna.study_name, storage=config.optuna.storage_url)
            logger.info(f"Deleted study '{config.optuna.study_name}'")

        # Load datasets once before defining the objective
        logger.info("Loading datasets once for all trials.")
        train_dataset, val_dataset = prepare_data(directory=config.training.training_data_dir, return_datasets=True)
        
        # Log dataset sizes
        logger.info(f"Loaded {len(train_dataset)} training examples")
        logger.info(f"Loaded {len(val_dataset)} validation examples")

        study = optuna.create_study(
            study_name=config.optuna.study_name,
            storage=config.optuna.storage_url,
            direction="minimize",
            load_if_exists=True
        )

        # Pass preloaded datasets to create_objective
        objective = create_objective(config, train_dataset, val_dataset)

        # Run optimization
        logger.info(f"Running {config.optuna.n_trials} trials")
        study.optimize(objective, n_trials=config.optuna.n_trials)

        # Add logging for optimization results
        logger.info("Optimization completed")
        best_trial = study.best_trial
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best value: {best_trial.value}")
        logger.info("Best params:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
        
        return best_trial

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise
                                                                                                    
if __name__ == "__main__":                                                                             
    try:                                                                                               
        # Instantiate the Config class first
        config = Config()

        # Import argparse here to ensure it's only loaded when running the script                      
        import argparse                                                                                
        parser = argparse.ArgumentParser(description="Run Optuna optimization for JARC Reactor")       
        parser.add_argument("--n_trials", type=int, default=config.optuna.n_trials, help="Number of Optuna trials to run") 
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")               
        parser.add_argument("--delete_study", action="store_true", help="Delete existing Optuna study before running")
        args = parser.parse_args()
                                                                                                    
        if args.debug:                                                                                 
            logging.getLogger().setLevel(logging.DEBUG)                                                
            logger.debug("Debug mode enabled")                                                         
                                                                                                    
        # Update the number of trials based on user input                            
        config.optuna.n_trials = args.n_trials                                                         
                                                                                                    
        # Run the optimization with the updated config
        best_trial = run_optimization(config, delete_study=args.delete_study)                                                          
        print(f"\nOptimization completed. Best trial: {best_trial.number}")                            
        print(f"Best parameters: {best_trial.params}")                                                 
                                                                                                    
    except KeyboardInterrupt:                                                                          
        logger.info("Optimization interrupted by user")                                                
        sys.exit(0)                                                                                    
    except Exception as e:                                                                             
        logger.error(f"Script failed: {str(e)}")                                                       
        logger.error("Stack trace:", exc_info=True)                                                    
        sys.exit(1)  
