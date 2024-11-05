# Utils/optuna/optimize.py
import optuna                                                                                          
import logging                                                                                         
import torch                                                                                           
import sys                                                                                             
from pathlib import Path                                                                               
                                                                                                    
# Add project root to path to ensure modules can be imported correctly                                 
project_root = Path(__file__).resolve().parents[2]  # Navigate two levels up to root
sys.path.append(str(project_root))                                                                     
                                                                                                    
from config import Config                                                                              
from Utils.optuna.objective import create_objective                                                    
                                                                                                    
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

        # Optionally delete existing study
        if delete_study:
            logger.info(f"Deleting existing study '{config.optuna.study_name}'")
            optuna.delete_study(study_name=config.optuna.study_name, storage=config.optuna.storage_url)
            logger.info(f"Deleted study '{config.optuna.study_name}'")

        # Create study
        study = optuna.create_study(
            study_name=config.optuna.study_name,
            storage=config.optuna.storage_url,
            direction="minimize",
            load_if_exists=True
        )

        # Create objective
        objective = create_objective(config)

        # Run optimization
        logger.info(f"Running {config.optuna.n_trials} trials")
        study.optimize(objective, n_trials=config.optuna.n_trials)

        # Retrieve metrics
        metrics = TrialMetrics(
            val_loss=float(trainer.callback_metrics.get("val_loss", float('inf'))),
            train_loss=float(trainer.callback_metrics.get("train_loss", float('inf'))),
            val_accuracy=float(trainer.callback_metrics.get("val_accuracy", 0.0))
        )

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise
                                                                                                    
if __name__ == "__main__":                                                                             
    try:                                                                                               
        # Import argparse here to ensure it's only loaded when running the script                      
        import argparse                                                                                
        parser = argparse.ArgumentParser(description="Run Optuna optimization for JARC Reactor")       
        parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials to run") 
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")               
        parser.add_argument("--delete_study", action="store_true", help="Delete existing Optuna study before running")
        args = parser.parse_args()
                                                                                                    
        if args.debug:                                                                                 
            logging.getLogger().setLevel(logging.DEBUG)                                                
            logger.debug("Debug mode enabled")                                                         
                                                                                                    
        # Create config and update the number of trials based on user input                            
        config = Config()                                                                              
        config.optuna.n_trials = args.n_trials                                                         
                                                                                                    
        # Run the optimization                                                                         
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
