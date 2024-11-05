import optuna                                                                                          
import logging                                                                                         
import torch                                                                                           
import sys                                                                                             
from pathlib import Path                                                                               
                                                                                                    
# Add project root to path to ensure modules can be imported correctly                                 
project_root = Path(__file__).parent                                                                   
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
                                                                                                    
def run_optimization(config):                                                                          
    """Main optimization entry point"""                                                                
    try:                                                                                               
        logger.info("Starting optimization")                                                           
        logger.debug(f"Using config: {vars(config.optuna)}")                                           
                                                                                                    
        # Log system info                                                                              
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")                                   
        if torch.cuda.is_available():                                                                  
            logger.debug(f"GPU: {torch.cuda.get_device_name(0)}")                                      
            logger.debug(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
                                                                                                    
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
                                                                                                    
        # Log results                                                                                  
        logger.info("Optimization completed")                                                          
        logger.info(f"Best trial: {study.best_trial.number}")                                          
        logger.info(f"Best value: {study.best_trial.value}")                                           
        logger.info("Best params:")                                                                    
        for key, value in study.best_trial.params.items():                                             
            logger.info(f"  {key}: {value}")                                                           
                                                                                                    
        return study.best_trial                                                                        
                                                                                                    
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
        args = parser.parse_args()                                                                     
                                                                                                    
        if args.debug:                                                                                 
            logging.getLogger().setLevel(logging.DEBUG)                                                
            logger.debug("Debug mode enabled")                                                         
                                                                                                    
        # Create config and update the number of trials based on user input                            
        config = Config()                                                                              
        config.optuna.n_trials = args.n_trials                                                         
                                                                                                    
        # Run the optimization                                                                         
        best_trial = run_optimization(config)                                                          
        print(f"\nOptimization completed. Best trial: {best_trial.number}")                            
        print(f"Best parameters: {best_trial.params}")                                                 
                                                                                                    
    except KeyboardInterrupt:                                                                          
        logger.info("Optimization interrupted by user")                                                
        sys.exit(0)                                                                                    
    except Exception as e:                                                                             
        logger.error(f"Script failed: {str(e)}")                                                       
        logger.error("Stack trace:", exc_info=True)                                                    
        sys.exit(1)  
