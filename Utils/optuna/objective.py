import torch                                                                                           
import logging                                                                                         
from dataclasses import dataclass                                                                      
import optuna                                                                                          
from config import Config                                                                              
from train import TransformerTrainer                                                                   
from pytorch_lightning import Trainer                                                                  
from pytorch_lightning.callbacks import EarlyStopping                                                  
                                                                                                    
# Setup logging                                                                                        
logging.basicConfig(level=logging.DEBUG)                                                               
logger = logging.getLogger(__name__)                                                                   
                                                                                                    
@dataclass                                                                                             
class TrialMetrics:                                                                                    
    val_loss: float                                                                                    
    train_loss: float                                                                                  
    val_accuracy: float                                                                                
    memory_used: float = 0.0                                                                           
                                                                                                    
    def log_metrics(self):                                                                             
        """Print all metrics for debugging"""                                                          
        logger.debug("Trial Metrics:")                                                                 
        logger.debug(f"  Val Loss: {self.val_loss:.4f}")                                               
        logger.debug(f"  Train Loss: {self.train_loss:.4f}")                                           
        logger.debug(f"  Val Accuracy: {self.val_accuracy:.4f}")                                       
        logger.debug(f"  Memory Used: {self.memory_used:.2f} MB")                                      
                                                                                                    
def create_trial_config(trial, base_config):                                                           
    """Create a new config with trial-suggested values"""                                              
    try:                                                                                               
        logger.debug("Creating trial config")                                                          
                                                                                                    
        # Get ranges from OptunaConfig                                                                 
        ranges = base_config.optuna.param_ranges                                                       
                                                                                                    
        # Create new config objects based on existing classes                                          
        model_config = base_config.model.__class__()                                                   
        training_config = base_config.training.__class__()                                             
                                                                                                    
        # Suggest hyperparameters using Optuna                                                         
        model_config.d_model = trial.suggest_int("d_model", *ranges["d_model"], step=4)                        
        model_config.decoder_layers = trial.suggest_int("decoder_layers", *ranges["decoder_layers"])   
        model_config.heads = trial.suggest_int("heads", *ranges["heads"])                              
        model_config.d_ff = trial.suggest_int("d_ff", *ranges["d_ff"])                                 
        model_config.dropout = trial.suggest_float("dropout", *ranges["dropout"])                      
                                                                                                    
        training_config.batch_size = trial.suggest_int("batch_size", *ranges["batch_size"])            
        training_config.learning_rate = trial.suggest_float(                                           
            "learning_rate", *ranges["learning_rate"], log=True                                        
        )                                                                                              
                                                                                                    
        logger.debug(f"Trial config created with params: {trial.params}")                              
        return Config(model=model_config, training=training_config)                                    
                                                                                                    
    except Exception as e:                                                                             
        logger.error(f"Error creating trial config: {str(e)}")                                         
        raise                                                                                          
                                                                                                    
def create_objective(base_config):                                                                     
    """Creates an objective function with closure over base_config"""                                  
                                                                                                    
    def objective(trial):                                                                              
        logger.debug(f"\nStarting trial {trial.number}")                                               
        try:                                                                                           
            # Create trial-specific config                                                             
            trial_config = create_trial_config(trial, base_config)                                     
                                                                                                    
            # Log memory usage before training                                                         
            if torch.cuda.is_available():                                                              
                torch.cuda.empty_cache()                                                               
                memory_before = torch.cuda.memory_allocated() / 1024**2                                
                logger.debug(f"GPU memory before training: {memory_before:.2f} MB")                    
                                                                                                    
            # Initialize the model with trial-specific hyperparameters                                 
            model = TransformerTrainer(                                                                
                input_dim=trial_config.model.input_dim,                                                
                d_model=trial_config.model.d_model,                                                    
                encoder_layers=trial_config.model.encoder_layers,                                      
                decoder_layers=trial_config.model.decoder_layers,                                      
                heads=trial_config.model.heads,                                                        
                d_ff=trial_config.model.d_ff,                                                          
                output_dim=trial_config.model.output_dim,                                              
                learning_rate=trial_config.training.learning_rate,                                     
                include_sythtraining_data=trial_config.training.include_sythtraining_data              
            )                                                                                          
                                                                                                    
            # Setup early stopping callback                                                            
            early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")                     
                                                                                                    
            # Initialize Trainer                                                                       
            trainer = Trainer(                                                                         
                max_epochs=10,  # Start with fewer epochs for testing                                  
                callbacks=[early_stop],                                                                
                enable_progress_bar=True,                                                              
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',                             
                devices=1                                                                              
            )                                                                                          
                                                                                                    
            # Train the model                                                                          
            logger.debug("Starting training")                                                          
            trainer.fit(model)                                                                         
                                                                                                    
            # Retrieve metrics                                                                         
            metrics = TrialMetrics(                                                                    
                val_loss=trainer.callback_metrics.get("val_loss", float('inf')),                       
                train_loss=trainer.callback_metrics.get("train_loss", float('inf')),                   
                val_accuracy=trainer.callback_metrics.get("val_accuracy", 0.0)                         
            )                                                                                          
                                                                                                    
            # Log memory usage after training                                                          
            if torch.cuda.is_available():                                                              
                memory_after = torch.cuda.memory_allocated() / 1024**2                                 
                metrics.memory_used = memory_after - memory_before                                     
                logger.debug(f"GPU memory used: {metrics.memory_used:.2f} MB")                         
                                                                                                    
            # Log metrics                                                                              
            metrics.log_metrics()                                                                      
                                                                                                    
            # Prune trial if validation loss is invalid                                                
            if not metrics.val_loss or torch.isnan(torch.tensor(metrics.val_loss)):                    
                logger.warning("Invalid validation loss detected")                                     
                raise optuna.TrialPruned()                                                             
                                                                                                    
            return metrics.val_loss                                                                    
                                                                                                    
        except Exception as e:                                                                         
            logger.error(f"Trial {trial.number} failed: {str(e)}")                                     
            logger.error("Stack trace:", exc_info=True)                                                
            raise optuna.TrialPruned()                                                                 
                                                                                                    
    return objective   
